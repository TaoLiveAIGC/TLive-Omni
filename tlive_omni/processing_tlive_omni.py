import copy
import re
import threading
from contextlib import contextmanager
from typing import Optional, Union

import numpy as np

from transformers.audio_utils import AudioInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, VideosKwargs
from transformers.tokenization_utils_base import TextInput
from transformers.utils import auto_docstring
from transformers.video_utils import VideoInput

from .configuration_tlive_omni import SUPPORTED_TRANSFORMERS_VERSION

QWEN35_IMAGE_SIZE = {
    "shortest_edge": 65536,
    "longest_edge": 16777216,
}
QWEN35_VIDEO_SIZE = {
    "shortest_edge": 4096,
    "longest_edge": 25165824,
}
NATIVE_VIDEO_FPS = 1.0
NATIVE_VIDEO_MIN_FRAMES = 2
NATIVE_VIDEO_MAX_FRAMES = 768
NATIVE_PATCH_SIZE = 16
NATIVE_TEMPORAL_PATCH_SIZE = 2
NATIVE_MERGE_SIZE = 2
NATIVE_IMAGE_MEAN = [0.5, 0.5, 0.5]
NATIVE_IMAGE_STD = [0.5, 0.5, 0.5]
_VIDEO_PROCESSOR_LOCK = threading.RLock()


def _validate_video_backend(backend):
    if backend is None:
        return None
    if not isinstance(backend, str) or not backend:
        raise TypeError(f"video_backend must be a non-empty string or None, got {backend!r}")

    from transformers.video_utils import VIDEO_DECODERS

    if backend not in VIDEO_DECODERS:
        valid = ", ".join(sorted(VIDEO_DECODERS))
        raise ValueError(f"Invalid video_backend {backend!r}; expected one of: {valid}")
    return backend


@contextmanager
def _force_transformers_video_backend(backend):
    with _VIDEO_PROCESSOR_LOCK:
        if backend is None:
            yield
            return

        import transformers.video_processing_utils as video_processing_utils
        import transformers.video_utils as video_utils

        modules = [video_utils, video_processing_utils]
        try:
            import transformers.models.qwen3_vl.video_processing_qwen3_vl as qwen3_vl_video_processing
        except ImportError:
            qwen3_vl_video_processing = None
        if qwen3_vl_video_processing is not None:
            modules.append(qwen3_vl_video_processing)

        original_load_video = video_utils.load_video
        originals = {}

        def forced_load_video(video, *args, **kwargs):
            args = list(args)
            if args and isinstance(args[0], str) and args[0] in {
                "torchcodec",
                "torchvision",
                "decord",
                "pyav",
                "opencv",
            }:
                args[0] = backend
                kwargs.pop("backend", None)
            else:
                kwargs["backend"] = backend
            return original_load_video(video, *args, **kwargs)

        for module in modules:
            if hasattr(module, "load_video"):
                originals[module] = module.load_video
                module.load_video = forced_load_video
        try:
            yield
        finally:
            for module, original in originals.items():
                module.load_video = original


def _load_video_audio(video, sampling_rate):
    import av

    chunks = []
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=sampling_rate)
    with av.open(_normalize_media_path(video)) as container:
        if not container.streams.audio:
            raise ValueError(f"Video has no audio track: {video}")
        for frame in container.decode(container.streams.audio[0]):
            converted_frames = resampler.resample(frame)
            if not isinstance(converted_frames, list):
                converted_frames = [converted_frames]
            chunks.extend(
                converted.to_ndarray().reshape(-1)
                for converted in converted_frames
                if converted is not None
            )
        flushed_frames = resampler.resample(None)
        if not isinstance(flushed_frames, list):
            flushed_frames = [flushed_frames]
        chunks.extend(
            converted.to_ndarray().reshape(-1)
            for converted in flushed_frames
            if converted is not None
        )
    if not chunks:
        raise ValueError(f"Could not decode the audio track from video: {video}")
    return np.concatenate(chunks)


@contextmanager
def _force_transformers_video_audio_loader(enabled):
    with _VIDEO_PROCESSOR_LOCK:
        if not enabled:
            yield
            return

        import transformers.processing_utils as processing_utils

        original_load_audio = processing_utils.load_audio

        def load_video_audio(video, sampling_rate=16000, timeout=None):
            del timeout
            return _load_video_audio(video, sampling_rate)

        processing_utils.load_audio = load_video_audio
        try:
            yield
        finally:
            processing_utils.load_audio = original_load_audio


def _merge_size(default_size, size):
    if size is None:
        return dict(default_size)
    if not isinstance(size, dict):
        raise TypeError(f"size must be a dict, got {type(size).__name__}")
    merged = dict(default_size)
    merged.update(size)
    return merged


def _normalize_media_path(media):
    if isinstance(media, str) and media.startswith("file://"):
        return media[len("file://"):]
    return media


def normalize_media_paths(media):
    if isinstance(media, (list, tuple)):
        return type(media)(normalize_media_paths(x) for x in media)
    return _normalize_media_path(media)


def get_video_metadata_value(metadata, key, default=None):
    if metadata is None:
        return default
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return getattr(metadata, key, default)


def _conversation_media_modes(conversations):
    modes = set()
    aliases = {
        "image": {"image", "image_url"},
        "audio": {"audio", "audio_url"},
        "video": {"video", "video_url"},
    }

    def visit(value):
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        if "content" in value and "role" in value:
            visit(value["content"])
            return
        content_type = value.get("type")
        for mode, names in aliases.items():
            if content_type in names or any(name in value for name in names):
                modes.add(mode)

    visit(conversations)
    return modes


def _metadata_sampled_fps(metadata, fallback_fps):
    sampled_fps = get_video_metadata_value(metadata, "sampled_fps", None)
    if sampled_fps is not None:
        return sampled_fps
    frames_indices = get_video_metadata_value(metadata, "frames_indices", None)
    total_num_frames = get_video_metadata_value(metadata, "total_num_frames", None)
    original_fps = get_video_metadata_value(metadata, "fps", None)
    if frames_indices is not None and total_num_frames and original_fps:
        return len(frames_indices) / total_num_frames * original_fps
    return fallback_fps


def _round_video_second_per_grid(value):
    import torch

    return float(torch.tensor(float(value), dtype=torch.bfloat16).item())


class TLiveOmniVideosKwargs(VideosKwargs, total=False):
    fps: Optional[Union[int, float]]
    video_backend: Optional[str]
    use_audio_in_video: Optional[bool]
    seconds_per_chunk: Optional[float]
    position_id_per_seconds: Optional[int]


class TLiveOmniImagesKwargs(ImagesKwargs, total=False):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]


class TLiveOmniProcessorKwargs(ProcessingKwargs, total=False):
    videos_kwargs: TLiveOmniVideosKwargs
    images_kwargs: TLiveOmniImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
        "videos_kwargs": {
            "return_metadata": False,
            "do_sample_frames": True,
            "fps": NATIVE_VIDEO_FPS,
            "min_frames": NATIVE_VIDEO_MIN_FRAMES,
            "max_frames": NATIVE_VIDEO_MAX_FRAMES,
            "num_frames": None,
            "seconds_per_chunk": 2.0,
            "position_id_per_seconds": 13,
            "use_audio_in_video": False,
            "size": dict(QWEN35_VIDEO_SIZE),
        },
        "images_kwargs": {
            "size": dict(QWEN35_IMAGE_SIZE),
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": True,
            "truncation": False,
            "return_attention_mask": True,
        },
    }


def _get_feat_extract_output_lengths(input_lengths):
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


@auto_docstring
class TLiveOmniProcessor(ProcessorMixin):
    required_transformers_version = SUPPORTED_TRANSFORMERS_VERSION
    attributes = ["image_processor", "video_processor", "feature_extractor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    @classmethod
    def get_attributes(cls):
        return cls.attributes

    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
    ):
        super().__init__(image_processor, video_processor, feature_extractor, tokenizer, chat_template=chat_template)
        self._tlive_video_backend = None
        if self.video_processor is not None and self.video_processor.__class__.__name__ != "Qwen3VLVideoProcessor":
            from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

            self.video_processor = Qwen3VLVideoProcessor(
                patch_size=NATIVE_PATCH_SIZE,
                temporal_patch_size=NATIVE_TEMPORAL_PATCH_SIZE,
                merge_size=NATIVE_MERGE_SIZE,
                image_mean=list(NATIVE_IMAGE_MEAN),
                image_std=list(NATIVE_IMAGE_STD),
                size=dict(QWEN35_VIDEO_SIZE),
                do_sample_frames=True,
                fps=NATIVE_VIDEO_FPS,
                min_frames=NATIVE_VIDEO_MIN_FRAMES,
                max_frames=NATIVE_VIDEO_MAX_FRAMES,
            )
        if self.video_processor is not None:
            self.video_processor.size = _merge_size(
                QWEN35_VIDEO_SIZE,
                getattr(self.video_processor, "size", None),
            )
            self.video_processor.do_sample_frames = bool(
                getattr(self.video_processor, "do_sample_frames", True)
            )
            self.video_processor.fps = getattr(self.video_processor, "fps", None) or NATIVE_VIDEO_FPS
            self.video_processor.min_frames = (
                getattr(self.video_processor, "min_frames", None) or NATIVE_VIDEO_MIN_FRAMES
            )
            self.video_processor.max_frames = (
                getattr(self.video_processor, "max_frames", None) or NATIVE_VIDEO_MAX_FRAMES
            )
            self.video_processor.patch_size = NATIVE_PATCH_SIZE
            self.video_processor.temporal_patch_size = NATIVE_TEMPORAL_PATCH_SIZE
            self.video_processor.merge_size = NATIVE_MERGE_SIZE
            self.video_processor.image_mean = list(NATIVE_IMAGE_MEAN)
            self.video_processor.image_std = list(NATIVE_IMAGE_STD)
        if self.image_processor is not None:
            self.image_processor.size = dict(QWEN35_IMAGE_SIZE)
            self.image_processor.patch_size = NATIVE_PATCH_SIZE
            self.image_processor.temporal_patch_size = NATIVE_TEMPORAL_PATCH_SIZE
            self.image_processor.merge_size = NATIVE_MERGE_SIZE
            self.image_processor.image_mean = list(NATIVE_IMAGE_MEAN)
            self.image_processor.image_std = list(NATIVE_IMAGE_STD)
        self.image_token = self.tokenizer.image_token
        self.audio_token = self.tokenizer.audio_token
        self.video_token = self.tokenizer.video_token
        self.vision_bos_token = self.tokenizer.vision_bos_token
        self.vision_eos_token = self.tokenizer.vision_eos_token
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_eos_token = self.tokenizer.audio_eos_token

    @auto_docstring
    def __call__(
        self,
        text: TextInput = None,
        images: ImageInput = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        videos_kwargs = kwargs.get("videos_kwargs") or {}
        images_kwargs = kwargs.get("images_kwargs") or {}
        if not isinstance(videos_kwargs, dict):
            raise TypeError(f"videos_kwargs must be a dict, got {type(videos_kwargs).__name__}")
        if not isinstance(images_kwargs, dict):
            raise TypeError(f"images_kwargs must be a dict, got {type(images_kwargs).__name__}")

        videos_kwargs = dict(videos_kwargs)
        video_size = dict(videos_kwargs.get("size") or {})
        for alias, edge in (("min_pixels", "shortest_edge"), ("max_pixels", "longest_edge")):
            if alias in videos_kwargs:
                if edge in video_size:
                    raise ValueError(f"Specify either videos_kwargs['{alias}'] or size['{edge}'], not both.")
                video_size[edge] = videos_kwargs.pop(alias)
        if video_size:
            videos_kwargs["size"] = video_size
        if videos_kwargs:
            kwargs["videos_kwargs"] = videos_kwargs

        structural_overrides = ("patch_size", "temporal_patch_size", "merge_size")
        invalid_structural = [
            name for name in structural_overrides if name in kwargs or name in videos_kwargs or name in images_kwargs
        ]
        if invalid_structural:
            raise ValueError(
                "Per-call architecture overrides are not supported: " + ", ".join(sorted(set(invalid_structural)))
            )
        if "return_metadata" in videos_kwargs:
            return_video_metadata = bool(videos_kwargs["return_metadata"])
        elif "return_metadata" in kwargs:
            return_video_metadata = bool(kwargs["return_metadata"])
        else:
            return_video_metadata = bool(getattr(self, "_tlive_return_video_metadata", False))

        fps_is_explicit = kwargs.get("fps") is not None or videos_kwargs.get("fps") is not None
        video_size_is_explicit = kwargs.get("size") is not None or videos_kwargs.get("size") is not None
        image_size_is_explicit = kwargs.get("size") is not None or images_kwargs.get("size") is not None
        video_option_is_explicit = {
            name: kwargs.get(name) is not None or videos_kwargs.get(name) is not None
            for name in ("do_sample_frames", "min_frames", "max_frames")
        }

        output_kwargs = self._merge_kwargs(
            TLiveOmniProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        default_video_size = _merge_size(
            QWEN35_VIDEO_SIZE,
            getattr(self.video_processor, "size", None),
        )
        default_video_fps = getattr(self.video_processor, "fps", NATIVE_VIDEO_FPS)
        if default_video_fps is None:
            default_video_fps = NATIVE_VIDEO_FPS

        output_kwargs["videos_kwargs"]["size"] = _merge_size(
            default_video_size,
            output_kwargs["videos_kwargs"].get("size") if video_size_is_explicit else None,
        )
        output_kwargs["images_kwargs"]["size"] = _merge_size(
            QWEN35_IMAGE_SIZE,
            output_kwargs["images_kwargs"].get("size") if image_size_is_explicit else None,
        )
        for name, fallback in (
            ("do_sample_frames", True),
            ("min_frames", NATIVE_VIDEO_MIN_FRAMES),
            ("max_frames", NATIVE_VIDEO_MAX_FRAMES),
        ):
            if not video_option_is_explicit[name]:
                output_kwargs["videos_kwargs"][name] = fallback
        seconds_per_chunk = output_kwargs["videos_kwargs"].pop("seconds_per_chunk")
        position_id_per_seconds = output_kwargs["videos_kwargs"].pop("position_id_per_seconds")
        use_audio_in_video = output_kwargs["videos_kwargs"].pop("use_audio_in_video")
        call_video_backend = output_kwargs["videos_kwargs"].pop("video_backend", None)
        video_backend = _validate_video_backend(
            getattr(self, "_tlive_video_backend", None) if call_video_backend is None else call_video_backend
        )
        if not fps_is_explicit:
            output_kwargs["videos_kwargs"]["fps"] = default_video_fps
        fps = output_kwargs["videos_kwargs"].get("fps", default_video_fps)
        video_second_per_grid_fps = fps if fps is not None else default_video_fps
        if output_kwargs["videos_kwargs"].get("num_frames") is not None and fps_is_explicit:
            raise ValueError("Native video process accepts either `fps` or `num_frames`, not both.")
        if output_kwargs["videos_kwargs"].get("num_frames") is not None:
            output_kwargs["videos_kwargs"]["fps"] = None
            fps = None
        video_frame_limits = {}
        for key in ("min_frames", "max_frames"):
            value = output_kwargs["videos_kwargs"].pop(key, None)
            if value is not None:
                video_frame_limits[key] = int(value)
        if videos is not None:
            output_kwargs["videos_kwargs"]["return_metadata"] = True

        if audio is not None:
            output_kwargs["audio_kwargs"]["padding"] = True  # Setting to True to avoid default truncation
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")  # rename feature_attention_mask to prevent conflicts later on
            audio_inputs["input_features"] = audio_inputs.pop("input_features")  # rename input_features to prevent conflicts later on
            audio_lengths = iter(_get_feat_extract_output_lengths(audio_inputs["feature_attention_mask"].sum(-1)))
        else:
            audio_inputs = {}
            audio_lengths = iter([])

        if videos is not None and use_audio_in_video and audio is None:
            raise ValueError(
                "use_audio_in_video=True requires an audio input for each video. "
                "apply_chat_template(tokenize=True, use_audio_in_video=True) loads video audio automatically."
            )

        if images is not None:
            images = normalize_media_paths(images)
            images_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = iter(images_inputs["image_grid_thw"])
        else:
            images_inputs = {}
            image_grid_thw = iter([])

        if videos is not None:
            videos = normalize_media_paths(videos)
            video_processor = copy.copy(self.video_processor)
            for key, value in video_frame_limits.items():
                setattr(video_processor, key, value)
            with _force_transformers_video_backend(video_backend):
                videos_inputs = video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_metadata = videos_inputs.get("video_metadata")
            videos_inputs["video_second_per_grid"] = [
                _round_video_second_per_grid(
                    video_processor.temporal_patch_size
                    / _metadata_sampled_fps(metadata, video_second_per_grid_fps)
                )
                for metadata in video_metadata
            ]
            if not return_video_metadata:
                videos_inputs.pop("video_metadata", None)
            video_grid_thw = iter(videos_inputs["video_grid_thw"])
            video_second_per_grid = iter(videos_inputs["video_second_per_grid"])

        else:
            videos_inputs = {}
            video_metadata = None
            video_grid_thw = iter([])
            video_second_per_grid = iter([])

        if not isinstance(text, list):
            text = [text]

        text = self.replace_multimodal_special_tokens(
            text,
            audio_lengths,
            image_grid_thw,
            video_grid_thw,
            video_second_per_grid=video_second_per_grid,
            use_audio_in_video=use_audio_in_video,
            position_id_per_seconds=position_id_per_seconds,
            seconds_per_chunk=seconds_per_chunk,
            video_metadatas=video_metadata,
        )

        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={
                **texts_inputs,
                **images_inputs,
                **videos_inputs,
                **audio_inputs,
                "use_audio_in_video": use_audio_in_video,
            },
            tensor_type=kwargs.get("return_tensors"),
        )

    def _calculate_timestamps(self, indices: Union[list[int], np.ndarray], video_fps: float, merge_size: int = 2, return_in_seconds: bool = False):
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        timestamps = [idx / video_fps for idx in indices]
        # Use the midpoint timestamp for each temporal patch.
        if not return_in_seconds:
            timestamps = [
                (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
            ]
        else:
            timestamps = [
                i for i in range(0, len(timestamps), merge_size)
            ]
        return timestamps
    def replace_multimodal_special_tokens(
        self,
        text,
        audio_lengths,
        image_grid_thw,
        video_grid_thw,
        video_second_per_grid,
        use_audio_in_video,
        position_id_per_seconds,
        seconds_per_chunk,
        video_metadatas=None,
    ):
        merge_length_image = self.image_processor.merge_size**2

        # Materialize the media iterators so cursors can span the full batch.
        audio_lengths_list = list(audio_lengths) if audio_lengths else []
        image_grid_thw_list = list(image_grid_thw) if image_grid_thw else []
        video_grid_thw_list = list(video_grid_thw) if video_grid_thw else []
        video_second_per_grid_list = list(video_second_per_grid) if video_second_per_grid else []
        
        processed_text = []
        audio_idx = 0
        image_idx = 0
        video_idx = 0

        for sample in text:
            special_tokens = [re.escape(tok) for tok in [self.audio_token, self.image_token, self.video_token]]
            pattern = "|".join(special_tokens)
            positions = sorted([(match.start(), match.group()) for match in re.finditer(pattern, sample)])
            positions.sort(key=lambda x: x[0])
            sample_video_count = 0

            for _, special_token in positions:
                if special_token == self.audio_token:
                    if audio_idx >= len(audio_lengths_list):
                        raise ValueError("The prompt contains more audio tokens than supplied audio inputs.")
                    audio_length = audio_lengths_list[audio_idx]
                    sample = sample.replace(self.audio_token, "<|audio_placeholder|>" * audio_length, 1)
                    audio_idx += 1
                elif special_token == self.image_token:
                    if image_idx >= len(image_grid_thw_list):
                        raise ValueError("The prompt contains more image tokens than supplied image inputs.")
                    image_seq_length = image_grid_thw_list[image_idx].prod() // merge_length_image
                    sample = sample.replace(self.image_token, "<|image_placeholder|>" * image_seq_length, 1)
                    image_idx += 1
                elif special_token == self.video_token:
                    if video_idx >= len(video_grid_thw_list):
                        raise ValueError("The prompt contains more video tokens than supplied video inputs.")
                    if use_audio_in_video and audio_idx >= len(audio_lengths_list):
                        raise ValueError("Each vocal video requires one matching audio track.")
                    placeholder = self._generate_video_placeholder(
                        video_grid_thw_list[video_idx],
                        video_metadatas[video_idx] if video_metadatas else None,
                        use_audio_in_video,
                        audio_lengths_list[audio_idx] if use_audio_in_video else None,
                        video_second_per_grid_list[video_idx] if use_audio_in_video else None,
                        position_id_per_seconds,
                        seconds_per_chunk,
                    )

                    wrapped_video_token = self.vision_bos_token + self.video_token + self.vision_eos_token
                    if wrapped_video_token in sample:
                        sample = sample.replace(wrapped_video_token, placeholder, 1)
                    else:
                        sample = sample.replace(self.video_token, placeholder, 1)
                    if use_audio_in_video:
                        audio_idx += 1
                    video_idx += 1
                    sample_video_count += 1

            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            sample = sample.replace("<|image_placeholder|>", self.image_token)
            sample = sample.replace("<|video_placeholder|>", self.video_token)
            processed_text.append(sample)
            if use_audio_in_video and sample_video_count:
                token_ids = self.tokenizer.encode(sample, add_special_tokens=False)
                self._validate_video_audio_tokens(token_ids, None, position_id_per_seconds)

        consumed = {
            "audio": (audio_idx, len(audio_lengths_list)),
            "image": (image_idx, len(image_grid_thw_list)),
            "video": (video_idx, len(video_grid_thw_list)),
        }
        unconsumed = [name for name, (used, available) in consumed.items() if used != available]
        media_only_call = all(sample == "" for sample in text)
        if unconsumed and not media_only_call:
            details = ", ".join(
                f"{name}={consumed[name][0]}/{consumed[name][1]}" for name in unconsumed
            )
            raise ValueError(f"Prompt/media count mismatch: {details}.")

        return processed_text
    def _validate_video_audio_tokens(
        self,
        token_ids,
        video_second_per_grid,
        position_id_per_seconds,
    ):
        video_token_start = self.tokenizer.convert_tokens_to_ids(self.vision_bos_token)
        video_token_end = self.tokenizer.convert_tokens_to_ids(self.vision_eos_token)
        audio_token_start = self.tokenizer.convert_tokens_to_ids(self.audio_bos_token)
        audio_token_end = self.tokenizer.convert_tokens_to_ids(self.audio_eos_token)

        video_start_indices = [idx for idx, token in enumerate(token_ids) if token == video_token_start]
        video_end_indices = [idx for idx, token in enumerate(token_ids) if token == video_token_end]
        audio_start_indices = [idx for idx, token in enumerate(token_ids) if token == audio_token_start]
        audio_end_indices = [idx for idx, token in enumerate(token_ids) if token == audio_token_end]

        if len(video_start_indices) != len(video_end_indices):
            raise ValueError(
                "Video start/end token count mismatch: "
                f"{len(video_start_indices)} != {len(video_end_indices)}"
            )
        if len(audio_start_indices) != len(audio_end_indices):
            raise ValueError(
                "Audio start/end token count mismatch: "
                f"{len(audio_start_indices)} != {len(audio_end_indices)}"
            )

        return True
    def _generate_video_placeholder(
        self,
        video_grid_thw,
        video_metadata,
        use_audio_in_video,
        audio_length=None,
        video_second_per_grid=None,
        position_id_per_seconds=None,
        seconds_per_chunk=None,
    ):
        merge_length_video = self.video_processor.merge_size**2
        
        if not use_audio_in_video:
            # Visual-only video uses one timestamped span per temporal grid.
            num_frames = video_grid_thw[0]
            token_len_per_frame = video_grid_thw[1:].prod() // merge_length_video
            
            # Compute one timestamp per temporal grid.
            if video_metadata:
                timestamps = self._calculate_timestamps(
                    video_metadata['frames_indices'],
                    video_metadata['fps'],
                    self.video_processor.temporal_patch_size,
                    return_in_seconds=False
                )
            else:
                timestamps = list(range(num_frames))
            
            # Build the visual-only placeholder spans.
            placeholder_parts = []
            for frame_idx in range(num_frames):
                curr_time = timestamps[frame_idx] if frame_idx < len(timestamps) else frame_idx
                placeholder_parts.append(f"<{curr_time:.1f} seconds>")
                placeholder_parts.append(self.vision_bos_token)
                placeholder_parts.append("<|video_placeholder|>" * token_len_per_frame)
                placeholder_parts.append(self.vision_eos_token)
            
            return "".join(placeholder_parts)
        
        else:
            # video+audio mode: one span per temporal grid.
            height = video_grid_thw[1] // self.video_processor.merge_size
            width = video_grid_thw[2] // self.video_processor.merge_size
            
            # Compute one timestamp per temporal grid.
            if video_metadata:
                cur_timestamp = self._calculate_timestamps(
                    video_metadata['frames_indices'],
                    video_metadata['fps'],
                    self.video_processor.temporal_patch_size,
                    return_in_seconds=False
                )
            else:
                cur_timestamp = list(range(video_grid_thw[0]))
            
            video_tokens_per_grid = height * width
            grid_t = int(video_grid_thw[0].item() if hasattr(video_grid_thw[0], "item") else video_grid_thw[0])
            audio_len = int(audio_length.item() if hasattr(audio_length, "item") else audio_length)
            seconds_per_grid = float(
                video_second_per_grid.item() if hasattr(video_second_per_grid, "item") else video_second_per_grid
            )
            audio_boundaries = [0]
            for grid_idx in range(1, grid_t):
                boundary = int(grid_idx * seconds_per_grid * position_id_per_seconds + 1e-6)
                boundary = min(audio_len, max(boundary, audio_boundaries[-1]))
                audio_boundaries.append(boundary)
            audio_boundaries.append(audio_len)
            
            # Build one interleaved video-audio span per temporal grid.
            placeholder_parts = []
            
            for grid_idx in range(grid_t):
                timestamp_idx = min(grid_idx, len(cur_timestamp) - 1)
                curr_time = cur_timestamp[timestamp_idx] if cur_timestamp else grid_idx
                placeholder_parts.append(f"<{curr_time:.1f} seconds>")
                placeholder_parts.append(self.vision_bos_token)
                placeholder_parts.append(self.audio_bos_token)
                placeholder_parts.append("<|video_placeholder|>" * video_tokens_per_grid)
                audio_seq_length = audio_boundaries[grid_idx + 1] - audio_boundaries[grid_idx]
                placeholder_parts.append("<|audio_placeholder|>" * audio_seq_length)
                placeholder_parts.append(self.audio_eos_token)
                placeholder_parts.append(self.vision_eos_token)
            
            return "".join(placeholder_parts)
    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        media_modes = _conversation_media_modes(conversations)
        if len(media_modes) > 1:
            raise ValueError(
                "TLive-Omni supports one media mode per apply_chat_template call; "
                f"received: {', '.join(sorted(media_modes))}."
            )

        processor_kwargs = kwargs.pop("processor_kwargs", None)
        if processor_kwargs is not None:
            if not isinstance(processor_kwargs, dict):
                raise TypeError(f"processor_kwargs must be a dict, got {type(processor_kwargs).__name__}")
            duplicate_keys = sorted(set(processor_kwargs).intersection(kwargs))
            if duplicate_keys:
                raise ValueError(
                    "Processor arguments were provided both directly and in processor_kwargs: "
                    + ", ".join(duplicate_keys)
                )
            kwargs.update(processor_kwargs)

        videos_kwargs = kwargs.get("videos_kwargs")
        if videos_kwargs is not None and not isinstance(videos_kwargs, dict):
            raise TypeError(f"videos_kwargs must be a dict, got {type(videos_kwargs).__name__}")

        unset = object()
        top_level_use_audio = kwargs.get("use_audio_in_video", unset)
        nested_use_audio = (
            videos_kwargs.get("use_audio_in_video", unset) if videos_kwargs is not None else unset
        )
        explicit_use_audio_values = [
            value
            for value in (top_level_use_audio, nested_use_audio)
            if value is not unset and value is not None
        ]
        for value in explicit_use_audio_values:
            if not isinstance(value, bool):
                raise TypeError(f"use_audio_in_video must be a bool, got {type(value).__name__}")
        if explicit_use_audio_values and any(
            value != explicit_use_audio_values[0] for value in explicit_use_audio_values[1:]
        ):
            raise ValueError("Conflicting explicit use_audio_in_video values.")
        use_audio_in_video = explicit_use_audio_values[0] if explicit_use_audio_values else False
        if use_audio_in_video and media_modes != {"video"}:
            raise ValueError("use_audio_in_video=True requires a video-only conversation.")

        load_audio_from_video = kwargs.get("load_audio_from_video", unset)
        if load_audio_from_video is not unset and load_audio_from_video is not None:
            if not isinstance(load_audio_from_video, bool):
                raise TypeError(
                    f"load_audio_from_video must be a bool, got {type(load_audio_from_video).__name__}"
                )
            if load_audio_from_video != use_audio_in_video:
                raise ValueError(
                    "load_audio_from_video must match use_audio_in_video; "
                    "use use_audio_in_video as the public switch."
                )

        if nested_use_audio is not unset:
            videos_kwargs = dict(videos_kwargs)
            videos_kwargs.pop("use_audio_in_video")

        videos_kwargs = dict(videos_kwargs or {})
        videos_kwargs["use_audio_in_video"] = use_audio_in_video
        kwargs["videos_kwargs"] = videos_kwargs
        kwargs.pop("use_audio_in_video", None)
        kwargs["load_audio_from_video"] = use_audio_in_video

        with _force_transformers_video_audio_loader(use_audio_in_video):
            output = super().apply_chat_template(conversations, chat_template, **kwargs)
        if isinstance(output, BatchFeature):
            output["use_audio_in_video"] = use_audio_in_video
        return output

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        video_processor_input_names = self.video_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + feature_extractor_input_names
                + image_processor_input_names
                + video_processor_input_names
                + ["feature_attention_mask", "video_second_per_grid", "use_audio_in_video"]
            )
        )
