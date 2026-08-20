# Copyright 2025 The vLLM team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only TLiveOmni model.

Architecture:
- audio_tower: Qwen3OmniMoeAudioEncoder (from Qwen3-Omni)
- visual: Qwen3_VisionTransformer (from Qwen3.5, NO deepstack)
- language_model: Qwen3_5ForCausalLM (hybrid linear_attention + full_attention)
- Flat config (no thinker_config wrapper)
"""

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from functools import partial
from typing import Any, Literal, cast
import json
import os

import numpy as np
import torch
from transformers import video_processing_utils


class _TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)


# Fast video processors keep `image_mean`/`image_std` as tensors, which the
# stock `to_json_string` cannot serialize.
if hasattr(video_processing_utils, 'BaseVideoProcessor'):
    def _patched_to_json_string(self, *args, **kwargs):
        dictionary = self.to_dict()
        return json.dumps(dictionary, indent=2, sort_keys=True, cls=_TensorEncoder) + "\n"
    video_processing_utils.BaseVideoProcessor.to_json_string = _patched_to_json_string


import torch.nn as nn
from packaging.version import Version
from transformers import PretrainedConfig
from transformers import __version__ as TRANSFORMERS_VERSION
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
    IsHybrid,
    HasInnerState
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen2_audio import Qwen2AudioProcessingInfo
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniAudioFeatureInputs,
    Qwen2_5OmniConditionalGenerationMixin,
    Qwen2_5OmniThinkerDummyInputsBuilder,
    Qwen2_5OmniThinkerMultiModalProcessor,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
    _get_feat_extract_output_lengths,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItems, MultiModalFieldConfig
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing.processor import (
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.forward_context import set_forward_context
from vllm.transformers_utils.processor import cached_processor_from_config

from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models.tlive_omni_processing import TLiveOmniProcessor

logger = init_logger(__name__)

ISO639_1_SUPPORTED_LANGS = {
    "en": "English",
    "zh": "Chinese",
    "ko": "Korean",
    "ja": "Japanese",
    "de": "German",
    "ru": "Russian",
    "it": "Italian",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "ms": "Malay",
    "nl": "Dutch",
    "id": "Indonesian",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "ar": "Arabic",
    "ur": "Urdu",
}


class TLiveOmniDataParser(MultiModalDataParser):
    def _get_video_with_metadata(self, video):
        if isinstance(video, (str, os.PathLike)):
            return os.fspath(video), None
        return super()._get_video_with_metadata(video)


class TLiveOmniProcessingInfo(Qwen2AudioProcessingInfo, Qwen3VLProcessingInfo):
    """Flat config - no thinker_config wrapper."""

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_feature_extractor()

        return TLiveOmniDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_hf_processor(self, **kwargs: object) -> TLiveOmniProcessor:
        processor = self.ctx.get_hf_processor(
            TLiveOmniProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )
        if not hasattr(processor, "audio_token"):
            processor.audio_token = "<|audio_pad|>"
        if not hasattr(processor, "image_token"):
            processor.image_token = "<|image_pad|>"
        if not hasattr(processor, "video_token"):
            processor.video_token = "<|video_pad|>"
        return processor

    def get_feature_extractor(self, **kwargs: object):
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None, "image": None, "video": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
    ) -> Mapping[str, int] | None:
        mm_counts = mm_counts or {}
        requested_modalities = {m for m, c in mm_counts.items() if c > 0}
        mm_max_tokens: dict[str, int] = {}

        if requested_modalities & {"image", "video"}:
            vl_tokens = Qwen3VLProcessingInfo.get_mm_max_tokens_per_item(
                self, seq_len=seq_len, mm_counts=mm_counts,
            )
            mm_max_tokens.update(
                {
                    m: vl_tokens[m]
                    for m in ["image", "video"]
                    if m in requested_modalities
                }
            )

        if "audio" in requested_modalities:
            audio_tokens = Qwen2AudioProcessingInfo.get_mm_max_tokens_per_item(
                self,
                seq_len=seq_len,
                mm_counts=mm_counts,
            )
            mm_max_tokens["audio"] = audio_tokens["audio"]

        return mm_max_tokens


TLiveOmniDummyInputsBuilder = Qwen2_5OmniThinkerDummyInputsBuilder


class TLiveOmniMultiModalProcessor(Qwen2_5OmniThinkerMultiModalProcessor):
    """Processor adapted for flat config structure."""

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        spatial_merge_size = self.info.get_hf_config().vision_config.spatial_merge_size
        audio_feature_lengths = hf_inputs.get("audio_feature_lengths", torch.empty((0,)))
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        image_pixel_grid_sizes = image_grid_thw.prod(-1)
        image_embed_grid_sizes = image_pixel_grid_sizes // spatial_merge_size // spatial_merge_size
        video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
        video_grid_sizes = video_grid_thw.prod(-1)
        video_embed_grid_sizes = video_grid_sizes // spatial_merge_size // spatial_merge_size
        num_videos = len(video_grid_sizes)
        return {
            "input_audio_features": MultiModalFieldConfig.flat_from_sizes("audio", audio_feature_lengths, dim=1),
            "feature_attention_mask": MultiModalFieldConfig.batched("audio"),
            "audio_feature_lengths": MultiModalFieldConfig.batched("audio"),
            "pixel_values": MultiModalFieldConfig.flat_from_sizes("image", image_pixel_grid_sizes),
            "image_embeds": MultiModalFieldConfig.flat_from_sizes("image", image_embed_grid_sizes),
            "image_grid_thw": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            "pixel_values_videos": MultiModalFieldConfig.flat_from_sizes("video", video_grid_sizes),
            "video_embeds": MultiModalFieldConfig.flat_from_sizes("video", video_embed_grid_sizes),
            "video_grid_thw": MultiModalFieldConfig.batched("video", keep_on_cpu=True),
            "second_per_grid_ts": MultiModalFieldConfig.batched("video", keep_on_cpu=True),
            "timestamps": MultiModalFieldConfig.batched("video", keep_on_cpu=True),
            "use_audio_in_video": MultiModalFieldConfig.shared("video", num_videos),
        }

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)

        mm_kwargs = dict(mm_kwargs)
        videos_kwargs = dict(mm_kwargs.get("videos_kwargs") or {})
        mm_kwargs["videos_kwargs"] = videos_kwargs
        for duplicated_video_key in ("use_audio_in_video", "fps", "size", "min_frames", "max_frames", "num_frames"):
            mm_kwargs.pop(duplicated_video_key, None)

        def is_mm_only_prompt(text: str) -> bool:
            stripped = text.strip()
            if not stripped:
                return False
            tmp = stripped
            for token in (
                "<|audio_start|>",
                "<|audio_pad|>",
                "<|audio_end|>",
                "<|vision_start|>",
                "<|image_pad|>",
                "<|video_pad|>",
                "<|vision_end|>",
            ):
                tmp = tmp.replace(token, "")
            return tmp == ""

        if videos_kwargs.get("use_audio_in_video", False) and "<|video_pad|>" in prompt and is_mm_only_prompt(prompt):
            prompt = prompt.replace("<|audio_start|><|audio_pad|><|audio_end|>", "")
            prompt = prompt.replace("<|audio_pad|>", "")

        audios = mm_data.pop("audios", None)
        if audios is None:
            audios = mm_data.pop("audio", [])

        def pad_to_hop_length(x: np.ndarray, hop_length: int) -> np.ndarray:
            length = x.shape[-1]
            if length % hop_length != 0:
                pad_length = hop_length - (length % hop_length)
                x = np.pad(x, (0, pad_length), mode="constant", constant_values=0)
            return x

        # NOTE: WhisperFeatureExtractor cannot handle empty list of audios
        feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
        hop_length = feature_extractor.hop_length
        if audios:
            _orig_audio_lengths = [
                len(a[0]) if isinstance(a, tuple) else len(a)
                for a in audios
            ]
            # NOTE: Qwen3-Omni processor accept "audio"
            # To make sure the cache works with padding=True, we pre-padded
            # the audio to multiple of hop_length.
            mm_data["audio"] = [
                pad_to_hop_length(audio, hop_length)
                if isinstance(audio, np.ndarray)
                else (pad_to_hop_length(audio[0], hop_length), audio[1])
                for audio in audios
            ]

            mm_kwargs = dict(mm_kwargs)
            tok_kwargs = dict(tok_kwargs)
            mm_kwargs["audio_kwargs"] = dict(mm_kwargs.get("audio_kwargs") or {})
            mm_kwargs["text_kwargs"] = dict(mm_kwargs.get("text_kwargs") or {})
            if Version(TRANSFORMERS_VERSION) < Version("4.58.0"):
                # Extract audio_sample_rate before restructuring
                audio_sample_rate = mm_kwargs.pop("audio_sample_rate", None)

                # move truncation to audio_kwargs level to avoid conflict
                # with tok_kwargs
                mm_kwargs["audio_kwargs"].setdefault(
                    "truncation", mm_kwargs.pop("truncation", False)
                )
                mm_kwargs["text_kwargs"].setdefault(
                    "truncation", tok_kwargs.pop("truncation", False)
                )

                # Validate and conditionally pass audio_sample_rate
                # WhisperFeatureExtractor has a fixed sampling rate, and vLLM's
                # audio loader already resamples audio to the target rate.
                # Only pass the value if it matches to avoid unexpected behavior.
                if audio_sample_rate is not None:
                    expected_sr = feature_extractor.sampling_rate
                    if audio_sample_rate != expected_sr:
                        logger.warning(
                            "[%s] audio_sample_rate mismatch: user provided %dHz "
                            "but model expects %dHz. Ignoring user value. "
                            "vLLM's audio loader already resampled to %dHz.",
                            self.__class__.__name__,
                            audio_sample_rate,
                            expected_sr,
                            expected_sr,
                        )
                    else:
                        # Sample rate matches, safe to pass
                        mm_kwargs["audio_kwargs"]["audio_sample_rate"] = (
                            audio_sample_rate
                        )

        hf_inputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        if (
            "audio_feature_lengths" in hf_inputs
            and "feature_attention_mask" in hf_inputs
            and (audios := mm_data.get("audio", []))
        ):
            audio_num_frames = []
            for i, audio in enumerate(audios):
                audio_length = _orig_audio_lengths[i]
                num_frame = audio_length // hop_length
                if mm_kwargs.get("truncation", False):
                    num_frame = min(
                        num_frame, feature_extractor.n_samples // hop_length
                    )
                audio_num_frames.append(num_frame)
            hf_inputs["feature_attention_mask"] = [
                torch.ones(num_frame) for num_frame in audio_num_frames
            ]
            hf_inputs["audio_feature_lengths"] = torch.tensor(audio_num_frames)
            if "input_audio_features" in hf_inputs:
                total_orig_frames = sum(audio_num_frames)
                feat = hf_inputs["input_audio_features"]
                # shape: (mel_bins, total_frames)
                if feat.shape[-1] > total_orig_frames:
                    hf_inputs["input_audio_features"] = feat[:, :total_orig_frames]
        use_audio_in_video_value = bool(videos_kwargs.get("use_audio_in_video", False))
        hf_inputs["use_audio_in_video"] = torch.tensor(use_audio_in_video_value)
        if "video_second_per_grid" in hf_inputs and "second_per_grid_ts" not in hf_inputs:
            hf_inputs["second_per_grid_ts"] = hf_inputs["video_second_per_grid"]
        if "video_metadata" in hf_inputs and "timestamps" not in hf_inputs:
            hf_processor = self.info.get_hf_processor(**mm_kwargs)
            temporal_patch_size = getattr(hf_processor.video_processor, "temporal_patch_size", 2)
            timestamps = []
            for metadata in hf_inputs["video_metadata"]:
                frames_indices = metadata.get("frames_indices") if isinstance(metadata, dict) else getattr(metadata, "frames_indices", None)
                video_fps = metadata.get("fps", 1.0) if isinstance(metadata, dict) else getattr(metadata, "fps", 1.0)
                if frames_indices is None:
                    grid_idx = len(timestamps)
                    timestamps.append(torch.arange(int(hf_inputs["video_grid_thw"][grid_idx][0]), dtype=torch.float64))
                    continue
                indices = list(frames_indices) if not isinstance(frames_indices, list) else frames_indices[:]
                if len(indices) % temporal_patch_size != 0:
                    indices.extend([indices[-1]] * (temporal_patch_size - len(indices) % temporal_patch_size))
                raw_ts = [int(idx.item() if hasattr(idx, "item") else idx) / float(video_fps) for idx in indices]
                ts = [
                    (raw_ts[i] + raw_ts[i + temporal_patch_size - 1]) / 2
                    for i in range(0, len(raw_ts), temporal_patch_size)
                ]
                timestamps.append(torch.tensor(ts, dtype=torch.float64))
            hf_inputs["timestamps"] = timestamps
        if "video_metadata" in hf_inputs and "video_metadata" not in mm_kwargs:
            mm_kwargs["video_metadata"] = hf_inputs["video_metadata"]
        return hf_inputs

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsItems,
        mm_prompt_updates: MultiModalPromptUpdates,
        is_update_applied: bool,
    ) -> tuple[list[int], str, Mapping[str, list[PlaceholderFeaturesInfo]]]:
        """
        Qwen3-Omni reimplements this function to handle `use_audio_in_video`.
        """
        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_kwargs, mm_item_counts)

        use_audio_in_video = False
        video_token_id = self.info.get_hf_config().video_token_id
        audio_token_id = self.info.get_hf_config().audio_token_id
        if "video" in mm_kwargs:
            for item in mm_kwargs["video"]:
                if item and item.get("use_audio_in_video") and item["use_audio_in_video"].data.numel() > 0 and item["use_audio_in_video"].data.item():
                    use_audio_in_video = True
                    break
            if not use_audio_in_video:
                prompt_has_video_token = video_token_id in prompt_ids
                prompt_has_audio_token = audio_token_id in prompt_ids
                has_video_items = mm_item_counts.get("video", 0) > 0
                has_audio_items = mm_item_counts.get("audio", 0) > 0
                if has_video_items and has_audio_items and prompt_has_video_token and not prompt_has_audio_token:
                    use_audio_in_video = True
                elif prompt_has_video_token and prompt_has_audio_token and "audio" in mm_prompt_updates and "video" in mm_prompt_updates:
                    use_audio_in_video = True
            if any(item is None for item in mm_kwargs["video"]):
                video_audio_item_num = sum(
                    id in (video_token_id, audio_token_id) for id in prompt_ids
                )
                audio_updates_num = len(mm_prompt_updates.get("audio", []))
                video_updates_num = len(mm_prompt_updates.get("video", []))
                if video_audio_item_num != video_updates_num + audio_updates_num:
                    use_audio_in_video = True

        # normal case with `use_audio_in_video=False`
        if is_update_applied:
            mm_placeholders = self._find_mm_placeholders(
                prompt_ids,
                mm_prompt_updates,
            )
            if any(
                len(mm_placeholders.get(modality, [])) != int(count)
                for modality, count in mm_item_counts.items()
                if int(count) > 0
            ):
                prompt_ids, mm_placeholders = self._apply_prompt_updates(
                    prompt_ids,
                    mm_prompt_updates,
                )
        else:
            if use_audio_in_video:
                filtered_updates = {
                    k: v for k, v in mm_prompt_updates.items() if k != "audio"
                }
                prompt_ids, mm_placeholders = self._apply_prompt_updates(
                    prompt_ids,
                    filtered_updates,
                )
                mm_placeholders = self._derive_audio_from_video_placeholders(
                    mm_placeholders, mm_prompt_updates, mm_item_counts
                )
                if len(mm_placeholders.get("video", [])) != int(mm_item_counts.get("video", 0)):
                    prompt_ids, mm_placeholders = self._apply_prompt_updates(
                        prompt_ids,
                        mm_prompt_updates,
                    )
            else:
                prompt_ids, mm_placeholders = self._apply_prompt_updates(
                    prompt_ids,
                    mm_prompt_updates,
                )

        self._validate_mm_placeholders(
            mm_placeholders,
            mm_item_counts,
        )

        return prompt_ids, mm_placeholders

    def get_updates_use_audio_in_video(
        self,
        thinker_config: PretrainedConfig,
        audio_len: int,
        video_grid_thw: list[int] | torch.Tensor,
        video_second_per_grid_t: float,
        tokenizer,
        timestamps: list[float] | None = None,
    ) -> list[int]:
        """Build the new-processor video+audio token sequence."""
        audio_token_id = thinker_config.audio_token_id
        video_token_id = thinker_config.video_token_id
        audio_start_token_id = thinker_config.audio_start_token_id
        audio_end_token_id = thinker_config.audio_end_token_id
        vision_start_token_id = tokenizer.get_vocab()[tokenizer.vision_bos_token]
        vision_end_token_id = tokenizer.get_vocab()[tokenizer.vision_eos_token]
        spatial_merge_size = thinker_config.vision_config.spatial_merge_size
        position_id_per_seconds = thinker_config.position_id_per_seconds

        curr_video_grid_thw = video_grid_thw
        grid_t = int(curr_video_grid_thw[0])
        height = int(curr_video_grid_thw[1]) // spatial_merge_size
        width = int(curr_video_grid_thw[2]) // spatial_merge_size
        video_tokens_per_grid = height * width

        if timestamps is None:
            temporal_patch_size = 2
            ts_offset = video_second_per_grid_t / temporal_patch_size * 0.5
            timestamps = [float(i) * video_second_per_grid_t + ts_offset for i in range(grid_t)]

        audio_boundaries = [0]
        for grid_idx in range(1, grid_t):
            boundary = int(grid_idx * video_second_per_grid_t * position_id_per_seconds + 1e-6)
            boundary = min(audio_len, max(boundary, audio_boundaries[-1]))
            audio_boundaries.append(boundary)
        audio_boundaries.append(audio_len)

        updates: list[int] = []
        for grid_idx in range(grid_t):
            curr_time = timestamps[min(grid_idx, len(timestamps) - 1)] if timestamps else float(grid_idx)
            updates.extend(tokenizer.encode(f"<{curr_time:.1f} seconds>", add_special_tokens=False))
            updates.append(vision_start_token_id)
            updates.append(audio_start_token_id)
            updates.extend([video_token_id] * video_tokens_per_grid)
            audio_seq_len = audio_boundaries[grid_idx + 1] - audio_boundaries[grid_idx]
            updates.extend([audio_token_id] * audio_seq_len)
            updates.append(audio_end_token_id)
            updates.append(vision_end_token_id)

        return updates

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        vocab = tokenizer.get_vocab()

        audio_token = processor.audio_token
        image_token = processor.image_token
        video_token = processor.video_token
        audio_token_id = vocab[audio_token]
        image_token_id = vocab[image_token]
        video_token_id = vocab[video_token]

        out_mm_data = out_mm_kwargs.get_data()
        audio_feature_lengths = out_mm_data.get("audio_feature_lengths")
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        if audio_feature_lengths is None and feature_attention_mask is None:
            audio_output_lengths = []
        elif audio_feature_lengths is not None:
            audio_output_lens = _get_feat_extract_output_lengths(audio_feature_lengths)
            audio_output_lengths = audio_output_lens.tolist()
        elif feature_attention_mask is not None:
            assert isinstance(feature_attention_mask, torch.Tensor)
            audio_output_lens = _get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )
            audio_output_lengths = audio_output_lens.tolist()

        # number of audios read from video.
        audio_in_video_item_idx = 0
        audio_item_idx = 0

        def get_replacement_audio(item_idx: int):
            nonlocal audio_item_idx
            item_idx += audio_in_video_item_idx

            audio_item_idx += 1

            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio = audios.get(item_idx)
                raise ValueError(
                    f"The audio {audio} (len={len(audio)}) is too short "
                    "to be represented inside the model"
                )

            return [audio_token_id] * num_features

        def get_replacement_vision(item_idx: int, modality: str):
            grid_thw = out_mm_data[f"{modality}_grid_thw"][item_idx]
            assert isinstance(grid_thw, torch.Tensor)
            merge_length = image_processor.merge_size**2

            if modality == "image":
                return [image_token_id] * (int(grid_thw.prod()) // merge_length)

            # video: insert per-frame timestamp tokens (same as Qwen3VL get_video_repl)
            num_frames = int(grid_thw[0].item())
            token_len_per_frame = int(grid_thw[1:].prod().item()) // merge_length

            # get second_per_grid_t for this video item
            second_per_grid_ts = out_mm_data.get("second_per_grid_ts")
            if second_per_grid_ts is not None:
                video_second_per_grid_t = float(second_per_grid_ts[item_idx])
            else:
                video_second_per_grid_t = hf_processor_mm_kwargs.get(
                    "second_per_grid_ts", [2.0]
                )
                if isinstance(video_second_per_grid_t, (list, tuple)):
                    video_second_per_grid_t = (
                        video_second_per_grid_t[item_idx]
                        if item_idx < len(video_second_per_grid_t)
                        else video_second_per_grid_t[-1]
                    )
                video_second_per_grid_t = float(video_second_per_grid_t)

            timestamps_from_data = out_mm_data.get("timestamps")
            if timestamps_from_data is not None:
                timestamps = timestamps_from_data[item_idx]
                if isinstance(timestamps, torch.Tensor):
                    timestamps = timestamps.tolist()
            else:
                video_metadata_list = out_mm_data.get("video_metadata") or hf_processor_mm_kwargs.get("video_metadata")
                timestamps = None
                if video_metadata_list is not None and item_idx < len(video_metadata_list):
                    metadata = video_metadata_list[item_idx]
                    if hasattr(metadata, "data"):
                        metadata = metadata.data
                    frames_indices = metadata.get("frames_indices") if isinstance(metadata, dict) else getattr(metadata, "frames_indices", None)
                    video_fps = metadata.get("fps", 1.0) if isinstance(metadata, dict) else getattr(metadata, "fps", 1.0)
                    if frames_indices is not None:
                        merge_size = getattr(image_processor, "temporal_patch_size", 2)
                        indices = list(frames_indices) if not isinstance(frames_indices, list) else frames_indices[:]
                        if len(indices) % merge_size != 0:
                            indices.extend([indices[-1]] * (merge_size - len(indices) % merge_size))
                        raw_ts = [int(idx.item() if hasattr(idx, "item") else idx) / float(video_fps) for idx in indices]
                        timestamps = [
                            (raw_ts[i] + raw_ts[i + merge_size - 1]) / 2
                            for i in range(0, len(raw_ts), merge_size)
                        ]
                if timestamps is None:
                    timestamps = list(range(num_frames))

            vision_start_token_id = vocab[tokenizer.vision_bos_token]
            vision_end_token_id = vocab[tokenizer.vision_eos_token]

            all_token_ids = []
            for ts in timestamps:
                # encode "<X.X seconds>" as text tokens
                ts_ids = tokenizer.encode(
                    f"<{ts:.1f} seconds>", add_special_tokens=False
                )
                all_token_ids.extend(ts_ids)
                all_token_ids.append(vision_start_token_id)
                all_token_ids.extend([video_token_id] * token_len_per_frame)
                all_token_ids.append(vision_end_token_id)

            return PromptUpdateDetails.select_token_id(all_token_ids, video_token_id)

        use_audio_in_video = (hf_processor_mm_kwargs.get("videos_kwargs") or {}).get(
            "use_audio_in_video",
            hf_processor_mm_kwargs.get("use_audio_in_video", False),
        )
        thinker_config = self.info.get_hf_config()

        def get_replacement_use_audio_in_video(item_idx: int):
            nonlocal audio_in_video_item_idx
            audio_num_features = audio_output_lengths[audio_in_video_item_idx]
            video_grid_thw = out_mm_data["video_grid_thw"][item_idx]

            audio_in_video_item_idx += 1

            second_per_grid_ts = out_mm_data.get("second_per_grid_ts")
            if second_per_grid_ts is not None:
                video_second_per_grid_t = float(second_per_grid_ts[item_idx])
            else:
                second_per_grid_ts = hf_processor_mm_kwargs.get("second_per_grid_ts", None)
                if second_per_grid_ts:
                    video_second_per_grid_t = second_per_grid_ts[item_idx]
                else:
                    video_second_per_grid_t = 2.0

            # Compute timestamps from video_metadata first to match HF _calculate_timestamps.
            timestamps = None
            video_metadata_list = out_mm_data.get("video_metadata") or hf_processor_mm_kwargs.get("video_metadata")
            if video_metadata_list and item_idx < len(video_metadata_list):
                vm = video_metadata_list[item_idx]
                if hasattr(vm, "data"):
                    vm = vm.data
                frames_indices = vm.get("frames_indices", None) if isinstance(vm, dict) else getattr(vm, "frames_indices", None)
                video_fps = vm.get("fps", 1.0) if isinstance(vm, dict) else getattr(vm, "fps", 1.0)
                if frames_indices is not None:
                    merge_size = getattr(image_processor, "temporal_patch_size", 2)
                    indices = list(frames_indices) if not isinstance(frames_indices, list) else frames_indices[:]
                    if len(indices) % merge_size != 0:
                        indices.extend([indices[-1]] * (merge_size - len(indices) % merge_size))
                    raw_ts = [int(idx.item() if hasattr(idx, "item") else idx) / float(video_fps) for idx in indices]
                    timestamps = [
                        (raw_ts[i] + raw_ts[i + merge_size - 1]) / 2
                        for i in range(0, len(raw_ts), merge_size)
                    ]
            timestamps_from_data = out_mm_data.get("timestamps")
            if timestamps is None and timestamps_from_data is not None:
                timestamps = timestamps_from_data[item_idx]
                if isinstance(timestamps, torch.Tensor):
                    timestamps = timestamps.tolist()

            placeholder = self.get_updates_use_audio_in_video(
                thinker_config=thinker_config,
                audio_len=audio_num_features,
                video_grid_thw=video_grid_thw,
                video_second_per_grid_t=video_second_per_grid_t,
                tokenizer=tokenizer,
                timestamps=timestamps,
            )
            return PromptUpdateDetails.select_token_id(
                placeholder, embed_token_id=video_token_id
            )

        video_replacement_fn = (
            get_replacement_use_audio_in_video
            if use_audio_in_video
            else partial(get_replacement_vision, modality="video")
        )

        return [
            PromptReplacement(modality="audio", target=audio_token, replacement=get_replacement_audio),
            PromptReplacement(modality="image", target=image_token, replacement=partial(get_replacement_vision, modality="image")),
            PromptReplacement(modality="video", target=video_token, replacement=video_replacement_fn),
        ]

    def _derive_audio_from_video_placeholders(
        self,
        placeholders: Mapping[str, list[PlaceholderFeaturesInfo]],
        mm_prompt_updates: MultiModalPromptUpdates,
        mm_item_counts: Mapping[str, int],
    ) -> Mapping[str, list[PlaceholderFeaturesInfo]]:
        """
        Helper to derive audio placeholders from video placeholders when
        use_audio_in_video=True.
        """
        if "video" not in placeholders:
            return placeholders

        num_videos = len(placeholders["video"])
        num_audios = int(mm_item_counts.get("audio", 0))
        if num_audios != num_videos:
            raise ValueError(
                f"use_audio_in_video requires equal number of audio and video items, "
                f"got {num_audios=}, {num_videos=}"
            )

        tokenizer = self.info.get_tokenizer()
        processor = self.info.get_hf_processor()
        audio_token_id = tokenizer.get_vocab()[processor.audio_token]

        result_placeholders = dict(placeholders)
        audio_placeholders = []

        # Each video is paired with one audio
        for video_idx, video_placeholder in enumerate(placeholders["video"]):
            # Create is_embed mask selecting only audio tokens
            audio_is_embed = torch.tensor(video_placeholder.tokens) == audio_token_id

            audio_placeholder = PlaceholderFeaturesInfo(
                modality="audio",
                item_idx=video_idx,
                start_idx=video_placeholder.start_idx,
                tokens=video_placeholder.tokens,
                is_embed=audio_is_embed,
            )
            audio_placeholders.append(audio_placeholder)

        result_placeholders["audio"] = audio_placeholders
        return result_placeholders

    def _get_raw_input_ids(
        self,
        token_ids: list[int],
        use_audio_in_video: bool = False,
    ) -> list[int]:
        tokenizer = self.info.get_tokenizer()
        vision_bos_token = tokenizer.encode(tokenizer.vision_bos_token)[0]
        vision_eos_token = tokenizer.encode(tokenizer.vision_eos_token)[0]
        audio_bos_token = tokenizer.encode(tokenizer.audio_bos_token)[0]
        audio_eos_token = tokenizer.encode(tokenizer.audio_eos_token)[0]
        audio_token = tokenizer.encode("<|audio_pad|>")[0]
        image_token = tokenizer.encode("<|image_pad|>")[0]
        video_token = tokenizer.encode("<|video_pad|>")[0]

        result = token_ids[:]
        if use_audio_in_video:
            while True:
                start = None
                for i in range(len(result) - 1):
                    if result[i : i + 2] == [vision_bos_token, audio_bos_token]:
                        start = i
                        break
                if start is not None:
                    end = None
                    for i in range(start + 2, len(result) - 1):
                        if result[i : i + 2] == [audio_eos_token, vision_eos_token]:
                            end = i
                            break
                    if end is not None:
                        result = (
                            result[:start]
                            + [vision_bos_token, video_token, vision_eos_token]
                            + result[end + 2 :]
                        )
                else:
                    break

        for mm_token in [audio_token, image_token, video_token]:
            compressed = []
            for x in result:
                if x != mm_token or (not compressed or compressed[-1] != mm_token):
                    compressed.append(x)
            result = compressed
        return result


class TLiveOmniConditionalGenerationMixin(Qwen2_5OmniConditionalGenerationMixin):
    def _process_audio_input(
        self,
        audio_input: Qwen2_5OmniAudioFeatureInputs,
        audio_hashes: list[str] | None = None,
        cached_audio_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        input_features = audio_input["input_features"]
        audio_feature_lengths = audio_input["audio_feature_lengths"]

        audio_output_lengths = _get_feat_extract_output_lengths(audio_feature_lengths)
        tower_input_features = input_features.to(self.audio_tower.dtype)
        audio_features = self.audio_tower(
            tower_input_features,
            feature_lens=audio_feature_lengths,
            aftercnn_lens=audio_output_lengths,
        )
        return audio_features.split(audio_output_lengths.tolist())

    def _process_image_input(self, image_input) -> tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return image_input["image_embeds"].type(self.visual.dtype)

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        with set_forward_context(None, self.vllm_config):
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size
        return image_embeds.split(sizes.tolist())

    def _process_video_input(
        self,
        video_input,
        video_hashes: list[str] | None = None,
        cached_video_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if video_input["type"] == "video_embeds":
            return video_input["video_embeds"].type(self.visual.dtype)

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
        with set_forward_context(None, self.vllm_config):
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size
        return video_embeds.split(sizes.tolist())


@MULTIMODAL_REGISTRY.register_processor(
    TLiveOmniMultiModalProcessor,
    info=TLiveOmniProcessingInfo,
    dummy_inputs=TLiveOmniDummyInputsBuilder,
)
class TLiveOmniForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
    TLiveOmniConditionalGenerationMixin,
    SupportsTranscription,
    IsHybrid,
    HasInnerState
):
    """TLiveOmni = Qwen3OmniMoe Audio + Qwen3.5 Vision + Qwen3.5 Text"""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    supported_languages = ISO639_1_SUPPORTED_LANGS

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()



    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        if modality.startswith("audio"):
            return "<|audio_start|><|audio_pad|><|audio_end|>"
        raise ValueError("Only image, video or audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config

        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = Qwen3OmniMoeAudioEncoder(
                config.audio_config,
                prefix=maybe_prefix(prefix, "audio_tower"),
            )

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                vision_config=config.vision_config,
                norm_eps=getattr(config.text_config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5ForCausalLM(
                vllm_config=vllm_config.with_hf_config(
                    config.text_config,
                    architectures=["Qwen3_5ForCausalLM"],
                ),
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}
        for input_key in kwargs:
            if input_key in ("pixel_values", "image_embeds") and "image" not in mm_input_by_modality:
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(**kwargs)
            if input_key in ("pixel_values_videos", "video_embeds") and "video" not in mm_input_by_modality:
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(**kwargs)
            if input_key in ("input_audio_features",) and "audio" not in mm_input_by_modality:
                mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(**kwargs)
        return mm_input_by_modality

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            self._tlive_mm_embedding_modalities = []
            return []
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()
        modalities: list[str] = list(getattr(self, "_tlive_mm_embedding_modalities", []))
        for modality in mm_input_by_modality:
            mm_input = mm_input_by_modality[modality]
            if modality == "image":
                chunks = tuple(self._process_image_input(mm_input))
            elif modality == "video":
                chunks = tuple(self._process_video_input(mm_input))
            elif modality == "audio":
                chunks = tuple(self._process_audio_input(mm_input))
            else:
                chunks = ()
            multimodal_embeddings += chunks
            modalities.extend([modality] * len(chunks))
        self._tlive_mm_embedding_modalities = modalities
        return multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids, self.language_model.embed_input_ids, is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if is_multimodal is None:
            return super().embed_input_ids(
                input_ids, multimodal_embeddings=multimodal_embeddings, is_multimodal=is_multimodal,
            )

        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        is_video = is_multimodal & (input_ids == video_token_id)
        is_audio = is_multimodal & (input_ids == audio_token_id)

        if bool(is_video.any() and is_audio.any()):
            return self._tlive_merge_embeddings_by_modality(
                inputs_embeds,
                input_ids,
                multimodal_embeddings,
                is_multimodal,
            )

        return super().embed_input_ids(
            input_ids, multimodal_embeddings=multimodal_embeddings, is_multimodal=is_multimodal,
        )

    def _tlive_merge_embeddings_by_modality(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings,
        is_multimodal: torch.Tensor,
    ) -> torch.Tensor:
        token_ids = {
            "image": self.config.image_token_id,
            "video": self.config.video_token_id,
            "audio": self.config.audio_token_id,
        }
        token_counts = {
            modality: int((is_multimodal & (input_ids == token_id)).sum().item())
            for modality, token_id in token_ids.items()
        }
        modalities = getattr(self, "_tlive_mm_embedding_modalities", None)
        embeddings_list = list(multimodal_embeddings)

        def is_valid_assignment(candidate: list[str]) -> bool:
            if len(candidate) != len(embeddings_list):
                return False
            assigned_counts = {modality: 0 for modality in token_ids}
            for modality, embedding in zip(candidate, embeddings_list, strict=True):
                if modality not in token_ids:
                    return False
                assigned_counts[modality] += int(embedding.shape[0])
            return all(
                assigned_counts[modality] == count
                for modality, count in token_counts.items()
                if count > 0 or assigned_counts[modality] > 0
            )

        if modalities is not None and len(modalities) >= len(embeddings_list):
            all_modalities = modalities
            window_size = len(embeddings_list)
            for start in range(len(all_modalities) - window_size + 1):
                candidate = all_modalities[start:start + window_size]
                if is_valid_assignment(candidate):
                    modalities = candidate
                    self._tlive_mm_embedding_modalities = all_modalities[start + window_size:]
                    break
            else:
                if len(all_modalities) == len(embeddings_list) and all(
                    modality in token_ids and token_counts[modality] > 0
                    for modality in all_modalities
                ):
                    modalities = all_modalities
                else:
                    modalities = None
        if modalities is None or len(modalities) != len(embeddings_list):
            present_modalities = [
                modality for modality, count in token_counts.items() if count > 0
            ]
            if len(present_modalities) == 1:
                modalities = [present_modalities[0]] * len(embeddings_list)
            else:
                inferred_modalities: list[str] | None = None

                def backtrack(
                    index: int,
                    assigned_counts: dict[str, int],
                    assignment: list[str],
                ) -> bool:
                    nonlocal inferred_modalities
                    if index == len(embeddings_list):
                        if all(
                            assigned_counts[modality] == token_counts[modality]
                            for modality in present_modalities
                        ):
                            inferred_modalities = list(assignment)
                            return True
                        return False
                    rows = int(embeddings_list[index].shape[0])
                    for modality in present_modalities:
                        next_count = assigned_counts[modality] + rows
                        if next_count > token_counts[modality]:
                            continue
                        assigned_counts[modality] = next_count
                        assignment.append(modality)
                        if backtrack(index + 1, assigned_counts, assignment):
                            return True
                        assignment.pop()
                        assigned_counts[modality] -= rows
                    return False

                backtrack(0, {modality: 0 for modality in present_modalities}, [])
                modalities = inferred_modalities or []
        if modalities is None or len(modalities) != len(embeddings_list):
            raise ValueError(
                "Missing modality metadata for multimodal embeddings; "
                f"got {len(embeddings_list)} embedding chunks and "
                f"modalities={modalities}."
            )

        embeddings_by_modality: dict[str, list[torch.Tensor]] = defaultdict(list)
        for modality, embedding in zip(modalities, embeddings_list, strict=True):
            embeddings_by_modality[modality].append(embedding)

        merged = inputs_embeds.clone()
        for modality, chunks in embeddings_by_modality.items():
            if modality not in token_ids:
                raise ValueError(f"Unsupported multimodal embedding modality: {modality}")
            positions = (is_multimodal & (input_ids == token_ids[modality])).nonzero(
                as_tuple=True
            )[0]
            embeddings = torch.cat(chunks, dim=0)
            if embeddings.shape[0] != positions.numel():
                raise ValueError(
                    f"{modality} embedding/token count mismatch: "
                    f"embeddings={embeddings.shape[0]}, tokens={positions.numel()}, "
                    f"chunk_sizes={[chunk.shape[0] for chunk in chunks]}"
                )
            merged[positions] = embeddings
        self._tlive_mm_embedding_modalities = []
        return merged

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["talker.", "code2wav."])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    # ---- MRoPE & Transcription ----

    def _compute_audio_token_count(self, audio_feature_length: int) -> int:
        return _get_feat_extract_output_lengths(torch.tensor([audio_feature_length])).item()

    def _get_audio_for_video_mapping(
        self, mm_features: list[MultiModalFeatureSpec]
    ) -> tuple[dict[int, int], set[int]]:
        videos_with_audio = [
            f for f in mm_features
            if f.modality == "video" and f.data.get("use_audio_in_video") and f.data["use_audio_in_video"].data.item()
        ]
        audios = [f for f in mm_features if f.modality == "audio"]
        mapping: dict[int, int] = {}
        paired_audio_offsets: set[int] = set()
        for i, video_f in enumerate(videos_with_audio):
            if i < len(audios):
                audio_len = audios[i].data["audio_feature_lengths"].data.item()
                mapping[video_f.mm_position.offset] = audio_len
                paired_audio_offsets.add(audios[i].mm_position.offset)
        return mapping, paired_audio_offsets

    def iter_mm_features(
        self, mm_features: list[MultiModalFeatureSpec]
    ) -> Iterator[tuple[int, str, dict[str, Any]]]:
        """
        Iterate over multimodal features sorted by position offset.

        Yields: (offset, modality, feature_data) where feature_data contains:
        - image: {"grid_t", "grid_h", "grid_w", "t_factor"}
        - video: {"grid_t", "grid_h", "grid_w", "t_factor",
                  "use_audio_in_video", "audio_feature_length"}
        - audio: {"audio_feature_length"}
        """
        config = self.config
        spatial_merge_size = config.vision_config.spatial_merge_size
        position_id_per_seconds = config.position_id_per_seconds
        sorted_features = sorted(mm_features, key=lambda f: f.mm_position.offset)
        audio_for_video, paired_audio_offsets = self._get_audio_for_video_mapping(sorted_features)

        for mm_feature in sorted_features:
            offset = mm_feature.mm_position.offset
            modality = mm_feature.modality
            if modality == "image":
                t, h, w = mm_feature.data["image_grid_thw"].data.tolist()
                yield (offset, "image", {
                    "grid_t": t, "grid_h": h // spatial_merge_size,
                    "grid_w": w // spatial_merge_size, "t_factor": position_id_per_seconds,
                })
            elif modality == "video":
                t, h, w = mm_feature.data["video_grid_thw"].data.tolist()
                second_per_grid_ts = 2.0
                if mm_feature.data.get("second_per_grid_ts"):
                    second_per_grid_ts = mm_feature.data["second_per_grid_ts"].data.item()
                use_audio_in_video = bool(
                    mm_feature.data.get("use_audio_in_video") and mm_feature.data["use_audio_in_video"].data.item()
                )
                yield (offset, "video", {
                    "grid_t": t, "grid_h": h // spatial_merge_size,
                    "grid_w": w // spatial_merge_size,
                    "t_factor": second_per_grid_ts * position_id_per_seconds,
                    "use_audio_in_video": use_audio_in_video,
                    "audio_feature_length": audio_for_video.get(offset),
                })
            elif modality == "audio":
                if offset not in paired_audio_offsets:
                    audio_len = mm_feature.data["audio_feature_lengths"].data.item()
                    yield offset, "audio", {"audio_feature_length": audio_len}

    def _compute_interleaved_positions(
        self, start_idx: int, data: dict[str, Any]
    ) -> tuple[np.ndarray, int]:
        grid_t, grid_h, grid_w = data["grid_t"], data["grid_h"], data["grid_w"]
        t_factor = data["t_factor"]
        audio_len = self._compute_audio_token_count(data["audio_feature_length"])

        h_index = np.tile(np.arange(grid_h).reshape(1, -1, 1), (grid_t, 1, grid_w)).flatten()
        w_index = np.tile(np.arange(grid_w).reshape(1, 1, -1), (grid_t, grid_h, 1)).flatten()
        t_index = np.repeat((np.arange(grid_t) * t_factor).astype(np.int64), grid_h * grid_w)

        video_pos = np.stack([t_index, h_index, w_index]) + start_idx
        audio_pos = np.broadcast_to(np.arange(audio_len), (3, audio_len)) + start_idx

        pos_ids_list: list[np.ndarray] = []
        video_idx, audio_idx = 0, 0
        num_video = grid_t * grid_h * grid_w
        while video_idx < num_video and audio_idx < audio_len:
            if video_pos[0, video_idx] <= audio_pos[0, audio_idx]:
                pos_ids_list.append(video_pos[:, video_idx:video_idx+1])
                video_idx += 1
            else:
                pos_ids_list.append(audio_pos[:, audio_idx:audio_idx+1])
                audio_idx += 1
        if video_idx < num_video:
            pos_ids_list.append(video_pos[:, video_idx:])
        if audio_idx < audio_len:
            pos_ids_list.append(audio_pos[:, audio_idx:])
        return np.concatenate(pos_ids_list, axis=1), num_video + audio_len

    @classmethod
    def get_speech_to_text_config(cls, model_config: ModelConfig, task_type: str) -> SpeechToTextConfig:
        processor = cached_processor_from_config(model_config, processor_cls=TLiveOmniProcessor)
        return SpeechToTextConfig(
            max_audio_clip_s=processor.feature_extractor.chunk_length,
            sample_rate=processor.feature_extractor.sampling_rate,
            min_energy_split_window_size=None,
        )

    @classmethod
    def get_generation_prompt(
        cls, audio: np.ndarray, stt_config: SpeechToTextConfig,
        model_config: ModelConfig, language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str, to_language: str | None,
    ) -> PromptType:
        instruction = "Transcribe" if task_type == "transcribe" else "Translate"
        instruction += " this audio"
        if task_type == "translate" and to_language is None:
            to_language = "en"
        full_lang_name = cls.supported_languages.get(language, "")
        full_lang_name_to = cls.supported_languages.get(to_language, "")
        if task_type == "transcribe" and full_lang_name:
            instruction += f" into {full_lang_name}"
        elif task_type == "translate":
            if full_lang_name: instruction += f" from {full_lang_name}"
            if full_lang_name_to: instruction += f" into {full_lang_name_to}"
        instruction += "."
        if request_prompt: instruction += f" {request_prompt}"

        processor = cached_processor_from_config(model_config, processor_cls=TLiveOmniProcessor)
        messages = [{"role": "user", "content": f"<|audio_start|><|audio_pad|><|audio_end|>{instruction}"}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return cast(PromptType, {"multi_modal_data": {"audio": (audio, stt_config.sample_rate)}, "prompt": prompt})

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        config = self.config
        spatial_merge_size = config.vision_config.spatial_merge_size
        image_token_id = config.image_token_id
        video_token_id = config.video_token_id
        audio_token_id = config.audio_token_id
        vision_start_token_id = config.vision_start_token_id
        vision_end_token_id = config.vision_end_token_id
        audio_start_token_id = config.audio_start_token_id
        audio_end_token_id = config.audio_end_token_id
        position_id_per_seconds = config.position_id_per_seconds

        sorted_features = sorted(mm_features, key=lambda f: f.mm_position.offset)
        image_grid_thw_list: list[list[int]] = []
        video_grid_thw_list: list[list[int]] = []
        audio_seqlens_list: list[int] = []
        second_per_grids_list: list[float] = []
        use_audio_in_video = False

        for feature in sorted_features:
            data = feature.data
            if feature.modality == "image":
                image_grid_thw_list.append(data["image_grid_thw"].data.tolist())
            elif feature.modality == "video":
                video_grid_thw_list.append(data["video_grid_thw"].data.tolist())
                if data.get("second_per_grid_ts"):
                    second_per_grids_list.append(float(data["second_per_grid_ts"].data.item()))
                else:
                    second_per_grids_list.append(1.0)
                if data.get("use_audio_in_video") and data["use_audio_in_video"].data.item():
                    use_audio_in_video = True
            elif feature.modality == "audio":
                audio_seqlens_list.append(int(data["audio_feature_lengths"].data.item()))

        image_grid_thw = torch.tensor(image_grid_thw_list, dtype=torch.long) if image_grid_thw_list else None
        video_grid_thw = torch.tensor(video_grid_thw_list, dtype=torch.long) if video_grid_thw_list else None
        audio_seqlens = torch.tensor(audio_seqlens_list, dtype=torch.long) if audio_seqlens_list else None
        second_per_grids = torch.tensor(second_per_grids_list, dtype=torch.float) if second_per_grids_list else None

        video_chunk_end_indices: set[int] = set()
        if video_grid_thw is not None:
            if second_per_grids is None:
                second_per_grids = torch.ones(video_grid_thw.shape[0], dtype=torch.float)
            second_per_grids = torch.repeat_interleave(
                second_per_grids,
                video_grid_thw[:, 0].detach().cpu(),
                dim=0,
            )
            video_chunk_end_indices = set(torch.cumsum(video_grid_thw[:, 0], dim=0).tolist())
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        if image_grid_thw is None and video_grid_thw is None:
            llm_positions = np.broadcast_to(np.arange(len(input_tokens)), (3, len(input_tokens)))
            positions_tensor = torch.from_numpy(llm_positions)
            return positions_tensor, 0

        vision_end_indices = [idx for idx, token in enumerate(input_tokens) if token == vision_end_token_id]
        audio_end_indices = [idx for idx, token in enumerate(input_tokens) if token == audio_end_token_id]
        vision_tokens = [input_tokens[idx - 1] for idx in vision_end_indices]
        audio_start_nums = sum(token == audio_start_token_id for token in input_tokens)
        image_nums = sum(token == image_token_id for token in vision_tokens)
        video_audio_nums = sum(token == audio_end_token_id for token in vision_tokens)
        video_only_nums = sum(token == video_token_id for token in vision_tokens)
        video_nums = video_audio_nums if use_audio_in_video and video_audio_nums > 0 else video_only_nums
        audio_nums = audio_start_nums - video_audio_nums

        llm_pos_ids_list: list[np.ndarray] = []
        st = 0
        image_idx = 0
        video_idx = 0
        audio_idx = 0
        remain_images = image_nums
        remain_videos = video_nums
        remain_audios = audio_nums
        multimodal_nums = image_nums + video_nums + audio_nums

        def _next_position() -> int:
            return int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0

        def _append_1d(length: int, start: int | None = None) -> None:
            if length <= 0:
                return
            base = _next_position() if start is None else start
            llm_pos_ids_list.append(np.broadcast_to(np.arange(length), (3, length)) + base)

        def _append_vision_grid(start: int, grid_t: int, grid_h: int, grid_w: int, t_factor: float) -> int:
            llm_grid_h = int(grid_h) // spatial_merge_size
            llm_grid_w = int(grid_w) // spatial_merge_size
            t_index = (
                np.arange(int(grid_t), dtype=np.float32) * float(t_factor) + 1e-6
            ).astype(np.int64)
            t_index = np.repeat(t_index, llm_grid_h * llm_grid_w)
            h_index = np.tile(
                np.arange(llm_grid_h).reshape(1, -1, 1),
                (int(grid_t), 1, llm_grid_w),
            ).flatten()
            w_index = np.tile(
                np.arange(llm_grid_w).reshape(1, 1, -1),
                (int(grid_t), llm_grid_h, 1),
            ).flatten()
            llm_pos_ids_list.append(np.stack([t_index, h_index, w_index]) + start)
            return int(grid_t) * int(grid_h) * int(grid_w) // (spatial_merge_size**2)

        for _ in range(int(multimodal_nums)):
            st_idx = _next_position()
            if (image_token_id in input_tokens or video_token_id in input_tokens) and (
                remain_videos > 0 or remain_images > 0
            ):
                ed_vision_start = input_tokens.index(vision_start_token_id, st)
                ed_vision_end = input_tokens.index(vision_end_token_id, st)
            else:
                ed_vision_start = len(input_tokens) + 1
                ed_vision_end = len(input_tokens) + 1

            try:
                next_audio_start = input_tokens.index(audio_start_token_id, st)
            except ValueError:
                next_audio_start = len(input_tokens) + 1

            if use_audio_in_video and ed_vision_start < next_audio_start < ed_vision_end:
                ed_audio_start = next_audio_start
            elif remain_audios > 0:
                ed_audio_start = next_audio_start
            else:
                ed_audio_start = len(input_tokens) + 1

            min_ed = min(ed_vision_start, ed_audio_start)
            text_len = min_ed - st
            if text_len:
                _append_1d(text_len, st_idx)
                st_idx += text_len

            bos_len = 1
            eos_len = 1
            _append_1d(bos_len, st_idx)
            st_idx += bos_len

            if min_ed == ed_audio_start:
                if audio_seqlens is None:
                    raise RuntimeError("audio_seqlens is required for audio position ids")
                try:
                    audio_end_token_indice = input_tokens.index(audio_end_token_id, ed_audio_start)
                except ValueError as exc:
                    raise ValueError("Standalone audio span missing audio_end token.") from exc
                audio_len = audio_end_token_indice - ed_audio_start - 1
                if audio_len < 0:
                    raise ValueError("Standalone audio span has invalid token order.")
                if any(token != audio_token_id for token in input_tokens[ed_audio_start + 1:audio_end_token_indice]):
                    raise ValueError("Standalone audio span expected only audio_pad tokens between audio_start and audio_end.")
                _append_1d(audio_len, st_idx)
                st += int(text_len + bos_len + audio_len + eos_len)
                audio_idx += 1
                remain_audios -= 1

            elif min_ed == ed_vision_start and input_tokens[ed_vision_start + 1] == image_token_id:
                assert image_grid_thw is not None
                grid_t, grid_h, grid_w = image_grid_thw[image_idx].tolist()
                image_len = _append_vision_grid(
                    st_idx,
                    grid_t,
                    grid_h,
                    grid_w,
                    position_id_per_seconds,
                )
                st += int(text_len + bos_len + image_len + eos_len)
                image_idx += 1
                remain_images -= 1

            elif min_ed == ed_vision_start and ed_vision_end < ed_audio_start:
                assert video_grid_thw is not None and second_per_grids is not None
                _, grid_h, grid_w = video_grid_thw[video_idx].tolist()
                video_len = _append_vision_grid(
                    st_idx,
                    1,
                    grid_h,
                    grid_w,
                    float(second_per_grids[video_idx].item()) * position_id_per_seconds,
                )
                st += int(text_len + bos_len + video_len + eos_len)
                video_idx += 1
                remain_videos -= 1

            elif min_ed == ed_vision_start and ed_vision_end > ed_audio_start:
                assert video_grid_thw is not None and second_per_grids is not None
                eos_len = 2
                if ed_audio_start != ed_vision_start + bos_len:
                    raise ValueError("Audio-in-video span expected audio_start immediately after vision_start.")
                _, grid_h, grid_w = video_grid_thw[video_idx].tolist()
                grid_tokens = int(grid_h) * int(grid_w) // (spatial_merge_size**2)
                audio_end_token_indice = next(
                    idx for idx in audio_end_indices if idx > ed_audio_start
                )
                if audio_end_token_indice != ed_vision_end - 1:
                    raise ValueError("Audio-in-video span expected audio_end immediately before vision_end.")
                video_token_start = ed_audio_start + 1
                video_token_end = video_token_start
                while video_token_end < audio_end_token_indice and input_tokens[video_token_end] == video_token_id:
                    video_token_end += 1
                actual_video_len = video_token_end - video_token_start
                if actual_video_len % grid_tokens != 0:
                    raise ValueError(
                        f"Video/audio span has {actual_video_len} video tokens, which is not divisible by one temporal grid of {grid_tokens} tokens."
                    )
                grid_t = actual_video_len // grid_tokens
                if grid_t != 1:
                    raise ValueError(f"Audio-in-video span expected exactly one temporal grid, got {grid_t}.")

                audio_start_st_idx = st_idx
                content_st_idx = st_idx + 1
                _append_1d(1, audio_start_st_idx)
                _append_vision_grid(
                    content_st_idx,
                    1,
                    grid_h,
                    grid_w,
                    float(second_per_grids[video_idx].item()) * position_id_per_seconds,
                )
                audio_len = audio_end_token_indice - video_token_end
                if audio_len > 0:
                    if any(token != audio_token_id for token in input_tokens[video_token_end:audio_end_token_indice]):
                        raise ValueError("Audio-in-video span expected only audio_pad tokens between video_pad and audio_end.")
                    _append_1d(audio_len, content_st_idx)
                st += int(text_len + bos_len + 1 + audio_len + actual_video_len + eos_len)
                video_idx += int(grid_t)
                if video_idx in video_chunk_end_indices:
                    audio_idx += 1
                remain_videos -= 1

            st_idx = _next_position()
            _append_1d(eos_len, st_idx)

        if st < len(input_tokens):
            _append_1d(len(input_tokens) - st, _next_position())

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        if llm_positions.shape[1] != len(input_tokens):
            raise RuntimeError(
                f"Position ids length {llm_positions.shape[1]} != input ids length {len(input_tokens)}"
            )

        mrope_position_delta = int(llm_positions.max()) + 1 - len(input_tokens)
        positions_tensor = torch.from_numpy(llm_positions)
        return positions_tensor, mrope_position_delta


    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.merger",
            tower_model=["visual.", "audio_tower."],
        )


if __name__ == '__main__':
    pass