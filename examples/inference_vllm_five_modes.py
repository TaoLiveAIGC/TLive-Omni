#!/usr/bin/env python3
"""Run TLive-Omni with vLLM in one of five mutually exclusive modes.

This is the official offline inference entry point. It uses a normally
installed vLLM package (no source overlay). Install the TLive-adapted vLLM
wheel first; see the vLLM installation section in README.md.

Depending on the environment, `from vllm import LLM` may require preloading
a compatible libstdc++, e.g.:

    LD_PRELOAD=/path/to/libstdc++.so.6 python inference_vllm_five_modes.py ...

Examples:

    python inference_vllm_five_modes.py --model <MODEL_DIR> --mode text
    python inference_vllm_five_modes.py --model <MODEL_DIR> --mode image --image data/image.jpg
    python inference_vllm_five_modes.py --model <MODEL_DIR> --mode audio --audio data/audio.mp3
    python inference_vllm_five_modes.py --model <MODEL_DIR> --mode vocal-video --video data/vocal_video.mp4
    python inference_vllm_five_modes.py --model <MODEL_DIR> --mode silence-video --video data/silence_video.mp4
"""

from __future__ import annotations

import argparse
from typing import Any


DEFAULT_PROMPTS = {
    "text": "Briefly explain why multimodal context can improve an answer.",
    "image": "Describe this image.",
    "audio": "Transcribe and summarize this audio.",
    "vocal-video": "Describe the video, including relevant speech and sounds.",
    "silence-video": "Describe the visual events in this video.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local TLive HF snapshot or model ID")
    parser.add_argument("--mode", required=True, choices=tuple(DEFAULT_PROMPTS))
    parser.add_argument("--prompt", help="User prompt; each mode has a default")
    parser.add_argument("--image", help="Local image path, only for image mode")
    parser.add_argument("--audio", help="Local audio path, only for audio mode")
    parser.add_argument("--video", help="Local video path, only for a video mode")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--mm-processor-cache-gb", type=float, default=0.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--video-total-pixels", type=int)
    parser.add_argument("--video-shortest-edge", type=int)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    validate_args(parser, args)
    return args


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    expected_media = {
        "text": None,
        "image": "image",
        "audio": "audio",
        "vocal-video": "video",
        "silence-video": "video",
    }[args.mode]
    supplied_media = {name for name in ("image", "audio", "video") if getattr(args, name) is not None}
    expected = set() if expected_media is None else {expected_media}
    if supplied_media != expected:
        parser.error(
            f"mode {args.mode!r} accepts exactly {sorted(expected) or ['no media']}; "
            f"received {sorted(supplied_media) or ['no media']}"
        )
    if args.max_model_len <= 0:
        parser.error("--max-model-len must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.max_num_seqs <= 0:
        parser.error("--max-num-seqs must be positive")
    if args.max_num_batched_tokens is not None and args.max_num_batched_tokens <= 0:
        parser.error("--max-num-batched-tokens must be positive")
    if not 0 <= args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in [0, 1]")
    if args.mm_processor_cache_gb < 0:
        parser.error("--mm-processor-cache-gb must be non-negative")
    for name in ("fps", "video_total_pixels", "video_shortest_edge", "max_frames"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")


def build_messages(args: argparse.Namespace) -> list[dict[str, Any]]:
    content = []
    if args.mode == "image":
        content.append({"type": "image", "path": args.image})
    elif args.mode == "audio":
        content.append({"type": "audio", "audio": args.audio})
    elif args.mode in {"vocal-video", "silence-video"}:
        content.append({"type": "video", "video": args.video})
    content.append({"type": "text", "text": args.prompt or DEFAULT_PROMPTS[args.mode]})
    return [{"role": "user", "content": content}]


def build_videos_kwargs(args: argparse.Namespace, use_audio_in_video: bool) -> dict[str, Any]:
    videos_kwargs: dict[str, Any] = {"use_audio_in_video": use_audio_in_video, "return_metadata": True}
    if args.fps is not None:
        videos_kwargs["fps"] = args.fps
    if args.max_frames is not None:
        videos_kwargs["max_frames"] = args.max_frames
    if args.video_total_pixels is not None or args.video_shortest_edge is not None:
        size: dict[str, int] = {}
        if args.video_total_pixels is not None:
            size["longest_edge"] = args.video_total_pixels
        if args.video_shortest_edge is not None:
            size["shortest_edge"] = args.video_shortest_edge
        videos_kwargs["size"] = size
    return videos_kwargs


def build_vllm_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, int]]:
    from transformers import AutoProcessor
    from vllm.model_executor.models.tlive_omni_processing import process_audio_info

    messages = build_messages(args)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )

    inputs: dict[str, Any] = {"prompt": prompt}
    limit_mm_per_prompt: dict[str, int] = {}

    if args.mode == "image":
        inputs["multi_modal_data"] = {"image": [args.image]}
        limit_mm_per_prompt = {"image": 1}
    elif args.mode == "audio":
        inputs["multi_modal_data"] = {"audio": process_audio_info(messages, use_audio_in_video=False)}
        limit_mm_per_prompt = {"audio": 1}
    elif args.mode == "silence-video":
        inputs["multi_modal_data"] = {"video": [args.video]}
        inputs["mm_processor_kwargs"] = {"videos_kwargs": build_videos_kwargs(args, use_audio_in_video=False)}
        limit_mm_per_prompt = {"video": 1}
    elif args.mode == "vocal-video":
        inputs["multi_modal_data"] = {
            "video": [args.video],
            "audio": process_audio_info(messages, use_audio_in_video=True),
        }
        inputs["mm_processor_kwargs"] = {"videos_kwargs": build_videos_kwargs(args, use_audio_in_video=True)}
        limit_mm_per_prompt = {"audio": 1, "video": 1}

    return inputs, limit_mm_per_prompt


def main() -> None:
    args = parse_args()
    inputs, limit_mm_per_prompt = build_vllm_inputs(args)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens or args.max_model_len,
        limit_mm_per_prompt=limit_mm_per_prompt or None,
        mm_processor_cache_gb=args.mm_processor_cache_gb,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    print(outputs[0].outputs[0].text.strip())


if __name__ == "__main__":
    main()
