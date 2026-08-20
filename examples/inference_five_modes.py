#!/usr/bin/env python3
"""Run TLive-Omni inference in one of five mutually exclusive modes."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


DEFAULT_PROMPTS = {
    "text": "Briefly explain why multimodal context can improve an answer.",
    "image": "Describe this image.",
    "audio": "Transcribe and summarize this audio.",
    "vocal-video": "Describe the video, including relevant speech and sounds.",
    "silence-video": "Describe the visual events in this video.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hub model ID or exported snapshot directory")
    parser.add_argument(
        "--mode",
        required=True,
        choices=tuple(DEFAULT_PROMPTS),
        help="Exactly one input modality mode",
    )
    parser.add_argument("--prompt", help="User prompt; each mode has a default")
    parser.add_argument("--image", help="Local image path, only for image mode")
    parser.add_argument("--audio", help="Local audio path, only for audio mode")
    parser.add_argument("--video", help="Local video path, only for a video mode")
    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument("--fps", type=float, help="Sampled frames per second for a video mode")
    frame_group.add_argument("--num-frames", type=int, help="Requested sampled frame count for a video mode")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--attn-implementation",
        choices=("flash_attention_2", "sdpa"),
        default="flash_attention_2",
        help="SDPA is a fallback/debug path; FlashAttention 2 is the primary path",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
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

    video_options = (args.fps, args.num_frames)
    if args.mode not in {"vocal-video", "silence-video"} and any(value is not None for value in video_options):
        parser.error("--fps and --num-frames are only valid in a video mode")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")
    if args.num_frames is not None and args.num_frames <= 0:
        parser.error("--num-frames must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")


def build_messages(args: argparse.Namespace) -> list[dict]:
    content = []
    if args.mode == "image":
        content.append({"type": "image", "path": args.image})
    elif args.mode == "audio":
        content.append({"type": "audio", "path": args.audio})
    elif args.mode in {"vocal-video", "silence-video"}:
        content.append({"type": "video", "path": args.video})
    content.append({"type": "text", "text": args.prompt or DEFAULT_PROMPTS[args.mode]})
    return [{"role": "user", "content": content}]


def main() -> None:
    args = parse_args()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    ).eval()

    videos_kwargs = {}
    if args.fps is not None:
        videos_kwargs["fps"] = args.fps
    if args.num_frames is not None:
        videos_kwargs["num_frames"] = args.num_frames
    inputs = processor.apply_chat_template(
        build_messages(args),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
        use_audio_in_video=args.mode == "vocal-video",
        videos_kwargs=videos_kwargs,
    )

    prompt_length = inputs["input_ids"].shape[-1]
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )
    answer_ids = generated_ids[:, prompt_length:]
    answer = processor.batch_decode(
        answer_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print(answer.strip())


if __name__ == "__main__":
    main()
