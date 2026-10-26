# Continuation-prompt fixture for MOSS-TTS-v1.5 (the reference's mode="continuation").
#
# Added by the Shorts Factory fork (boksajang/audio.cpp), 2026-10; not part of
# upstream audio.cpp. See FORK_NOTES.md.
#
# Prompt-only: the processor is built for its tokenizer and message rendering,
# and the codec encodes the two recordings. The 8B weights are never loaded, so
# this runs on CPU. The output is read by moss_tts_v15_prompt_parity:
#
#   python dump_continuation_fixture.py --model <MOSS-TTS-v1.5 dir> \
#       --codec <MOSS-Audio-Tokenizer dir> --reference voice.wav --prefix tail.wav \
#       --continuation-text "Transcript of tail.wav." --text "The next paragraph." \
#       --language Korean --out ref_prompt_ko_continuation.json
#   moss_tts_v15_prompt_parity --model <MOSS-TTS-v1.5 dir> --prompt ref_prompt_ko_continuation.json
import argparse
import json
import sys

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor

SAMPLES_PER_FRAME = 1920  # 24 kHz / 12.5 frames a second


def join_text(prefix_text: str, text: str) -> str:
    """Same rule as moss_tts_v15::continuation::join_text."""
    if not prefix_text:
        return text
    if not text:
        return prefix_text
    if prefix_text[-1].isspace() or text[0].isspace():
        return prefix_text + text
    return prefix_text + " " + text


def encode(tokenizer, path: str, rate: int) -> torch.Tensor:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.T.mean(axis=0, keepdims=True)
    if sr != rate:
        raise SystemExit(f"{path}: expected {rate} Hz (resample first so both sides see the same samples), got {sr}")
    x = x[:, : (x.shape[1] // SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME]
    with torch.no_grad():
        enc = tokenizer.encode(
            torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(tokenizer.device), return_dict=True)
    return enc.audio_codes.squeeze(1)  # (n_vq, frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--codec", required=True)
    ap.add_argument("--reference", required=True, help="channel voice recording (24 kHz)")
    ap.add_argument("--prefix", required=True, help="audio the take continues (24 kHz)")
    ap.add_argument("--continuation-text", required=True, help="exact transcript of --prefix")
    ap.add_argument("--text", required=True, help="the new text")
    ap.add_argument("--language", default=None)
    ap.add_argument("--n-vq", type=int, default=32)
    ap.add_argument("--name", default="continuation")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, codec_path=args.codec)
    pm = sys.modules[type(proc).__module__]
    rate = int(getattr(proc.model_config, "sampling_rate", 24000))
    tok = proc.audio_tokenizer

    ref_codes = encode(tok, args.reference, rate)[: args.n_vq]
    prefix_codes = encode(tok, args.prefix, rate)[: args.n_vq]
    print("reference codes:", tuple(ref_codes.shape), "prefix codes:", tuple(prefix_codes.shape), flush=True)

    text = join_text(args.continuation_text, args.text)
    user = pm.UserMessage(text=text, reference=[ref_codes.transpose(0, 1).contiguous()], language=args.language)
    assistant = pm.AssistantMessage(audio_codes_list=[prefix_codes.transpose(0, 1).contiguous()])
    ids = proc([[user, assistant]], mode="continuation", n_vq=args.n_vq)["input_ids"]
    print("prompt rows:", tuple(ids.shape), flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "name": args.name,
            "text": user.text, "instruction": user.instruction, "language": user.language, "tokens": user.tokens,
            "continuation_text": args.continuation_text, "new_text": args.text,
            "content": user._content,
            "reference_codes": ref_codes.cpu().tolist(),       # [n_vq][frames], codec layout
            "assistant_codes": prefix_codes.cpu().tolist(),    # [n_vq][frames], codec layout
            "input_ids": ids[0].tolist(),
        }, f, ensure_ascii=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
