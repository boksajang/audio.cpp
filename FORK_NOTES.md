# Fork notes: boksajang/audio.cpp

This repository is a fork of [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp),
maintained for the Shorts Factory desktop app, which loads the audio.cpp C API
(`audiocpp.dll` / `libaudiocpp.dylib`) to run MOSS-TTS v1.5. Shorts Factory
downloads its audio.cpp binaries from this fork's GitHub Releases.

audio.cpp is licensed under the Apache License 2.0 (`LICENSE`, Copyright 2026
ShugoAI LLC). The upstream license and copyright notices are kept unchanged.
Files changed in this fork are listed below and carry a "Modified by the Shorts
Factory fork" note where the file format allows comments, as required by
Apache-2.0 §4(b).

## Branches and tags

| Ref | Meaning |
|---|---|
| `sf/ci` | Upstream tag `v0.9.2` (commit `0097d365a00b52a6991ad7618f0a04e8d2935e47`) plus the CI changes below. No source code changes. |
| `sf/continuation` | `sf/ci` plus the MOSS-TTS-v1.5 continuation patch below. |
| `sf-v0.9.2-base` | Release built from `sf/ci`: upstream v0.9.2 sources, fork CI. |
| `sf-v0.9.2-cont1` | Release built from `sf/continuation`. |

Release tags use the form `sf-vX.Y.Z[-suffix]`. Assets are named
`audio-<tag>-<kind>-<platform>...`, and the embedded version string is
`X.Y.Z-sf[-suffix]`.

## Modifications

### CI (`sf/ci`)

| File | Change |
|---|---|
| `.github/workflows/release.yml` | Triggered by `sf-v*` tags (upstream `v*` trigger removed) or `workflow_dispatch`. Build matrix restricted to Windows x64 CUDA 13.3, Windows x64 CUDA 12.4 and macOS arm64 Metal; all other jobs removed. Release tag/version mapping for `sf-` tags. Release notes written explicitly instead of `--generate-notes`; `SHA256SUMS.txt` attached to each release. Build steps, toolchains and CMake options are unchanged from upstream v0.9.2. |
| `FORK_NOTES.md` | This file. |

### MOSS-TTS-v1.5 continuation (`sf/continuation`)

The reference implementation's `mode="continuation"` for `moss_tts_v15`: a take
continues a given piece of audio instead of starting fresh. The C API is
unchanged (ABI 0.2.0, no new symbols): the audio to continue is the request's
audio input (`audiocpp_request_set_audio`) and its transcript the
`continuation_text` request option. Details: `docs/models/moss_tts_v15.md`,
section "Continuation (Shorts Factory fork)".

| File | Change |
|---|---|
| `include/engine/community_models/moss_tts_v15/continuation.h` | New. Header-only helpers: text join, length-bound shift, prefix + generated code join, ratio trim. |
| `include/engine/community_models/moss_tts_v15/prompt.h`, `src/community_models/moss_tts_v15/prompt.cpp` | `build_continuation_prefix` (wraps the family's `build_moss_continuation_prefix` with the v1.5 user turn). |
| `include/engine/community_models/moss_tts_v15/session.h`, `src/community_models/moss_tts_v15/session.cpp` | Continuation path in `run()` (one unchunked take, `begin_continuation`, bounds moved past the prefix frames, prefix + generated decoded together and trimmed by ratio). Session option `moss_tts_v15.encoder_precision` (`f16` default = upstream, `f32` = encoder widened to f32 in the graph as `moss_ttsd` does; alias `moss_tts_v15.continuation_encoder`). Behaviour without `continuation_text`/audio input and with the default encoder precision is unchanged from upstream. |
| `src/community_models/moss_tts_v15/loader.cpp`, `model_specs/moss_tts_v15.json` | Declare the `continuation_text` request option and the `encoder_precision` session option. |
| `docs/models/moss_tts_v15.md` | Continuation section. |
| `tests/unittests/test_moss_tts_v15_continuation.cpp`, `CMakeLists.txt` | New unit test `moss_tts_v15_continuation_test` (ENGINE_BUILD_TESTS). |
| `tests/moss_tts_v15/prompt_parity.cpp` | Accepts continuation fixtures (`assistant_codes`). |
| `tests/moss_tts_v15/reference/dump_continuation_fixture.py` | New. Dumps a continuation fixture from the reference processor. |
| `.github/workflows/sf-check.yml` | New. Compiles the moss_tts_v15 objects and runs its unit tests on every push to `sf/**`. |
