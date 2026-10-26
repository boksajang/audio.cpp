# Fork notes: boksajang/audio.cpp

This repository is a fork of [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp),
maintained for the Shorts Factory desktop app, which loads the audio.cpp C API
(`audiocpp.dll` / `libaudiocpp.dylib`) to run MOSS-TTS v1.5. Shorts Factory
downloads its audio.cpp binaries from this fork's GitHub Releases.

audio.cpp is licensed under the Apache License 2.0 (`LICENSE`, Copyright 2026
ShugoAI LLC). The upstream license and copyright notices are kept unchanged.
Files changed in this fork are listed below, as required by Apache-2.0 §4(b).

## Branches and tags

| Ref | Meaning |
|---|---|
| `sf/ci` | Upstream tag `v0.9.2` (commit `0097d365a00b52a6991ad7618f0a04e8d2935e47`) plus the CI changes below. No source code changes. |
| `sf-v0.9.2-base` | Baseline release built from `sf/ci`: upstream v0.9.2 sources, fork CI. |

Release tags use the form `sf-vX.Y.Z[-suffix]`. Assets are named
`audio-<tag>-<kind>-<platform>...`, and the embedded version string is
`X.Y.Z-sf[-suffix]`.

## Modifications

| File | Change |
|---|---|
| `.github/workflows/release.yml` | Triggered by `sf-v*` tags (upstream `v*` trigger removed) or `workflow_dispatch`. Build matrix restricted to Windows x64 CUDA 13.3, Windows x64 CUDA 12.4 and macOS arm64 Metal; all other jobs removed. Release tag/version mapping for `sf-` tags. Release notes written explicitly instead of `--generate-notes`; `SHA256SUMS.txt` attached to each release. Build steps, toolchains and CMake options are unchanged from upstream v0.9.2. |
| `FORK_NOTES.md` | This file. |
