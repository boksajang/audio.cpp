#pragma once

// Added by the Shorts Factory fork (boksajang/audio.cpp), 2026-10: MOSS-TTS-v1.5
// continuation. Not part of upstream audio.cpp; see FORK_NOTES.md.
//
// Continuation is the reference's mode="continuation" (processing_moss_tts.py):
// the user turn carries the prefix audio's transcript followed by the new text,
// and an unfinished assistant turn carries the prefix audio's codes, so the model
// carries on speaking from where that audio stopped instead of starting afresh.
// The session wiring lives in session.cpp; the small pieces of arithmetic it
// needs are kept here, free of the runtime, so a unit test can pin them.

#include "engine/framework/decoders/moss_tts_delay/delay_decoder.h"
#include "engine/framework/decoders/moss_tts_delay/prompt.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::moss_tts_v15::continuation {

// The user turn's text: the prefix transcript, then the new text. The reference
// concatenates the two; one space keeps the sentences apart, and the v1.5 text
// normaliser leaves a single space between sentences as it is.
inline std::string join_text(const std::string & prefix_text, const std::string & text) {
    if (prefix_text.empty()) {
        return text;
    }
    if (text.empty()) {
        return prefix_text;
    }
    const auto ends_space = std::isspace(static_cast<unsigned char>(prefix_text.back())) != 0;
    const auto starts_space = std::isspace(static_cast<unsigned char>(text.front())) != 0;
    return ends_space || starts_space ? prefix_text + text : prefix_text + " " + text;
}

// The decoder counts frames from the start of the assistant audio, and in
// continuation the prefix frames are already on that clock. Bounds derived for
// the new text alone are moved past them, the floor as well as the ceiling: v1.5
// floors the length, and an unmoved floor would be met by the prefix alone.
inline decoders::MossTtsDelayLengthBounds shift_bounds(
    decoders::MossTtsDelayLengthBounds bounds, int64_t prefix_frames) {
    if (prefix_frames <= 0) {
        return bounds;
    }
    if (bounds.min_frames > 0) {
        bounds.min_frames += prefix_frames;
    }
    if (bounds.max_frames > 0) {
        bounds.max_frames += prefix_frames;
    }
    return bounds;
}

// The prefix codes ([n_vq][prefix_frames], codec layout) followed by the
// generated ones ([codebooks, frames] row-major), as one [codebooks, total]
// row-major block. Decoding the two together keeps the causal codec's context;
// decoding the generated frames alone starts it cold and leaves a seam.
inline std::vector<int32_t> join_codes(
    const decoders::MossReferenceAudio & prefix,
    const std::vector<int32_t> & generated,
    int64_t codebooks,
    int64_t frames) {
    if (codebooks <= 0 || frames < 0) {
        throw std::runtime_error("MOSS-TTS-v1.5 continuation: invalid generated code shape");
    }
    if (static_cast<int64_t>(generated.size()) != codebooks * frames) {
        throw std::runtime_error("MOSS-TTS-v1.5 continuation: generated codes do not match their shape");
    }
    if (static_cast<int64_t>(prefix.codes.size()) < codebooks) {
        throw std::runtime_error("MOSS-TTS-v1.5 continuation: prefix has fewer codebooks than the generated codes");
    }
    for (int64_t cb = 0; cb < codebooks; ++cb) {
        if (static_cast<int64_t>(prefix.codes[static_cast<size_t>(cb)].size()) != prefix.frames) {
            throw std::runtime_error("MOSS-TTS-v1.5 continuation: prefix codebook length does not match its frame count");
        }
    }
    const int64_t total = prefix.frames + frames;
    std::vector<int32_t> out(static_cast<size_t>(codebooks * total), 0);
    for (int64_t cb = 0; cb < codebooks; ++cb) {
        auto * row = out.data() + static_cast<size_t>(cb * total);
        const auto & p = prefix.codes[static_cast<size_t>(cb)];
        std::copy(p.begin(), p.end(), row);
        std::copy(generated.begin() + static_cast<std::ptrdiff_t>(cb * frames),
                  generated.begin() + static_cast<std::ptrdiff_t>((cb + 1) * frames),
                  row + prefix.frames);
    }
    return out;
}

// Number of leading samples that belong to the prefix, by frame ratio. The
// reference trims the same way ("trim at waveform level according to
// start_length ratio"): the cut follows the audio the codec actually returned,
// so it cannot drift from the codec's own frame-to-sample ratio.
inline size_t prefix_sample_count(size_t total_samples, int64_t prefix_frames, int64_t total_frames) {
    if (prefix_frames <= 0 || total_frames <= 0) {
        return 0;
    }
    const double ratio = static_cast<double>(prefix_frames) / static_cast<double>(total_frames);
    const auto cut = static_cast<size_t>(static_cast<double>(total_samples) * ratio);
    return std::min(cut, total_samples);
}

}  // namespace engine::models::moss_tts_v15::continuation
