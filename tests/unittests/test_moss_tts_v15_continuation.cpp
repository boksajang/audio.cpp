/*
 * MOSS-TTS-v1.5 continuation arithmetic (Shorts Factory fork, 2026-10).
 *
 * Added by the Shorts Factory fork (boksajang/audio.cpp); not part of upstream
 * audio.cpp. See FORK_NOTES.md.
 *
 * The session needs no model for these pieces, and each one fails quietly when it
 * is wrong: a missing space runs the prefix transcript into the new text, an
 * unmoved floor is met by the prefix alone, a misplaced copy decodes the wrong
 * codes, and a wrong cut leaves part of the prefix in the output.
 */

#include "engine/community_models/moss_tts_v15/continuation.h"
#include "test_assert.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace cont = engine::models::moss_tts_v15::continuation;

void test_join_text() {
    engine::test::require(cont::join_text("Hello.", "World.") == "Hello. World.", "space between sentences");
    engine::test::require(cont::join_text("Hello. ", "World.") == "Hello. World.", "no double space after a trailing one");
    engine::test::require(cont::join_text("Hello.", " World.") == "Hello. World.", "no double space before a leading one");
    engine::test::require(cont::join_text("", "World.") == "World.", "empty prefix");
    engine::test::require(cont::join_text("Hello.", "") == "Hello.", "empty text");
    // Korean, the case this exists for: UTF-8 bytes pass through untouched.
    engine::test::require(
        cont::join_text("\xEC\x95\x88\xEB\x85\x95.", "\xEB\x8B\xA4\xEC\x9D\x8C.")
            == "\xEC\x95\x88\xEB\x85\x95. \xEB\x8B\xA4\xEC\x9D\x8C.",
        "UTF-8 text joined with one space");
}

void test_shift_bounds() {
    engine::decoders::MossTtsDelayLengthBounds bounds;
    bounds.min_frames = 20;
    bounds.max_frames = 90;
    const auto moved = cont::shift_bounds(bounds, 50);
    engine::test::require(moved.min_frames == 70 && moved.max_frames == 140, "both bounds move past the prefix");

    const auto unchanged = cont::shift_bounds(bounds, 0);
    engine::test::require(unchanged.min_frames == 20 && unchanged.max_frames == 90, "no prefix, no move");

    engine::decoders::MossTtsDelayLengthBounds disabled;
    const auto still_disabled = cont::shift_bounds(disabled, 50);
    engine::test::require(
        still_disabled.min_frames == 0 && still_disabled.max_frames == 0, "a disabled bound stays disabled");
}

void test_join_codes() {
    engine::decoders::MossReferenceAudio prefix;
    prefix.frames = 2;
    prefix.codes = {{1, 2}, {11, 12}, {21, 22}};  // 3 codebooks in the prefix
    // 2 codebooks x 3 frames generated, row-major.
    const std::vector<int32_t> generated = {3, 4, 5, 13, 14, 15};
    const auto joined = cont::join_codes(prefix, generated, 2, 3);
    const std::vector<int32_t> expected = {1, 2, 3, 4, 5, 11, 12, 13, 14, 15};
    engine::test::require(joined == expected, "prefix codes then generated codes, per codebook");

    bool threw = false;
    try {
        engine::decoders::MossReferenceAudio short_prefix;
        short_prefix.frames = 2;
        short_prefix.codes = {{1, 2}};
        (void) cont::join_codes(short_prefix, generated, 2, 3);
    } catch (const std::exception &) {
        threw = true;
    }
    engine::test::require(threw, "a prefix with fewer codebooks is rejected");

    threw = false;
    try {
        engine::decoders::MossReferenceAudio ragged;
        ragged.frames = 2;
        ragged.codes = {{1, 2}, {11}};
        (void) cont::join_codes(ragged, generated, 2, 3);
    } catch (const std::exception &) {
        threw = true;
    }
    engine::test::require(threw, "a ragged prefix is rejected");

    threw = false;
    try {
        (void) cont::join_codes(prefix, generated, 2, 4);
    } catch (const std::exception &) {
        threw = true;
    }
    engine::test::require(threw, "generated codes that do not match their shape are rejected");
}

void test_prefix_sample_count() {
    // 1920 samples per frame at 24 kHz / 12.5 Hz.
    engine::test::require(cont::prefix_sample_count(1920 * 10, 4, 10) == 1920 * 4, "exact frame ratio");
    engine::test::require(cont::prefix_sample_count(1000, 0, 10) == 0, "no prefix, no cut");
    engine::test::require(cont::prefix_sample_count(1000, 10, 10) == 1000, "all prefix, all cut");
    engine::test::require(cont::prefix_sample_count(0, 4, 10) == 0, "no audio, no cut");
}

}  // namespace

int main() {
    try {
        test_join_text();
        test_shift_bounds();
        test_join_codes();
        test_prefix_sample_count();
    } catch (const std::exception & error) {
        std::cerr << "moss_tts_v15_continuation_test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "moss_tts_v15_continuation_test: ok\n";
    return 0;
}
