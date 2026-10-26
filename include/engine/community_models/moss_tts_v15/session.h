#pragma once

// Modified by the Shorts Factory fork (boksajang/audio.cpp), 2026-10: MOSS-TTS-v1.5
// continuation (request audio_input + continuation_text) and the encoder_precision
// session option. See FORK_NOTES.md.

#include "engine/community_models/moss_tts_v15/assets.h"
#include "engine/community_models/moss_tts_v15/prompt.h"
#include "engine/framework/decoders/moss_tts_delay/backbone.h"
#include "engine/framework/decoders/moss_tts_delay/delay_decoder.h"
#include "engine/framework/decoders/moss_tts_delay/heads.h"
#include "engine/community_models/moss_tts_v15/assets.h"
#include "engine/community_models/moss_tts_v15/prompt.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/codecs/moss_audio_tokenizer_codec_runtime.h"
#include "engine/framework/modules/multi_codebook_embedding.h"
#include "engine/framework/runtime/session_base.h"

#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>

namespace engine::models::moss_tts_v15 {

class MossTtsV15Session final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    MossTtsV15Session(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const Assets> assets);
    ~MossTtsV15Session() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    struct GeneratedChunk {
        std::vector<int32_t> codes;  // [n_vq, frames] row-major
        int64_t codebooks = 0;
        int64_t frames = 0;
        bool started_audio = false;
        bool hit_frame_ceiling = false;
    };

    // Takes the whole prompt rather than text/instruction/language: v1.5 also carries
    // reference audio and a duration budget, and threading those as further positional
    // parameters would be worse than passing the block the builder already needs.
    GeneratedChunk generate_chunk(
        const PromptFields & fields,
        const decoders::MossTtsDelaySamplingOptions & sampling,
        uint32_t seed,
        decoders::MossTtsDelayLengthBounds bounds_override,
        int64_t requested_frames,
        // [Shorts Factory fork] Continuation: the prefix audio's codes, carried as an
        // unfinished assistant turn, and the length of the NEW text alone (the prompt
        // text also holds the prefix transcript, which the prefix audio already speaks).
        const ReferenceAudio * assistant_audio = nullptr,
        int64_t spoken_characters = -1);
    std::vector<float> decode_codes(const GeneratedChunk & chunk);
    // [Shorts Factory fork] Decodes the prefix codes and the generated ones together and
    // trims the prefix back off by frame ratio. The codec is causal: decoding the
    // generated frames alone starts it cold and leaves an audible seam.
    std::vector<float> decode_with_prefix(const GeneratedChunk & chunk, const ReferenceAudio & prefix);
    // [Shorts Factory fork] One take that continues `request.audio_input`.
    runtime::TaskResult run_continuation(
        const runtime::TaskRequest & request,
        const std::string & continuation_text,
        const std::optional<std::string> & instruction,
        const std::optional<std::string> & language,
        std::vector<ReferenceAudio> references,
        const decoders::MossTtsDelaySamplingOptions & sampling,
        uint32_t seed,
        decoders::MossTtsDelayLengthBounds bounds_override,
        int64_t requested_frames,
        std::chrono::steady_clock::time_point wall_start);
    // Resamples the caller's recording to the codec rate if needed, then encodes it.
    ReferenceAudio encode_reference(const runtime::AudioBuffer & audio);

    runtime::TaskSpec task_;
    std::shared_ptr<const Assets> assets_;
    // Native, not BF16: a GGUF package carries its own type, and forcing BF16
    // dequantises it on load, so q4_k and q8_0 cost exactly as much VRAM as bf16
    // and the quantisation buys nothing. Measured on a 3090: q4_k TTS peaks at
    // 21.2 GiB forced to BF16 and 10.6 GiB left native. Safetensors are bf16
    // upstream, so native is the same thing there.
    engine::assets::TensorStorageType weight_storage_type_ = engine::assets::TensorStorageType::Native;
    // [Shorts Factory fork] Codec encoder precision on GPU backends. false (f16, the
    // default) is upstream's behaviour. true computes the encoder at f32 the way
    // moss_ttsd does: weights held at their stored type and widened in the graph.
    // Continuation carries on directly from the encoded prefix, and on moss_ttsd an
    // f16 encoder changed 5-14% of the finer codebooks.
    bool encoder_f32_ = false;
    size_t backbone_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
#if defined(INTPTR_MAX) && (INTPTR_MAX == INT32_MAX)
    size_t backbone_weight_context_bytes_ = 1024ull * 1024ull * 1024ull;
    size_t heads_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    size_t heads_weight_context_bytes_ = 1024ull * 1024ull * 1024ull;
    size_t codec_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t codec_weight_context_bytes_ = 1024ull * 1024ull * 1024ull;
#else
    size_t backbone_weight_context_bytes_ = 8192ull * 1024ull * 1024ull;
    size_t heads_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    size_t heads_weight_context_bytes_ = 4096ull * 1024ull * 1024ull;
    // Charged twice -- the codec runtime takes one arena size for the encoder and
    // one for the decoder -- and this model, unlike moss_voicegen, holds both
    // halves resident because cloning encodes a reference. The graphs are small
    // (a few hundred codec frames at 12.5 a second), so moss_voicegen's 2 GiB was
    // 4 GiB of VRAM here for no benefit, and on a 24 GiB card it was the
    // difference between cloning running and failing to allocate the encoder.
    size_t codec_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t codec_weight_context_bytes_ = 4096ull * 1024ull * 1024ull;
#endif

    // The execution context comes from RuntimeSessionBase; the runtimes below borrow it.
    std::shared_ptr<tokenizers::LlamaBpeTokenizer> tokenizer_;
    std::unique_ptr<engine::modules::MultiCodebookEmbedding> codebooks_;
    std::unique_ptr<decoders::MossTtsDelayBackboneRuntime> backbone_;
    std::unique_ptr<decoders::MossTtsDelayHeadsRuntime> heads_;
    std::unique_ptr<engine::codecs::MossAudioTokenizerCodecRuntime> codec_;
};

}  // namespace engine::models::moss_tts_v15
