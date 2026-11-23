// SPDX-License-Identifier: Apache-2.0
// TitaNet speaker-embedding extractor on the ggml runtime.
//
// NeMo EncDecSpeakerLabelModel (e.g. nvidia/speakerverification_en_titanet_large):
//
//   waveform (16 kHz)
//     -> NeMo log-mel (80 bins, per_feature normalization)
//     -> ConvASREncoder: Jasper blocks of
//          [depthwise conv k -> pointwise conv (BN folded) -> ReLU] x repeat
//          -> squeeze-excitation (global mean, FC/ReLU/FC, sigmoid)
//          (+ residual pointwise conv with folded BN) -> ReLU
//     -> attentive statistics pooling (mean/std context, TDNN, tanh, softmax over time)
//     -> embedding: affine map 2C -> 192 (BN folded)
//
// The converter (conversion/speaker.py) folds every BatchNorm that follows a
// linear operation, so the graph only needs matmuls, depthwise convs and
// element-wise ops. One utterance runs per Session::run (no batching).
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "fe.h"
#include "runtime.h"

namespace nemo_speech::asr {

struct TitanetBlockConfig {
    int filters = 1024;
    int repeat = 1;
    int kernel = 3;
    bool residual = false;
};

struct TitanetConfig {
    // FE (speaker.preprocessor.*)
    int sample_rate = 16000;
    float window_size = 0.025f;
    float window_stride = 0.01f;
    int n_fft = 512;
    int n_mels = 80;
    float preemph = 0.97f;
    float log_zero_guard = 5.9604645e-8f;  // 2^-24

    int feat_in = 80;
    std::vector<TitanetBlockConfig> blocks;
    int se_reduction = 8;
    int attention_channels = 128;
    int embedding_dim = 192;
};

class TitanetGraph;

// Loads a speaker GGUF and computes one embedding per utterance. Owns its
// backend; Session::run serializes concurrent callers.
class SpeakerEmbedder {
   public:
    static std::shared_ptr<SpeakerEmbedder> load(int gpu, const std::string& model_path);

    SpeakerEmbedder(int gpu, const std::string& model_path);
    ~SpeakerEmbedder();
    SpeakerEmbedder(const SpeakerEmbedder&) = delete;
    SpeakerEmbedder& operator=(const SpeakerEmbedder&) = delete;

    // Raw TitaNet embedding (embedding_dim floats, not normalized) of one
    // utterance. `sample_rate` 0 means the model rate; other rates are
    // resampled. Throws if the audio is shorter than `min_seconds()`.
    std::vector<float> embed(const float* samples, size_t n_samples, int sample_rate = 0) const;

    // L2-normalized mean of the normalized embeddings of consecutive windows of
    // at most `window_seconds`. Keeps activation memory bounded for long audio.
    std::vector<float> embed_long(
        const float* samples, size_t n_samples, int sample_rate = 0,
        double window_seconds = 10.0) const;

    static void l2_normalize(std::vector<float>& v);

    const TitanetConfig& config() const { return cfg_; }
    int embedding_dim() const { return cfg_.embedding_dim; }
    int sample_rate() const { return cfg_.sample_rate; }
    double min_seconds() const { return 0.5; }
    const std::string& model_name() const { return model_name_; }

   private:
    std::vector<float> embed_model_rate(const float* samples, size_t n_samples) const;

    TitanetConfig cfg_;
    std::string model_name_;
    std::unique_ptr<ggml_runtime::BackendManager> backend_;
    std::unique_ptr<ggml_runtime::GGUFLoader> loader_;
    std::unique_ptr<TitanetGraph> graph_;
    std::unique_ptr<ggml_runtime::Session> session_;
    std::unique_ptr<MelSpectrogramExtractor> fe_;
};

}  // namespace nemo_speech::asr
