// SPDX-License-Identifier: Apache-2.0
#include "titanet_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "audio_resampler.h"
#include "nn.h"

namespace nemo_speech::asr {
namespace {

ggml_runtime::Params
backend_params(int gpu) {
    ggml_runtime::Params params;
    params.use_gpu = gpu >= 0;
    params.gpu_device_idx = gpu < 0 ? 0 : gpu;
    params.pe_bin_path = const_cast<char*>("");
    return params;
}

TitanetConfig
parse_config(const ggml_runtime::GGUFLoader& loader) {
    if (loader.get_str("general.architecture") != "speaker")
        throw std::runtime_error("speaker model: GGUF architecture must be 'speaker'");
    TitanetConfig cfg;
    cfg.sample_rate = loader.get_u32("speaker.preprocessor.sample_rate", 16000);
    cfg.window_size = loader.get_f32("speaker.preprocessor.window_size", 0.025f);
    cfg.window_stride = loader.get_f32("speaker.preprocessor.window_stride", 0.01f);
    cfg.n_fft = loader.get_u32("speaker.preprocessor.n_fft", 512);
    cfg.n_mels = loader.get_u32("speaker.preprocessor.features", 80);
    cfg.preemph = loader.get_f32("speaker.preprocessor.preemph", 0.97f);
    cfg.log_zero_guard = loader.get_f32("speaker.preprocessor.log_zero_guard", cfg.log_zero_guard);
    cfg.feat_in = loader.get_u32("speaker.encoder.feat_in", cfg.n_mels);
    cfg.se_reduction = loader.get_u32("speaker.encoder.se_reduction", 8);
    cfg.attention_channels = loader.get_u32("speaker.pool.attention_channels", 128);
    cfg.embedding_dim = loader.get_u32("speaker.embedding_dim", 192);

    const auto filters = loader.get_i32_array("speaker.encoder.filters");
    const auto repeat = loader.get_i32_array("speaker.encoder.repeat");
    const auto kernel = loader.get_i32_array("speaker.encoder.kernel");
    const auto residual = loader.get_i32_array("speaker.encoder.residual");
    const size_t n = filters.size();
    if (n == 0 || repeat.size() != n || kernel.size() != n || residual.size() != n)
        throw std::runtime_error("speaker model: inconsistent speaker.encoder.* block metadata");
    for (size_t i = 0; i < n; ++i) {
        TitanetBlockConfig block;
        block.filters = filters[i];
        block.repeat = repeat[i];
        block.kernel = kernel[i];
        block.residual = residual[i] != 0;
        if (block.filters <= 0 || block.repeat <= 0 || block.kernel <= 0 || block.kernel % 2 == 0)
            throw std::runtime_error("speaker model: invalid encoder block geometry");
        cfg.blocks.push_back(block);
    }
    if (cfg.feat_in != cfg.n_mels)
        throw std::runtime_error("speaker model: encoder feat_in must match the mel bins");
    return cfg;
}

MelSpecConfig
make_fe_cfg(const TitanetConfig& c) {
    MelSpecConfig fe;
    fe.sample_rate = c.sample_rate;
    fe.n_fft = c.n_fft;
    fe.n_mels = c.n_mels;
    fe.window_size = c.window_size;
    fe.window_stride = c.window_stride;
    fe.preemph = c.preemph;
    fe.log_zero_guard = c.log_zero_guard;
    fe.normalize_per_feature = true;
    // NeMo FilterbankFeatures parity: centered symmetric Hann window.
    fe.stft_center_window = true;
    fe.hann_periodic = false;
    return fe;
}

using ggml_runtime::ggml_bf_tensor;
using ggml_runtime::TensorBag;

ggml_tensor*
transpose(ggml_context* ctx, ggml_tensor* x) {
    return ggml_cont(ctx, ggml_transpose(ctx, x));
}

}  // namespace

// Activations are kept channel-major, (C, T): ne0 = channels, ne1 = frames.
// That makes every pointwise conv a plain mul_mat; depthwise convs and
// reductions over time transpose to (T, C) locally.
class TitanetGraph : public ggml_runtime::Module {
   public:
    explicit TitanetGraph(const TitanetConfig& cfg) : cfg_(cfg) {
        int in = cfg.feat_in;
        for (size_t b = 0; b < cfg.blocks.size(); ++b) {
            const auto& block = cfg.blocks[b];
            const std::string p = "encoder." + std::to_string(b);
            Block layers;
            int rep_in = in;
            for (int r = 0; r < block.repeat; ++r) {
                const std::string q = p + ".rep." + std::to_string(r);
                if (block.kernel > 1) {
                    layers.dw.push_back(std::make_unique<ggml_runtime::Conv1D>(
                        q + ".dw", rep_in, rep_in, block.kernel, 1, block.kernel / 2, 1,
                        /*use_bias=*/false, /*is_dw=*/true));
                }
                layers.pw.push_back(
                    std::make_unique<ggml_runtime::Linear>(q + ".pw", rep_in, block.filters));
                rep_in = block.filters;
            }
            const int se_channels = block.filters / cfg.se_reduction;
            layers.se1 = std::make_unique<ggml_runtime::Linear>(
                p + ".se.fc1", block.filters, se_channels, false);
            layers.se2 = std::make_unique<ggml_runtime::Linear>(
                p + ".se.fc2", se_channels, block.filters, false);
            if (block.residual)
                layers.res = std::make_unique<ggml_runtime::Linear>(p + ".res", in, block.filters);
            blocks_.push_back(std::move(layers));
            in = block.filters;
        }
        channels_ = in;
        tdnn_ = std::make_unique<ggml_runtime::Linear>(
            "pool.tdnn", 3 * channels_, cfg.attention_channels);
        attn_ = std::make_unique<ggml_runtime::Linear>(
            "pool.attn", cfg.attention_channels, channels_);
        emb_ = std::make_unique<ggml_runtime::Linear>("emb", 2 * channels_, cfg.embedding_dim);
    }

    void define_tensors(ggml_runtime::Session* session) override {
        for_each_module([&](ggml_runtime::Module& m) { m.define_tensors(session); });
        auto* tc = session->model_tensor_container.get();
        tc->create_tensor_1d("pool.tdnn.bn_scale", GGML_TYPE_F32, cfg_.attention_channels);
        tc->create_tensor_1d("pool.tdnn.bn_shift", GGML_TYPE_F32, cfg_.attention_channels);
    }

    void set_data(ggml_runtime::Session* session) override {
        for_each_module([&](ggml_runtime::Module& m) { m.set_data(session); });
        session->load_weight("pool.tdnn.bn_scale");
        session->load_weight("pool.tdnn.bn_shift");
    }

    TensorBag build_graph(
        ggml_runtime::Session* session, TensorBag input_tensors,
        ggml_runtime::TensorContainer* tc) override {
        (void)input_tensors;
        auto mel = tc->get_tensor_by_name("input.mel");  // (n_mels, T)
        auto buft = mel.buft;
        ggml_context* ctx = tc->get_ctx_of_buffer_type(buft).ctx;

        auto linear = [&](ggml_runtime::Linear& layer, ggml_tensor* x) {
            TensorBag in;
            in.add_tensor(ggml_bf_tensor(x, buft));
            return layer.build_graph(session, in, tc).get_tensor(0).tensor;
        };
        auto depthwise = [&](ggml_runtime::Conv1D& layer, ggml_tensor* x) {
            // (C, T) -> (T, C, 1) -> conv -> (C, T)
            ggml_tensor* t = transpose(ctx, x);
            t = ggml_reshape_3d(ctx, t, t->ne[0], t->ne[1], 1);
            TensorBag in;
            in.add_tensor(ggml_bf_tensor(t, buft));
            ggml_tensor* y = layer.build_graph(session, in, tc).get_tensor(0).tensor;
            y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
            return transpose(ctx, y);
        };
        // Mean over time of a (C, T) tensor, returned as (C, 1).
        auto time_mean = [&](ggml_tensor* x) {
            ggml_tensor* m = ggml_mean(ctx, transpose(ctx, x));  // (1, C)
            return ggml_reshape_2d(ctx, m, x->ne[0], 1);
        };

        ggml_tensor* x = mel.tensor;
        for (size_t b = 0; b < blocks_.size(); ++b) {
            auto& layers = blocks_[b];
            const auto& block = cfg_.blocks[b];
            ggml_tensor* block_in = x;
            ggml_tensor* y = x;
            for (int r = 0; r < block.repeat; ++r) {
                if (!layers.dw.empty())
                    y = depthwise(*layers.dw[static_cast<size_t>(r)], y);
                y = linear(*layers.pw[static_cast<size_t>(r)], y);
                if (r + 1 < block.repeat)
                    y = ggml_relu(ctx, y);
            }
            // Squeeze-excitation over the whole utterance.
            ggml_tensor* s = time_mean(y);
            s = ggml_relu(ctx, linear(*layers.se1, s));
            s = ggml_sigmoid(ctx, linear(*layers.se2, s));
            y = ggml_mul(ctx, y, s);
            if (layers.res)
                y = ggml_add(ctx, y, linear(*layers.res, block_in));
            x = ggml_relu(ctx, y);
        }

        // Attentive statistics pooling (NeMo AttentivePoolLayer).
        const float eps = 1e-10f;
        ggml_tensor* mean = time_mean(x);  // (C, 1)
        ggml_tensor* centered = ggml_sub(ctx, x, mean);
        ggml_tensor* std = ggml_sqrt(ctx, ggml_clamp(ctx, time_mean(ggml_sqr(ctx, centered)), eps, INFINITY));
        ggml_tensor* context = ggml_concat(ctx, x, ggml_repeat(ctx, mean, x), 0);
        context = ggml_concat(ctx, context, ggml_repeat(ctx, std, x), 0);  // (3C, T)
        auto* bn_scale = session->model_tensor_container->get_tensor_by_name("pool.tdnn.bn_scale").tensor;
        auto* bn_shift = session->model_tensor_container->get_tensor_by_name("pool.tdnn.bn_shift").tensor;
        ggml_tensor* h = ggml_relu(ctx, linear(*tdnn_, context));
        h = ggml_add(ctx, ggml_mul(ctx, h, bn_scale), bn_shift);
        h = ggml_tanh(ctx, h);
        ggml_tensor* logits = linear(*attn_, h);                    // (C, T)
        ggml_tensor* alpha = ggml_soft_max(ctx, transpose(ctx, logits));  // (T, C), over time
        ggml_tensor* xt = transpose(ctx, x);                         // (T, C)
        ggml_tensor* mu = ggml_sum_rows(ctx, ggml_mul(ctx, alpha, xt));  // (1, C)
        ggml_tensor* dev = ggml_sqr(ctx, ggml_sub(ctx, xt, mu));
        ggml_tensor* sg = ggml_sqrt(
            ctx, ggml_clamp(ctx, ggml_sum_rows(ctx, ggml_mul(ctx, alpha, dev)), eps, INFINITY));
        ggml_tensor* pooled = ggml_concat(ctx, mu, sg, 1);           // (1, 2C)
        pooled = ggml_reshape_2d(ctx, pooled, 2 * channels_, 1);

        TensorBag out;
        out.add_tensor(ggml_bf_tensor(linear(*emb_, pooled), buft));  // (emb_dim, 1)
        return out;
    }

   private:
    struct Block {
        std::vector<std::unique_ptr<ggml_runtime::Conv1D>> dw;
        std::vector<std::unique_ptr<ggml_runtime::Linear>> pw;
        std::unique_ptr<ggml_runtime::Linear> se1;
        std::unique_ptr<ggml_runtime::Linear> se2;
        std::unique_ptr<ggml_runtime::Linear> res;
    };

    template <typename F>
    void for_each_module(F&& f) {
        for (auto& block : blocks_) {
            for (auto& dw : block.dw) f(*dw);
            for (auto& pw : block.pw) f(*pw);
            f(*block.se1);
            f(*block.se2);
            if (block.res)
                f(*block.res);
        }
        f(*tdnn_);
        f(*attn_);
        f(*emb_);
    }

    TitanetConfig cfg_;
    int channels_ = 0;
    std::vector<Block> blocks_;
    std::unique_ptr<ggml_runtime::Linear> tdnn_;
    std::unique_ptr<ggml_runtime::Linear> attn_;
    std::unique_ptr<ggml_runtime::Linear> emb_;
};

std::shared_ptr<SpeakerEmbedder>
SpeakerEmbedder::load(int gpu, const std::string& model_path) {
    return std::make_shared<SpeakerEmbedder>(gpu, model_path);
}

SpeakerEmbedder::SpeakerEmbedder(int gpu, const std::string& model_path) {
    backend_ = std::make_unique<ggml_runtime::BackendManager>(backend_params(gpu));
    loader_ = std::make_unique<ggml_runtime::GGUFLoader>(model_path);
    cfg_ = parse_config(*loader_);
    model_name_ = loader_->get_str("general.name", "speaker");

    fe_ = std::make_unique<MelSpectrogramExtractor>(make_fe_cfg(cfg_), /*bm=*/nullptr);
    const int n_bins = cfg_.n_fft / 2 + 1;
    std::vector<float> fb(static_cast<size_t>(cfg_.n_mels) * n_bins);
    const char* data = loader_->get_tensor_file_data("preprocessor.fb", fb.size() * sizeof(float));
    std::memcpy(fb.data(), data, fb.size() * sizeof(float));
    fe_->set_mel_basis(fb.data(), cfg_.n_mels, n_bins);

    graph_ = std::make_unique<TitanetGraph>(cfg_);
    session_ = std::make_unique<ggml_runtime::Session>(*backend_, graph_.get(), loader_.get());
    session_->set_run_cache_capacity(8);
    session_->setup();
}

SpeakerEmbedder::~SpeakerEmbedder() = default;

void
SpeakerEmbedder::l2_normalize(std::vector<float>& v) {
    double norm = 0.0;
    for (float x : v) norm += static_cast<double>(x) * x;
    norm = std::sqrt(norm);
    if (norm > 0.0)
        for (float& x : v) x = static_cast<float>(x / norm);
}

std::vector<float>
SpeakerEmbedder::embed_model_rate(const float* samples, size_t n_samples) const {
    if (n_samples < static_cast<size_t>(min_seconds() * cfg_.sample_rate))
        throw std::invalid_argument("speaker embedding needs at least 0.5 s of audio");
    std::vector<float> mel;
    int t_mel = 0;
    fe_->compute(samples, n_samples, mel, t_mel, /*reflect_left=*/true, /*normalize=*/true);
    // NeMo's feature length is floor(samples / hop); the trailing centered frame
    // is padding and excluded from every masked encoder/pooling statistic.
    const int valid = std::min(t_mel, static_cast<int>(n_samples / fe_->hop_length()));
    mel.resize(static_cast<size_t>(valid) * cfg_.n_mels);

    std::vector<ggml_runtime::Session::Input> inputs;
    inputs.push_back({"input.mel", GGML_TYPE_F32, mel.data(), {cfg_.n_mels, valid}});
    std::vector<float> embedding(static_cast<size_t>(cfg_.embedding_dim));
    std::vector<ggml_runtime::Session::Output> outputs(1);
    outputs[0].index = 0;
    outputs[0].host_buffer = embedding.data();
    outputs[0].nbytes = embedding.size() * sizeof(float);
    session_->run(inputs, outputs);
    return embedding;
}

std::vector<float>
SpeakerEmbedder::embed(const float* samples, size_t n_samples, int sample_rate) const {
    if (!samples && n_samples > 0)
        throw std::invalid_argument("samples must not be null");
    if (sample_rate == 0 || sample_rate == cfg_.sample_rate)
        return embed_model_rate(samples, n_samples);
    if (!audio::supported_input_sample_rate(sample_rate))
        throw std::invalid_argument("input sample rate must be between 8000 and 96000 Hz");
    audio::AudioResampler resampler(sample_rate, cfg_.sample_rate);
    std::vector<float> resampled;
    resampler.process(samples, n_samples, &resampled);
    resampler.finish(&resampled);
    return embed_model_rate(resampled.data(), resampled.size());
}

std::vector<float>
SpeakerEmbedder::embed_long(
    const float* samples, size_t n_samples, int sample_rate, double window_seconds) const {
    std::vector<float> resampled;
    if (sample_rate != 0 && sample_rate != cfg_.sample_rate) {
        if (!audio::supported_input_sample_rate(sample_rate))
            throw std::invalid_argument("input sample rate must be between 8000 and 96000 Hz");
        audio::AudioResampler resampler(sample_rate, cfg_.sample_rate);
        resampler.process(samples, n_samples, &resampled);
        resampler.finish(&resampled);
        samples = resampled.data();
        n_samples = resampled.size();
    }
    const size_t window =
        std::max<size_t>(1, static_cast<size_t>(window_seconds * cfg_.sample_rate));
    const size_t min_samples = static_cast<size_t>(min_seconds() * cfg_.sample_rate);
    const size_t n_windows = std::max<size_t>(1, (n_samples + window - 1) / window);
    // Spread the samples evenly so no trailing window is a sliver.
    const size_t step = (n_samples + n_windows - 1) / n_windows;
    std::vector<float> sum(static_cast<size_t>(cfg_.embedding_dim), 0.0f);
    int used = 0;
    for (size_t start = 0; start < n_samples; start += step) {
        const size_t count = std::min(step, n_samples - start);
        if (count < min_samples && used > 0)
            break;
        auto e = embed_model_rate(samples + start, count);
        l2_normalize(e);
        for (size_t i = 0; i < sum.size(); ++i) sum[i] += e[i];
        ++used;
    }
    l2_normalize(sum);
    return sum;
}

}  // namespace nemo_speech::asr
