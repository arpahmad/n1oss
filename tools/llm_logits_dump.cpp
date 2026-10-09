// tools/llm_logits_dump.cpp - teacher-forced logits from upstream llama.cpp, for the second-oracle
// diff (REPORT-GLM5-FLASH.md §12.1/§12.2): load ANY llama.cpp GGUF, decode explicit token ids in
// ONE ubatch with logits enabled at every position, and dump (n_tokens, n_vocab) float32
// row-major plus the per-position argmax to stdout.  CPU-only by construction (n_gpu_layers=0).
//
//   llm_logits_dump <model.gguf> <out.bin> <token> [<token>...]
//
// Built out-of-tree against a llama.cpp build:
//   g++ -O2 -std=c++17 -I <llama.cpp>/include llm_logits_dump.cpp -o llm_logits_dump \
//       -L <build>/src -llama -L <build>/ggml/src -lggml -lggml-base -lggml-cpu -lm -pthread
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// cb_eval: dump every NAMED node's data (the graph's cb() seams) to <dir>/<name>.bin so the
// per-layer diff against ref/glm.py's dumps can bisect to the first divergent seam.  Non-contiguous
// tensors are gathered element-wise by their strides (a raw memcpy of t->data would be wrong).
static std::string g_cb_dir;
static void gather_strided(const struct ggml_tensor* t, float* dst) {
    const int64_t ne[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    const size_t  nb[4] = {t->nb[0], t->nb[1], t->nb[2], t->nb[3]};
    for (int64_t i3 = 0; i3 < ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < ne[1]; ++i1) {
                const char* row = (const char*) t->data + i3 * nb[3] + i2 * nb[2] + i1 * nb[1];
                for (int64_t i0 = 0; i0 < ne[0]; ++i0) {
                    float v;
                    memcpy(&v, row + i0 * nb[0], sizeof(float));
                    *dst++ = v;
                }
            }
}
static bool eval_cb(struct ggml_tensor* t, bool ask, void*) {
    if (ask) return t->name && t->name[0] && g_cb_dir[0];
    if (!t->data || !t->name || !t->name[0]) return true;
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) return true;
    char path[512];
    std::string name = t->name;
    for (auto& c : name) if (c == '/') c = '_';
    std::snprintf(path, sizeof(path), "%s/%s.%s", g_cb_dir.c_str(), name.c_str(),
                  t->type == GGML_TYPE_I32 ? "i32" : "f32");
    const int64_t n = t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
    const size_t bytes = (size_t) n * (t->type == GGML_TYPE_I32 ? 4 : 4);
    if (!ggml_is_contiguous(t)) {
        float* buf = (float*) std::malloc(bytes);
        if (!buf) return true;
        gather_strided(t, buf);
        std::FILE* f = std::fopen(path, "wb");
        if (f) { std::fwrite(buf, 1, bytes, f); std::fclose(f); }
        std::free(buf);
        return true;
    }
    std::FILE* f = std::fopen(path, "wb");
    if (f) {
        std::fwrite(t->data, 1, bytes, f);
        std::fclose(f);
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <out.bin> <token> [<token>...]\n", argv[0]);
        return 2;
    }
    std::vector<llama_token> toks;
    for (int i = 3; i < argc; ++i) toks.push_back((llama_token) std::atoi(argv[i]));
    const size_t n_tok = toks.size();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;                       // the CPU backend IS the point: it is the second oracle
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        std::fprintf(stderr, "llm_logits_dump: model load failed\n");
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = (uint32_t) (n_tok + 8);
    cp.n_batch  = (uint32_t) (n_tok + 8);
    cp.n_ubatch = (uint32_t) (n_tok + 8);      // force ONE ubatch: the pooled indexer's new-pool
    cp.no_perf  = true;                        // path must run exactly like the oracle's batched pass
    g_cb_dir = std::getenv("LLM_CB_DIR") ? std::getenv("LLM_CB_DIR") : "";
    if (g_cb_dir[0]) cp.cb_eval = eval_cb;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "llm_logits_dump: context init failed\n");
        return 1;
    }

    // Decode ONE token per llama_decode call.  The single-ubatch variant aborts inside llama.cpp's
    // graph_reserve (ggml_set_rows shape assert under resolve_fused_ops) at 272aad8b for this
    // model; streaming is also the mode the Strata runner is verified in, so per-token is the
    // comparison that matters.
    std::FILE* out = std::fopen(argv[2], "wb");
    if (!out) {
        std::fprintf(stderr, "llm_logits_dump: cannot write %s\n", argv[2]);
        return 1;
    }
    for (size_t t = 0; t < n_tok; ++t) {
        llama_token one = toks[t];
        llama_batch batch = llama_batch_get_one(&one, 1);
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "llm_logits_dump: decode failed at position %zu\n", t);
            return 1;
        }
        const float* lg = llama_get_logits_ith(ctx, 0);
        std::fwrite(lg, sizeof(float), (size_t) n_vocab, out);
        int am = 0;
        for (int v = 1; v < n_vocab; ++v)
            if (lg[v] > lg[am]) am = v;
        std::printf("pos %zu tok %d argmax %d (%.6f)\n", t, toks[t], am, lg[am]);
    }
    std::fclose(out);
    llama_free(ctx);
    llama_model_free(model);
    std::fprintf(stderr, "llm_logits_dump: wrote %zu x %d floats to %s\n", n_tok, n_vocab, argv[2]);
    return 0;
}
