// Reference dump: prefill a ChatML prompt with llama.cpp, print the top-10
// logits of the last position, then greedy-decode N tokens and print ids.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

static void quiet(enum ggml_log_level, const char *, void *) {}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model prompt n_predict [n_top]\n", argv[0]); return 1; }
    const char *model_path = argv[1];
    std::string prompt = argv[2];
    if (!prompt.empty() && prompt[0] == '@') {
        FILE *f = fopen(prompt.c_str() + 1, "rb");
        if (!f) { fprintf(stderr, "cannot open prompt file\n"); return 1; }
        std::string data; char buf[4096]; size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) data.append(buf, n);
        fclose(f);
        prompt = data;
    }
    int n_predict = atoi(argv[3]);
    int n_top = argc > 4 ? atoi(argv[4]) : 10;
    int dump_step = argc > 6 ? atoi(argv[5]) : -1;
    const char *dump_path = argc > 6 ? argv[6] : nullptr;
    llama_log_set(quiet, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab *vocab = llama_model_get_vocab(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 16384; cp.n_batch = 2048; cp.n_ubatch = 2048; cp.n_threads = 8; cp.n_threads_batch = 8;
    llama_context *ctx = llama_init_from_model(model, cp);
    std::vector<llama_token> toks(prompt.size() + 16);
    int n = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), toks.data(), (int)toks.size(), false, true);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(n);
    printf("tokens(%d):", n);
    for (int i = 0; i < n; i++) printf(" %d", toks[i]);
    printf("\n");
    // prefill in n_batch-sized chunks so long prompts work
    for (int i0 = 0; i0 < n; i0 += 2048) {
        int len = std::min(2048, n - i0);
        llama_batch batch = llama_batch_get_one(toks.data() + i0, len);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at %d\n", i0); return 1; }
    }
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> out;
    for (int step = 0; step < n_predict + 1; step++) {
        const float *logits = llama_get_logits_ith(ctx, -1);
        std::vector<int> idx(n_vocab);
        for (int i = 0; i < n_vocab; i++) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + n_top, idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        if (step == dump_step && dump_path) {
            FILE *f = fopen(dump_path, "wb");
            if (f) { fwrite(logits, sizeof(float), n_vocab, f); fclose(f); }
            printf("step%d top%d:", step, n_top);
            for (int i = 0; i < n_top; i++) printf(" %d:%.4f", idx[i], logits[idx[i]]);
            printf("\n");
        }
        if (step == 0) {
            printf("top%d:", n_top);
            for (int i = 0; i < n_top; i++) printf(" %d:%.4f", idx[i], logits[idx[i]]);
            printf("\n");
        }
        if (step == n_predict) break;
        llama_token best = idx[0];
        out.push_back(best);
        if (llama_vocab_is_eog(vocab, best)) break;
        llama_batch b1 = llama_batch_get_one(&best, 1);
        if (llama_decode(ctx, b1) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    }
    printf("greedy(%zu):", out.size());
    for (auto t : out) printf(" %d", t);
    printf("\ntext: ");
    for (auto t : out) {
        char buf[256];
        int len = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        if (len > 0) fwrite(buf, 1, len, stdout);
    }
    printf("\n");
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
