/* Checks the classic dense matvec kernels for the mixed-recipe quant types
 * (Q3_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, IQ3_S, and Q4_K) against the generic
 * mul_mv_ext path on real tensors from a GGUF such as the Unsloth Qwen3.8
 * UD-Q4_K_M file.  Usage: test_qwen_quant_mv <model.gguf> */
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ds4.h"
#include "ds4_gpu.h"

typedef struct {
    const uint8_t *p;
    const uint8_t *end;
} cursor;

static uint64_t rd_u64(cursor *c) { uint64_t v; memcpy(&v, c->p, 8); c->p += 8; return v; }
static uint32_t rd_u32(cursor *c) { uint32_t v; memcpy(&v, c->p, 4); c->p += 4; return v; }
static void rd_str(cursor *c, const char **s, uint64_t *n) {
    *n = rd_u64(c);
    *s = (const char *)c->p;
    c->p += *n;
}

/* GGUF value sizes by type id; strings and arrays are walked explicitly. */
static void skip_value(cursor *c, uint32_t type) {
    static const uint8_t sizes[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (type == 8) {
        const char *s; uint64_t n;
        rd_str(c, &s, &n);
    } else if (type == 9) {
        uint32_t et = rd_u32(c);
        uint64_t n = rd_u64(c);
        for (uint64_t i = 0; i < n; i++) skip_value(c, et);
    } else {
        c->p += sizes[type < 13 ? type : 0];
    }
}

typedef struct {
    uint32_t type;
    uint64_t offset;
    uint64_t in_dim;
    uint64_t out_dim;
    char name[96];
} pick;

static const uint32_t wanted[] = {11, 12, 13, 14, 20, 21, 23};

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : getenv("DS4_QWEN_TEST_MODEL");
    if (!path || !path[0]) {
        fprintf(stderr, "usage: test_qwen_quant_mv <model.gguf>\n");
        return 2;
    }
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return 2; }
    struct stat st;
    fstat(fd, &st);
    const uint8_t *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 2; }

    cursor c = { map + 4, map + st.st_size };
    if (memcmp(map, "GGUF", 4) != 0 || rd_u32(&c) != 3) {
        fprintf(stderr, "not a GGUF v3 file\n");
        return 2;
    }
    const uint64_t n_tensors = rd_u64(&c);
    const uint64_t n_kv = rd_u64(&c);
    for (uint64_t i = 0; i < n_kv; i++) {
        const char *k; uint64_t kn;
        rd_str(&c, &k, &kn);
        skip_value(&c, rd_u32(&c));
    }
    pick picks[sizeof(wanted) / sizeof(wanted[0])] = {{0}};
    size_t n_picks = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        const char *name; uint64_t name_len;
        rd_str(&c, &name, &name_len);
        const uint32_t ndim = rd_u32(&c);
        uint64_t dims[4] = {0};
        for (uint32_t d = 0; d < ndim; d++) dims[d] = rd_u64(&c);
        const uint32_t type = rd_u32(&c);
        const uint64_t off = rd_u64(&c);
        if (ndim != 2) continue;
        for (size_t w = 0; w < sizeof(wanted) / sizeof(wanted[0]); w++) {
            bool have = false;
            for (size_t k = 0; k < n_picks; k++) have |= picks[k].type == wanted[w];
            if (type != wanted[w] || have) continue;
            pick *pk = &picks[n_picks++];
            pk->type = type;
            pk->offset = off;
            pk->in_dim = dims[0];
            pk->out_dim = dims[1];
            snprintf(pk->name, sizeof(pk->name), "%.*s", (int)name_len, name);
        }
    }
    const uint64_t data_start = (((uint64_t)(c.p - map)) + 31u) & ~(uint64_t)31u;

    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(map, (uint64_t)st.st_size)) {
        fprintf(stderr, "GPU initialization failed\n");
        return 2;
    }
    int failures = 0;
    for (size_t k = 0; k < n_picks; k++) {
        const pick *pk = &picks[k];
        const uint64_t off = data_start + pk->offset;
        for (uint32_t n_tok = 1; n_tok <= 4; n_tok += 3) {
            const uint64_t xn = n_tok * pk->in_dim, on = n_tok * pk->out_dim;
            float *x = malloc(xn * sizeof(float));
            srand(1234);
            for (uint64_t i = 0; i < xn; i++) x[i] = (float)rand() / RAND_MAX - 0.5f;
            ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(xn * sizeof(float));
            ds4_gpu_tensor *classic = ds4_gpu_tensor_alloc(on * sizeof(float));
            ds4_gpu_tensor *ext = ds4_gpu_tensor_alloc(on * sizeof(float));
            ds4_gpu_tensor_write(xt, 0, x, xn * sizeof(float));
            unsetenv("DS4_METAL_DISABLE_CLASSIC_QUANT_MV");
            unsetenv("DS4_METAL_DISABLE_Q4_MV_CLASSIC");
            int ok = ds4_gpu_matmul_quant_tensor(classic, map, (uint64_t)st.st_size, off,
                                                 pk->type, pk->in_dim, pk->out_dim, xt, n_tok);
            setenv("DS4_METAL_DISABLE_CLASSIC_QUANT_MV", "1", 1);
            setenv("DS4_METAL_DISABLE_Q4_MV_CLASSIC", "1", 1);
            ok &= ds4_gpu_matmul_quant_tensor(ext, map, (uint64_t)st.st_size, off,
                                              pk->type, pk->in_dim, pk->out_dim, xt, n_tok);
            ds4_gpu_synchronize();
            float *a = malloc(on * sizeof(float)), *b = malloc(on * sizeof(float));
            ds4_gpu_tensor_read(classic, 0, a, on * sizeof(float));
            ds4_gpu_tensor_read(ext, 0, b, on * sizeof(float));
            double max_abs = 0.0, max_diff = 0.0;
            for (uint64_t i = 0; i < on; i++) {
                if (fabs(b[i]) > max_abs) max_abs = fabs(b[i]);
                if (fabs(a[i] - b[i]) > max_diff) max_diff = fabs(a[i] - b[i]);
            }
            const bool pass = ok && max_diff <= 1e-3 * max_abs + 1e-4;
            printf("%-28s type %2u n_tok %u in=%llu out=%llu max=%.4f diff=%.6f %s\n",
                   pk->name, pk->type, n_tok,
                   (unsigned long long)pk->in_dim, (unsigned long long)pk->out_dim,
                   max_abs, max_diff, pass ? "ok" : "FAIL");
            if (!pass) failures++;
            free(x); free(a); free(b);
            ds4_gpu_tensor_free(xt);
            ds4_gpu_tensor_free(classic);
            ds4_gpu_tensor_free(ext);
        }
    }
    if (n_picks == 0) {
        fprintf(stderr, "no 2-D tensors of the tested quant types in %s\n", path);
        return 2;
    }
    return failures ? 1 : 0;
}
