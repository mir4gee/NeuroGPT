/*
 * Neuro-GPT encoder-only model (EEG Conformer + FC head), W8A8 inference in portable C.
 * Integer (int8 x int8 -> int32) for every convolution, linear layer and attention product;
 * float for LayerNorm, softmax, GELU, ELU, pooling and residual additions (an MCU with an FPU
 * does these directly; an FPGA would use fixed-point/LUT units).
 *
 * Memory: weights live in flash (const). The front end is computed one time step at a time
 * (temporal conv -> spatial conv -> ELU -> running average-pool), so the 40x22x476 intermediate
 * tensor of the PyTorch model is never stored. Working RAM is a few tens of KB.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "model_weights.h"
#include "neurogpt_int8.h"

#define C 22
#define T 500
#define F 40
#define K1 25
#define T1 (T - K1 + 1)   /* 476 */
#define POOL 75
#define STRIDE 15
#define NT 27            /* (476 - 75) / 15 + 1 */
#define H 10
#define HD 4
#define FF 160
#define FC1 256
#define FC2 32
#define NCLS 4

uint64_t ng_macs;  /* integer multiply-accumulates of the last inference (for the report) */

static inline int8_t q8(float x, float s) {
    float r = nearbyintf(x / s);  /* round half to even, like torch.round */
    if (r > 127.f) r = 127.f;
    if (r < -127.f) r = -127.f;
    return (int8_t)r;
}

static inline uint8_t qu8(float x, float s) {
    float r = nearbyintf(x / s);
    if (r > 255.f) r = 255.f;
    if (r < 0.f) r = 0.f;
    return (uint8_t)r;
}

static inline float elu(float x) { return x > 0.f ? x : expm1f(x); }
static inline float gelu(float x) { return 0.5f * x * (1.f + erff(x * 0.70710678118654752f)); }

/* y[o] = (sum_i W[o][i] * q(x[i])) * s_x * s_w[o] + b[o] */
static void qlinear(const float *x, int in, int out, const int8_t *w, const float *ws, const float *b, float as,
                    float *y) {
    int8_t xq[2160];
    for (int i = 0; i < in; i++) xq[i] = q8(x[i], as);
    for (int o = 0; o < out; o++) {
        const int8_t *row = w + (size_t)o * in;
        int32_t acc = 0;
        for (int i = 0; i < in; i++) acc += (int32_t)row[i] * xq[i];
        y[o] = (float)acc * as * ws[o] + b[o];
    }
    ng_macs += (uint64_t)in * out;
}

static void layernorm(const float *x, const float *g, const float *be, float *y) {
    float mu = 0.f, var = 0.f;
    for (int i = 0; i < F; i++) mu += x[i];
    mu /= F;
    for (int i = 0; i < F; i++) var += (x[i] - mu) * (x[i] - mu);
    var /= F;
    float inv = 1.f / sqrtf(var + 1e-5f);
    for (int i = 0; i < F; i++) y[i] = (x[i] - mu) * inv * g[i] + be[i];
}

/* One 2-s chunk [22][500] -> 27 tokens of 40 (patch embedding). */
static void patch_embed(const float *x, float tok[NT][F]) {
    static int8_t xq[C][T];
    float pool[F][NT];
    memset(pool, 0, sizeof pool);
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++) xq[c][t] = q8(x[c * T + t], conv1_as);

    for (int t = 0; t < T1; t++) {
        int jlo = t >= POOL - 1 ? (t - (POOL - 1) + STRIDE - 1) / STRIDE : 0;
        int jhi = t / STRIDE;
        if (jhi > NT - 1) jhi = NT - 1;
#ifdef FUSED_FRONTEND
        /* one 22x25 spatio-temporal convolution (temporal and spatial convs merged offline) */
        for (int o = 0; o < F; o++) {
            const int8_t *wo = conv1_w + o * C * K1;  /* [o][c][k] */
            int32_t acc = 0;
            for (int c = 0; c < C; c++)
                for (int k = 0; k < K1; k++) acc += (int32_t)wo[c * K1 + k] * xq[c][t + k];
            float z = elu((float)acc * conv1_as * conv1_ws[o] + conv1_b[o]);
            for (int j = jlo; j <= jhi; j++) pool[o][j] += z;
        }
#else
        int8_t y1q[F][C];  /* temporal-conv output at time t, quantized for the spatial conv */
        for (int f = 0; f < F; f++) {
            const int8_t *wf = conv1_w + f * K1;
            for (int c = 0; c < C; c++) {
                int32_t acc = 0;
                for (int k = 0; k < K1; k++) acc += (int32_t)wf[k] * xq[c][t + k];
                y1q[f][c] = q8((float)acc * conv1_as * conv1_ws[f] + conv1_b[f], conv2_as);
            }
        }
        for (int o = 0; o < F; o++) {
            const int8_t *wo = conv2_w + o * F * C;  /* [o][f][c] */
            int32_t acc = 0;
            for (int f = 0; f < F; f++)
                for (int c = 0; c < C; c++) acc += (int32_t)wo[f * C + c] * y1q[f][c];
            float z = elu((float)acc * conv2_as * conv2_ws[o] + conv2_b[o]);  /* BatchNorm folded into conv2 */
            for (int j = jlo; j <= jhi; j++) pool[o][j] += z;
        }
#endif
    }
#ifdef FUSED_FRONTEND
    ng_macs += (uint64_t)T1 * F * C * K1;
#else
    ng_macs += (uint64_t)T1 * F * C * K1 + (uint64_t)T1 * F * F * C;
#endif

    for (int j = 0; j < NT; j++) {
        float pin[F];
        for (int o = 0; o < F; o++) pin[o] = pool[o][j] / POOL;
        qlinear(pin, F, F, proj_w, proj_ws, proj_b, proj_as, tok[j]);
    }
}

#define ATT_BLOCK(b)                                                                                          \
    static void block_##b(float x[NT][F]) {                                                                   \
        float ln[NT][F], q[NT][F], k[NT][F], v[NT][F], ao[NT][F], tmp[F];                                      \
        int8_t qi[NT][F], ki[NT][F], vi[NT][F];                                                                \
        for (int n = 0; n < NT; n++) {                                                                         \
            layernorm(x[n], ln1_##b##_g, ln1_##b##_be, ln[n]);                                                 \
            qlinear(ln[n], F, F, q_##b##_w, q_##b##_ws, q_##b##_b, q_##b##_as, q[n]);                           \
            qlinear(ln[n], F, F, k_##b##_w, k_##b##_ws, k_##b##_b, k_##b##_as, k[n]);                           \
            qlinear(ln[n], F, F, v_##b##_w, v_##b##_ws, v_##b##_b, v_##b##_as, v[n]);                           \
            for (int i = 0; i < F; i++) {                                                                      \
                qi[n][i] = q8(q[n][i], qa_##b##_s);                                                            \
                ki[n][i] = q8(k[n][i], ka_##b##_s);                                                            \
                vi[n][i] = q8(v[n][i], va_##b##_s);                                                            \
            }                                                                                                  \
        }                                                                                                      \
        const float escale = qa_##b##_s * ka_##b##_s / sqrtf((float)F);                                        \
        for (int h = 0; h < H; h++)                                                                            \
            for (int i = 0; i < NT; i++) {                                                                     \
                float e[NT], m = -INFINITY, sum = 0.f;                                                         \
                uint8_t p[NT];                                                                                 \
                for (int j = 0; j < NT; j++) {                                                                 \
                    int32_t acc = 0;                                                                           \
                    for (int d = 0; d < HD; d++) acc += (int32_t)qi[i][h * HD + d] * ki[j][h * HD + d];         \
                    e[j] = (float)acc * escale;                                                                \
                    if (e[j] > m) m = e[j];                                                                    \
                }                                                                                              \
                for (int j = 0; j < NT; j++) { e[j] = expf(e[j] - m); sum += e[j]; }                          \
                for (int j = 0; j < NT; j++) p[j] = qu8(e[j] / sum, pa_##b##_s);                              \
                for (int d = 0; d < HD; d++) {                                                                 \
                    int32_t acc = 0;                                                                           \
                    for (int j = 0; j < NT; j++) acc += (int32_t)p[j] * vi[j][h * HD + d];                      \
                    ao[i][h * HD + d] = (float)acc * pa_##b##_s * va_##b##_s;                                  \
                }                                                                                              \
            }                                                                                                  \
        ng_macs += 2ull * H * NT * NT * HD;                                                                    \
        for (int n = 0; n < NT; n++) {                                                                         \
            qlinear(ao[n], F, F, o_##b##_w, o_##b##_ws, o_##b##_b, o_##b##_as, tmp);                            \
            for (int i = 0; i < F; i++) x[n][i] += tmp[i];                                                     \
        }                                                                                                      \
        for (int n = 0; n < NT; n++) {                                                                         \
            float h1[FF];                                                                                      \
            layernorm(x[n], ln2_##b##_g, ln2_##b##_be, ln[n]);                                                 \
            qlinear(ln[n], F, FF, ff1_##b##_w, ff1_##b##_ws, ff1_##b##_b, ff1_##b##_as, h1);                    \
            for (int i = 0; i < FF; i++) h1[i] = gelu(h1[i]);                                                  \
            qlinear(h1, FF, F, ff2_##b##_w, ff2_##b##_ws, ff2_##b##_b, ff2_##b##_as, tmp);                      \
            for (int i = 0; i < F; i++) x[n][i] += tmp[i];                                                     \
        }                                                                                                      \
    }
BLK_TABLE(ATT_BLOCK)

void ng_infer(const float *trial, float logits[NCLS]) {
    static float feat[2][NT][F];
    ng_macs = 0;
    for (int ch = 0; ch < 2; ch++) {
        patch_embed(trial + (size_t)ch * C * T, feat[ch]);
#define RUN_BLOCK(b) block_##b(feat[ch]);
        BLK_TABLE(RUN_BLOCK)
#undef RUN_BLOCK
    }
    float h1[FC1], h2[FC2];
    qlinear(&feat[0][0][0], 2 * NT * F, FC1, fc1_w, fc1_ws, fc1_b, fc1_as, h1);
    for (int i = 0; i < FC1; i++) h1[i] = elu(h1[i]);
    qlinear(h1, FC1, FC2, fc2_w, fc2_ws, fc2_b, fc2_as, h2);
    for (int i = 0; i < FC2; i++) h2[i] = elu(h2[i]);
    qlinear(h2, FC2, NCLS, fc3_w, fc3_ws, fc3_b, fc3_as, logits);
    float m = logits[0], s = 0.f;  /* log-softmax, as in the PyTorch model */
    for (int i = 1; i < NCLS; i++) if (logits[i] > m) m = logits[i];
    for (int i = 0; i < NCLS; i++) s += expf(logits[i] - m);
    for (int i = 0; i < NCLS; i++) logits[i] = logits[i] - m - logf(s);
}
