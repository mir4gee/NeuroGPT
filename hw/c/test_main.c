#define _POSIX_C_SOURCE 199309L
/* Host test: run the C model on exported test vectors, compare with PyTorch, time it.
   usage: ./ng_test DIR   (DIR holds inputs.bin, labels.bin, ref_logits.bin from export_c.py) */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "model_weights.h"
#include "neurogpt_int8.h"

#define TRIAL (2 * 22 * 500)

static void *slurp(const char *dir, const char *name, long *bytes) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    *bytes = ftell(f);
    rewind(f);
    void *buf = malloc(*bytes);
    if (fread(buf, 1, *bytes, f) != (size_t)*bytes) { perror("read"); exit(1); }
    fclose(f);
    return buf;
}

static int argmax4(const float *v) {
    int a = 0;
    for (int i = 1; i < 4; i++) if (v[i] > v[a]) a = i;
    return a;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s DIR\n", argv[0]); return 2; }
    long nb;
    float *x = slurp(argv[1], "inputs.bin", &nb);
    int n = (int)(nb / (TRIAL * sizeof(float)));
    int *y = slurp(argv[1], "labels.bin", &nb);
    float *ref = slurp(argv[1], "ref_logits.bin", &nb);

    int correct = 0, ref_correct = 0, agree = 0;
    double maxdiff = 0.0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < n; i++) {
        float lg[4];
        ng_infer(x + (size_t)i * TRIAL, lg);
        int p = argmax4(lg), pr = argmax4(ref + 4 * i);
        correct += p == y[i];
        ref_correct += pr == y[i];
        agree += p == pr;
        for (int k = 0; k < 4; k++) {
            double d = fabs((double)lg[k] - ref[4 * i + k]);
            if (d > maxdiff) maxdiff = d;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = ((t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6) / n;
    printf("trials %d | C accuracy %.3f | PyTorch W8A8 accuracy %.3f | prediction agreement %.4f | max |logit diff| %.2e\n",
           n, (double)correct / n, (double)ref_correct / n, (double)agree / n, maxdiff);
    printf("int8 weights %d bytes | integer MACs per trial %llu | host latency %.2f ms/trial (single core)\n",
           N_INT8_WEIGHTS, (unsigned long long)ng_macs, ms);
    return 0;
}
