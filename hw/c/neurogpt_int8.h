#ifndef NEUROGPT_INT8_H
#define NEUROGPT_INT8_H
#include <stdint.h>

/* trial: float32 [2 chunks][22 channels][500 samples], preprocessed as in training.
   logits: log-softmax scores for the 4 classes (left hand, right hand, feet, tongue). */
void ng_infer(const float *trial, float logits[4]);
extern uint64_t ng_macs;

#endif
