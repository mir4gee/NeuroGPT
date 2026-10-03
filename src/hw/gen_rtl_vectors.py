#!/usr/bin/env python3
"""
Test vectors for hw/rtl/frontend_accel.v from a fused export (export_c.py ... fused):
  w.hex        550 lines, 40 int8 weights per line (lane 39 in the most significant byte), index c*25+k
  x.hex        11,000 int8 samples of the first chunk of trial 0, index c*500+t, quantized as in the C model
  expected.hex 19,040 int32 accumulators acc[t][o], index t*40+o (exact integer reference)
usage: python src/hw/gen_rtl_vectors.py EXPORT_DIR OUT_DIR
"""
import os
import re
import sys

import numpy as np

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
hdr = open(os.path.join(src, 'model_weights.h')).read()
assert '#define FUSED_FRONTEND 1' in hdr, 'needs a fused export'
w = np.array([int(v) for v in re.search(r'conv1_w\[\d+\] = \{([^}]*)\}', hdr).group(1).split(',')], dtype=np.int64)
w = w.reshape(40, 22, 25)
s = np.float32(re.search(r'#define conv1_as ([0-9.eE+-]+)f', hdr).group(1))

x = np.fromfile(os.path.join(src, 'inputs.bin'), dtype=np.float32)[:22 * 500].reshape(22, 500)
xq = np.clip(np.rint(x / s), -127, 127).astype(np.int64)  # float32 divide, round half to even

acc = np.zeros((476, 40), dtype=np.int64)
for t in range(476):
    acc[t] = np.einsum('ock,ck->o', w, xq[:, t:t + 25])
assert np.abs(acc).max() < 2 ** 31

with open(os.path.join(out, 'w.hex'), 'w') as f:
    for c in range(22):
        for k in range(25):
            f.write(''.join(f'{int(w[o, c, k]) & 0xff:02x}' for o in reversed(range(40))) + '\n')
with open(os.path.join(out, 'x.hex'), 'w') as f:
    f.write('\n'.join(f'{int(v) & 0xff:02x}' for v in xq.ravel()) + '\n')
with open(os.path.join(out, 'expected.hex'), 'w') as f:
    f.write('\n'.join(f'{int(v) & 0xffffffff:08x}' for v in acc.ravel()) + '\n')
print('vectors written to', out, '| max |acc|', int(np.abs(acc).max()))
