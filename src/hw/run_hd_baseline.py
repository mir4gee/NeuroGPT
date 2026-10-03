#!/usr/bin/env python3
"""
Run the Hersche, Benini & Rahimi (AICAS 2020) binary motor-imagery baseline (github.com/MHersche/HDembedding-BCI)
on our converted BCI-IV-2a files, with the settings of its run_hd.py for IV2a: multiscale Riemannian features
(43 bands, t = 2.5-6 s), binarized SVM / binarized LDA (precision 2), and sparse bipolar random projection to a
binary HD space with SVM-learned class vectors (sparsity 0.9). Protocol as in the repo: train on session T,
test on session E, trials marked as artifacts excluded. Their code is used unmodified; only get_data() is
replaced, because the repo reads the BNCI .mat release and we have the GDF release.
usage: [SUBJECTS=7,8,9] python src/hw/run_hd_baseline.py REPO_DIR NPZ_DIR HD_DIM [HD_DIM ...]
"""
import os
import sys
import types

import numpy as np
from scipy.io import loadmat

REPO, NPZ = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
HD_DIMS = [int(v) for v in sys.argv[3:]] or [10000]


def get_data(subject, training, PATH):
    name = f'A0{subject}{"T" if training else "E"}'
    d = np.load(os.path.join(NPZ, name + '.npz'))
    s, etyp, epos = d['s'], d['etyp'].ravel(), d['epos'].ravel()
    starts = epos[etyp == 768]
    rejected = set(epos[etyp == 1023].tolist())
    labels = loadmat(os.path.join(NPZ, 'true_labels', name + '.mat'))['classlabel'].ravel()
    X, y = [], []
    for st, lab in zip(starts, labels):
        if int(st) in rejected:
            continue
        X.append(s[int(st):int(st) + 1750, :22].T)
        y.append(int(lab))
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)


shim = types.ModuleType('get_data_IV2a')
shim.get_data = get_data
sys.modules['get_data_IV2a'] = shim
os.chdir(REPO)
sys.path.insert(0, REPO)
from main_hd import Hd_model  # noqa: E402
import lda_multires as _lda  # noqa: E402
import svm_multires as _svm  # noqa: E402

# Newer scikit-learn validates every __init__ argument as an attribute; the 2020 classes do not store
# `precision`. Expose it as a class attribute (value 2 = binarized, the setting used here).
_svm.svm_multires.precision = 2
_lda.lda_multires.precision = 2

for subj in range(1, 10):
    Xt, yt = get_data(subj, True, None)
    Xe, ye = get_data(subj, False, None)
    print(f'A0{subj}: train {len(yt)} test {len(ye)} trials (artifact trials removed)')

model = Hd_model('IV2a', './dataset/IV2a/', False, 'assotiative', 'cpu', 2, False, False)
model.feat_type, model.code, model.encoding, model.learning = 'Riemann', 'random_proj_bp', 'single', 'SVM'
model.t_vec, model.f_band, model.k, model.sparsity = np.array([[2.5, 6]]), np.arange(43), 2, 0.9
model.N_feat_per_band = 10879
os.makedirs('./dataset/IV2a/results', exist_ok=True)
SUBJECTS = [int(v) for v in os.environ.get('SUBJECTS', '').split(',') if v]
for D in HD_DIMS:
    model.HD_dim = D
    model.save_path = f'./dataset/IV2a/results/ours_D{D}'
    print(f'\n=== HD dimension {D} ===  columns: binarized SVM | binarized LDA | HD (binary) | HD train')
    if not SUBJECTS:
        model.run()
        continue
    model.N_bands = len(model.f_band)  # what run() sets before its subject loop
    for model.subject in SUBJECTS:
        suc = model.test_hd(0)
        print('Subject{:}; {:0.4f};\t{:0.4f};\t{:0.4f};\t{:0.4f}'.format(model.subject, *suc), flush=True)
