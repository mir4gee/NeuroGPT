#!/usr/bin/env python3
"""
Convert the raw BCI Competition IV-2a files into the .npz layout that
src/batcher/downstream_dataset.py::MotorImageryDataset expects (keys s, etyp,
epos, edur, artifacts, plus true_labels/<name>.mat).

Usage: python scripts/convert_bci2a.py RAW_DIR OUT_DIR
RAW_DIR holds A01T.gdf ... A09E.gdf and a true_labels/ folder with the .mat files.
"""
import os
import shutil
import sys
import warnings

import mne
import numpy as np

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')


def main(raw_dir: str, out_dir: str) -> None:
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(os.path.join(out_dir, 'true_labels'))
    for f in sorted(x for x in os.listdir(raw_dir) if x.endswith('.gdf')):
        raw = mne.io.read_raw_gdf(os.path.join(raw_dir, f), preload=True)
        sf = raw.info['sfreq']
        etyp = np.array([int(d) for d in raw.annotations.description]).reshape(-1, 1)
        np.savez(
            os.path.join(out_dir, f[:-4] + '.npz'),
            s=(raw.get_data() * 1e6).T,  # microvolts, (time, channels)
            etyp=etyp,
            epos=np.round(raw.annotations.onset * sf).astype(int).reshape(-1, 1),
            edur=np.round(raw.annotations.duration * sf).astype(int).reshape(-1, 1),
            artifacts=np.zeros((len(etyp), 1)),
        )
        print(f'{f[:-4]}: {raw.n_times} samples @ {sf:.0f} Hz, {int((etyp == 768).sum())} trials (expect 288)')
    labels = os.path.join(raw_dir, 'true_labels')
    for f in os.listdir(labels):
        shutil.copy(os.path.join(labels, f), os.path.join(out_dir, 'true_labels', f))
    print('converted', len([x for x in os.listdir(out_dir) if x.endswith('.npz')]), 'files ->', out_dir)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
