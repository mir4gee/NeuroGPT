#!/usr/bin/env python3
"""
Fine-tune NeuroGPT on BCI-IV-2a with the authors' own recipe, optionally with
per-session Euclidean Alignment of the input (--align ea). With --align none this
is the vendored decoding path (same model, trainer, split, hyper-parameters), so
`--align none` vs `--align ea` run in the same environment is a controlled
comparison. Reuses train_gpt.py's argument parser, config, model constructor and
HF Trainer plumbing unmodified; only the dataset class is swapped.

After training it evaluates on the held-out subject at the final step and writes
heldout_metrics.json (the repo's own test_metrics.csv is computed on the
*training* subjects because of a dataset swap in train_gpt.py, so it is not used).
"""
import json
import os
import sys
from typing import Dict

import numpy as np
from numpy import random
from torch import manual_seed

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import train_gpt  # noqa: E402  (vendored NeuroGPT training entrypoint)
from alignment.ea_dataset import AlignedMotorImageryDataset  # noqa: E402


def get_args():
    parser = train_gpt.get_args()
    parser.add_argument(
        '--align', metavar='STR', default='none', choices=('none', 'ea'), type=str,
        help="input alignment: 'none' (vendored behaviour) or 'ea' (per-session "
             "Euclidean Alignment, see src/alignment/ea_dataset.py) (default: none)"
    )
    parser.add_argument('--margin', metavar='INT', default=0, type=int,
                        help='extra samples on each side of the t=2..6 s window; training then uses random '
                             'crops inside it, evaluation uses the centre crop (default: 0 = off)')
    parser.add_argument('--noise-std', metavar='FLOAT', default=0.0, type=float, help='train-time Gaussian noise std')
    parser.add_argument('--ch-drop', metavar='FLOAT', default=0.0, type=float, help='train-time channel dropout prob')
    parser.add_argument('--amp-scale', metavar='FLOAT', default=0.0, type=float, help='train-time amplitude jitter')
    parser.add_argument('--val-subjects', metavar='STR', default='', type=str,
                        help="comma-separated 1-based subject numbers to use as validation instead of the fold's "
                             "test subject (for hyper-parameter selection on TRAINING subjects only)")
    parser.add_argument('--exclude-subjects', metavar='STR', default='', type=str,
                        help='comma-separated 1-based subject numbers removed from training (e.g. the outer test subject)')
    parser.add_argument('--init-from', metavar='PATH', default='', type=str,
                        help='model.safetensors of an already fine-tuned run to start from (e.g. the LOSO model)')
    parser.add_argument('--calib-subject', metavar='INT', default=0, type=int,
                        help='1-based subject: train ONLY on its session T (labeled calibration data) and evaluate on '
                             'its session E. Standard competition protocol, NOT the paper\'s cross-subject protocol.')
    parser.add_argument('--calib-val-frac', metavar='FLOAT', default=0.0, type=float,
                        help='with --calib-subject: hold out this fraction of session T for validation, train on the '
                             'rest, and additionally predict session E (written as test_*.npy / test_metrics.json)')
    return parser


def _subject_files(files, subjects):
    out = []
    for s in subjects:
        out += files[2 * (s - 1): 2 * (s - 1) + 2]
    return out


def _parse_subjects(text):
    return [int(x) for x in str(text or '').split(',') if x.strip()]


def train(config: Dict = None):
    if config is None:
        config = train_gpt.get_config(get_args().parse_args())

    assert config['training_style'] == 'decoding', 'alignment is implemented for the decoding setting only'

    os.makedirs(config['log_dir'], exist_ok=True)
    with open(os.path.join(config['log_dir'], 'train_config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)
    config['resume_from'] = None

    if config['set_seed']:
        random.seed(config['seed'])
        manual_seed(config['seed'])

    downstream_path = config['dst_data_path']
    train_folds, test_folds = train_gpt.cv_split_bci(sorted(os.listdir(downstream_path))[:18])
    train_files = train_folds[config['fold_i']]
    test_files = test_folds[config['fold_i']]
    val_subj = _parse_subjects(config.get('val_subjects'))
    if val_subj:
        all_files = sorted(os.listdir(downstream_path))[:18]
        test_files = _subject_files(all_files, val_subj)
        banned = set(test_files) | set(_subject_files(all_files, _parse_subjects(config.get('exclude_subjects'))))
        train_files = [f for f in all_files if f not in banned]

    if config.get('calib_subject'):
        all_files = sorted(os.listdir(downstream_path))[:18]
        s_ = int(config['calib_subject'])
        test_files = [all_files[2 * (s_ - 1)]]       # A0sE.npz, a later session
        train_files = [all_files[2 * (s_ - 1) + 1]]  # A0sT.npz, calibration session

    dataset_kwargs = dict(
        sample_keys=['inputs', 'attention_mask'],
        chunk_len=config['chunk_len'],
        num_chunks=config['num_chunks'],
        ovlp=config['chunk_ovlp'],
        root_path=downstream_path,
        gpt_only=not config['use_encoder'],
        align=config['align'],
        margin=config['margin'],
    )
    vfrac = float(config.get('calib_val_frac') or 0.0)
    split = config.get('calib_subject') and vfrac > 0
    train_dataset = AlignedMotorImageryDataset(
        train_files, augment=True, noise_std=config['noise_std'], ch_drop=config['ch_drop'],
        amp_scale=config['amp_scale'], subset='train' if split else None, subset_frac=vfrac, **dataset_kwargs)
    final_test_dataset = None
    if split:
        validation_dataset = AlignedMotorImageryDataset(train_files, augment=False, subset='val', subset_frac=vfrac,
                                                        **dataset_kwargs)
        final_test_dataset = AlignedMotorImageryDataset(test_files, augment=False, **dataset_kwargs)
    else:
        validation_dataset = AlignedMotorImageryDataset(test_files, augment=False, **dataset_kwargs)

    def model_init(params: Dict = None):
        model_config = dict(config)
        if params is not None:
            model_config |= params
        model = train_gpt.make_model(model_config)
        if config.get('init_from'):
            from safetensors.torch import load_file
            model.load_state_dict(load_file(config['init_from']), strict=True)
        return model

    trainer = train_gpt.make_trainer(
        model_init=model_init,
        training_style=config['training_style'],
        run_name=config['run_name'],
        output_dir=config['log_dir'],
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        per_device_train_batch_size=config['per_device_training_batch_size'],
        per_device_eval_batch_size=config['per_device_validation_batch_size'],
        dataloader_num_workers=config['num_workers'],
        optim=config['optim'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        adam_beta1=config['adam_beta_1'],
        adam_beta2=config['adam_beta_1'],
        adam_epsilon=config['adam_epsilon'],
        max_grad_norm=config['max_grad_norm'],
        lr_scheduler_type=config['lr_scheduler'],
        warmup_ratio=config['warmup_ratio'],
        max_steps=config['training_steps'],
        save_steps=config['training_steps'] * 2,
        logging_steps=config['log_every_n_steps'],
        eval_steps=config['eval_every_n_steps'],
        seed=config['seed'] if config['set_seed'] else int(np.random.choice(range(1, 100000))),
        fp16=config['fp16'],
        deepspeed=config['deepspeed'],
    )

    trainer.train(resume_from_checkpoint=None)
    trainer.save_model(os.path.join(config['log_dir'], 'model_final'))  # small (encoder-only), enables calibration/ensembling later

    metrics = trainer.evaluate(validation_dataset)
    with open(os.path.join(config['log_dir'], 'heldout_metrics.json'), 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    pred = trainer.predict(validation_dataset)
    np.save(os.path.join(config['log_dir'], 'heldout_logits.npy'), pred.predictions)
    np.save(os.path.join(config['log_dir'], 'heldout_labels.npy'), pred.label_ids)
    if final_test_dataset is not None:
        tp = trainer.predict(final_test_dataset)
        np.save(os.path.join(config['log_dir'], 'test_logits.npy'), tp.predictions)
        np.save(os.path.join(config['log_dir'], 'test_labels.npy'), tp.label_ids)
        with open(os.path.join(config['log_dir'], 'test_metrics.json'), 'w') as f:
            json.dump({k: float(v) for k, v in tp.metrics.items()}, f, indent=2)
    print('HELDOUT_FINAL', json.dumps({k: float(v) for k, v in metrics.items()}))
    return trainer


if __name__ == '__main__':
    train()
