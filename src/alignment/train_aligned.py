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
    return parser


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

    dataset_kwargs = dict(
        sample_keys=['inputs', 'attention_mask'],
        chunk_len=config['chunk_len'],
        num_chunks=config['num_chunks'],
        ovlp=config['chunk_ovlp'],
        root_path=downstream_path,
        gpt_only=not config['use_encoder'],
        align=config['align'],
    )
    train_dataset = AlignedMotorImageryDataset(train_files, **dataset_kwargs)
    validation_dataset = AlignedMotorImageryDataset(test_files, **dataset_kwargs)

    def model_init(params: Dict = None):
        model_config = dict(config)
        if params is not None:
            model_config |= params
        return train_gpt.make_model(model_config)

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

    metrics = trainer.evaluate(validation_dataset)
    with open(os.path.join(config['log_dir'], 'heldout_metrics.json'), 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print('HELDOUT_FINAL', json.dumps({k: float(v) for k, v in metrics.items()}))
    return trainer


if __name__ == '__main__':
    train()
