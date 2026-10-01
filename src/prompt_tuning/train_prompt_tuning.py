#!/usr/bin/env python3
"""
Fine-tune NeuroGPT for downstream classification with the class-conditional
soft-prompt-tuning novelty (see PROMPT_TUNING.md), instead of the paper's
own fine-tuning approach (that's what train_gpt.py + scripts/finetune.sh
already do -- see PROMPT_TUNING.md for the "actual" baseline command).

This script is decoding-mode only. It reuses train_gpt.py's argument
parser, dataset construction, and HF Trainer plumbing (make_trainer)
unmodified, swapping in only the model constructor: instead of a plain
frozen/unfrozen Model, it builds a PromptTunedModel and applies
freeze_for_prompt_tuning.

BCI-IV-2a's 4 motor-imagery classes stand in here for "mental-state
conditions" -- no focused/mind-wandering EEG dataset ships with the
NeuroGPT repo. The mechanism itself is dataset-agnostic: swapping in a
real attention-state dataset only requires a different Dataset class and
--num-decoding-classes.

Leave the original --freeze-* flags at their defaults (False) when using
this script; --freeze-backbone (default True) does the freezing instead,
uniformly across encoder/embedder/decoder-transformer.
"""
import json
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
from numpy import random
from torch import manual_seed

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import train_gpt  # noqa: E402  (vendored NeuroGPT training entrypoint)
from prompt_tuning.soft_prompt import PromptTunedModel, freeze_for_prompt_tuning  # noqa: E402


def get_args():
    parser = train_gpt.get_args()
    parser.add_argument(
        '--num-prompt-tokens-per-condition',
        metavar='INT', default=4, type=int,
        help='number of learned virtual prompt tokens per mental-state '
             'condition; 0 disables prompting (frozen linear-probe baseline) '
             '(default: 4)'
    )
    parser.add_argument(
        '--freeze-backbone',
        metavar='BOOL', default='True', choices=('True', 'False'), type=str,
        help='whether to freeze encoder/embedder/decoder-transformer and '
             'train only the soft-prompt menu + pooler + decoding head '
             '(default: True). Leave the original --freeze-* flags at '
             'their defaults (False) when using this flag.'
    )
    return parser


def make_prompt_tuned_model(config: Dict):
    base_model = train_gpt.make_model(dict(config))

    model = PromptTunedModel(
        encoder=base_model.encoder,
        embedder=base_model.embedder,
        decoder=base_model.decoder,
        unembedder=base_model.unembedder,
        num_conditions=config['num_decoding_classes'],
        num_tokens_per_condition=config['num_prompt_tokens_per_condition'],
    )
    # base_model already ran switch_decoding_mode() inside make_model()
    # (same shared decoder/embedder objects), so just copy the top-level
    # flags rather than re-invoking switch_decoding_mode (which would
    # re-warn about the pooler/head "already existing").
    model.is_decoding_mode = base_model.is_decoding_mode
    model.ft_only_encoder = base_model.ft_only_encoder

    if config['freeze_backbone']:
        freeze_for_prompt_tuning(model)

    return model


def train(config: Dict = None):
    if config is None:
        config = train_gpt.get_config(get_args().parse_args())

    assert config['training_style'] == 'decoding', (
        "Soft prompt-tuning is implemented for the decoding/classification "
        "setting only (see PROMPT_TUNING.md); CSM pretraining is unaffected "
        "and should keep using train_gpt.py directly."
    )

    if config['do_train']:
        os.makedirs(config['log_dir'], exist_ok=True)
        config_filepath = os.path.join(config['log_dir'], 'train_config.json')
        with open(config_filepath, 'w') as f:
            json.dump(config, f, indent=2)
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
    )
    train_dataset = train_gpt.MotorImageryDataset(train_files, **dataset_kwargs)
    test_dataset = train_gpt.MotorImageryDataset(test_files, **dataset_kwargs)
    # Matches train_gpt.py's own decoding-mode branch exactly (including
    # this validation/test swap), for apples-to-apples comparison.
    validation_dataset = test_dataset
    test_dataset = train_dataset

    def model_init(params: Dict = None):
        model_config = dict(config)
        if params is not None:
            model_config |= params
        return make_prompt_tuned_model(model_config)

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
        seed=config['seed'] if config['set_seed'] else np.random.choice(range(1, 100000)),
        fp16=config['fp16'],
        deepspeed=config['deepspeed'],
    )

    if config['do_train']:
        trainer.train(resume_from_checkpoint=config['resume_from'])
        trainer.save_model(os.path.join(config['log_dir'], 'model_final'))

    test_prediction = trainer.predict(test_dataset)
    pd.DataFrame(test_prediction.metrics, index=[0]).to_csv(
        os.path.join(config['log_dir'], 'test_metrics.csv'), index=False
    )
    np.save(os.path.join(config['log_dir'], 'test_predictions.npy'), test_prediction.predictions)
    np.save(os.path.join(config['log_dir'], 'test_label_ids.npy'), test_prediction.label_ids)

    return trainer


if __name__ == '__main__':
    trainer = train()
