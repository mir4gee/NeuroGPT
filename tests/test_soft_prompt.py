#!/usr/bin/env python3
"""
Smoke test for the class-conditional soft prompt-tuning novelty.

No real EEG data or the released pretrained checkpoint is available in
this environment, so this builds a tiny random GPT2Config decoder and
fabricated tensors matching the shapes NeuroGPT's own pipeline produces,
and checks that:
  - the soft-prompt menu prepends the right number of tokens,
  - forward + backward runs in decoding mode,
  - num_tokens_per_condition=0 degenerates to a no-op (frozen baseline),
  - freezing leaves only the prompt menu + pooler + decoding head trainable.

Run with: python tests/test_soft_prompt.py
"""
import os
import sys

import torch

_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
_SRC_DIR = os.path.join(os.path.dirname(_TEST_DIR), 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from decoder.gpt import GPTModel  # noqa: E402
from embedder.base import BaseEmbedder  # noqa: E402
from prompt_tuning.soft_prompt import ConditionSoftPrompt, PromptTunedModel, freeze_for_prompt_tuning  # noqa: E402

EMBED_DIM = 32
NUM_CONDITIONS = 4
NUM_TOKENS_PER_CONDITION = 3
BATCH_SIZE = 2
NUM_CHUNKS = 5


def _make_model(num_tokens_per_condition):
    embedder = BaseEmbedder(in_dim=EMBED_DIM, embed_dim=EMBED_DIM, num_hidden_layers=1)
    decoder = GPTModel(num_hidden_layers=2, num_attention_heads=4, embed_dim=EMBED_DIM, n_positions=64)
    model = PromptTunedModel(
        encoder=None, embedder=embedder, decoder=decoder, unembedder=None,
        num_conditions=NUM_CONDITIONS, num_tokens_per_condition=num_tokens_per_condition,
    )
    model.switch_decoding_mode(is_decoding_mode=True, num_decoding_classes=NUM_CONDITIONS)
    return model


def _make_batch():
    inputs = torch.randn(BATCH_SIZE, NUM_CHUNKS, EMBED_DIM)
    attention_mask = torch.ones(BATCH_SIZE, NUM_CHUNKS, dtype=torch.long)
    attention_mask[0, -1] = 0  # exercise padding / last-token indexing
    labels = torch.randint(0, NUM_CONDITIONS, (BATCH_SIZE,))
    return {'inputs': inputs, 'attention_mask': attention_mask, 'labels': labels}


def test_prompt_prepends_correct_shape():
    prompt = ConditionSoftPrompt(NUM_CONDITIONS, NUM_TOKENS_PER_CONDITION, EMBED_DIM)
    inputs_embeds = torch.randn(BATCH_SIZE, NUM_CHUNKS, EMBED_DIM)
    attention_mask = torch.ones(BATCH_SIZE, NUM_CHUNKS, dtype=torch.long)

    new_embeds, new_mask = prompt(inputs_embeds, attention_mask)

    expected_prompt_len = NUM_CONDITIONS * NUM_TOKENS_PER_CONDITION
    assert new_embeds.shape == (BATCH_SIZE, expected_prompt_len + NUM_CHUNKS, EMBED_DIM)
    assert new_mask.shape == (BATCH_SIZE, expected_prompt_len + NUM_CHUNKS)
    assert torch.all(new_mask[:, :expected_prompt_len] == 1)


def test_zero_tokens_is_a_noop():
    prompt = ConditionSoftPrompt(NUM_CONDITIONS, 0, EMBED_DIM)
    inputs_embeds = torch.randn(BATCH_SIZE, NUM_CHUNKS, EMBED_DIM)
    attention_mask = torch.ones(BATCH_SIZE, NUM_CHUNKS, dtype=torch.long)

    new_embeds, new_mask = prompt(inputs_embeds, attention_mask)

    assert torch.equal(new_embeds, inputs_embeds)
    assert torch.equal(new_mask, attention_mask)


def test_forward_and_backward_run():
    model = _make_model(NUM_TOKENS_PER_CONDITION)
    batch = _make_batch()

    losses, outputs = model.compute_loss(batch=batch, return_outputs=True)

    assert 'decoding_logits' in outputs
    assert outputs['decoding_logits'].shape == (BATCH_SIZE, NUM_CONDITIONS)

    losses['loss'].backward()
    assert model.soft_prompt.prompt_embeds.grad is not None


def test_zero_prompt_tokens_baseline_runs():
    model = _make_model(0)
    batch = _make_batch()

    losses, outputs = model.compute_loss(batch=batch, return_outputs=True)
    assert outputs['decoding_logits'].shape == (BATCH_SIZE, NUM_CONDITIONS)
    losses['loss'].backward()


def test_freezing_leaves_only_prompt_and_head_trainable():
    model = _make_model(NUM_TOKENS_PER_CONDITION)
    freeze_for_prompt_tuning(model)

    for n, p in model.named_parameters():
        is_trainable_group = n.startswith('soft_prompt.') or 'pooler_layer' in n or 'decoding_head' in n
        assert p.requires_grad == is_trainable_group, f'unexpected requires_grad for {n}: {p.requires_grad}'

    assert any(n.startswith('embedder.') and not p.requires_grad for n, p in model.named_parameters())


class _DummyEncoder(torch.nn.Module):
    """Maps (B, chunks, 2, 16) -> (B*chunks, 2, 16), like the real encoder's output layout."""
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(16, 16)

    def forward(self, x):
        b, n, c, t = x.shape
        return self.lin(x.reshape(b * n, c, t))


def test_skip_readout_forward_backward_and_freeze():
    embedder = BaseEmbedder(in_dim=EMBED_DIM, embed_dim=EMBED_DIM, num_hidden_layers=1)
    decoder = GPTModel(num_hidden_layers=2, num_attention_heads=4, embed_dim=EMBED_DIM, n_positions=64)
    model = PromptTunedModel(
        encoder=_DummyEncoder(), embedder=embedder, decoder=decoder, unembedder=None,
        num_conditions=NUM_CONDITIONS, num_tokens_per_condition=NUM_TOKENS_PER_CONDITION,
        readout='skip', enc_dim=EMBED_DIM,
    )
    model.switch_decoding_mode(is_decoding_mode=True, num_decoding_classes=NUM_CONDITIONS)
    freeze_for_prompt_tuning(model)

    model.train()
    assert not model.encoder.training and not model.embedder.training and not model.decoder.transformer.training, \
        'frozen backbone must stay in eval mode (no dropout / BN updates) during training'
    assert model.decoder.pooler_layer.training and model.decoder.decoding_head.training

    batch = {
        'inputs': torch.randn(BATCH_SIZE, NUM_CHUNKS, 2, 16),
        'attention_mask': torch.ones(BATCH_SIZE, NUM_CHUNKS, dtype=torch.long),
        'labels': torch.randint(0, NUM_CONDITIONS, (BATCH_SIZE,)),
    }
    losses, outputs = model.compute_loss(batch=batch, return_outputs=True)
    assert outputs['decoding_logits'].shape == (BATCH_SIZE, NUM_CONDITIONS)
    losses['loss'].backward()

    assert model.skip_proj.weight.requires_grad and model.skip_proj.weight.grad is not None
    assert model.soft_prompt.prompt_embeds.grad is not None
    assert all(not p.requires_grad for p in model.encoder.parameters())

    try:
        PromptTunedModel(encoder=None, embedder=embedder, decoder=decoder, readout='skip', enc_dim=EMBED_DIM)
    except ValueError:
        pass
    else:
        raise AssertionError("readout='skip' without an encoder must raise")


if __name__ == '__main__':
    test_skip_readout_forward_backward_and_freeze()
    test_prompt_prepends_correct_shape()
    test_zero_tokens_is_a_noop()
    test_forward_and_backward_run()
    test_zero_prompt_tokens_baseline_runs()
    test_freezing_leaves_only_prompt_and_head_trainable()
    print('All soft-prompt smoke tests passed.')
