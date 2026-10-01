#!/usr/bin/env python3
"""
Class-conditional soft prompt-tuning for NeuroGPT.

Novelty (see PROMPT_TUNING.md): learn a small, fixed "menu" of virtual
prompt tokens, one contiguous block per mental-state condition, and prepend
the *entire* menu to every input sequence (train and inference alike). The
frozen backbone must learn to attend to the block relevant to the trial's
actual EEG content -- no label is used to select a prompt, so nothing about
the answer leaks into the input. Freeze everything except the prompt menu
and the existing pooler/decoding head.
"""
import os
import sys
from typing import Dict

import torch
import torch.nn as nn

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from model import Model  # noqa: E402  (vendored NeuroGPT model)


class ConditionSoftPrompt(nn.Module):
    """
    Holds a (num_conditions, num_tokens_per_condition, embed_dim) parameter
    and prepends the flattened, batch-broadcast menu to inputs_embeds /
    attention_mask. num_tokens_per_condition=0 is a no-op (frozen linear
    probe baseline).
    """

    def __init__(self, num_conditions: int, num_tokens_per_condition: int, embed_dim: int) -> None:
        super().__init__()
        self.num_conditions = num_conditions
        self.num_tokens_per_condition = num_tokens_per_condition
        self.embed_dim = embed_dim
        self.num_prompt_tokens = num_conditions * num_tokens_per_condition

        if self.num_prompt_tokens > 0:
            self.prompt_embeds = nn.Parameter(
                torch.randn(num_conditions, num_tokens_per_condition, embed_dim) * 0.02
            )
        else:
            self.prompt_embeds = None

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        if self.prompt_embeds is None:
            return inputs_embeds, attention_mask

        batch_size = inputs_embeds.size(0)
        menu = self.prompt_embeds.reshape(1, self.num_prompt_tokens, self.embed_dim)
        menu = menu.expand(batch_size, -1, -1).to(dtype=inputs_embeds.dtype)

        prompted_embeds = torch.cat([menu, inputs_embeds], dim=1)

        prompt_mask = torch.ones(
            batch_size, self.num_prompt_tokens,
            dtype=attention_mask.dtype, device=attention_mask.device,
        )
        prompted_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        return prompted_embeds, prompted_mask


class PromptTunedModel(Model):
    """
    Wraps the vendored Model with a ConditionSoftPrompt. forward() is a full
    method override (not a monkeypatch or super()-partial-reuse) since
    Model.forward is a single monolithic method not designed for partial
    reuse -- this is the standard "extend via subclass" pattern for
    vendored code we don't own.

    Decoding-mode only: the reconstruction loss used in CSM pretraining
    assumes the sequence length is unmodified, so this wrapper refuses to
    prepend prompts outside decoding mode.
    """

    def __init__(
        self,
        encoder,
        embedder,
        decoder,
        unembedder=None,
        num_conditions: int = 4,
        num_tokens_per_condition: int = 4,
        readout: str = 'gpt',
        enc_dim: int = None,
    ) -> None:
        super().__init__(encoder=encoder, embedder=embedder, decoder=decoder, unembedder=unembedder)
        self.soft_prompt = ConditionSoftPrompt(
            num_conditions=num_conditions,
            num_tokens_per_condition=num_tokens_per_condition,
            embed_dim=decoder.embed_dim,
        )
        # readout='gpt' (default, the original spec): the classifier reads only
        # the frozen GPT's last-token state. Diagnostic on BCI-IV-2a fold 0
        # showed that state is almost constant across samples (mean pairwise
        # cosine 0.97) and a linear probe on it reaches only 0.36 held-out,
        # versus 0.49 on the frozen encoder output, so the frozen GPT discards
        # most class information. readout='skip' adds a trainable linear
        # projection of the (mean-over-chunks) frozen encoder features to the
        # GPT last-token state before the pooler, so the head can use the
        # encoder information while the prompts still steer the frozen GPT.
        if readout not in ('gpt', 'skip'):
            raise ValueError(f"readout must be 'gpt' or 'skip', got {readout!r}")
        self.readout = readout
        self.skip_proj = None
        if readout == 'skip':
            if encoder is None or enc_dim is None:
                raise ValueError("readout='skip' needs an encoder and enc_dim")
            self.skip_proj = nn.Linear(enc_dim, decoder.embed_dim)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        prep_batch: bool = True,
        return_batch: bool = False,
    ):
        if self.encoder is not None:
            features = self.encoder(batch['inputs'])
            if self.is_decoding_mode and self.ft_only_encoder:
                outputs = {'outputs': features, 'decoding_logits': features}
                return (outputs, batch) if return_batch else outputs

            b, f1, f2 = features.size()
            nchunks = batch['inputs'].size()[1]
            batch['inputs'] = features.view(b // nchunks, nchunks, f1 * f2)
            if self.readout == 'skip':
                chunk_mask = batch['attention_mask'].to(batch['inputs'].dtype).unsqueeze(-1)
                enc_feats = (batch['inputs'] * chunk_mask).sum(dim=1) / chunk_mask.sum(dim=1).clamp(min=1)

        if prep_batch:
            if len(batch['inputs'].size()) > 3:
                bsize, chunk, chann, time = batch['inputs'].size()
                batch['inputs'] = batch['inputs'].view(bsize, chunk, chann * time)
            batch = self.prep_batch(batch=batch)
        else:
            assert 'inputs_embeds' in batch, 'inputs_embeds not in batch'

        batch['inputs_embeds'] = self.embedder(batch=batch)

        if not self.is_decoding_mode and self.soft_prompt.num_prompt_tokens > 0:
            raise RuntimeError(
                "PromptTunedModel only supports prepending prompt tokens in "
                "decoding mode (is_decoding_mode=True); the CSM pretraining "
                "reconstruction loss assumes the sequence length equals the "
                "original number of chunks, which prompt-prepending breaks."
            )

        batch['inputs_embeds'], batch['attention_mask'] = self.soft_prompt(
            inputs_embeds=batch['inputs_embeds'],
            attention_mask=batch['attention_mask'],
        )
        # Defensive: keep attention_mask an integer dtype for the pooler's
        # advanced indexing (attention_mask.sum(dim=1)-1), regardless of
        # what dtype BaseEmbedder.prep_batch cast it to upstream.
        batch['attention_mask'] = batch['attention_mask'].long()

        if self.readout == 'skip' and self.is_decoding_mode:
            hidden = self.decoder.transformer.forward(
                inputs_embeds=batch['inputs_embeds'],
                attention_mask=batch['attention_mask'],
                return_dict=True,
            )['last_hidden_state']
            last = hidden[torch.arange(hidden.size(0), device=hidden.device), batch['attention_mask'].sum(dim=1) - 1]
            pooled = self.decoder.pooler_layer(last + self.skip_proj(enc_feats))
            outputs = {
                'outputs': hidden,
                'pooler_outputs': pooled,
                'decoding_logits': self.decoder.decoding_head(pooled),
            }
        else:
            outputs = self.decoder(batch=batch)

        if self.unembedder is not None and not self.is_decoding_mode:
            outputs['outputs'] = self.unembedder(inputs=outputs['outputs'])['outputs']

        return (outputs, batch) if return_batch else outputs


def freeze_for_prompt_tuning(model: "PromptTunedModel") -> None:
    """
    Freezes encoder, embedder, and the decoder's transformer backbone,
    leaving only the soft-prompt menu and the existing pooler/decoding head
    trainable. Mirrors the granularity of train_gpt.py's --freeze-* flags,
    but is specific to this method so it lives here rather than editing
    vendored code.
    """
    if model.encoder is not None:
        for param in model.encoder.parameters():
            param.requires_grad = False

    for param in model.embedder.parameters():
        param.requires_grad = False

    for name, param in model.decoder.named_parameters():
        if 'pooler_layer' in name or 'decoding_head' in name:
            continue
        param.requires_grad = False

    if model.unembedder is not None:
        for param in model.unembedder.parameters():
            param.requires_grad = False

    # model.soft_prompt.prompt_embeds keeps requires_grad=True (nn.Parameter default)
