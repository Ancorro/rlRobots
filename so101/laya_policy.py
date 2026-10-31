"""Laya as a trainable policy: the same per-option probabilities `system_one` returns, but with
gradients, so it can be tuned with log loss (proper scoring) and policy gradient.

Uses Laya internals (`_encode_state`, `collate_items`, temperatures) from laya 0.x; the pre-flight
check in train_decider.py compares against `system_one` so a changed internal shows up at once.
"""
import shutil
from pathlib import Path

import torch
from laya.agent import temp_bucket
from laya.common import QTYPES, collate_items
from safetensors.torch import save_file


class LayaPolicy:
    def __init__(self, agent, questions):
        self.agent, self.model = agent, agent.model.float()
        self.model.eval()  # no dropout: sampled and trained probabilities must match
        self.amp = True  # bf16 autocast; False = full precision (checks only)
        self.ids = list(questions)
        for q in self.ids:
            agent._check_question(q, questions[q])
        self.internal = {q: agent._to_internal(questions[q]) for q in self.ids}
        self.temps = [agent.temperature_by_options.get(temp_bucket(QTYPES[self.internal[q]["t"]],
                                                                    len(questions[q]["criteria"])),
                                                        agent.temperature[QTYPES[self.internal[q]["t"]]])
                      for q in self.ids]

    def logits(self, states):
        """For each state, a list of tempered option-logit tensors, one per question (in question order)."""
        encoded = [self.agent._encode_state(s, self.ids, self.internal) for s in states]
        b = collate_items(encoded, self.agent.tok.pad_token_id)
        dev = self.agent.device
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            raw, _ = self.model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev),
                                b["marker_mask"].to(dev), b["qtype"].to(dev))
        raw = raw.float()
        n = len(self.ids)
        return [[raw[i * n + j, :len(encoded[i][j]["markers"])] / self.temps[j] for j in range(n)]
                for i in range(len(states))]

    def save(self, folder, source_dir):
        """Copy the original checkpoint folder (config, tokenizer, encoder) with the new weights."""
        folder = Path(folder)
        if folder.exists():
            shutil.rmtree(folder)
        shutil.copytree(source_dir, folder, ignore=shutil.ignore_patterns("model.safetensors"))
        save_file({k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items()},
                  str(folder / "model.safetensors"))
