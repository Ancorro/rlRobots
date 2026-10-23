"""LoRA for Gemma 4 12B as a decision model (owner 2026-10-21): the base model stays the planner; the adapter is switched
on only for decisions. One forward pass per question; the loss is cross-entropy over the option letters at the answer
position (each option's logit = logsumexp of its bare and space-prefixed letter tokens), so the adapter learns a
classifier over the options, not text. Prompt text and letter tokens come from so101/deciders.py (letter_prompt,
letter_chat, letter_ids), the same code the vLLM decider serves with.

  python vla/train_decide_lora.py --train=runs/decide/train.jsonl --eval=runs/decide/heldout.jsonl --out=runs/decide/lora
         [--rank=16] [--lr=1e-4] [--epochs=1] [--accum=8] [--limit=0] [--eval_only=0]
Writes the adapter (PEFT format, language-model layers only) and <out>/eval.json: held-out accuracy, mean probability
of the right option and accuracy by answer letter (position bias), for the base model and the adapter.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from so101.deciders import letter_chat, letter_ids, letter_prompt  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
BASE = ARGS.get("base", "google/gemma-4-12B-it")
OUT = Path(ARGS.get("out", "runs/decide/lora"))
RANK, LR, EPOCHS, ACCUM = int(ARGS.get("rank", 16)), float(ARGS.get("lr", 1e-4)), int(ARGS.get("epochs", 1)), int(ARGS.get("accum", 8))


def load(path, limit=0):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    return rows[:limit] if limit else rows


def encode(tok, r):
    text = letter_chat(tok, letter_prompt(r["state"], dict(instructions=r["instructions"], criteria=r["criteria"]), r["names"]))
    return torch.tensor([tok.encode(text, add_special_tokens=False)])


def option_logits(model, tok, r, dev):
    try:
        out = model(input_ids=encode(tok, r).to(dev), logits_to_keep=1)
    except TypeError:  # a model class without logits_to_keep
        out = model(input_ids=encode(tok, r).to(dev))
    last = out.logits[0, -1].float()
    return torch.stack([torch.logsumexp(last[ids], 0) for ids in letter_ids(tok, len(r["names"]))])


@torch.no_grad()
def evaluate(model, tok, rows, dev):
    model.eval()
    right, p_right, by_letter, by_q, t0 = 0, 0.0, {}, {}, time.time()
    for r in rows:
        p = torch.softmax(option_logits(model, tok, r, dev), 0)
        ok = int(p.argmax()) == r["answer"]
        right += ok
        p_right += float(p[r["answer"]])
        L = chr(65 + r["answer"])
        by_letter.setdefault(L, [0, 0])[0] += ok
        by_letter[L][1] += 1
        by_q.setdefault(r.get("skill", r["q"]), [0, 0])[0] += ok
        by_q[r.get("skill", r["q"])][1] += 1
    n = len(rows)
    return dict(n=n, accuracy=round(right / n, 4), mean_p_right=round(p_right / n, 4), seconds_per=round((time.time() - t0) / n, 3),
                by_right_skill={k: f"{a}/{b}" for k, (a, b) in by_q.items()},
                by_answer_letter={k: f"{a}/{b}" for k, (a, b) in sorted(by_letter.items())})


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE)
    try:
        model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16).to(dev)
    except ValueError:  # a text-only class for this checkpoint
        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(dev)
    print(type(model).__name__, flush=True)
    held = load(ARGS["eval"], int(ARGS.get("eval_limit", 0)))
    print("letter tokens:", letter_ids(tok, 6), flush=True)
    print("example prompt:\n" + letter_chat(tok, letter_prompt(held[0]["state"], held[0], held[0]["names"]))[-700:], flush=True)
    res = dict(base=evaluate(model, tok, held, dev))
    print("base model, held-out:", res["base"], flush=True)
    if ARGS.get("eval_only", "0") == "1":
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "eval.json").write_text(json.dumps(res, indent=1))
        return
    cfg = LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    train = load(ARGS["train"], int(ARGS.get("limit", 0)))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    steps = math.ceil(len(train) * EPOCHS / ACCUM)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
    rng, k, loss_sum, t0 = np.random.default_rng(0), 0, 0.0, time.time()
    model.train()
    for ep in range(EPOCHS):
        for i in rng.permutation(len(train)):
            r = train[i]
            loss = torch.nn.functional.cross_entropy(option_logits(model, tok, r, dev)[None], torch.tensor([r["answer"]], device=dev))
            (loss / ACCUM).backward()
            loss_sum += float(loss)
            k += 1
            if k % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if k % 400 == 0:
                print(f"epoch {ep} example {k}: loss {loss_sum / 400:.4f}, {(time.time() - t0) / k:.2f} s/example", flush=True)
                loss_sum = 0.0
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    res["adapter"] = evaluate(model, tok, held, dev)
    res["train"] = dict(examples=len(train), epochs=EPOCHS, rank=RANK, lr=LR, minutes=round((time.time() - t0) / 60, 1))
    print("adapter, held-out:", res["adapter"], flush=True)
    (OUT / "eval.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
