"""Decision models that can drive any decision task (reach, pick, sort), all behind one Jev-shaped interface:

    decider.decide(env) -> {question_id: {"choice": option, "probabilities": {option: p, ...}}, ...}

The request every model gets is the TypeSafe Jev / Laya schema: env.state_text() + env.questions.
Add a model by adding a class and a line in make().
"""
import os

import numpy as np


def options(env):
    """Options of each question, in order: action index i of question q means options(env)[q][i]."""
    return {q: list(spec["criteria"]) for q, spec in env.questions.items()}


class Rule:
    """Hand-written baseline: each task's oracle_action (EmbodiedJev's rule baseline)."""
    name = "rule"

    def decide(self, env):
        return {q: {"choice": opts[i], "probabilities": {}} for (q, opts), i in zip(options(env).items(), env.oracle_action())}


class Random:
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def decide(self, env):
        return {q: {"choice": opts[self.rng.integers(len(opts))], "probabilities": {o: 1 / len(opts) for o in opts}}
                for q, opts in options(env).items()}


class Laya:
    """Open-source Laya decision model (Apache-2.0), local GPU. Any Laya checkpoint id or folder."""

    def __init__(self, model_id="convaiinnovations/laya"):
        import laya
        self.agent = laya.load(model_id)
        self.name = model_id.rstrip("/").split("/")[-1]

    def decide(self, env):
        out = self.agent.system_one(env.state_text(), env.questions)["answers"]
        return {q: {"choice": out[q]["choice"], "probabilities": out[q]["probabilities"]} for q in env.questions}


class Jev:
    """TypeSafe Jev (closed, paid API). Needs TYPESAFE_API_KEY; each decision is a billed call."""
    name = "jev"

    def __init__(self):
        import httpx
        self.key = os.environ["TYPESAFE_API_KEY"]
        self.http = httpx.Client(timeout=25)

    def decide(self, env):
        r = self.http.post("https://api.typesafe.ai/v1/systemone",
                           headers={"Authorization": f"Bearer {self.key}"},
                           json={"model": "jev-latest", "state": env.state_text(), "questions": env.questions})
        r.raise_for_status()
        out = r.json()["answers"]
        return {q: {"choice": out[q]["choice"], "probabilities": out[q].get("probabilities", {})} for q in env.questions}


class JevLocal:
    """AutoTrust's open JEV models (JEV-9B, JEV-27B-VL) served locally by their vl/serve_decide.py: POST /v1/decide,
    untuned (owner 2026-10-07: the decision model, swapped in for Laya, no tuning).
    one choice question per call. Options go in as "name: description" (their card: one-line descriptions help) and come
    back by position. make("jev-local") reads the server address from JEV_URL (default http://127.0.0.1:8000)."""
    name = "jev-9b"

    def __init__(self, url=None):
        import httpx
        self.url = (url or os.environ.get("JEV_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.http = httpx.Client(timeout=60)

    def decide(self, env):
        """Questions in order (skill, then object). With the env's `valid_targets` (skill -> allowed objects), the object
        question offers only what the chosen skill accepts: pick an object, place on a tray or shelf (owner 2026-10-07)."""
        out, state = {}, env.state_text()
        valid = getattr(env, "valid_targets", None)
        for q, spec in env.questions.items():
            names = list(spec["criteria"])
            if q == "object" and valid and out.get("skill", {}).get("choice") in valid:
                names = [n for n in names if n in valid[out["skill"]["choice"]]]
            if len(names) == 1:  # the server takes 2+ options; one left is the answer
                out[q] = {"choice": names[0], "probabilities": {names[0]: 1.0}}
                continue
            r = self.http.post(self.url + "/v1/decide", json=dict(
                kind="choice", state=state, question=spec["instructions"],
                options=[f"{o}: {spec['criteria'][o]}" for o in names]))
            r.raise_for_status()
            d = r.json()
            out[q] = {"choice": names[d["choice_index"]], "probabilities": dict(zip(names, d["probabilities"]))}
        return out


class SubQ:
    """SubQ Snap (Subquadratic, paid API; owner 2026-10-14: replaces the decision model "for now"). Same System One
    request as TypeSafe Jev: POST https://api.subq.ai/v1/systemone, model subq-snap-preview; billed on input tokens.
    Key from SUBQ_API_KEY or ~/.subq_key (cluster, mode 600; never in code or logs). Questions in order (skill, then
    object), one call each, so the object question offers only what the chosen skill accepts (as JevLocal)."""
    name = "subq-snap"
    URL = "https://api.subq.ai/v1/systemone"

    def __init__(self, model="subq-snap-preview"):
        from pathlib import Path

        import httpx
        key = os.environ.get("SUBQ_API_KEY") or (Path.home() / ".subq_key").read_text().strip()
        self.model, self.headers = model, {"Authorization": f"Bearer {key}"}
        self.http = httpx.Client(timeout=60)
        self.input_tokens = 0

    def ask(self, state, q, spec):
        import time

        import httpx
        body = dict(model=self.model, state=state, questions={q: dict(spec, type="choice")})
        for wait in (1, 2, 4, 8, 16, None):  # docs: retry 429, 5xx and network errors with backoff; other 4xx are our bugs
            try:
                r = self.http.post(self.URL, headers=self.headers, json=body)
            except httpx.TransportError:  # dev60-vision-subq stopped at episode 26 on a read timeout (2026-10-14)
                if wait is None:
                    raise
                time.sleep(wait)
                continue
            if (r.status_code != 429 and r.status_code < 500) or wait is None:
                break
            time.sleep(wait)
        if r.status_code >= 400:
            raise RuntimeError(f"SubQ {r.status_code}: {r.text[:300]}")
        d = r.json()
        self.input_tokens += d.get("usage", {}).get("input_tokens", 0)
        return d["answers"][q]

    def decide(self, env):
        out, state = {}, env.state_text()
        valid = getattr(env, "valid_targets", None)
        for q, spec in env.questions.items():
            names = list(spec["criteria"])
            if q == "object" and valid and out.get("skill", {}).get("choice") in valid:
                names = [n for n in names if n in valid[out["skill"]["choice"]]]
            if len(names) == 1:
                out[q] = {"choice": names[0], "probabilities": {names[0]: 1.0}}
                continue
            a = self.ask(state, q, dict(instructions=spec["instructions"], criteria={n: spec["criteria"][n] for n in names}))
            out[q] = {"choice": a["choice"], "probabilities": a.get("probabilities", {})}
        return out


def letter_prompt(state, spec, names):
    """The one-pass decision prompt shared by GemmaLetter, GemmaVLLM and vla/train_decide_lora.py: options lettered A, B, ..."""
    opts = "\n".join(f"{chr(65 + i)}) {n}: {spec['criteria'][n]}" for i, n in enumerate(names))
    return (f"{state}\n\nQuestion: {spec['instructions']}\nOptions:\n{opts}\n\n"
            "Reply with the letter of the best option only.")


def letter_chat(tok, prompt):
    """The exact text the model reads: chat template, assistant turn opened with "Answer:" (next token: the letter)."""
    return tok.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": "Answer:"}],
                                   continue_final_message=True, tokenize=False)


def letter_ids(tok, n):
    """Token ids that spell each of the first n letters as one token, bare and with a leading space: [[ids of A], ...]."""
    out = []
    for i in range(n):
        ids = {t[0] for v in (chr(65 + i), " " + chr(65 + i)) if len(t := tok.encode(v, add_special_tokens=False)) == 1}
        out.append(sorted(ids))
    return out


JOINT_Q = ("What should the robot do next? Pick grasps an object and lifts it; place puts the held object on a tray "
           "or the shelf; done ends the task.")


def joint_question(env):
    """A step's skill and object questions as one question over the valid (skill, object) pairs, so one forward pass
    decides the step (owner 2026-10-21: one request per step, as SubQ Snap answers a whole step at once). With the
    gripper empty the pairs are pick <each object> + done; holding something, place <each place> + done.
    Returns (spec, names, pairs) or None when the env does not ask skill-then-object."""
    qs, valid = env.questions, getattr(env, "valid_targets", None)
    if set(qs) != {"skill", "object"} or not valid:
        return None
    sk, ob = qs["skill"]["criteria"], qs["object"]["criteria"]
    pairs = [(s, o) for s in sk if s != "done" for o in valid.get(s, []) if o in ob] + [("done", None)] * ("done" in sk)
    names = [f"{s} {o}" if o else s for s, o in pairs]
    verb = {"pick": "pick up {}", "place": "put the held object on {}"}
    crit = {n: (verb.get(s, s + " {}").format(ob[o]) if o else sk[s]) for n, (s, o) in zip(names, pairs)}
    return dict(instructions=JOINT_Q, criteria=crit), names, pairs


def split_joint(p, names, pairs, sk_opts):
    """Pair probabilities -> the stack's two answers: p(skill) summed over its pairs, p(object | chosen skill)."""
    ps = {s: sum(p[n] for n, (s2, _) in zip(names, pairs) if s2 == s) for s in sk_opts}
    skill = max(ps, key=ps.get)
    po = {o: p[n] for n, (s2, o) in zip(names, pairs) if s2 == skill and o}
    z = sum(po.values()) or 1.0
    po = {o: v / z for o, v in po.items()}
    out = {"skill": {"choice": skill, "probabilities": ps}}
    out["object"] = {"choice": max(po, key=po.get), "probabilities": po} if po else {"choice": None, "probabilities": {}}
    return out


class GemmaLetter:
    """The planner's own model as the decision model, untrained (owner 2026-10-21: one model for planning and deciding):
    Gemma 4 12B on the same Ollama server, one forward pass per question. Options are lettered A, B, ...; the prompt ends
    at "Answer:" and the model writes one token with thinking off; the probability of each option is its letter's
    probability among the top token log-probabilities, renormalized over the allowed letters (docs/plan.md "Laya mode").
    A letter outside the top 20 gets 0. One question per step over the valid (skill, object) pairs (joint_question)."""
    name = "gemma-letter"

    def __init__(self, model="gemma4:12b"):
        import httpx
        host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
        self.url = (host if host.startswith("http") else "http://" + host) + "/api/chat"
        self.model, self.http = model, httpx.Client(timeout=120)
        self.misses = 0  # forward passes whose top tokens held no allowed letter (then the first option is chosen)

    def ask(self, state, spec, names):
        letters = [chr(65 + i) for i in range(len(names))]
        prompt = letter_prompt(state, spec, names)
        r = self.http.post(self.url, json=dict(
            model=self.model, stream=False, think=False, logprobs=True, top_logprobs=20,
            messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": "Answer:"}],
            options=dict(temperature=0, num_predict=1, num_ctx=16384)))
        r.raise_for_status()
        lp = r.json().get("logprobs") or []
        top = {}
        for t in (lp[0].get("top_logprobs") or [] if lp else []):
            k = t["token"].strip().strip("()*.").upper()
            if k in letters:
                top[k] = top.get(k, 0.0) + float(np.exp(t["logprob"]))
        if not top:
            self.misses += 1
            return names[0], {n: 1.0 / len(names) for n in names}
        z = sum(top.values())
        p = {n: top.get(c, 0.0) / z for c, n in zip(letters, names)}
        return max(p, key=p.get), p

    def decide(self, env):
        """One forward pass per step: the joint (skill, object) question when the env asks skill-then-object."""
        state, j = env.state_text(), joint_question(env)
        if j:
            spec, names, pairs = j
            if len(names) == 1:
                p = {names[0]: 1.0}
            else:
                _, p = self.ask(state, spec, names)
            out = split_joint(p, names, pairs, list(env.questions["skill"]["criteria"]))
            if out["object"]["choice"] is None:  # done: any object; the stack ignores it
                out["object"] = {"choice": list(env.questions["object"]["criteria"])[0], "probabilities": {}}
            return out
        out = {}
        for q, spec in env.questions.items():
            names = list(spec["criteria"])
            if len(names) == 1:
                out[q] = {"choice": names[0], "probabilities": {names[0]: 1.0}}
                continue
            choice, p = self.ask(state, spec, names)
            out[q] = {"choice": choice, "probabilities": p}
        return out


class GemmaVLLM(GemmaLetter):
    """Gemma 4 12B on vLLM, one forward pass, exact constrained choice: only the option letters may be produced
    (allowed_token_ids), the server returns their log-probabilities after that mask (--logprobs-mode processed_logprobs),
    and they are renormalized per option (bare and space-prefixed letter tokens summed). adapter: a LoRA module name the
    server was started with (--lora-modules), used for decisions only; None = the base model. vLLM renders the chat
    template and tokenizes (continue_final_message: the reply opened with "Answer:"), the same template training reads
    through apply_chat_template (vla/train_decide_lora.py); no local tokenizer needed."""
    name = "gemma-vllm"

    def __init__(self, adapter=None):
        import httpx
        self.root = os.environ.get("GEMMA_URL", "http://127.0.0.1:8100").rstrip("/")
        self.base = os.environ.get("GEMMA_MODEL", "google/gemma-4-12B-it")
        self.model = adapter or self.base
        self.http, self.misses, self.ids = httpx.Client(timeout=120), 0, {}

    def letter_token(self, text):
        if text not in self.ids:
            r = self.http.post(self.root + "/tokenize", json=dict(model=self.base, prompt=text, add_special_tokens=False))
            r.raise_for_status()
            t = r.json()["tokens"]
            self.ids[text] = t[0] if len(t) == 1 else None
        return self.ids[text]

    def ask(self, state, spec, names):
        allowed = sorted({t for i in range(len(names)) for v in (chr(65 + i), " " + chr(65 + i))
                          if (t := self.letter_token(v)) is not None})
        r = self.http.post(self.root + "/v1/chat/completions", json=dict(
            model=self.model, messages=[{"role": "user", "content": letter_prompt(state, spec, names)},
                                        {"role": "assistant", "content": "Answer:"}],
            continue_final_message=True, add_generation_prompt=False, max_tokens=1,
            # temperature 1, top_p 1: processed log-probabilities are then the plain softmax over the allowed letters
            # (at temperature 0 they come back one-hot, every answer p = 1.0); the choice is our argmax, not the sample
            temperature=1.0, top_p=1.0, logprobs=True, top_logprobs=len(allowed), allowed_token_ids=allowed))
        r.raise_for_status()
        top = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        by_text = {}
        for t in top:
            k = t["token"].strip().upper()
            by_text[k] = by_text.get(k, 0.0) + float(np.exp(t["logprob"]))
        p = np.array([by_text.get(chr(65 + i), 0.0) for i in range(len(names))])
        if p.sum() <= 0:
            self.misses += 1
            return names[0], {n: 1.0 / len(names) for n in names}
        p = p / p.sum()
        return names[int(p.argmax())], dict(zip(names, p.tolist()))


def make(name):
    if name == "rule":
        return Rule()
    if name == "random":
        return Random()
    if name == "jev":
        return Jev()
    if name == "laya":
        return Laya()
    if name == "jev-local":
        return JevLocal()
    if name == "subq":
        return SubQ()
    if name == "gemma":
        return GemmaLetter()
    if name == "gemma-vllm":
        return GemmaVLLM()
    if name.startswith("gemma-lora"):  # gemma-lora[:module name], the LoRA module the vLLM server was started with
        return GemmaVLLM(name.split(":", 1)[1] if ":" in name else "gemma-decide")
    return Laya(name)  # any other value: a Laya checkpoint id or local folder (e.g. an RL-tuned one)
