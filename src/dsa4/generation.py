"""Prompts and Qwen generation helpers used by generate_block.py.

The scoring helpers (token log-probability and entropy) work on plain
NumPy arrays, so they can be tested without a GPU.
"""
import json
import re

import numpy as np

ANSWER_SYSTEM = "You answer trivia questions. Reply with the short answer only: a name, number, date or short phrase. No explanation."

VARIANT_PROMPT = """Rewrite the question below in three ways and reply with JSON only.

Question: {question}
Answer: {answer}

1. "paraphrases": two rewordings with exactly the same meaning and the same answer.
2. "lookalike": one question that looks almost the same but changes ONE entity, date or number, so the correct answer is different.
3. "changed": which kind of thing you changed in the look-alike: "entity", "date" or "number".

Reply in this format:
{{"paraphrases": ["...", "..."], "lookalike": "...", "changed": "entity"}}"""


def answer_messages(question: str, passages=None):
    if passages:
        ctx = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
        user = f"Use the passages to answer.\n\n{ctx}\n\nQuestion: {question}\nAnswer:"
    else:
        user = f"Question: {question}\nAnswer:"
    return [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}]


def variant_messages(question: str, answer: str):
    return [{"role": "user", "content": VARIANT_PROMPT.format(question=question, answer=answer)}]


def parse_variants(text: str):
    """Pull the JSON object out of the model reply; return None if it is unusable."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    paras = [p.strip() for p in obj.get("paraphrases", []) if isinstance(p, str) and p.strip()]
    look = obj.get("lookalike", "")
    if len(paras) < 2 or not isinstance(look, str) or not look.strip():
        return None
    changed = obj.get("changed", "")
    changed = changed if changed in ("entity", "date", "number") else "other"
    return {"paraphrases": paras[:2], "lookalike": look.strip(), "changed": changed}


def token_scores(step_logits: np.ndarray, chosen_ids: np.ndarray):
    """From per-step logits (steps x vocab) and the chosen token ids, return
    (mean log-probability of chosen tokens, mean predictive entropy in nats)."""
    logits = np.asarray(step_logits, dtype=np.float64)
    if logits.size == 0:
        return float("nan"), float("nan")
    logits = logits - logits.max(axis=1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    chosen = logp[np.arange(len(chosen_ids)), np.asarray(chosen_ids)]
    p = np.exp(logp)
    entropy = -(p * logp).sum(axis=1)
    return float(chosen.mean()), float(entropy.mean())


def clean_answer(text: str) -> str:
    text = text.strip().split("\n")[0]
    return re.sub(r"^(answer\s*:\s*)", "", text, flags=re.I).strip().rstrip(".")


class QwenGenerator:
    """Thin wrapper around a Hugging Face causal LM (Qwen2.5-Instruct)."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", load_4bit: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        kwargs = {"device_map": "auto"}
        if load_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()

    def generate(self, messages, max_new_tokens: int = 24, with_scores: bool = True):
        """Greedy generation. Returns dict(text, logprob_mean, entropy_mean, n_tokens)."""
        torch = self.torch
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      output_scores=with_scores, return_dict_in_generate=True,
                                      pad_token_id=self.tok.eos_token_id)
        new_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
        # drop end-of-sequence tokens from scoring
        keep = [i for i, t in enumerate(new_ids.tolist()) if t not in (self.tok.eos_token_id,
                                                                        self.tok.pad_token_id)]
        text = self.tok.decode(new_ids, skip_special_tokens=True)
        res = {"text": text, "n_tokens": len(keep)}
        if with_scores and keep:
            logits = torch.stack([out.scores[i][0] for i in keep]).float().cpu().numpy()
            lp, ent = token_scores(logits, new_ids[keep].cpu().numpy())
            res.update(logprob_mean=lp, entropy_mean=ent)
        else:
            res.update(logprob_mean=float("nan"), entropy_mean=float("nan"))
        return res
