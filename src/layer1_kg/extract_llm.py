"""
Layer 1 LLM extraction: use Qwen to extract Article 9 requirements
from raw paragraph text into a candidate knowledge graph.

Each paragraph is extracted 3 times; the run yielding the most requirements
that parse as valid JSON is kept (majority-by-consistency).
"""

import json
import re
from pathlib import Path
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

ROOT = Path(__file__).resolve().parents[2]
PARAS = ROOT / "data" / "article9" / "article9_paragraphs.json"
PROMPT = ROOT / "prompts" / "extraction_prompt.txt"
OUT = ROOT / "data" / "gold_standard" / "llm_extracted.json"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_RUNS = 3

def load_model():
    print("Loading Qwen2.5-7B-Instruct...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto", device_map="auto")
    print("Model loaded on:", next(model.parameters()).device)
    return tok, model

def extract_json(text):
    """Pull the first JSON array out of the model's reply."""
    # strip markdown fences if present
    text = re.sub(r"```(json)?", "", text).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end+1])
    except json.JSONDecodeError:
        return None

def run_one(tok, model, prompt_text):
    messages = [{"role": "user", "content": prompt_text}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=1200, temperature=0.2, do_sample=True)
    reply = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return extract_json(reply)

def main():
    paras = json.load(open(PARAS))
    template = open(PROMPT).read()
    tok, model = load_model()

    all_reqs = []
    for pnum, ptext in paras.items():
        prompt_text = template.replace("{PARA_NUM}", pnum).replace("{PARA_TEXT}", ptext)
        # run N times, keep the run with the most valid requirements
        candidates = []
        for run in range(N_RUNS):
            result = run_one(tok, model, prompt_text)
            if result is not None:
                candidates.append(result)
        if candidates:
            best = max(candidates, key=len)  # majority-by-consistency (most complete valid parse)
            for r in best:
                r["paragraph"] = int(pnum)
            all_reqs.extend(best)
            print(f"Paragraph {pnum}: extracted {len(best)} requirements "
                  f"({len(candidates)}/{N_RUNS} runs parsed)")
        else:
            print(f"Paragraph {pnum}: FAILED to parse any run")

    json.dump(all_reqs, open(OUT, "w"), indent=2)
    print(f"\nTotal requirements extracted: {len(all_reqs)}")
    print(f"Saved to: {OUT}")

if __name__ == "__main__":
    main()
