"""Layer 2: semantic verification of hierarchy and severity claims."""
import json, csv, re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
HIER_P = (ROOT/"prompts"/"hierarchy_prompt.txt").read_text()
SEV_P = (ROOT/"prompts"/"severity_prompt.txt").read_text()
MODEL = "Qwen/Qwen2.5-7B-Instruct"

_tok = _model = None
def _load():
    global _tok, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(MODEL)
        _model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto", device_map="auto")
    return _tok, _model

def _ask(prompt):
    tok, model = _load()
    msgs=[{"role":"user","content":prompt}]
    text=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp=tok(text, return_tensors="pt").to(model.device)
    out=model.generate(**inp, max_new_tokens=200, temperature=0.0, do_sample=False)
    reply=tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    m=re.search(r"\{.*\}", reply, re.DOTALL)
    if not m: return {"flag":"AMBIGUOUS","severity":0,"justification":"unparseable"}
    try: return json.loads(m.group(0))
    except: return {"flag":"AMBIGUOUS","severity":0,"justification":"parse error"}

def check_hierarchy(desc, level):
    return _ask(HIER_P.replace("{DESCRIPTION}",str(desc)).replace("{LEVEL}",str(level)))

def check_severity(desc, sev):
    return _ask(SEV_P.replace("{DESCRIPTION}",str(desc)).replace("{SEVERITY}",str(sev)))

def verify_package(pkg_dir):
    """Return semantic flags for all hierarchy + severity claims in a package."""
    d=Path(pkg_dir); flags=[]
    mp=d/"mitigation_tracker.csv"
    if mp.exists():
        for r in csv.DictReader(open(mp)):
            res=check_hierarchy(r["description"], r["hierarchy_level"])
            res.update({"check":"hierarchy","record_id":r["mitigation_id"]})
            flags.append(res)
    rp=d/"clinical_risk_register.csv"
    if rp.exists():
        for r in csv.DictReader(open(rp)):
            res=check_severity(r["hazard"], r["severity"])
            res.update({"check":"severity","record_id":r["risk_id"]})
            flags.append(res)
    return flags

if __name__ == "__main__":
    import sys
    pkg = sys.argv[1] if len(sys.argv)>1 else "data/synthetic/base/PKG_C0000"
    print(f"Verifying: {pkg}")
    for f in verify_package(pkg):
        print(f"  [{f['flag']:9}] {f['check']:9} {f['record_id']}: {f.get('justification','')[:60]}")
