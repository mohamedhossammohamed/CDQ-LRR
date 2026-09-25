"""Smartness benchmark: baseline BF16 vs CDQ-LRR tiers on Qwen2.5-0.5B-Instruct.

48 hand-authored multiple-choice questions, MMLU-style scoring: the model
picks the answer letter with the highest next-token logprob. Deterministic
(single forward per question, no sampling), offline, CPU-friendly.

Models compared:
  - bf16      : models/qwen-local untouched
  - cdq26     : + CDQ-LRR 26-state / STC 1.56% (production v1)
  - cdq16zero : + CDQ-LRR 16-state zero-clamped / STC 1.56% (draft tier)

Writes logs/cdq_smartness.json
"""
import json
import time
import torch
import torch.nn as nn

MODEL_DIR = "/Users/mohammedhossam/Desktop/MZSAE/models/qwen-local"
OUT = "/Users/mohammedhossam/Desktop/MZSAE/logs/cdq_smartness.json"

# (category, question, [A,B,C,D], gold_index)
QS = [
    ("arithmetic", "What is 47 + 58?", ["103", "105", "115", "95"], 1),
    ("arithmetic", "What is 12 x 13?", ["146", "156", "166", "169"], 1),
    ("arithmetic", "What is three quarters of 96?", ["68", "70", "72", "74"], 2),
    ("arithmetic", "What is 15% of 200?", ["25", "30", "35", "40"], 1),
    ("arithmetic", "What is 7 squared plus 5 squared?", ["72", "74", "76", "84"], 1),
    ("arithmetic", "What is 1001 - 487?", ["504", "514", "524", "614"], 1),
    ("arithmetic", "What is the next prime number after 47?", ["49", "51", "53", "57"], 2),
    ("arithmetic", "What is the greatest common divisor of 48 and 180?", ["6", "12", "18", "24"], 1),
    ("algebra", "If x + 7 = 15, what is x?", ["6", "7", "8", "9"], 2),
    ("algebra", "If 2x - 3 = 11, what is x?", ["5", "6", "7", "8"], 2),
    ("algebra", "If y = 3x + 2 and x = 4, what is y?", ["12", "13", "14", "15"], 2),
    ("algebra", "If x squared equals 81 and x is positive, what is x?", ["7", "8", "9", "10"], 2),
    ("algebra", "Expanding (x+3)(x+2), what is the coefficient of x?", ["5", "6", "7", "8"], 0),
    ("algebra", "If 5x = 3x + 12, what is x?", ["4", "5", "6", "7"], 2),
    ("algebra", "What is the average of 4, 8, 15, 16 and 22?", ["12", "13", "14", "15"], 1),
    ("algebra", "If 3 pencils cost 90 cents, how much do 7 pencils cost?", ["180 cents", "200 cents", "210 cents", "240 cents"], 2),
    ("science", "What is the chemical symbol for gold?", ["Ag", "Au", "Gd", "Go"], 1),
    ("science", "Which planet is closest to the Sun?", ["Venus", "Mercury", "Mars", "Earth"], 1),
    ("science", "Which gas do plants absorb for photosynthesis?", ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"], 2),
    ("science", "How many pairs of chromosomes do humans normally have?", ["22", "23", "24", "46"], 1),
    ("science", "What is the boiling point of water at sea level in Celsius?", ["90", "95", "100", "110"], 2),
    ("science", "What is the SI unit of force?", ["Joule", "Watt", "Newton", "Volt"], 2),
    ("science", "Which organelle is known as the powerhouse of the cell?", ["Nucleus", "Ribosome", "Mitochondria", "Chloroplast"], 2),
    ("science", "Approximately how fast does light travel in km/s?", ["150,000", "300,000", "450,000", "600,000"], 1),
    ("science", "What is the pH of a neutral solution?", ["0", "1", "7", "14"], 2),
    ("science", "In which organ does human digestion begin?", ["Stomach", "Mouth", "Small intestine", "Esophagus"], 1),
    ("logic", "All bloops are razzies. All razzies are lazzies. So all bloops are lazzies. This:", ["Follows", "Contradicts", "Is unrelated", "Cannot be judged"], 0),
    ("logic", "If it rains, the ground gets wet. The ground is dry. So:", ["It rained", "It did not rain", "It will rain", "Nothing follows"], 1),
    ("logic", "What comes next: 2, 4, 8, 16, ...?", ["24", "30", "32", "20"], 2),
    ("logic", "What comes next: 1, 1, 2, 3, 5, 8, ...?", ["11", "12", "13", "14"], 2),
    ("logic", "A is taller than B, and B is taller than C. Who is shortest?", ["A", "B", "C", "Cannot tell"], 2),
    ("logic", "5 machines take 5 minutes to make 5 widgets. How long do 100 machines take to make 100 widgets?", ["5 minutes", "20 minutes", "100 minutes", "500 minutes"], 0),
    ("logic", "A book costs $1 plus half its price. What is the full price?", ["$1", "$1.50", "$2", "$3"], 2),
    ("logic", "All Zogs eat rocks. Rex is a Zog. So Rex eats rocks. This:", ["Follows", "Contradicts", "Is unrelated", "Cannot be judged"], 0),
    ("commonsense", "How many days are in a leap year?", ["365", "366", "367", "364"], 1),
    ("commonsense", "Ice floats on water because ice is:", ["Heavier", "Less dense", "Colder", "Saltier"], 1),
    ("commonsense", "Which of these animals is a mammal?", ["Shark", "Dolphin", "Octopus", "Penguin"], 1),
    ("commonsense", "How many legs does a spider have?", ["6", "8", "10", "12"], 1),
    ("commonsense", "Which instrument measures temperature?", ["Barometer", "Thermometer", "Speedometer", "Odometer"], 1),
    ("commonsense", "Which of these is a fruit?", ["Carrot", "Potato", "Apple", "Broccoli"], 2),
    ("commonsense", "The Sun rises in the:", ["North", "East", "South", "West"], 1),
    ("commonsense", "How many minutes are in 2.5 hours?", ["120", "130", "150", "250"], 2),
    ("language", "Which word is a synonym of rapid?", ["Slow", "Fast", "Late", "Quiet"], 1),
    ("language", "Which word is an antonym of ancient?", ["Old", "Antique", "Modern", "Aged"], 2),
    ("language", "Which word is spelled correctly?", ["Occasion", "Ocassion", "Occassion", "Ocasion"], 0),
    ("language", "What is the past tense of go?", ["Goed", "Went", "Gone", "Goes"], 1),
    ("language", "Which sentence is grammatically correct?", ["She don't like apples", "She doesn't likes apples", "She doesn't like apples", "She not like apples"], 2),
    ("language", "What is the plural of child?", ["Childs", "Childes", "Children", "Childrens"], 2),
]

LETTERS = ["A", "B", "C", "D"]


def fmt(q, choices):
    opts = "\n".join(f"{L}) {c}" for L, c in zip(LETTERS, choices))
    return (f"Answer the following multiple choice question. Reply with only the letter.\n\n"
            f"Question: {q}\n{opts}\nAnswer:")


def letter_ids(tok):
    out = {}
    for L in LETTERS:
        ids = set()
        for s in (L, " " + L):
            e = tok.encode(s, add_special_tokens=False)
            if e:
                ids.add(e[-1])
        out[L] = sorted(ids)
    return out


@torch.no_grad()
def score_model(model, tok, lids):
    preds = []
    for cat, q, choices, gold in QS:
        enc = tok(fmt(q, choices), return_tensors="pt")
        logits = model(**enc).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        best, best_s = "A", -1e9
        for L in LETTERS:
            s = max(float(lp[i]) for i in lids[L])
            if s > best_s:
                best, best_s = L, s
        preds.append(LETTERS.index(best))
    return preds


def patch_cdq(model, lattice, stc_frac):
    import sys
    sys.path.insert(0, "/Users/mohammedhossam/Desktop/MZSAE/scripts")
    from cdq_lrr_qwen05 import quantize_cdq_lrr, CDQLinear, TARGET_SUBSTRINGS
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and any(s in name for s in TARGET_SUBSTRINGS):
            parent_path, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            pack = quantize_cdq_lrr(mod.weight.data, lattice=lattice, stc_frac=stc_frac)
            setattr(parent, attr, CDQLinear(
                pack, mod.in_features, mod.out_features,
                bias=mod.bias.data if mod.bias is not None else None))
    return model


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sys
    sys.path.insert(0, "/Users/mohammedhossam/Desktop/MZSAE/scripts")
    from cdq_lrr_qwen05 import CAMKII_LATTICE, CAMKII_LATTICE_16

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    lids = letter_ids(tok)
    print("letter token ids:", lids)

    def load():
        return AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cpu",
            trust_remote_code=True, low_cpu_mem_usage=True).eval()

    results = {}
    plan = [("bf16", None, None),
            ("cdq26_156", CAMKII_LATTICE, 0.0156),
            ("cdq16zero_156", CAMKII_LATTICE_16, 0.0156)]
    for tag, lat, frac in plan:
        t0 = time.time()
        m = load()
        if lat is not None:
            patch_cdq(m, lat, frac)
            n = sum(1 for _ in m.modules() if isinstance(_, nn.Linear) and "CDQ" in type(_).__name__)
        preds = score_model(m, tok, lids)
        golds = [g for _, _, _, g in QS]
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        cats = {}
        for (cat, _, _, g), p in zip(QS, preds):
            c = cats.setdefault(cat, [0, 0])
            c[1] += 1
            c[0] += (p == g)
        results[tag] = {"acc": acc, "preds": preds,
                        "cats": {k: f"{v[0]}/{v[1]}" for k, v in cats.items()},
                        "time_s": round(time.time() - t0, 1)}
        print(f"[{tag}] acc={acc:.3f} ({sum(p==g for p,g in zip(preds,golds))}/{len(golds)}) "
              f"cats={results[tag]['cats']} {results[tag]['time_s']}s", flush=True)
        del m

    # agreement: how many answers flip vs baseline
    base = results["bf16"]["preds"]
    for tag in ("cdq26_156", "cdq16zero_156"):
        p = results[tag]["preds"]
        flips = sum(a != b for a, b in zip(base, p))
        results[tag]["flips_vs_bf16"] = flips
        print(f"[{tag}] flips vs bf16: {flips}/{len(base)}", flush=True)

    with open(OUT, "w") as f:
        json.dump({"golds": [g for _, _, _, g in QS], **results}, f)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
