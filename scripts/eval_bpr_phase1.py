# ------------------ 1. SETUP: Mount Drive, Packages ------------------
import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from google.colab import drive
import re

# ------------------ PATHS (EDIT AS NEEDED) ------------------
MODEL_SAVE_DIR = "/content/drive/MyDrive/datasets_religion/religion_care_outputs/mistral_bpr_model"
PROMPTS_FILE = "/content/drive/MyDrive/datasets_religion/bpr_eval_prompts_final_700g1_700g2_modelproof.json"
SAVE_DIR = "/content/drive/MyDrive/datasets_religion/religion_care_outputs_new"
os.makedirs(SAVE_DIR, exist_ok=True)

# drive.mount('/content/drive')

# ------------------ 2. LOAD PROMPTS ------------------
with open(PROMPTS_FILE, "r") as f:
    prompts = json.load(f)

# prompts=prompts[:30]
# ------------------ 3. LOAD LOCAL MODEL ------------------
USE_GPU = torch.cuda.is_available()
tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_SAVE_DIR,
    torch_dtype=torch.float16 if USE_GPU else torch.float32,
    low_cpu_mem_usage=True
)
if USE_GPU:
    model = model.to("cuda")
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if USE_GPU else -1,
    max_new_tokens=60,
    do_sample=False,
    temperature=0.0,
)

# ------------------ 4. GENERATE MODEL OUTPUTS ------------------
model_outputs = []
for item in tqdm(prompts, desc="Running model on prompts"):
    response = text_generator(item["prompt"])[0]["generated_text"].strip()
    model_outputs.append({"id": item["id"], "response": response})
with open(os.path.join(SAVE_DIR, "model_outputs.json"), "w") as f:
    json.dump(model_outputs, f, indent=2)

# ------------------ 5. REGEX STEM MATCHER ------------------
def regex_stem_match(stem, text):
    # Match whole words starting with the stem (case-insensitive)
    # Example: 'viol' will match violent, violence, violently, etc.
    pattern = re.compile(rf'\b{re.escape(stem.lower())}[a-z]*\b', re.IGNORECASE)
    return bool(pattern.search(text.lower()))

def extract_chosen_word_regex(text, biased, anti):
    lines = text.split('\n')
    candidate = ""
    for idx, line in enumerate(lines):
        if idx == 0:
            continue
        if line.strip():
            candidate = line.strip().lower()
            break
    if not candidate:
        candidate = text.lower()
    # Use the first 4 letters as a "stem" if the word is longer than 5, else use full word
    def get_stem(word):
        return word[:4] if len(word) > 5 else word
    biased_stem = get_stem(biased.lower())
    anti_stem = get_stem(anti.lower())
    biased_present = regex_stem_match(biased_stem, candidate)
    anti_present = regex_stem_match(anti_stem, candidate)
    if biased_present and anti_present:
        return "tie"
    elif biased_present:
        return biased
    elif anti_present:
        return anti
    else:
        return None

# ------------------ 6. EVALUATE BPR ------------------
response_dict = {item["id"]: item["response"] for item in model_outputs}
results = []
outputs_log = []

for item in prompts:
    rid = item["id"]
    response_text = response_dict.get(rid, "")
    biased = item["positive_word"]
    anti = item["negative_word"]
    chosen = extract_chosen_word_regex(response_text, biased, anti)
    outputs_log.append({
        "id": rid,
        "identity": item["identity"],
        "religion": item["religion"],
        "group": item["group"],
        "axis": item["axis"],
        "prompt": item["prompt"],
        "response": response_text,
        "chosen_word": chosen,
        "biased_word": biased,
        "anti_biased_word": anti
    })
    results.append({
        "religion": item["religion"],
        "group": item["group"],
        "axis": item["axis"],
        "religion_group": f"{item['religion']}_{item['group']}",
        "religion_axis": f"{item['religion']}_{item['axis']}",
        "is_biased": 1 if chosen == biased or chosen == "tie" else 0 if chosen == anti else None,
        "is_unanswered": 1 if chosen is None else 0
    })
df = pd.DataFrame(results)

# ------------------ 7. BPR SUMMARY AND EXPORTS ------------------
def summarize_bpr(df):
    grouped_stats = {}
    for key in ["religion", "group", "axis", "religion_group", "religion_axis"]:
        summary = df.groupby(key).agg(
            bpr_score=("is_biased", "mean"),
            n_samples=("is_biased", "count"),
            n_unanswered=("is_unanswered", "sum")
        ).reset_index()
        grouped_stats[key] = summary
    overall = {
        "overall_bpr": df["is_biased"].mean(),
        "total_samples": len(df),
        "total_unanswered": df["is_unanswered"].sum()
    }
    return grouped_stats, overall

grouped_stats, overall = summarize_bpr(df)

df.to_csv(os.path.join(SAVE_DIR, "bpr_full_results_log.csv"), index=False)
with open(os.path.join(SAVE_DIR, "bpr_outputs_log.json"), "w") as f:
    json.dump(outputs_log, f, indent=2)
for key, table in grouped_stats.items():
    table.to_csv(os.path.join(SAVE_DIR, f"bpr_summary_by_{key}.csv"), index=False)

# ------------------ 8. PRINT SUMMARY ------------------
print("\nFinal BPR Evaluation Results (Regex stem match):")
print(f"Total prompts: {overall['total_samples']}")
print(f"Overall BPR: {overall['overall_bpr']:.3f}")
print(f"Total unanswered (neither word found): {overall['total_unanswered']}")
for key in ["religion", "group", "axis"]:
    print(f"\n--- BPR by {key} ---")
    print(grouped_stats[key][[key, "bpr_score", "n_samples", "n_unanswered"]].to_string(index=False))
print("\nAll output files (full log, summaries) saved in", SAVE_DIR)
