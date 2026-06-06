"""
图文匹配检索: 从候选描述中选最优匹配
方案A: 使用 BLIP ITM 模型直出匹配分数
方案B: 用 CLIP/BLIP 双塔做 embedding 余弦相似度（兼容性更好）
用法: python caption_retrieval.py
"""
import torch, os
import torch.nn.functional as F
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForImageTextRetrieval,  # BLIP 的 ITM 模型
)

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

# ── 1. 加载模型 ────────────────────────────────────────────
print("加载 BLIP Captioning...")
cap_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
cap_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
cap_model.eval()

print("加载 BLIP ITM (图文匹配)...")
itm_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
itm_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
itm_model.eval()

# ── 2. 生成候选描述 ────────────────────────────────────────
def generate_candidates(image, n=5):
    """用 Beam Search 生成 n 个候选描述"""
    inputs = cap_processor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = cap_model.generate(
            **inputs, max_new_tokens=50,
            min_length=3,
            num_beams=n * 2,
            num_return_sequences=n,
            early_stopping=True,
            output_scores=True, return_dict_in_generate=True,
        )
    captions = []
    for seq in outputs.sequences:
        captions.append(cap_processor.decode(seq, skip_special_tokens=True))
    return captions

# ── 3. ITM 打分 ────────────────────────────────────────────
def score_image_text_match(image, captions):
    """
    对每个候选描述计算图文匹配分数
    返回: [(caption, itm_score), ...]  sorted by score desc
    """
    scores = []
    for cap in captions:
        inputs = itm_processor(image, cap, return_tensors="pt")
        with torch.no_grad():
            outputs = itm_model(**inputs)
            # itm_score: 匹配(1) vs 不匹配(0) 的 logit
            itm_score = torch.softmax(outputs.itm_score, dim=1)[0, 1].item()
        scores.append((cap, itm_score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

# ── 4. 批量处理 ────────────────────────────────────────────
img_dir = "images"
exts = {".jpg", ".jpeg", ".png", ".bmp"}
img_files = sorted([f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in exts])

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(f"{RESULTS_DIR}/itm_results.txt", "w", encoding="utf-8") as f:
    for fname in img_files:
        image = Image.open(os.path.join(img_dir, fname)).convert("RGB")

        # 生成候选
        candidates = generate_candidates(image, n=5)
        # ITM 排序
        ranked = score_image_text_match(image, candidates)

        f.write(f"=== {fname} ===\n")
        for i, (cap, score) in enumerate(ranked):
            marker = " ← 最优" if i == 0 else ""
            f.write(f"  [{i+1}] ITM={score:.4f}: {cap}{marker}\n")
        f.write("\n")

        print(f"{fname}: best → {ranked[0][0]} (ITM={ranked[0][1]:.4f})")

print("\nITM 匹配结果已保存到 " + RESULTS_DIR + "/itm_results.txt")
