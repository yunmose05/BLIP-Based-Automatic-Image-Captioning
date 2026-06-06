"""
Beam Search 多候选生成 + 自动择优
用法: python caption_beam.py
"""
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import MarianMTModel, MarianTokenizer
import os

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
translator = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

def generate_caption_beam(image, num_beams=5, num_candidates=3):
    """
    Beam Search 生成多个候选描述
    返回: [(caption, score), ...]  按 score 降序
    """
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            min_length=3,
            num_beams=num_beams * 2,
            num_return_sequences=num_candidates,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
    # 计算每个序列的累积 log 概率作为置信度分数
    captions = []
    for i, seq in enumerate(outputs.sequences):
        cap = processor.decode(seq, skip_special_tokens=True).strip()
        # 过滤过短或空的候选
        if len(cap.split()) < 3:
            continue
        score = outputs.sequences_scores[i].item() if outputs.sequences_scores is not None else 0.0
        captions.append((cap, score))
    captions.sort(key=lambda x: x[1], reverse=True)
    # 如果过滤后不够，用贪婪解码补充
    if len(captions) < num_candidates:
        # greedy fallback
        outputs_greedy = model.generate(**inputs, max_new_tokens=50)
        cap_greedy = processor.decode(outputs_greedy[0], skip_special_tokens=True).strip()
        if len(cap_greedy.split()) >= 3:
            captions.append((cap_greedy, 0.0))
    return captions

def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        output = translator.generate(**inputs)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ── 单图示例 ────────────────────────────────────────────────
# 自动选择 images/ 目录下的第一张图片
img_dir = "images"
exts = {".jpg", ".jpeg", ".png", ".bmp"}
img_files = sorted([f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in exts])
if not img_files:
    raise FileNotFoundError(f"未在 images/ 目录下找到图片文件（支持: {', '.join(exts)}）")
img_path = os.path.join(img_dir, img_files[0])
print(f"使用图片: {img_path}")
image = Image.open(img_path).convert("RGB")

candidates = generate_caption_beam(image, num_beams=5, num_candidates=3)
best_en = candidates[0][0]

print(f"=== Beam Search 候选结果 ===")
for i, (cap, score) in enumerate(candidates):
    print(f"  [{i+1}] score={score:.3f} → {cap}")

print(f"\n最佳描述: {best_en}")
print(f"中文翻译: {translate(best_en)}")

# ── 批量版本 ────────────────────────────────────────────────
img_dir = "images"
exts = {".jpg", ".jpeg", ".png", ".bmp"}
img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in exts]

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(f"{RESULTS_DIR}/beam_results.txt", "w", encoding="utf-8") as f:
    for fname in img_files:
        image = Image.open(os.path.join(img_dir, fname)).convert("RGB")
        candidates = generate_caption_beam(image)
        best_en = candidates[0][0]
        best_zh = translate(best_en)

        f.write(f"=== {fname} ===\n")
        for i, (cap, score) in enumerate(candidates):
            f.write(f"  [{i+1}] score={score:.3f}: {cap}\n")
        f.write(f"  → 最佳中文: {best_zh}\n\n")

print("Beam Search 批量结果已保存到 " + RESULTS_DIR + "/beam_results.txt")
