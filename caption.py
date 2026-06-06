"""
单图图像描述 + 中英翻译 + 可视化
用法: python caption.py
"""
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import MarianMTModel, MarianTokenizer
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免弹窗阻塞
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

# ── 1. 加载模型 ────────────────────────────────────────────
print("加载 BLIP 模型...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()
print("BLIP 加载完成")

# ── 2. 加载翻译模型 ────────────────────────────────────────
print("加载翻译模型...")
trans_name = "Helsinki-NLP/opus-mt-en-zh"
trans_tokenizer = MarianTokenizer.from_pretrained(trans_name)
trans_model = MarianMTModel.from_pretrained(trans_name)
print("翻译模型加载完成")

# ── 3. 读取图片 ────────────────────────────────────────────
# 自动选择 images/ 目录下的第一张图片
img_dir = "images"
exts = {".jpg", ".jpeg", ".png", ".bmp"}
img_files = sorted([f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in exts])
if not img_files:
    raise FileNotFoundError(f"未在 images/ 目录下找到图片文件（支持: {', '.join(exts)}）")
img_path = os.path.join(img_dir, img_files[0])
print(f"使用图片: {img_path}")
image = Image.open(img_path).convert("RGB")

# ── 4. 无条件生成 ──────────────────────────────────────────
inputs = processor(image, return_tensors="pt")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)
caption_en = processor.decode(output[0], skip_special_tokens=True)
print(f"无条件生成: {caption_en}")

# ── 5. 有条件生成（prompt 引导）────────────────────────────
prompt = "a photo of"
inputs_cond = processor(image, prompt, return_tensors="pt")
with torch.no_grad():
    output_cond = model.generate(**inputs_cond, max_new_tokens=50)
caption_cond = processor.decode(output_cond[0], skip_special_tokens=True)
print(f"条件生成: {caption_cond}")

# ── 6. 翻译中文 ────────────────────────────────────────────
inputs_zh = trans_tokenizer(caption_en, return_tensors="pt", padding=True)
with torch.no_grad():
    translated = trans_model.generate(**inputs_zh)
caption_zh = trans_tokenizer.decode(translated[0], skip_special_tokens=True)
print(f"中文翻译: {caption_zh}")

# ── 7. 可视化保存 ──────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 5))
ax.imshow(image)
ax.axis("off")
ax.set_title(f"EN: {caption_en}\nZH: {caption_zh}", fontsize=11, pad=10)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/caption_result.jpg", dpi=150, bbox_inches="tight")
plt.show()
print(f"结果已保存到 {RESULTS_DIR}/caption_result.jpg")
