"""
批量图像描述 + 翻译
用法: python batch_caption.py
"""
import torch, os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import MarianMTModel, MarianTokenizer

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

# ── 1. 加载模型（只加载一次）────────────────────────────────
print("加载模型中...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip.eval()

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
translator = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
print("模型加载完成\n")

# ── 2. 定义工具函数 ────────────────────────────────────────
def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = blip.generate(**inputs, max_new_tokens=50)
    return processor.decode(output[0], skip_special_tokens=True)

def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        output = translator.generate(**inputs)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ── 3. 批量处理 ────────────────────────────────────────────
img_dir = "images"
results = []
exts = {".jpg", ".jpeg", ".png", ".bmp"}
img_files = [f for f in os.listdir(img_dir)
             if os.path.splitext(f)[1].lower() in exts]

print(f"共 {len(img_files)} 张图片，开始处理...\n")

for fname in img_files:
    img_path = os.path.join(img_dir, fname)
    image = Image.open(img_path).convert("RGB")

    caption_en = generate_caption(image)
    caption_zh = translate(caption_en)

    print(f"[{fname}]")
    print(f"  EN: {caption_en}")
    print(f"  ZH: {caption_zh}\n")
    results.append((fname, caption_en, caption_zh))

# ── 4. 保存结果 ────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(f"{RESULTS_DIR}/results.txt", "w", encoding="utf-8") as f:
    for fname, en, zh in results:
        f.write(f"图片: {fname}\n")
        f.write(f"EN: {en}\n")
        f.write(f"ZH: {zh}\n")
        f.write("-" * 40 + "\n")

print(f"处理完成，结果保存到 {RESULTS_DIR}/results.txt")
