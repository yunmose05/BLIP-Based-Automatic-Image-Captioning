"""
DeepSeek API 英文→中文精翻（对比 MarianMT）
用法: python caption_translate_llm.py
"""
import torch, os, json, requests
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    MarianMTModel,
    MarianTokenizer,
)

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

# ── DeepSeek API 配置 ────────────────────────────────────
DEEPSEEK_API_KEY = "sk-5f0f5f75505d41b2bde9800ee39bf7e5"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"  # deepseek-chat (V3) 或 deepseek-reasoner (R1)

# ── 1. 加载 BLIP ─────────────────────────────────────────
print("[1/3] 加载 BLIP...")
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip.to(device)
blip.eval()
print(f"BLIP 加载完成 (device: {device})")

# ── 2. DeepSeek API 翻译函数 ──────────────────────────────
def translate_with_deepseek(text: str) -> str:
    """调用 DeepSeek API 进行中英翻译"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个专业的中文翻译。请将以下英文图像描述翻译成自然、地道的中文。只输出翻译结果，不要加任何解释。"},
            {"role": "user", "content": f"英文: {text}\n中文:"},
        ],
        "temperature": 0.1,
        "max_tokens": 128,
    }
    try:
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()
        return result
    except Exception as e:
        print(f"  DeepSeek API 错误: {e}")
        return "（翻译失败）"

def generate_caption_en(image):
    """BLIP 生成英文描述"""
    inputs = blip_processor(image, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        output = blip.generate(**inputs, max_new_tokens=50)
    return blip_processor.decode(output[0], skip_special_tokens=True)

# ── 3. 加载 MarianMT 用于对比 ─────────────────────────────
print("[2/3] 加载 MarianMT...")
marian_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
marian = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
print("MarianMT 加载完成")

# ── 4. 对比翻译 ──────────────────────────────────────────
print("[3/3] 开始翻译对比...\n")

img_dir = "images"
exts = {".jpg", ".jpeg", ".png", ".bmp"}
img_files = sorted([f for f in os.listdir(img_dir)
                    if os.path.splitext(f)[1].lower() in exts])

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(f"{RESULTS_DIR}/translation_compare.txt", "w", encoding="utf-8") as f:
    for fname in img_files:
        print(f"[{fname}]")
        image = Image.open(os.path.join(img_dir, fname)).convert("RGB")
        caption_en = generate_caption_en(image)
        print(f"  EN:        {caption_en}")

        # MarianMT 翻译
        inputs_mt = marian_tokenizer(caption_en, return_tensors="pt", padding=True)
        with torch.no_grad():
            zh_mt = marian_tokenizer.decode(
                marian.generate(**inputs_mt)[0], skip_special_tokens=True
            )
        print(f"  MarianMT:  {zh_mt}")

        # DeepSeek API 翻译
        zh_ds = translate_with_deepseek(caption_en)
        print(f"  DeepSeek:  {zh_ds}\n")

        f.write(f"=== {fname} ===\n")
        f.write(f"EN:        {caption_en}\n")
        f.write(f"MarianMT:  {zh_mt}\n")
        f.write(f"DeepSeek:  {zh_ds}\n\n")

print("翻译对比结果已保存到 " + RESULTS_DIR + "/translation_compare.txt")
