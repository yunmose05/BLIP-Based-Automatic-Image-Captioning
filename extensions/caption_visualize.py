"""
批量图像描述 + DeepSeek 丰富 + 精翻 + 可视化（每张图一个结果文件）
用法: python caption_visualize.py
"""
import torch, os, json, requests
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

DEEPSEEK_API_KEY = "sk-5f0f5f75505d41b2bde9800ee39bf7e5"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# ── 1. 加载 BLIP ────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[1/2] 加载 BLIP (device: {device})...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip.to(device)
blip.eval()


# ── 2. BLIP 生成英文描述 ────────────────────────────────────
def generate_blip_caption(image):
    """多策略生成最详细的 BLIP 原始描述"""
    prompts = ["a detailed photograph of", "a photo of", ""]
    candidates = []

    for prompt in prompts:
        try:
            if prompt:
                inputs = processor(image, prompt, return_tensors="pt")
            else:
                inputs = processor(image, return_tensors="pt")
            if device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = blip.generate(
                    **inputs,
                    max_new_tokens=80, min_length=6,
                    num_beams=7, repetition_penalty=1.3,
                    length_penalty=1.2, no_repeat_ngram_size=2,
                    early_stopping=True,
                )
            cap = processor.decode(outputs[0], skip_special_tokens=True).strip()
            words = cap.split()
            if len(words) >= 4 and cap[0].isalpha():
                candidates.append(cap)
        except Exception:
            continue

    if not candidates:
        inputs = processor(image, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            out = blip.generate(**inputs, max_new_tokens=50)
        return processor.decode(out[0], skip_special_tokens=True).strip()

    candidates.sort(key=lambda c: len(c.split()), reverse=True)
    best = candidates[0]
    for pf in ["a detailed photograph of ", "a clear image of ", "a photo of "]:
        if best.lower().startswith(pf):
            best = best[len(pf):]
            break
    return best.strip().lstrip("., ")


# ── 3. DeepSeek API：丰富描述 + 翻译中文 ────────────────────
def enrich_and_translate(blip_caption: str) -> tuple:
    """
    调用 DeepSeek 一次性完成：
    1. 将简洁的 BLIP 描述扩展为更丰富自然的英文
    2. 翻译为地道中文
    返回 (enriched_en, zh)
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": (
                "你是一个视觉描述专家。任务：\n"
                "1. 将简短的图像描述扩展为更丰富自然的英文描述，补充合理的细节（颜色、空间关系、氛围），"
                "但不要添加原文中没有的核心物体。用 1-2 句流畅英文表达。\n"
                "2. 然后将扩展后的英文翻译为自然地道的中文。\n"
                "严格按照以下格式输出（不要加任何额外内容）：\n"
                "ENRICHED: <扩展后的英文>\n"
                "CHINESE: <中文翻译>"
            )},
            {"role": "user", "content": f"原始描述: {blip_caption}"},
        ],
        "temperature": 0.3,
        "max_tokens": 300,
    }
    try:
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()

        en, zh = "", ""
        for line in result.split("\n"):
            line = line.strip()
            if line.upper().startswith("ENRICHED:"):
                en = line.split(":", 1)[1].strip()
            elif line.upper().startswith("CHINESE:"):
                zh = line.split(":", 1)[1].strip()

        return en if en else blip_caption, zh if zh else blip_caption
    except Exception as e:
        print(f"  DeepSeek API 错误: {e}")
        return blip_caption, blip_caption


# ── 4. 可视化 ──────────────────────────────────────────────
def create_visualization(image, blip_en, enriched_en, zh, save_path):
    """在图片顶部居中叠加文字，统一白色，自适应换行"""
    img = image.copy().convert("RGBA")
    W, H = img.size
    max_bar_h = int(H * 0.4)  # 顶部文字区域最多占 40% 高度

    # 自适应字体大小
    base_size = max(24, min(44, W // 28))

    # 尝试加载字体，逐级缩小直到文字能装下
    for size_offset in range(0, 20, 2):
        size = base_size - size_offset
        if size < 16:
            size = 16

        try:
            font_en = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
            font_zh = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", size + 2)
        except:
            font_en = font_zh = ImageFont.load_default()

        draw = ImageDraw.Draw(img)
        margin = 40
        max_w = W - margin * 2

        # 换行函数：英文按单词，中文逐字
        def wrap_text(text, font, max_w):
            lines = []
            # 中英文混合换行
            cur = ""
            i = 0
            while i < len(text):
                ch = text[i]
                cur += ch
                bbox = draw.textbbox((0, 0), cur, font=font)
                if bbox[2] - bbox[0] > max_w:
                    # 回退：找到合适的断点
                    break_at = len(cur) - 1
                    # 英文单词边界
                    if ch.isascii() and cur.rstrip()[-1].isalpha():
                        # 回退到上一个空格
                        last_space = cur.rfind(" ")
                        if last_space > 0:
                            break_at = last_space
                    if break_at > 0:
                        lines.append(cur[:break_at].strip())
                        cur = cur[break_at:].lstrip()
                    else:
                        lines.append(cur[:-1])
                        cur = ch
                i += 1
            if cur.strip():
                lines.append(cur.strip())
            return lines if lines else [text]

        # 组装所有文本行
        all_lines = []
        all_lines.append(("BLIP: " + blip_en, font_en))
        if enriched_en and enriched_en != blip_en:
            all_lines.append(("ENRICHED: " + enriched_en, font_en))
        all_lines.append(("中文翻译: " + zh, font_zh))

        # 计算总行数
        wrapped_lines = []
        for text, font in all_lines:
            for line in wrap_text(text, font, max_w):
                wrapped_lines.append((line, font))

        lh = size + 8
        total_h = len(wrapped_lines) * lh + 20

        if total_h <= max_bar_h:
            break  # 字体合适

    # 超出上限时强制缩小
    if total_h > max_bar_h:
        lh = max_bar_h // max(len(wrapped_lines), 1)
        lh = max(lh, 18)

    # 绘制背景条
    from PIL import Image as PILImage
    bar = PILImage.new("RGBA", (W, total_h), (0, 0, 0, 190))
    img.paste(bar, (0, 0), bar)

    draw = ImageDraw.Draw(img)
    y = 8

    for line, font in wrapped_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        x = max(5, (W - tw) // 2)  # 确保不超出左边界
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        y += lh

    img.convert("RGB").save(save_path, quality=95)


# ── 5. 批量处理 ────────────────────────────────────────────
print("[2/2] 开始处理...\n")

img_dir = "images"
exts = {".jpg", ".jpeg", ".png", ".bmp"}
img_files = sorted([f for f in os.listdir(img_dir)
                    if os.path.splitext(f)[1].lower() in exts])

print(f"共 {len(img_files)} 张图片\n")
os.makedirs(RESULTS_DIR, exist_ok=True)

results_txt = os.path.join(RESULTS_DIR, "captions.txt")
with open(results_txt, "w", encoding="utf-8") as log:
    for i, fname in enumerate(img_files, 1):
        img_path = os.path.join(img_dir, fname)
        name_no_ext = os.path.splitext(fname)[0]

        print(f"[{i}/{len(img_files)}] {fname}")
        image = Image.open(img_path).convert("RGB")

        blip_en = generate_blip_caption(image)
        print(f"  BLIP: {blip_en}")

        enriched_en, zh = enrich_and_translate(blip_en)
        if enriched_en != blip_en:
            print(f"  丰富: {enriched_en}")
        print(f"  中文: {zh}")

        out_name = f"{name_no_ext}_caption.jpg"
        create_visualization(image, blip_en, enriched_en, zh,
                           os.path.join(RESULTS_DIR, out_name))
        print(f"  -> {out_name}\n")

        log.write(f"=== {fname} ===\n")
        log.write(f"BLIP: {blip_en}\n")
        if enriched_en and enriched_en != blip_en:
            log.write(f"ENRICHED: {enriched_en}\n")
        log.write(f"中文翻译: {zh}\n\n")

print(f"完成！{len(img_files)} 张 -> {RESULTS_DIR}/")
