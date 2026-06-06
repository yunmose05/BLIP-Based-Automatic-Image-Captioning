"""
YOLO 目标检测 + BLIP 区域描述
用法: python caption_yolo_region.py
"""
import torch, os
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

# ── 1. 加载模型 ────────────────────────────────────────────
print("加载 YOLO...")
yolo = YOLO("yolo11n.pt")  # 自动下载，约 6MB；也可用 yolov8n.pt

print("加载 BLIP...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip.eval()

# ── 2. 从原图裁剪区域 ──────────────────────────────────────
def crop_region(image, box):
    """根据 [x1, y1, x2, y2] 裁剪区域"""
    x1, y1, x2, y2 = map(int, box)
    return image.crop((max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)))

def describe_region(region_image):
    """对裁剪区域生成描述"""
    inputs = blip_processor(region_image, return_tensors="pt")
    with torch.no_grad():
        output = blip.generate(**inputs, max_new_tokens=30)
    return blip_processor.decode(output[0], skip_special_tokens=True)

# ── 3. 处理单张图片 ────────────────────────────────────────
def analyze_image(img_path, conf_threshold=0.3):
    """
    对图片做完整分析: 整体描述 + 各区域描述
    返回: {整体描述, [(标签, 置信度, bbox, 区域描述), ...]}
    """
    image = Image.open(img_path).convert("RGB")

    # 3a. 整体描述
    global_caption = describe_region(image)

    # 3b. YOLO 检测
    results = yolo(image, conf=conf_threshold)
    detections = []
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        names = results[0].names  # {class_id: class_name}

        for box, cls_id, conf in zip(boxes, classes, confs):
            label = names[int(cls_id)]
            region = crop_region(image, box)
            if region.width > 10 and region.height > 10:  # 过滤过小区域
                region_desc = describe_region(region)
                detections.append({
                    "label": label, "confidence": float(conf),
                    "bbox": [int(v) for v in box],
                    "description": region_desc,
                })

    return {"global_caption": global_caption, "detections": detections}

# ── 4. 可视化 ──────────────────────────────────────────────
def draw_results(image, result, save_path):
    """绘制检测框 + 标注"""
    draw = ImageDraw.Draw(image)
    # 尝试加载中文字体，失败则用默认
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 16)
        font_small = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 12)
    except:
        font = font_small = ImageFont.load_default()

    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]
    for i, det in enumerate(result["detections"]):
        x1, y1, x2, y2 = det["bbox"]
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - 20), f"{det['label']}", fill=color, font=font)
        draw.text((x1, y2 + 4), det["description"][:40], fill=color, font=font_small)

    image.save(save_path)
    print(f"可视化结果: {save_path}")

# ── 5. 批量运行 ────────────────────────────────────────────
img_dir = "images"
exts = {".jpg", ".jpeg", ".png", ".bmp"}
img_files = sorted([f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in exts])

os.makedirs(f"{RESULTS_DIR}/yolo_regions", exist_ok=True)

with open(f"{RESULTS_DIR}/yolo_region_results.txt", "w", encoding="utf-8") as f:
    for fname in img_files:
        img_path = os.path.join(img_dir, fname)
        print(f"\n处理: {fname}")

        result = analyze_image(img_path)
        f.write(f"=== {fname} ===\n")
        f.write(f"整体描述: {result['global_caption']}\n")
        f.write(f"检测到 {len(result['detections'])} 个物体:\n")
        for i, det in enumerate(result["detections"]):
            f.write(f"  [{i+1}] {det['label']} (置信度: {det['confidence']:.2f})\n")
            f.write(f"      区域描述: {det['description']}\n")
            f.write(f"      位置: {det['bbox']}\n")
        f.write("\n")

        # 可视化
        image = Image.open(img_path).convert("RGB")
        draw_results(image, result, f"{RESULTS_DIR}/yolo_regions/{fname}")

print("\n所有结果已保存到 " + RESULTS_DIR + "/yolo_region_results.txt")
print("可视化图片保存在 " + RESULTS_DIR + "/yolo_regions/")
