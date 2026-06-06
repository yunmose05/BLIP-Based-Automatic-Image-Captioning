"""
GPU 批量加速推理: DataLoader + batch inference
用法: python caption_batch_gpu.py
"""
import torch, os, time
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import MarianMTModel, MarianTokenizer

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

# ── 1. 自定义 Dataset ─────────────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, img_dir, exts=None):
        if exts is None:
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.files = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = Image.open(path).convert("RGB")
        return image, os.path.basename(path)

# ── 2. 批量推理函数 ────────────────────────────────────────
def collate_fn(batch):
    """将多张图片处理为 batch tensor"""
    images, names = zip(*batch)
    # BLIP processor 原生支持 batch
    inputs = processor(
        images=list(images),
        return_tensors="pt",
        padding=True,
    )
    return inputs, names

def batch_generate_captions(inputs):
    """一次推理整个 batch"""
    # 移到 GPU
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=3,  # batch 内每张用 beam=3 兼顾质量速度
        )
    # 解码（processor.batch_decode）
    captions = processor.batch_decode(outputs, skip_special_tokens=True)
    return captions

# ── 3. 主流程 ──────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

print("加载模型...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.to(device)
model.eval()

# 翻译模型
marian_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
marian = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

# ── 4. 对比: 逐张 vs 批量 ─────────────────────────────────
dataset = ImageDataset("images")
print(f"共 {len(dataset)} 张图片\n")

batch_size = min(4, len(dataset))  # 根据显存调整
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

# 逐张基准测试
print("=== 逐张推理（基准）===")
t0 = time.time()
for image, name in dataset:
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)
    cap = processor.decode(out[0], skip_special_tokens=True)
    print(f"  {name}: {cap}")
t_single = time.time() - t0
print(f"逐张耗时: {t_single:.1f}s\n")

# 批量推理
print(f"=== 批量推理 (batch_size={batch_size}) ===")
t0 = time.time()
all_results = []
for inputs, names in dataloader:
    captions = batch_generate_captions(inputs)
    for name, cap in zip(names, captions):
        # 翻译
        zh_in = marian_tok(cap, return_tensors="pt", padding=True)
        with torch.no_grad():
            zh_out = marian.generate(**zh_in)
        zh = marian_tok.decode(zh_out[0], skip_special_tokens=True)
        all_results.append((name, cap, zh))
        print(f"  {name}: {cap} → {zh}")
t_batch = time.time() - t0

# ── 5. 速度对比 ─────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(f"{RESULTS_DIR}/batch_gpu_results.txt", "w", encoding="utf-8") as f:
    f.write(f"设备: {device}\n")
    f.write(f"图片数量: {len(dataset)}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"逐张耗时: {t_single:.1f}s\n")
    f.write(f"批量耗时: {t_batch:.1f}s\n")
    f.write(f"加速比: {t_single / t_batch:.1f}x\n")
    f.write("-" * 40 + "\n")
    for name, en, zh in all_results:
        f.write(f"{name}\n  EN: {en}\n  ZH: {zh}\n\n")

print(f"逐张: {t_single:.1f}s | 批量: {t_batch:.1f}s | 加速: {t_single / t_batch:.1f}x")
print("结果已保存到 " + RESULTS_DIR + "/batch_gpu_results.txt")
