"""
完成模型缓存：直接下载小文件 + 创建 snapshot
"""
import os, json, shutil
import requests

MIRROR = "https://hf-mirror.com"
CACHE = os.path.expanduser("~/.cache/huggingface/hub")

# 每个模型需要的小文件列表
MODEL_FILES = {
    "Salesforce/blip-itm-base-coco": [
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ],
    "Helsinki-NLP/opus-mt-en-zh": [
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "source.spm",
        "target.spm",
        "sentencepiece.bpe.model",
    ],
}

def get_snapshot_id(model_id):
    """从 mirror 获取当前 snapshot commit hash"""
    url = f"{MIRROR}/api/models/{model_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json().get("sha", r.json().get("_id", "main"))
    except:
        pass
    return "main"

def download_file(model_id, filename, snapshot_id):
    """下载单个文件并放入 HF 缓存结构"""
    model_dir = os.path.join(CACHE, f"models--{model_id.replace('/', '--')}")
    blobs_dir = os.path.join(model_dir, "blobs")
    snap_dir = os.path.join(model_dir, "snapshots", snapshot_id)
    os.makedirs(blobs_dir, exist_ok=True)
    os.makedirs(snap_dir, exist_ok=True)

    # 目标路径
    snap_path = os.path.join(snap_dir, filename)
    if os.path.exists(snap_path):
        return True  # 已存在

    url = f"{MIRROR}/{model_id}/resolve/main/{filename}"
    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        # 写入 blobs
        blob_path = os.path.join(blobs_dir, filename)
        with open(blob_path, "wb") as f:
            f.write(r.content)
        # 拷贝到 snapshot
        shutil.copy2(blob_path, snap_path)
        print(f"  OK: {filename} ({len(r.content):,} bytes)")
        return True
    else:
        print(f"  FAIL: {filename} (HTTP {r.status_code})")
        return False

# 主流程
for model_id, files in MODEL_FILES.items():
    print(f"\n[{model_id}]")
    snapshot_id = get_snapshot_id(model_id)
    print(f"  Snapshot: {snapshot_id}")

    ok = 0
    for fname in files:
        if download_file(model_id, fname, snapshot_id):
            ok += 1
    print(f"  Result: {ok}/{len(files)} files")

    # 拷贝权重文件到 snapshot
    model_dir = os.path.join(CACHE, f"models--{model_id.replace('/', '--')}")
    blobs_dir = os.path.join(model_dir, "blobs")
    snap_dir = os.path.join(model_dir, "snapshots", snapshot_id)

    weight = os.path.join(blobs_dir, "pytorch_model.bin")
    if os.path.exists(weight):
        target = os.path.join(snap_dir, "pytorch_model.bin")
        if not os.path.exists(target):
            print(f"  Copying pytorch_model.bin to snapshot...")
            shutil.copy2(weight, target)
            print(f"  Done: {os.path.getsize(target):,} bytes")
        else:
            print(f"  pytorch_model.bin already in snapshot")

# 验证
print(f"\n{'='*60}")
print("Verification:")
print(f"{'='*60}")

from transformers import BlipForConditionalGeneration, BlipForImageTextRetrieval
from transformers import BlipProcessor, MarianMTModel, MarianTokenizer

tests = [
    ("BLIP Captioning", lambda: BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base") or BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")),
    ("BLIP ITM", lambda: BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco") or BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")),
    ("MarianMT", lambda: MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh") or MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh")),
]

for name, fn in tests:
    try:
        import torch
        with torch.no_grad():
            fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {str(e)[:120]}")

print("\nAll done!")
