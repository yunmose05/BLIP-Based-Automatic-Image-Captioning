"""
下载所有实验模型（直接下载大文件 + API 下小文件）
用法: python download_models.py
"""
import os, hashlib, shutil
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import requests
from tqdm import tqdm
from huggingface_hub import snapshot_download, hf_hub_download

CACHE = os.path.expanduser("~/.cache/huggingface/hub")
MIRROR = "https://hf-mirror.com"

MODELS = [
    ("Salesforce/blip-image-captioning-base", "blip-captioning", [
        "config.json", "preprocessor_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
    ]),
    ("Salesforce/blip-itm-base-coco", "blip-itm", [
        "config.json", "preprocessor_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
    ]),
    ("Helsinki-NLP/opus-mt-en-zh", "marianmt", [
        "config.json", "tokenizer_config.json", "vocab.json",
        "source.spm", "target.spm",
    ]),
]

def download_small_files(model_id):
    """下载模型的小文件（config, tokenizer 等）"""
    print(f"  Downloading small files...")
    try:
        snapshot_download(model_id, allow_patterns="*.json;*.txt;*.spm;*.md", max_workers=4)
        print(f"  Small files: OK")
        return True
    except Exception as e:
        print(f"  Small files: FAIL - {e}")
        return False

def download_big_file(model_id, filename, sha256=None):
    """直接下载大模型权重文件到 HF 缓存"""
    url = f"{MIRROR}/{model_id}/resolve/main/{filename}"
    # 构建目标路径：blobs/<sha256>
    # 如果不知道 sha256，先下载到临时位置
    model_dir = os.path.join(CACHE, f"models--{model_id.replace('/', '--')}")
    blobs_dir = os.path.join(model_dir, "blobs")
    snapshots_dir = os.path.join(model_dir, "snapshots")
    os.makedirs(blobs_dir, exist_ok=True)

    # 获取文件大小
    try:
        r = requests.head(url, timeout=10)
        total = int(r.headers.get("content-length", 0))
    except:
        print(f"  Cannot get file size for {filename}")
        return False

    # 临时下载位置
    tmp_path = os.path.join(blobs_dir, f"{filename}.download")

    downloaded = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}

    resp = requests.get(url, headers=headers, stream=True, timeout=60)
    if resp.status_code not in (200, 206):
        print(f"  HTTP {resp.status_code} for {filename}")
        return False

    # 计算实际大小（包括已下载部分）
    actual_total = downloaded + int(resp.headers.get("content-length", 0))

    mode = "ab" if downloaded else "wb"
    with open(tmp_path, mode) as f:
        with tqdm(total=actual_total, initial=downloaded,
                  unit="B", unit_scale=True, desc=f"  {filename}",
                  ncols=80) as bar:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    # 验证 SHA256 并移动到正确位置
    if sha256:
        print(f"  Verifying SHA256...")
        h = hashlib.sha256()
        with open(tmp_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != sha256:
            print(f"  SHA256 mismatch!\n  Expected: {sha256}\n  Got: {actual}")
            return False
        print(f"  SHA256: OK")

    # 移动到 blobs 目录（用 sha256 命名，未知则保留原名）
    if sha256:
        final_path = os.path.join(blobs_dir, sha256)
    else:
        final_path = os.path.join(blobs_dir, filename)
    shutil.move(tmp_path, final_path)

    # 找到最新的 snapshot 目录并拷贝过去
    if os.path.exists(snapshots_dir):
        snaps = os.listdir(snapshots_dir)
        if snaps:
            snap_dir = os.path.join(snapshots_dir, snaps[0])
            shutil.copy2(final_path, os.path.join(snap_dir, filename))
            print(f"  Copied to snapshot: {snaps[0]}")

    return True


for model_id, short_name, small_files in MODELS:
    print(f"\n{'='*60}")
    print(f"Model: {model_id}")
    print(f"{'='*60}")

    # Step 1: download small files via HF API
    download_small_files(model_id)

    # Step 2: download model weights directly
    # BLIP models use pytorch_model.bin; MarianMT uses pytorch_model.bin
    for weight_file in ["pytorch_model.bin", "model.safetensors"]:
        download_big_file(model_id, weight_file)
        break  # just try one format

print("\nDone! All models downloaded.")
