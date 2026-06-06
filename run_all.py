"""
一键运行所有实验模块（每次运行创建独立时间戳目录）
用法: python run_all.py
"""
import subprocess
import sys
import os
from datetime import datetime

# ── 创建带时间戳的结果目录 ────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join("results", f"run_{timestamp}")
os.makedirs(results_dir, exist_ok=True)
print(f"本轮结果目录: {results_dir}\n")

# 设置环境变量，所有子脚本自动使用此目录
env = os.environ.copy()
env["RESULTS_DIR"] = results_dir

scripts = [
    ("基础单图推理", "caption.py"),
    ("批量可视化（主输出）", "extensions/caption_visualize.py"),
    ("基础批量推理", "extensions/batch_caption.py"),
    ("Beam Search 优化", "extensions/caption_beam.py"),
    ("DeepSeek 翻译对比", "extensions/caption_translate_llm.py"),
    ("YOLO + BLIP 区域描述", "extensions/caption_yolo_region.py"),
    ("图文匹配检索", "extensions/caption_retrieval.py"),
    ("GPU 批量加速", "extensions/caption_batch_gpu.py"),
]

for name, script in scripts:
    print(f"\n{'='*60}")
    print(f"  {name}: {script}")
    print(f"{'='*60}")
    try:
        subprocess.run([sys.executable, script], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"  [WARNING] {script} 运行出错: {e}")
    except FileNotFoundError:
        print(f"  [SKIP] {script} 文件不存在，跳过")

print(f"\n所有模块执行完毕，结果保存在 {results_dir}/")
