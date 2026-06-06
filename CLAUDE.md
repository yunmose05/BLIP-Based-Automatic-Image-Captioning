# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI course Experiment 3: Image captioning using **BLIP** (`Salesforce/blip-image-captioning-base`) with English→Chinese translation, plus five optimization extensions. The reference document `实验流程.md` contains the full experiment guide with complete code listings for every module.

## Important: No active development

This is a completed/coursework directory. All code blocks are embedded in `实验流程.md`. The actual `.py` scripts and `images/` / `results/` directories described in the guide may not exist yet on disk. Before editing any script, check whether it exists as a standalone file or only as a code block in the guide.

## Architecture (7 modules)

| Script | Purpose | Key Model |
|--------|---------|-----------|
| `caption.py` | Single-image inference + visualization | BLIP + MarianMT (en→zh) |
| `batch_caption.py` | Batch inference over `images/`, output to `results/results.txt` | Same as above |
| `caption_beam.py` | Beam search (num_beams=5, top-3 candidates) instead of greedy decode | BLIP + MarianMT |
| `caption_translate_llm.py` | Replace MarianMT with local DeepSeek (`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, ~4GB VRAM) for better Chinese translation | BLIP + DeepSeek |
| `caption_yolo_region.py` | YOLO object detection → crop regions → BLIP per-region captions → annotated visualization | YOLO11n (~6MB) + BLIP |
| `caption_retrieval.py` | Generate N candidates via beam search → rank by ITM (Image-Text Matching) score using `Salesforce/blip-itm-base-coco` | BLIP Captioning + BLIP ITM |
| `caption_batch_gpu.py` | GPU batch inference with DataLoader, compare single vs batch speed | BLIP + MarianMT |

A master script `run_all.py` runs all 7 modules sequentially via `subprocess.run`.

## Required models (downloaded from HuggingFace on first run)

- `Salesforce/blip-image-captioning-base` (~900MB)
- `Salesforce/blip-itm-base-coco` (for retrieval module)
- `Helsinki-NLP/opus-mt-en-zh` (MarianMT translation)
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (~3GB, for LLM translation module)
- `yolo11n.pt` or `yolov8n.pt` (~6MB, auto-downloaded by ultralytics)

## Setup commands

```bash
pip install transformers>=4.40.0 torch>=2.0.0 pillow matplotlib sentencepiece sacremoses
pip install ultralytics>=8.0.0 accelerate
```

For slow HuggingFace downloads in China, set the mirror:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Verify CUDA availability:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Running

All scripts are self-contained — run individually:
```bash
python caption.py                    # single image
python batch_caption.py              # batch all images
python caption_beam.py               # beam search
python caption_translate_llm.py      # DeepSeek translation
python caption_yolo_region.py        # YOLO + BLIP
python caption_retrieval.py          # ITM matching
python caption_batch_gpu.py          # GPU batch
python run_all.py                    # run all sequentially
```

## Expected directory structure

```
deepseek/
├── images/                  # Input images (.jpg/.png/.bmp)
├── results/                 # All outputs (auto-created by scripts)
│   ├── caption_result.jpg
│   ├── results.txt
│   ├── beam_results.txt
│   ├── translation_compare.txt
│   ├── yolo_region_results.txt
│   ├── yolo_regions/
│   ├── itm_results.txt
│   └── batch_gpu_results.txt
├── requirements.txt
├── 【7 scripts + run_all.py】
└── 实验流程.md
```

## Common issues

| Symptom | Fix |
|---------|-----|
| `CUDA out of memory` | Reduce batch_size, use 1.5B DeepSeek instead of 7B/8B, or run on CPU |
| Chinese characters render as squares | Install SimHei font, or update `font_path` in visualization code |
| YOLO detects nothing | Lower `conf_threshold` from 0.3 to 0.15 |
| BLIP ITM model fails to load | Pre-download `blip-itm-base-coco` or check HuggingFace access |
