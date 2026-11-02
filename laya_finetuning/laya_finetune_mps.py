"""
Laya-SENTRY fine-tuning on Apple Silicon.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from laya.agent import _fix_tokenizer_config

MODEL_ID = "convaiinnovations/laya"
DATASET_PATH = "./data/train.jsonl"
DEFAULT_MODEL_DIR = "./laya_base"
DEFAULT_OUTPUT_DIR = "./laya_finetuned_SENTRY"


def choose_device(requested):
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS is unavailable; using CPU instead.")
        return torch.device("cpu")
    return torch.device(requested)


def prepare_model(model_dir):
    model_dir = Path(model_dir)
    if not (model_dir / "model.safetensors").exists():
        print(f"Downloading {MODEL_ID} to {model_dir} ...")
        snapshot_download(MODEL_ID, local_dir=str(model_dir))
    _fix_tokenizer_config(str(model_dir))
    return str(model_dir)

def train(args, model_dir, device):
    from laya.train import TrainConfig, finetune

    config = TrainConfig(
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        calib_max=args.calib_max,
        calib_seed=20260922,
        seed=42,
        max_len=1024,
        head_max_len=256,
        gradient_checkpointing=not args.no_checkpointing,
        amp=False,
        shuffle_options=("choice",),
    )
    summary = finetune(
        data=DATASET_PATH,
        model_dir=str(model_dir),
        output_dir=str(args.output_dir),
        config=config,
        device=device.type,
    )
    # Wrapper-visible metadata and persistence preserved from the old entry point.
    for rel in [Path(args.output_dir) / 'rl_agent_config.json', Path(args.output_dir) / 'checkpoint_latest' / 'rl_agent_config.json']:
        if rel.exists():
            cfg = json.loads(rel.read_text(encoding='utf-8'))
            for name, entry in (summary.get('calibration') or {}).items():
                if isinstance(entry, dict) and entry.get('items') == 0:
                    idx = {'choice': 0, 'score': 1, 'noul': 2}.get(name)
                    if idx is not None:
                        cfg['temperature'][idx] = 1.2
            cfg.pop('temperature_by_options', None)
            cfg['model_name'] = 'laya-SENTRY'
            cfg['max_tokens_per_batch'] = 2048
            rel.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    print(f'Train items: {summary["train_items"]}; calibration items: {summary["calibration_items"]}')
    print('Fitted temperatures (choice, score, noul):', [round(t, 3) for t in summary['temperature']])
    print(f'Model saved to {args.output_dir}')


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Laya on Apple Silicon MPS")
    parser.add_argument("model_dir", nargs="?", default=None)
    parser.add_argument("output_dir", nargs="?", default=None)
    parser.add_argument("--model-dir", dest="model_dir_option")
    parser.add_argument("--output-dir", dest="output_dir_option")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--calib-max", type=int, default=400)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--no-checkpointing", action="store_true")
    args = parser.parse_args()

    if args.model_id != MODEL_ID:
        # Keep the requested model ID local to preparation by updating the
        # module constant before prepare_model() is called.
        globals()["MODEL_ID"] = args.model_id

    args.model_dir = args.model_dir_option or args.model_dir or DEFAULT_MODEL_DIR
    args.output_dir = args.output_dir_option or args.output_dir or DEFAULT_OUTPUT_DIR

    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    model_dir = prepare_model(args.model_dir)
    train(args, model_dir, device)
    gc.collect()


if __name__ == "__main__":
    main()