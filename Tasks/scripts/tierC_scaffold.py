#!/usr/bin/env python3
"""
Tier C scaffold: prepare LoRA fine-tuning configuration for a transformer model.

This script does not start heavy training by default. Instead it prepares a small
training configuration and verifies that the training environment has the required
packages (transformers, accelerate, peft). If packages are missing, it prints clear
installation instructions and writes a JSON config to `data/analysis/tierC_config.json`.

When you're ready to run real training, use the produced config with a training runner
that leverages `accelerate` and a GPU. The script includes a short example command
that will run training with the config when the environment is provisioned.
"""
import argparse
import json
from pathlib import Path
import sys


def parse_args():
    p = argparse.ArgumentParser(description='Tier C LoRA scaffold for fine-tuning')
    p.add_argument('--out', type=str, default='data/analysis', help='Output directory for config')
    p.add_argument('--model', type=str, default='gpt2', help='Base model name (HuggingFace)')
    p.add_argument('--train-file', type=str, default='', help='Optional path to training JSONL or dataset')
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Check for heavy deps
    try:
        import transformers
        import accelerate
        import peft
        deps_ok = True
    except Exception:
        deps_ok = False

    config = {
        'model_name_or_path': args.model,
        'train_file': args.train_file,
        'task': 'causal-lm',
        'lora': {
            'r': 8,
            'lora_alpha': 32,
            'target_modules': ['c_attn', 'q_proj', 'v_proj', 'k_proj'],
            'dropout': 0.1
        },
        'training': {
            'per_device_train_batch_size': 8,
            'gradient_accumulation_steps': 4,
            'learning_rate': 2e-4,
            'num_train_epochs': 3,
            'logging_steps': 50,
        },
        'notes': 'This is a scaffold. Use accelerate and peft for efficient LoRA fine-tuning. Ensure you have a GPU.'
    }

    cfg_path = out / 'tierC_config.json'
    with open(cfg_path, 'w', encoding='utf-8') as fh:
        json.dump(config, fh, indent=2)

    print('Wrote LoRA scaffold config to', cfg_path)

    if not deps_ok:
        print('\nRequired packages for real LoRA training are not available in this environment.')
        print('To set up the training environment, run:')
        print('  pip install -r requirements_tierBC.txt')
        print('  pip install accelerate transformers peft')
        print('\nAfter installing and configuring `accelerate`, you can run a training command such as:')
        print('  accelerate launch run_clm_lora.py --config', str(cfg_path))
        print('\n(See run_clm_lora.py examples in the HuggingFace + PEFT docs for a full runner.)')
    else:
        print('\nEnvironment appears to have transformers/accelerate/peft installed.')
        print('You can use the generated config at', cfg_path, 'with your training runner.')


if __name__ == '__main__':
    main()
