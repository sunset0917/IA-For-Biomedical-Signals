import sys
sys.path.insert(0, "/home/ashley-bravo/SLAM-LLM/src")

import torch
import json
import logging
import numpy as np
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig
from typing import Optional
from torch.utils.data import DataLoader

# Imports del framework
from slam_llm.utils.model_utils import get_custom_model_factory

# Imports de tu proyecto ASR
sys.path.insert(0, "/home/ashley-bravo/SLAM-LLM/examples/asr_librispeech")
from asr_config import ModelConfig, TrainConfig, DataConfig, LogConfig, FSDPConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Paths ─────────────────────────────────────────────────────
CKPT_PATH    = "/home/ashley-bravo/outputs/slam_asr/experimento1_nuevo/asr_epoch_4_step_5249/model.pt"
LLM_PATH     = "/home/ashley-bravo/models/tinyllama"
ENCODER_PATH = "large-v3"
BIASED_JSONL = "/home/ashley-bravo/SLAM-LLM/examples/asr_librispeech/data/commonvoice/cv22_test_reducid.jsonl"
OUTPUT_PATH  = "/home/ashley-bravo/outputs/slam_asr/experimento1_nuevo/asr_epoch_4_step_5249/projector_debiased.pt"
CONF_PATH    = "/home/ashley-bravo/SLAM-LLM/examples/asr_librispeech/conf/prompt.yaml"

N_SAMPLES    = 18
TOP_N_NEURONS = 5
DEVICE       = "cuda"


def build_configs():
    """Construye train_config y model_config igual que tu bash script."""

    # Cargar yaml base
    cfg = OmegaConf.load(CONF_PATH)

    # Sobrescribir con los mismos valores que usas en tu bash script
    overrides = OmegaConf.create({
        "model_config": {
            "llm_name": "TinyLlama-1.1B-Chat-v1.0",
            "llm_path": LLM_PATH,
            "llm_dim": 2048,
            "encoder_name": "whisper",
            "encoder_projector_ds_rate": 5,
            "encoder_path": ENCODER_PATH,
            "encoder_dim": 1280,
            "encoder_projector": "linear",
        },
        "train_config": {
            "model_name": "asr",
            "freeze_encoder": True,
            "freeze_llm": True,
            "batching_strategy": "custom",
            "num_epochs": 1,
            "val_batch_size": 1,
            "num_workers_dataloader": 0,
            "output_dir": "/home/ashley-bravo/outputs/slam_asr/experimento1_nuevo",
        },
        "dataset_config": {
            "dataset": "speech_dataset",
            "val_data_path": BIASED_JSONL,
            "input_type": "mel",
            "mel_size": 128,
            "inference_mode": True,
        }
    })
    cfg = OmegaConf.merge(cfg, overrides)
    return cfg


def load_biased_samples(cfg, model, tokenizer, n_samples, device):
    """
    Carga muestras usando el mismo dataset/collate que usa tu inferencia normal.
    """
    from slam_llm.utils.dataset_utils import get_preprocessed_dataset

    # Obtener dataset de validación (el mismo que usa inference_batch)
    dataset_val = get_preprocessed_dataset(
        tokenizer,
        cfg.dataset_config,
        split="val",
    )

    # Tomar solo n_samples
    indices = list(range(min(n_samples, len(dataset_val))))
    subset = torch.utils.data.Subset(dataset_val, indices)

    dataloader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset_val.collater if hasattr(dataset_val, "collater") else None,
    )

    samples = []
    for batch in dataloader:
        # Mover tensores a device
        batch_device = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_device[k] = v.to(device)
            else:
                batch_device[k] = v

        # El texto golden está en el batch (campo "text" o "target_text")
        golden_text = batch.get("text", batch.get("target_text", [""]))[0]

        samples.append({
            "batch": batch_device,
            "golden_text": golden_text,
        })

    return samples


if __name__ == "__main__":
    # 1. Construir configs
    cfg = build_configs()

    # 2. Cargar modelo
    model_factory = get_custom_model_factory(cfg.model_config, logger)
    model, tokenizer = model_factory(
        cfg.train_config,
        cfg.model_config,
        ckpt_path=CKPT_PATH,
    )
    model = model.to(DEVICE)
    model.eval()

    # 3. Cargar muestras biasadas
    biased_samples = load_biased_samples(
        cfg, model, tokenizer,
        n_samples=N_SAMPLES,
        device=DEVICE,
    )
    print(f"Muestras cargadas: {len(biased_samples)}")

    # 4. Ejecutar CRISPR
    from crispr_debiasing import run_crispr
    top_indices, scores = run_crispr(
        model, tokenizer, biased_samples,
        top_n_neurons=TOP_N_NEURONS,
        device=DEVICE,
    )

    # 5. Guardar proyector debiased
    torch.save(
        model.encoder_projector.state_dict(),
        OUTPUT_PATH,
    )
    print(f"Proyector debiased guardado en: {OUTPUT_PATH}")
