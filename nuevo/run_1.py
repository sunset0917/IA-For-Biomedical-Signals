import torch
import json
from omegaconf import OmegaConf
# CORRECTO - reemplaza el import falso
from slam_llm.utils.model_utils import get_custom_model_factory

# Y para el dataset, lo que realmente existe en tu pipeline:
# No uses SpeechDataset directamente, usa get_preprocessed_dataset
from slam_llm.utils.dataset_utils import get_preprocessed_dataset
def load_biased_samples(jsonl_path, model, tokenizer, n_samples=20, device="cuda"):
    """
    Carga n_samples del dataset biasado ya procesados como tensores.
    Asume que tu dataset devuelve los mismos campos que el dataloader de inferencia.
    """
    samples = []
    with open(jsonl_path) as f:
        lines = [json.loads(l) for l in f][:n_samples]
    
    # Aquí debes usar tu propio collate/dataset para convertir
    # cada línea en el batch dict que espera forward_for_crispr
    # Esto depende de tu DataConfig — adapta según necesites
    for line in lines:
        # Ejemplo mínimo — reemplaza con tu pipeline real:
        batch = your_collate_fn(line, device=device)
        samples.append({
            "batch": batch,
            "golden_text": line["text"]  # campo con transcripción correcta
        })
    return samples


if __name__ == "__main__":
    # 1. Cargar modelo entrenado (igual que en inferencia)
    cfg = OmegaConf.load("/home/ashley-bravo/SLAM-LLM/examples/asr_librispeech/conf/prompt.yaml")
    model, tokenizer = model_factory(cfg.train_config, cfg.model_config,
                                      ckpt_path="/home/ashley-bravo/outputs/slam_asr/experimento1_nuevo/asr_epoch_4_step_5249/model.pt")
    device = "cuda"
    model = model.to(device)

    # 2. Cargar muestras biasadas (ej: audios de acento específico)
    biased_samples = load_biased_samples(
        "/home/ashley-bravo/SLAM-LLM/examples/asr_librispeech/data/commonvoice/cv22_test_reducid.jsonl",
        model, tokenizer,
        n_samples=18,
        device=device
    )

    # 3. Ejecutar CRISPR
    from crispr_debiasing import run_crispr
    top_indices, scores = run_crispr(
        model, tokenizer, biased_samples,
        top_n_neurons=5,  # empieza con pocos
        device=device
    )

    # 4. Guardar el proyector debbiasado
    torch.save(model.encoder_projector.state_dict(),
               "/home/ashley-bravo/outputs/slam_asr/experimento1_nuevo/asr_epoch_4_step_5249/model_debiased.pt")
    print("Proyector guardado.")
