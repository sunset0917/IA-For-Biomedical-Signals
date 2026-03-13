import json
import os
import re
import unicodedata
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURACIÓN ---
BASE_DATA_PATH = "/home/ashley-bravo/datasets/voxpopuli"
CONTROL_PATH = "/home/ashley-bravo/SLAM-LLM/examples/asr_librispeech/data/voxpopuli"
os.makedirs(BASE_DATA_PATH, exist_ok=True)
os.makedirs(CONTROL_PATH, exist_ok=True)

def clean_text(text):
    if not text: return ""
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    return " ".join(text.split())

def process_voxpopuli(max_hours=1.0):
    split_name = "test"
    print(f"\n🚀 Procesando VoxPopuli (en) - Split: {split_name}")

    dataset = load_dataset("facebook/voxpopuli", "en", split=split_name, streaming=True)
    
    out_asr = os.path.join(CONTROL_PATH, f"vox_{split_name}.jsonl")
    out_meta = os.path.join(CONTROL_PATH, f"vox_{split_name}_metadata.jsonl")
    audio_dir = os.path.join(BASE_DATA_PATH, split_name)
    os.makedirs(audio_dir, exist_ok=True)

    max_seconds = max_hours * 3600
    total_seconds = 0

    with open(out_asr, 'w') as f_asr, open(out_meta, 'w') as f_meta:
        for example in tqdm(dataset, desc="Extrayendo VoxPopuli"):
            if total_seconds >= max_seconds: break

            # VoxPopuli metadata
            gender = example.get('gender', 'unknown').lower()
            accent = example.get('accent', 'unknown').lower() # A veces es el country code
            
            # Filtro de seguridad (VoxPopuli a veces tiene 'nan' en gender)
            if gender in ["", "nan", "unknown"]: continue

            # Duración: VoxPopuli usa 16000Hz
            duration = len(example['audio']['array']) / 16000
            if duration < 2.0 or duration > 15.0: continue

            target_text = clean_text(example['normalized_text'])
            if not target_text: continue

            # Guardar Audio
            audio_key = f"vox_{example['audio_id']}"
            audio_path = os.path.join(audio_dir, f"{audio_key}.wav")
            
            if not os.path.exists(audio_path):
                sf.write(audio_path, example['audio']['array'], 16000)

            # Escribir JSONLs
            f_asr.write(json.dumps({
                "key": audio_key,
                "source": audio_path,
                "target": target_text
            }) + '\n')

            f_meta.write(json.dumps({
                "key": audio_key,
                "gender": gender,
                "accent": accent
            }) + '\n')

            total_seconds += duration

    print(f"✅ VoxPopuli completado: {total_seconds/3600:.2f} horas recolectadas.")

if __name__ == "__main__":
    process_voxpopuli(max_hours=1.0)
