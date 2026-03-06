import json
import os
import re
import unicodedata
import soundfile as sf
from datasets import load_dataset, Features, Value, Audio
from tqdm import tqdm

# --- CONFIGURACIÓN DEL CLUSTER ---
BASE_DATA_PATH = "/home/ashley-bravo/datasets/common_voice_22_storage"
CONTROL_PATH = "/home/ashley-bravo/SLAM-LLM/examples/asr_librispeech/data"
os.makedirs(BASE_DATA_PATH, exist_ok=True)
os.makedirs(CONTROL_PATH, exist_ok=True)

def clean_text(input_string):
    if not isinstance(input_string, str):
        input_string = str(input_string) if input_string is not None else ""
    normalized = unicodedata.normalize('NFD', input_string)
    stripped = "".join([char for char in normalized if unicodedata.category(char) != 'Mn'])
    text = unicodedata.normalize('NFC', stripped)
    text = text.lower()
    text = re.sub(r"[‘’“”—–&@#/$%,.?!;:\"\'\(\)\[\]\-]", "", text)
    text = " ".join(text.split())
    return text

custom_features = Features({
    'client_id': Value('string'),
    'path': Value('string'),
    'sentence_id': Value('string'),
    'sentence': Value('string'),
    'sentence_domain': Value('string'),
    'up_votes': Value('string'),
    'down_votes': Value('string'),
    'age': Value('string'),
    'gender': Value('string'),
    'variant': Value('string'),
    'locale': Value('string'),
    'segment': Value('string'),
    'accent': Value('string'),
    'audio': Audio(sampling_rate=16000),
})

def prepare_filtered_benchmark(output_name, max_hours=1.0):
    split_name = "test"
    print(f"\n🚀 Generando Benchmark filtrado (Solo con metadata de género)")
    
    out_jsonl = os.path.join(CONTROL_PATH, f"{output_name}.jsonl")
    out_gender = os.path.join(CONTROL_PATH, f"{output_name}_gender.jsonl")
    
    total_seconds = 0
    max_seconds = max_hours * 3600
    
    dataset = load_dataset(
        "fsicoli/common_voice_22_0", 
        "en", 
        split=split_name, 
        streaming=True, 
        trust_remote_code=True,
        features=custom_features
    )
    
    audio_save_dir = os.path.join(BASE_DATA_PATH, split_name)
    os.makedirs(audio_save_dir, exist_ok=True)

    with open(out_jsonl, 'w', encoding='utf-8') as f_asr, \
         open(out_gender, 'w', encoding='utf-8') as f_gen:
        
        count = 0
        # Usamos un contador para ver cuántos descartamos
        discarded_no_gender = 0
        
        for example in tqdm(dataset, desc="Buscando muestras válidas"):
            if total_seconds >= max_seconds:
                break

            # --- NUEVO FILTRO: GÉNERO OBLIGATORIO ---
            # Verificamos que el género exista y sea válido (male o female)
            raw_gender = example.get('gender')
            if not raw_gender or raw_gender.lower() in ["", "unknown", "other", "nan"]:
                discarded_no_gender += 1
                continue

            # Filtro de duración (el que ya tenías)
            duration = len(example['audio']['array']) / 16000
            if duration < 2.0 or duration > 10.0:
                continue

            target_text = clean_text(example['sentence'])
            if not target_text:
                continue

            # Si pasó todos los filtros, procedemos a guardar
            audio_filename = f"{example['sentence_id']}.wav"
            full_audio_path = os.path.join(audio_save_dir, audio_filename)
            
            if not os.path.exists(full_audio_path):
                sf.write(full_audio_path, example['audio']['array'], 16000)

            total_seconds += duration
            current_key = f"{example['client_id'][:8]}_{example['sentence_id']}_ASR"

            # Escribimos en ambos archivos
            f_asr.write(json.dumps({
                "key": current_key,
                "source": full_audio_path,
                "target": target_text
            }, ensure_ascii=False) + '\n')

            f_gen.write(json.dumps({
                "key": current_key,
                "gender": raw_gender.lower()
            }, ensure_ascii=False) + '\n')
            
            count += 1
            
    print(f"\n✅ Proceso completado con éxito.")
    print(f"📊 Muestras con género encontradas: {count}")
    print(f"📉 Muestras descartadas por falta de género: {discarded_no_gender}")
    print(f"⏱️ Tiempo total de audio recolectado: {total_seconds/3600:.2f} horas")

if __name__ == "__main__":
    # Aumenté a 2 horas para intentar capturar más datos con género
    prepare_filtered_benchmark("cv22_benchmark_test", max_hours=1.0)
