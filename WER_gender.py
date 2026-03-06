import os
import json
from jiwer import wer

# ==============================
# RUTAS
# ==============================

base_path = "/home/ashley-bravo/outputs/slam_asr/tiny_large/asr_epoch_7_step_36"

file_gt = os.path.join(base_path, "decode_train_beam4_gt")
file_pred = os.path.join(base_path, "decode_train_beam4_pred")

file_path = "/home/ashley-bravo/SLAM-LLM/examples/asr_librispeech/data/cv22_benchmark_test_gender.jsonl"


# ==============================
# 1. CARGAR MAPEO KEY -> GÉNERO
# ==============================

gender_map = {}

with open(file_path, "r") as f:
    for line in f:
        data = json.loads(line)

        key = data.get("key")
        gender = data.get("gender")

        if key and gender:

            # Normalizar género
            if gender.startswith("male"):
                gender = "male"
            elif gender.startswith("female"):
                gender = "female"
            else:
                gender = "other"

            gender_map[key] = gender


print("Gender labels encontrados:", set(gender_map.values()))
print("Total keys en gender_map:", len(gender_map))


# ==============================
# 2. FUNCIÓN WER POR GÉNERO
# ==============================

def calcular_wer_por_genero(path_gt, path_pred):

    data_by_gender = {
        "male": {"ref": [], "hyp": []},
        "female": {"ref": [], "hyp": []}
    }

    # --------------------------
    # Leer predicciones
    # --------------------------

    preds = {}

    with open(path_pred, "r") as f_pred:
        for line in f_pred:

            if "\t" not in line:
                continue

            key, text = line.strip().split("\t", 1)

            preds[key] = text

    print("Total predicciones:", len(preds))


    # --------------------------
    # Leer ground truth
    # --------------------------

    with open(path_gt, "r") as f_gt:

        for line in f_gt:

            if "\t" not in line:
                continue

            key, ref_text = line.strip().split("\t", 1)

            if key not in preds:
                continue

            genero = gender_map.get(key, "other")

            if genero in data_by_gender:

                data_by_gender[genero]["ref"].append(ref_text)
                data_by_gender[genero]["hyp"].append(preds[key])


    # ==========================
    # 3. CALCULAR WER
    # ==========================

    print("\n===== RESULTADOS WER POR GÉNERO =====\n")

    for g in ["male", "female"]:

        refs = data_by_gender[g]["ref"]
        hyps = data_by_gender[g]["hyp"]

        if len(refs) == 0:
            print(f"{g}: No samples encontrados")
            continue

        error = wer(refs, hyps)

        print(
            f"Género: {g.upper()} | Samples: {len(refs)} | WER: {error:.4f}"
        )


# ==============================
# 4. EJECUTAR
# ==============================

calcular_wer_por_genero(file_gt, file_pred)
