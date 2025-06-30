import numpy as np
from rapidfuzz import fuzz, process
from PIL import Image
import re

def extract_ocr_text(recto_file, verso_file=None, reader=None):

    images_np = []

    img_recto = Image.open(recto_file.stream).convert("RGB")
    img_recto = zoom_image_pil(img_recto)
    images_np.append(np.array(img_recto))

    if verso_file:
        img_verso = Image.open(verso_file.stream).convert("RGB")
        img_verso = zoom_image_pil(img_verso)
        images_np.append(np.array(img_verso))

    results = reader.readtext_batched(images_np)

    all_texts = []
    for res in results:
        all_texts.extend(res)

    return all_texts

def zoom_image_pil(img, target_min_height=50, target_min_width=120, max_scale=2.0):
    width, height = img.size

    # Calcul du ratio par rapport aux dimensions cibles
    scale_height = target_min_height / height
    scale_width = target_min_width / width
    scale_factor = max(scale_height, scale_width)

    # Limiter le zoom à max_scale
    if scale_factor > max_scale:
        scale_factor = max_scale

    if scale_factor <= 1.0:
        print(f"Image suffisante ({width}x{height}) → aucun zoom.")
        return img

    new_size = (int(width * scale_factor), int(height * scale_factor))
    print(f"Image zoomée : {width}x{height} → {new_size[0]}x{new_size[1]} (scale {scale_factor:.2f})")

    resized_img = img.resize(new_size, Image.LANCZOS)
    return resized_img



# Fonction de recherche avec tolérance (nom, prénom, etc.)
def find_in_ocr(user_input, extracted_texts, normalize=False, threshold=60):
    cleaned_input = normalize_date(user_input) if normalize else user_input.lower().strip()
    cleaned_texts = [normalize_date(t) if normalize else t.lower().strip() for t in extracted_texts]

    result = process.extractOne(
        cleaned_input,
        cleaned_texts,
        scorer=fuzz.token_set_ratio,
        score_cutoff=threshold
    )
    
    if result is None:
        return None, 0
    
    best_match, score, index = result

    if score >= threshold:
        return extracted_texts[index], score
    else:
        return None, 0

def normalize_date(date_str):
    s = date_str.lower() 
    s = re.sub(r'[^0-9]', '.', s)  # remplace tout caractère non numérique par un point
    s = re.sub(r'\.+', '.', s)     # remplace plusieurs points consécutifs par un seul
    s = s.strip('.')               # supprime les points au début/fin
    return s

# Fonction pour vérifier les informations utilisateur
def verifier_informations(extracted_texts, infos_utilisateur):
    # Extraire uniquement les chaînes de caractères des OCR
    only_texts = [item[1] for item in extracted_texts]

    verification = {}
    for key, value in infos_utilisateur.items():
        if key == "date de naissance":
            matched, score = find_in_ocr(value, only_texts, normalize=True)
        else:
            matched, score = find_in_ocr(value, only_texts)

        verification[key] = {
            "valide": score > 60,
            "valeur_ocr": normalize_date(matched) if key == "date de naissance" and matched else matched,
            "score": score
        }
    return verification
