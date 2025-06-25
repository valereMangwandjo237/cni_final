import easyocr
import numpy as np
from rapidfuzz import fuzz, process
from PIL import Image

def load_ocr_model():
    return easyocr.Reader(['fr', 'en'], gpu=False, download_enabled=False)

# Fonction OCR
def extract_ocr_text(img_path):
    image = zoom_image(img_path)
    img_array = np.array(image)

    reader = load_ocr_model()
    results = reader.readtext(img_array)

    extracted_texts = []
    for (_, text, _) in results:
        extracted_texts.append(text)
    
    return extracted_texts  # Liste des textes extraits

def zoom_image(image_path, target_min_height=50, target_min_width=120, max_scale=2.0):
    img = Image.open(image_path)
    width, height = img.size

    # Calcul du ratio par rapport aux dimensions cibles
    scale_height = target_min_height / height
    scale_width = target_min_width / width
    scale_factor = max(scale_height, scale_width)

    # Limiter le zoom à max_scale
    if scale_factor > max_scale:
        scale_factor = max_scale

    # Ne pas zoomer si image déjà assez grande
    if scale_factor <= 1.0:
        print(f"Image suffisante ({width}x{height}) → aucun zoom.")
        return img

    # Zoom dynamique
    new_size = (int(width * scale_factor), int(height * scale_factor))
    print(f"Image zoomée : {width}x{height} → {new_size[0]}x{new_size[1]} (scale {scale_factor:.2f})")

    resized_img = img.resize(new_size, Image.LANCZOS)
    return resized_img


# Fonction de recherche avec tolérance (nom, prénom, etc.)
def find_in_ocr(user_input, extracted_texts, threshold=60):
    cleaned_input = " ".join(user_input.lower().split())
    cleaned_texts = [" ".join(t.lower().split()) for t in extracted_texts]

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


# Fonction pour vérifier les informations utilisateur
def verifier_informations(extracted_texts, infos_utilisateur):
    verification = {}
    for key, value in infos_utilisateur.items():
        matched, score = find_in_ocr(value, extracted_texts)
        verification[key] = {
            "valide": score > 60,
            "valeur_ocr": matched,
            "score": score
        }
    return verification
