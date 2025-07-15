import numpy as np
from rapidfuzz import fuzz, process
from PIL import Image
import re
from datetime import datetime

def est_mot_valide(mot):
    # Supprime les ponctuations et vérifie si le mot est alphabétique et a au moins 3 lettres
    return mot.isalpha() and len(mot) >= 5


def zoom_image_for_extract(img, scale):
    width, height = img.size
    resized_img = img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    return resized_img


def rotate_image_if_needed(image_np, reader, seuil_mots=6):
    angles = [0, 180, 90, 270]
    zoom_levels = [1.0, 1.5, 2.0, 2.5]

    best_results = []
    max_word_count = 0
    best_angle = 0
    best_zoom = 1.0

    # Étape 1 : test de chaque rotation sans zoom
    for angle in angles:
        img = Image.fromarray(image_np).rotate(angle, expand=True)
        img_np_rotated = np.array(img)
        results = reader.readtext(img_np_rotated)

        mots_valides = []
        for item in results:
            if len(item) > 1:
                mot = item[1].strip()
                if est_mot_valide(mot):
                    mots_valides.append(mot)

        word_count = len(mots_valides)
        print(f"[DEBUG] Rotation {angle}°, Zoom ×1.0 → {word_count} mots valides")

        if word_count > max_word_count:
            best_results = results
            max_word_count = word_count
            best_angle = angle
            best_zoom = 1.0

        if word_count >= seuil_mots:
            print(f"[DEBUG] ✅ OCR suffisant à {angle}° sans zoom ({word_count} mots valides)")
            return results

    # Étape 2 : test de chaque zoom pour chaque angle
    for zoom_factor in zoom_levels[1:]:  # Ignore 1.0 (déjà fait)
        for angle in angles:
            img = Image.fromarray(image_np).rotate(angle, expand=True)
            img = zoom_image_for_extract(img, scale=zoom_factor)
            img_np_zoomed = np.array(img)
            results = reader.readtext(img_np_zoomed)

            mots_valides = []
            for item in results:
                if len(item) > 1:
                    mot = item[1].strip()
                    if est_mot_valide(mot):
                        mots_valides.append(mot)

            word_count = len(mots_valides)
            print(f"[DEBUG] Rotation {angle}°, Zoom ×{zoom_factor} → {word_count} mots valides")

            if word_count > max_word_count:
                best_results = results
                max_word_count = word_count
                best_angle = angle
                best_zoom = zoom_factor

            if word_count >= seuil_mots:
                print(f"[DEBUG] ✅ OCR suffisant à {angle}° avec zoom ×{zoom_factor} ({word_count} mots valides)")
                return results

    print(f"[DEBUG] ❌ Aucun angle/zoom n’a atteint {seuil_mots} mots valides")
    print(f"[DEBUG] Meilleur résultat : {max_word_count} mots à {best_angle}° zoom {best_zoom}")
    return best_results





def file_storage_to_ndarray(file_storage):
    try:
        image = Image.open(file_storage.stream).convert('RGB')
        image = zoom_image_pil(image)
        return np.array(image)
    except Exception as e:
        print(f"[Erreur de lecture d’image] {file_storage.filename} ➜ {e}")
        return None
    

def extract_ocr_text(recto_file, verso_file=None, reader=None):
    images_np = []

    img_recto = file_storage_to_ndarray(recto_file)
    if img_recto is not None:
        images_np.append(img_recto)

    if verso_file:
        img_verso = file_storage_to_ndarray(verso_file)
        if img_verso is not None:
            images_np.append(img_verso)

    if not images_np:
        raise ValueError("Aucune image valide fournie pour l'OCR.")
    
    # Debug : vérifier types et formes
    print(f"[DEBUG] Nombre d'images OCR : {len(images_np)}")
    for idx, img in enumerate(images_np):
        print(f"[DEBUG] Image {idx} - type: {type(img)}, shape: {getattr(img, 'shape', 'N/A')}")
    

    all_texts = []
    for img in images_np:
        results = rotate_image_if_needed(img, reader)
        all_texts.extend(results)

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

    #trouver la date d'expiration de la carte
    dates_trouvees = filtrer_dates(only_texts)
    date_max = trouver_date_max(dates_trouvees)

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
    return verification, date_max


def filtrer_dates(extracted_texts):
    pattern = r"\b\d{2}[\/\-,. ]+\d{2}[\/\-,. ]+\d{4}\b"
    dates = []

    for mot in extracted_texts:
        if isinstance(mot, str):
            texte = mot.strip()
            if re.fullmatch(pattern, texte):
                dates.append(normalize_date(texte))
    
    print("Date: ", dates)
    return dates


def trouver_date_max(liste_dates, formats=["%d.%m.%Y", "%d-%m-%Y", "%d/%m/%Y"]):
    dates_valides = []
    for d in liste_dates:
        for fmt in formats:
            try:
                dt = datetime.strptime(d, fmt)
                dates_valides.append((dt, d))
                break  # une fois un format validé, on ne teste pas les autres
            except ValueError:
                continue
    if not dates_valides:
        return None
    date_max = max(dates_valides, key=lambda x: x[0])
    return date_max[1]