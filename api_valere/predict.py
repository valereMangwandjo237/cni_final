import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# === Liste des classes dans le bon ordre ===
class_names = ['new_cni', 'old_cni', 'others', 'passport', 'recepisse']

# === Chargement du modèle de classification d'identité ===
model = load_model("model\\best_detection_model.h5")

def predict_type(image_np):
    """
    Prédit le type de document à partir d'une image (numpy array RGB).
    """
    try:
        resized = cv2.resize(image_np, (224, 224))
        img_array = preprocess_input(resized.astype("float32"))
        img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_label = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index])

        print(f"[PREDICT] Classe : {predicted_label}, Confiance : {confidence:.4f}")
        return predicted_label, confidence
    except Exception as e:
        print(f"[ERREUR] predict_type(image_np) → {str(e)}")
        return "inconnu", 0.0


def predict_type_by_keyword(extracted_texts):
    """
    Tente de deviner le type de document à partir des mots extraits par OCR.
    """
    if not extracted_texts:
        return "others", 1.0

    word_list = [item[1] for item in extracted_texts]
    print("[OCR] Mots extraits pour classification :", word_list)

    for word in word_list:
        word_lower = word.lower()

        if any(keyword in word_lower for keyword in [
            "kit", "tempory", "request", "presidence", "presidency", "provisoire"
        ]):
            return "recepisse", 1.0

        if word_lower.startswith("pocmr") or word_lower.startswith("aa") or word_lower.startswith("passeport"):
            return "passport", 1.0

    return "inconnu", 0.0
