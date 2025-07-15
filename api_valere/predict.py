from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import cv2

class_names = ['new_cni', 'old_cni', 'others', 'passport', 'recepisse']

model = load_model("model/mobile_net_valere.h5")

def predict_type(image_np):
    try:    
        # Redimensionner et normaliser l'image
        resized = cv2.resize(image_np, (224, 224))
        img_array = resized.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)

        # Prédiction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_label = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index])

        print("--------------prediction OK--------------")
        return predicted_label, confidence
    except Exception as e:
        print(f"[ERREUR] predict_type(image_np) → {str(e)}")
        return "inconnu", 0.0
   

def predict_type_by_keyword(extracted_texts):
    if not extracted_texts:
        return "others", 1.0
    
    # Extraire uniquement les chaînes de caractères des OCR
    word_list = [item[1] for item in extracted_texts]
    print("Texte extrait: ", word_list)

    for word in word_list:
        word_lower = word.lower()
        
        # Recherche de mots-clés pour "recepissé"
        if any(keyword in word_lower for keyword in ["kit", "tempory", "request", "presidence", "presidency", "provisoire"]):
            return "recepisse", 1.0
        
        # Recherche de préfixes pour "passeport"
        if word_lower.startswith("pocmr") or word_lower.startswith("aa") or word_lower.startswith("passeport"):
            return "passport", 1.0
        
        #full_text = " ".join(word.lower() for word in word_list)
        #if not any(keyword in full_text for keyword in ["republique", "cameroun", "republic", "cameroon"]):
        #    return "others", 1.0  # Confiance maximale car on est sûr que ce n’est pas un document officiel
    
    return "inconnu", 0.0
