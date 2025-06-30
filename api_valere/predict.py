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
   

