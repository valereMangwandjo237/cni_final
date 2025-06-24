from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np

class_names = ['new_cni', 'old_cni', 'others', 'passport', 'recepisse']

model = load_model("model/mobile_net_valere.h5")

def predict_type(img_path):
    try:
        img = keras_image.load_img(img_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_label = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index])
        print("--------------prediction OK--------------")
        return predicted_label, confidence
    except Exception as e:
        print(f"[ERREUR] predict_type({img_path}) → {str(e)}")
        return "inconnu", 0.0
   

