from flask import Flask, request, jsonify
import os
import uuid
import json

from predict import predict_type
from ocr_utils import extract_ocr_text, verifier_informations, zoom_image
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Bienvenue sur l'API Flask de vérification CNI/Recepissé"})


@app.route('/analyser', methods=['POST'])
def analyser():
    try:
        # Vérifier que les deux champs sont présents
        if 'recto' not in request.files or 'verso' not in request.files or 'data' not in request.form:
            return jsonify({"error": "Images (recto, verso) et données utilisateur (data) sont requis."}), 400

        # Récupérer les deux images
        file_recto = request.files['recto']
        file_verso = request.files['verso']
        json_data = json.loads(request.form['data'])

        # Vérification des champs requis
        required = ["nom", "prenom", "lieu_de_naisance", "numero", "date_de_naissance"]
        for key in required:
            if key not in json_data:
                return jsonify({"error": f"Champ manquant : {key}"}), 400

        # Sauvegarder les images temporairement
        img_recto_path = f"temp_{uuid.uuid4()}.jpg"
        img_verso_path = f"temp_{uuid.uuid4()}.jpg"

        file_recto.save(img_recto_path)
        file_verso.save(img_verso_path)

        # Étape 1 : prédire le type
        doc_type, confidence = predict_type(img_recto_path)

        # Étape 2 : OCR
        ocr_recto = extract_ocr_text(img_recto_path)
        ocr_verso = extract_ocr_text(img_verso_path)

        ocr_texts = ocr_recto + ocr_verso

        # Infos utilisateur
        infos = {
            "nom": json_data.get("nom", ""),
            "prenom": json_data.get("prenom", ""),
            "numero": json_data.get("numero", ""),
            "lieu de naissance": json_data.get("lieu_de_naisance", ""),
            "date de naissance": json_data.get("date_de_naissance", "")
        }

        # Étape 3 : Vérification
        verification = verifier_informations(ocr_texts, infos)

        # Supprimer les images temporaires
        if os.path.exists(img_recto_path):
            os.remove(img_recto_path)
        if os.path.exists(img_verso_path):
            os.remove(img_verso_path)

        return jsonify({
            "type_document": doc_type,
            "confiance": round(confidence, 3),
            "verification_informations": verification
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
