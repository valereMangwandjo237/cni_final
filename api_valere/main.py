from flask import Flask, request, jsonify
import os
import json
import easyocr
from PIL import Image
import traceback

from predict import predict_type
from ocr_utils import extract_ocr_text, verifier_informations
from global_score import *
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

app = Flask(__name__)

reader = easyocr.Reader(['fr', 'en'], gpu=False, download_enabled=False)

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

        img_pil = Image.open(file_recto.stream).convert("RGB")
        img_np = np.array(img_pil)

        # Étape 1 : prédire le type
        doc_type, confidence = predict_type(img_np)

        #si la confiance est inferieure à 60% ou si le type nest pas bon, on ne fait pas l'extraction
        if(doc_type!="others" and confidence > 0.6):

            print("bonne qualité")

            # Étape 2 : OCR
            ocr_texts = extract_ocr_text(file_recto, file_verso, reader=reader)

            # Infos utilisateur
            infos = {
                "nom": json_data.get("nom", ""),
                "prenom": json_data.get("prenom", ""),
                "numero": json_data.get("numero", ""),
                "lieu de naissance": json_data.get("lieu_de_naisance", ""),
                "date de naissance": json_data.get("date_de_naissance", "")
            }

            # Étape 3 : Vérification
            verification, date_expiration = verifier_informations(ocr_texts, infos)

            # Étape 3 : calcul score global
            score_global = calculer_score_global(verification)
            interpretation = interpret_score(score_global)

            return jsonify({
                "type_document": doc_type,
                "confiance": round(confidence, 3),
                "score_global": score_global,
                "interpretation": interpretation,
                "date_expiration": date_expiration,
                "verification_informations": verification
            })
        elif(doc_type=="others"):
            print("cest un others")
            return jsonify({
                "type_document": doc_type,
                "confiance": round(confidence, 3),
                "score_global": -1
            })
        else:
            print("mauvais")
            return jsonify({
                "type_document": doc_type,
                "confiance": round(confidence, 3),
                "score_global": -2
            })
        
    except Exception as e:
        print("Erreur interne :", traceback.format_exc())
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
