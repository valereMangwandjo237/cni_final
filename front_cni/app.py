import streamlit as st
import requests
import json
from datetime import datetime
from function_stream import *

# Configuration de la page
st.set_page_config(layout="wide")
st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #00FFFF ; font-size: 48px; margin: 0;">
            Vérification d'identité par CNI
        </h1>
    </div>


""", unsafe_allow_html=True)

# URL de votre API Flask (à adapter)
API_URL = "http://127.0.0.1:5000/analyser"

# Layout en 2 colonnes
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Informations personnelles")
    
    # Formulaire
    with st.form("identity_form"):
        nom = st.text_input("Nom *", help="Votre nom de famille", key="nom").strip()
        prenom = st.text_input("Prénom (optionnel)", help="Votre prénom", key="prenom").strip()
        lieu_naissance = st.text_input("Lieu de naissance *", help="Ville de naissance", key="lieu_naissance").strip()
        numero_cni = st.text_input("Numéro de CNI *", help="Numéro figurant sur la CNI", key="numero_cni").strip()
        date_naissance = st.date_input("Date de naissance *", min_value=datetime(1900,1,1), key="date_naissance")
        
        st.markdown("**Photos de la CNI (recto et verso)**")
        recto = st.file_uploader("Recto CNI (obligatoire) *", type=['jpg', 'png', 'jpeg'], key="recto")
        verso = st.file_uploader("Verso CNI (obligatoire) *", type=['jpg', 'png', 'jpeg'], key="verso")
        
        submitted = st.form_submit_button("Valider")

with col2:
    st.header("Vérification des documents")
    
    
    if submitted:
        # Validation des champs obligatoires
        errors = []
        if not nom: errors.append("Le nom est obligatoire")
        if not lieu_naissance: errors.append("Le lieu de naissance est obligatoire")
        if not numero_cni: errors.append("Le numéro de CNI est obligatoire")
        if not recto or not verso: errors.append("Les deux photos de la CNI sont obligatoires")
        
        if errors:
            for error in errors:
                st.error(error)
        else:
            with st.spinner("Analyse en cours..."):
                try:
                        # Préparation des données pour l'API
                    data = {
                        "nom": nom,
                        "prenom": prenom or "",  # Champ optionnel
                        "lieu_de_naisance": lieu_naissance,
                        "numero": numero_cni,
                        "date_de_naissance": date_naissance.strftime("%d.%m.%Y") 
                    }
                    
                    # Envoi à l'API Flask
                    recto_compressed = compress_image(recto)
                    verso_compressed = compress_image(verso)
                    files = {
                        'recto': (recto.name, recto_compressed, 'image/jpeg'),
                        'verso': (verso.name, verso_compressed, 'image/jpeg'),
                    }

                    response = requests.post(
                        API_URL,
                        files=files,
                        data={'data': json.dumps(data)}
                    )

                        # Traitement de la réponse
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Affichage des résultats
                        st.success("Analyse terminée avec succès!")

                        st.image(recto, caption="Recto CNI", width=300)
                        
                        st.markdown(f"""
                        <div class="result-box">
                            <h3>Type de document</h3>
                            <p><strong style="color: #ff4d4d; font-size: 1.4em;">{result.get('type_document', 'Inconnu')}</strong> (confiance: {result.get('confiance', 0)*100:.1f}%)</p>
                            
                            --------Vérification--------
                            <ul>
                        """, unsafe_allow_html=True)

                        # Création de la barre avec texte intégré
                        global_score = result.get("score_global")

                        if(global_score != -1 and global_score != -2):

                            st.header(f"Identité similaire à {global_score}%")


                            bar_color = display_score(global_score)

                            # HTML + CSS personnalisé
                            st.markdown(f"""
                            <div style="width: 100%; background-color: #ddd; border-radius: 8px;">
                            <div style="
                            width: {global_score}%;
                            background-color: {bar_color};
                            height: 20px;
                            border-radius: 8px;">
                            </div>
                            </div>
                            <p style="text-align: center;">Score : {global_score}%</p>
                            """, unsafe_allow_html=True)

                            for field, details in result.get('verification_informations', {}).items():
                                # Icône de validation
                                icon = "✅" if details.get('valide', False) else "❌"
                                
                                # Score arrondi
                                score = round(details.get('score', 0), 1)
                                
                                # Valeur OCR trouvée
                                valeur_ocr = details.get('valeur_ocr', 'Non détecté')
                                
                                # Affichage enrichi
                                st.markdown(f"""
                                <div style="margin-bottom: 10px;">
                                    <b>{field.capitalize()}</b> {icon}
                                    <div style="margin-left: 20px; color: #666;">
                                        Score: {score}%<br>
                                        Valeur détectée: <span style="color: white;">{valeur_ocr}</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            st.markdown("</ul></div>", unsafe_allow_html=True)
                        elif(global_score == -1):
                            st.error("Votre image semble ne pas respecter le bon type. Veillez inserer la bonne image et plus claire")
                        else:
                            st.warning("Profil à vérifier au POOL..........!")
                    else:
                        st.error(f"Erreur lors de l'analyse (code {response.status_code})")
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"Erreur de connexion à l'API : {str(e)}")
                except Exception as e:
                    st.error(f"Erreur inattendue : {str(e)}")