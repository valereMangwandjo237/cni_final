import streamlit as st
import numpy as np
from PIL import Image
import io

def display_score(score):
    # Définition des couleurs en fonction du score
    if score >= 90:
        color = '#2E7D32'  # Vert foncé - Excellence
    elif score >= 75:
        color = '#43A047'  # Vert clair - Très bon
    elif score >= 60:
        color = '#FFD600'  # Jaune vif - À vérifier
    elif score >= 50:
        color = '#FB8C00'  # Orange - Risque modéré
    else:
        color = '#C62828'  # Rouge - Danger

    return color


def compress_image(uploaded_file, max_size=(1000, 1000), quality=85):
    img = Image.open(uploaded_file).convert("RGB")# lire limage
    img.thumbnail(max_size)
    buffer = io.BytesIO()#creer une espace en RAM
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return buffer

def getType(name, confiance):
    if confiance >= 0.75:
        if name == "others":
            return "BLOQUE!!"
        if name != "others": return "VALIDE!!"
    else:
        return "A VERIFIER AU POOL"