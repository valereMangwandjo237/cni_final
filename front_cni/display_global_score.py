import streamlit as st
import numpy as np

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