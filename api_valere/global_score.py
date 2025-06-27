import numpy as np

def calculer_score_global(verification_infos):
    # Poids personnalisables par champ (en %)
    poids = {
        "nom": 30,
        "prenom": 15,
        "numero": 30,
        "date de naissance": 15,
        "lieu de naissance": 10 
    }
    
    # Normalisation des poids à 1
    total_poids = sum(poids.values())
    poids_normalises = {k: v/total_poids for k, v in poids.items()}
    
    # Calcul pondéré
    scores = []
    for champ, infos in verification_infos.items():
        score_pondere = infos["score"] * poids_normalises[champ]
        scores.append(score_pondere)
    
    # Score global (0-100)
    score_final = np.sum(scores)
    
    # Ajustement non-linéaire (pénalise les faibles scores)
    if any(infos["score"] < 50 for infos in verification_infos.values()):
        score_final *= 0.8  # Pénalité de 20%
    
    return min(100, max(0, round(score_final, 2)))  # Borné à 0-100


def interpret_score(score):
    if score >= 90:
        return "Correspondance très forte"
    elif score >= 75:
        return "Correspondance probable"
    elif score >= 60:
        return "Correspondance possible (vérification recommandée)"
    else:
        return "Risque d'erreur élevé"