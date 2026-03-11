import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def evaluate_predictions(gt_csv, pred_csv, vid="V1", cam="C1"):
    # Chargement des données
    df_gt = pd.read_csv(gt_csv)
    df_pred = pd.read_csv(pred_csv)
    
    # On filtre uniquement sur les piétons (classe 0 d'après le class_mapping)
    df_pred = df_pred[df_pred['Class'] == 0]
    
    # Sélectionner les colonnes de distance du véhicule/caméra cible (ex: Dist_V1C1_P1)
    gt_cols = [c for c in df_gt.columns if c.startswith(f"Dist_{vid}{cam}_")]
    
    matched_gts = []
    matched_preds = []
    
    for frame in df_pred['Frame'].unique():
        # Distances estimées par la stéréovision à cette frame
        preds = df_pred[df_pred['Frame'] == frame]['Dist_Est'].values
        
        # Distances réelles (ground truth) à cette frame
        gt_row = df_gt[df_gt['Frame'] == frame]
        if gt_row.empty:
            continue
            
        gts = gt_row[gt_cols].values[0]
        gts = gts[~np.isnan(gts)]  # Éliminer d'éventuels NaN
        
        if len(preds) == 0 or len(gts) == 0:
            continue
            
        # Matrice de coût pour trouver la meilleure association objet_détecté <-> objet_réel
        cost_matrix = np.abs(preds[:, None] - gts[None, :])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            err = cost_matrix[r, c]
            # On conserve l'association uniquement si l'erreur est physiquement cohérente (< 10 mètres)
            if err < 10.0:
                matched_preds.append(preds[r])
                matched_gts.append(gts[c])

    if not matched_preds:
        print("Aucune association valide n'a été trouvée entre les prédictions et la vérité terrain.")
        return
        
    matched_gts = np.array(matched_gts)
    matched_preds = np.array(matched_preds)
    
    # --- Calcul des Métriques ---
    errors = matched_preds - matched_gts  # Erreur signée (pour le biais)
    abs_errors = np.abs(errors)           # Erreur absolue
    
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors**2))
    bias = np.mean(errors)                # Moyenne des erreurs signées
    std_err = np.std(errors)              # Écart-type des erreurs
    mape = np.mean(abs_errors / matched_gts) * 100  # Erreur relative moyenne (%)
    
    acc_1m = np.mean(abs_errors <= 1.0) * 100
    acc_2m = np.mean(abs_errors <= 2.0) * 100
    
    # --- Affichage des métriques dans la console ---
    print(f"\n{'='*55}")
    print(f" STATISTIQUES D'ÉVALUATION (Véhicule {vid}, Caméra {cam})")
    print(f"{'='*55}")
    print(f"Échantillons valides appariés : {len(matched_preds)}")
    print(f"Erreur Absolue Moyenne (MAE)  : {mae:.3f} mètres")
    print(f"Racine Erreur Quadratique (RMSE): {rmse:.3f} mètres")
    print(f"Erreur Relative Moyenne (MAPE): {mape:.2f} %")
    print("-" * 55)
    print(f"Biais (Mean Error)            : {bias:.3f} mètres (>0 = surestimation)")
    print(f"Écart-type de l'erreur (Std)  : {std_err:.3f} mètres")
    print(f"Erreur Min / Max              : {np.min(abs_errors):.3f} m / {np.max(abs_errors):.3f} m")
    print("-" * 55)
    print(f"Précision (Erreur ≤ 1m)       : {acc_1m:.1f} %")
    print(f"Précision (Erreur ≤ 2m)       : {acc_2m:.1f} %")
    print(f"{'='*55}\n")
    
    # --- Visualisations Graphiques ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Histogramme des erreurs
    axs[0].hist(abs_errors, bins=25, color='royalblue', edgecolor='black', alpha=0.8)
    axs[0].axvline(mae, color='red', linestyle='dashed', linewidth=2, label=f'MAE: {mae:.2f}m')
    axs[0].set_title("Distribution des erreurs absolues")
    axs[0].set_xlabel("Erreur absolue (en mètres)")
    axs[0].set_ylabel("Fréquence")
    axs[0].legend()
    axs[0].grid(axis='y', alpha=0.4)
    
    # 2. Scatter plot : Prédictions vs Ground Truth
    # Ligne idéale y = x
    min_val = min(np.min(matched_gts), np.min(matched_preds))
    max_val = max(np.max(matched_gts), np.max(matched_preds))
    axs[1].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Idéal (y=x)')
    axs[1].scatter(matched_gts, matched_preds, color='teal', alpha=0.6, edgecolors='k')
    axs[1].set_title("Distance Estimée vs Vérité Terrain")
    axs[1].set_xlabel("Vérité Terrain (mètres)")
    axs[1].set_ylabel("Distance Estimée (mètres)")
    axs[1].legend()
    axs[1].grid(True, alpha=0.4)
    
    # 3. Erreur en fonction de la distance
    axs[2].scatter(matched_gts, abs_errors, color='darkorange', alpha=0.6, edgecolors='k')
    axs[2].axhline(mae, color='red', linestyle='dashed', linewidth=2, label=f'MAE Globale: {mae:.2f}m')
    axs[2].set_title("Évolution de l'erreur selon la distance")
    axs[2].set_xlabel("Vérité Terrain (mètres)")
    axs[2].set_ylabel("Erreur absolue (mètres)")
    axs[2].legend()
    axs[2].grid(True, alpha=0.4)
    
    plt.suptitle(f"Évaluation des performances Stéréo - {vid}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    fichier_csv_terrain = "Data/Simulation_4/distances_scenario_v4.csv"
    fichier_csv_predictions1 = "runs/predictions_V1.csv"
    fichier_csv_predictions2 = "runs/predictions_V2.csv"
    
    evaluate_predictions(fichier_csv_terrain, fichier_csv_predictions1, vid="V1")
    evaluate_predictions(fichier_csv_terrain, fichier_csv_predictions2, vid="V2")