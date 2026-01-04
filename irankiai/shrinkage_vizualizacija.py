# ============================================================================
# EMPIRICAL BAYES SHRINKAGE VIZUALIZACIJA
# ============================================================================
# Autorius: Magistro darbas
# Aprašymas: Universali funkcija Empirical Bayes Shrinkage efekto vizualizavimui
#            Palaiko Random Forest ir XGBoost modelius
# ============================================================================

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D


def plot_shrinkage_effect(imputer, model_name="Random Forest"):
    """
    Empirical Bayes Shrinkage grafikas - Raw vs Shrunken stilius.
    Parodo kaip shrinkage sumažina dispersiją (variance).

    UNIVERSALI VERSIJA: Palaiko abu raktų formatus:
    - Random Forest: 'y_ML', 'final_pred', 'y_region_mean', 'lambda'
    - XGBoost_old: 'original_pred', 'adjusted_pred', 'geo_mean', 'shrinkage_weight'

    Parametrai:
    -----------
    imputer : RandomForestImputer arba XGBoostImputer
        Modelio objektas su get_shrinkage_report() metodu
    model_name : str
        Modelio pavadinimas grafikui (default: "Random Forest")
    """
    shrinkage_data = imputer.get_shrinkage_report()

    if not shrinkage_data:
        print(f"Nėra shrinkage duomenų {model_name} modeliui.")
        return

    # Surenkame duomenis (UNIVERSALUS - palaiko abu formatus)
    all_original = []
    all_adjusted = []
    all_means = []
    all_lambdas = []

    for col, adjustments in shrinkage_data.items():
        for adj in adjustments:
            # Originali prognozė (RF: y_ML, XGB: original_pred)
            orig = adj.get('y_ML', adj.get('original_pred', 0))
            # Galutinė prognozė (RF: final_pred, XGB: adjusted_pred)
            final = adj.get('final_pred', adj.get('adjusted_pred', 0))
            # Regioninis vidurkis (RF: y_region_mean, XGB: geo_mean)
            mean = adj.get('y_region_mean', adj.get('geo_mean', 0))
            # Lambda\shrinkage svoris (RF: lambda, XGB: shrinkage_weight)
            lam = adj.get('lambda', adj.get('shrinkage_weight', 0))

            all_original.append(orig)
            all_adjusted.append(final)
            all_means.append(mean)
            all_lambdas.append(lam)

    if len(all_original) < 10:
        print(f"Per mažai duomenų {model_name} modeliui ({len(all_original)} įrašų).")
        return

    all_original = np.array(all_original)
    all_adjusted = np.array(all_adjusted)
    all_means = np.array(all_means)
    all_lambdas = np.array(all_lambdas)

    # X ašis = regioninis vidurkis (true value / target)
    # Y ašis = prognozė (raw arba shrunken)
    x_values = all_means
    y_raw = all_original
    y_shrunken = all_adjusted

    # Filtruojame outlierius (99 percentile) ir neigiamas reikšmes
    valid_mask = (x_values > 0) & (y_raw > 0) & (y_shrunken > 0)
    if valid_mask.sum() > 0:
        percentile_99 = np.percentile(np.abs(y_raw[valid_mask]), 99)
        mask = valid_mask & (np.abs(y_raw) < percentile_99)
    else:
        mask = valid_mask

    x_plot = x_values[mask]
    y_raw_plot = y_raw[mask]
    y_shrunken_plot = y_shrunken[mask]
    lambdas_plot = all_lambdas[mask]

    if len(x_plot) == 0:
        print(f"Nėra validžių duomenų po filtravimo {model_name} modeliui.")
        print(f"  - Pradinių įrašų: {len(all_original)}")
        print(f"  - Po filtravimo: {mask.sum()}")
        return

    # Sukuriame grafiką - shrinkage.png stilius
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Bendri parametrai
    alpha = 0.5
    size = 15
    point_color = '#6699CC'  # Šviesesnė mėlyna

    # Apskaičiuojame bendrą skalę
    min_val = min(x_plot.min(), y_raw_plot.min(), y_shrunken_plot.min())
    max_val = max(x_plot.max(), y_raw_plot.max(), y_shrunken_plot.max())

    # === KAIRĖ: Raw (prieš shrinkage) ===
    ax1 = axes[0]
    ax1.scatter(x_plot, y_raw_plot, c=point_color, s=size, alpha=alpha, edgecolors='none')

    # y = x linija (raudona, ryškesnė)
    ax1.plot([min_val, max_val], [min_val, max_val], color='#C44E52', linewidth=1.5, label='Ideali prognozė (y = x)')

    # Linear fit (pilka punktyrinė)
    slope, intercept, r, p, se = stats.linregress(x_plot, y_raw_plot)
    fit_line = slope * np.array([min_val, max_val]) + intercept
    ax1.plot([min_val, max_val], fit_line, color='#555555', linestyle='--', linewidth=1.5, label='Tiesinė regresija')

    ax1.set_xlabel('Regioninis vidurkis', fontsize=11)
    ax1.set_ylabel('Prognozė', fontsize=11)
    ax1.set_title('Prieš Shrinkage (Raw)', fontsize=12)
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.ticklabel_format(style='scientific', axis='both', scilimits=(6,6))

    # === DEŠINĖ: Shrunken (po shrinkage) ===
    ax2 = axes[1]
    ax2.scatter(x_plot, y_shrunken_plot, c=point_color, s=size, alpha=alpha, edgecolors='none')

    # y = x linija (raudona)
    ax2.plot([min_val, max_val], [min_val, max_val], color='#C44E52', linewidth=1.5)

    # Linear fit (pilka punktyrinė)
    slope2, intercept2, r2, p2, se2 = stats.linregress(x_plot, y_shrunken_plot)
    fit_line2 = slope2 * np.array([min_val, max_val]) + intercept2
    ax2.plot([min_val, max_val], fit_line2, color='#555555', linestyle='--', linewidth=1.5)

    ax2.set_xlabel('Regioninis vidurkis', fontsize=11)
    ax2.set_ylabel('Prognozė', fontsize=11)
    ax2.set_title('Po Shrinkage (Shrunken)', fontsize=12)
    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.ticklabel_format(style='scientific', axis='both', scilimits=(6,6))

    # Bendra legenda žemiau grafiko
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=point_color,
               markersize=8, alpha=0.7, label='Imputuotos prognozės'),
        Line2D([0], [0], color='#C44E52', linewidth=1.5, linestyle='-',
               label='Ideali prognozė (y = x)'),
        Line2D([0], [0], color='#555555', linewidth=1.5, linestyle='--',
               label='Tiesinė regresija')
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=10)

    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    plt.show()

    # Statistika
    var_raw = np.var(y_raw_plot - x_plot)
    var_shrunken = np.var(y_shrunken_plot - x_plot)
    var_reduction = (1 - var_shrunken / var_raw) * 100 if var_raw > 0 else 0

    print(f"\n{model_name} - Empirical Bayes Shrinkage efektas:")
    print(f"  Prognozių skaičius: {len(x_plot)}")
    print(f"  Dispersija (variance) PRIEŠ: {var_raw:.2f}")
    print(f"  Dispersija (variance) PO: {var_shrunken:.2f}")
    print(f"  Dispersijos sumažinimas: {var_reduction:.1f}%")
    print(f"  R² PRIEŠ: {r**2:.4f}")
    print(f"  R² PO: {r2**2:.4f}")
    print(f"\n  Lambda (λ) \ Shrinkage weight statistika:")
    print(f"    - Vidurkis: {np.mean(lambdas_plot):.4f} ({np.mean(lambdas_plot)*100:.2f}%)")
    print(f"    - Min:      {np.min(lambdas_plot):.4f} ({np.min(lambdas_plot)*100:.2f}%)")
    print(f"    - Max:      {np.max(lambdas_plot):.4f} ({np.max(lambdas_plot)*100:.2f}%)")
