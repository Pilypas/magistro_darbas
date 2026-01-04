# ======================================================================
# missingness_tools.py
# Įrankiai trūkstamų duomenų analizei:
# - Missingness Matrix (su geo ir year)
# - Little MCAR Testas (tik skaitiniams su NA)
# ======================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
from matplotlib.patches import Patch
from scipy.stats import chi2 # https://medium.com/@itk48/missing-data-imputation-with-chi-square-tests-mcar-mar-3278956387c8

# ======================================================================
# 1. Missingness Matrix (su geo ir year)
# ======================================================================

def plot_missing_matrix(df):
    """
    Nupiešia trūkstamų duomenų matricą.
    RODOMI VISI STULPELIAI, įskaitant 'geo' ir 'year'.
    """

    plt.figure(figsize=(18, 7))

    msno.matrix(
        df,
        labels=False,
        sparkline=False,
        color=(0.0, 0.0, 0.0)    # juoda = yra reikšmė
    )

    plt.gca().set_yticklabels([])
    plt.gca().set_ylabel("")
    plt.gca().set_xlabel("")

    legend_elements = [
        Patch(facecolor="black", edgecolor="black", label="Yra reikšmė"),
        Patch(facecolor="white", edgecolor="black", label="Trūksta reikšmės (NaN)")
    ]

    plt.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        fontsize=23,
        frameon=False,
        bbox_to_anchor=(0.5, -0.18)
    )

    plt.tight_layout()
    plt.show()


# ======================================================================
# 2. Little MCAR testas (tik skaitiniams stulpeliams su galimais NA)
# ======================================================================

def little_mcar_test(X):
    """
    Atlieka Little (1988) MCAR testą.
    Grąžina: chi-square, df ir p-value.
    """

    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X)
    else:
        X_df = X.copy()

    # Naudojami tik skaitiniai stulpeliai
    X_df = (
    X_df.select_dtypes(include=[np.number])
        .loc[:, X_df.isna().any()]   # paliekame tik stulpelius su NA
)
    print(f"Naudojamų stulpelių MCAR testui skaičius: {X_df.shape[1]}")
    #print("Stulpeliai naudojami MCAR testui:")
    #print(list(X_df.columns))

    var_names = X_df.columns
    n_vars = X_df.shape[1]

    # Globalūs vidurkiai ir kovariacinė matrica
    mu_hat = X_df.mean()
    sigma_hat = X_df.cov()

    # Missingness pattern'ai
    r = X_df.isna().astype(int)
    pattern_codes = r.values.dot(2 ** np.arange(n_vars))
    unique_codes, pattern_index = np.unique(pattern_codes, return_inverse=True)

    X_aug = X_df.copy()
    X_aug["_pattern_id"] = pattern_index

    d2 = 0.0
    pj = 0

    # Skaičiuojame pagal unikalius pattern'us
    for pid in range(len(unique_codes)):
        block = X_aug[X_aug["_pattern_id"] == pid].loc[:, var_names]

        observed_cols = block.columns[~block.isna().any(axis=0)]
        if len(observed_cols) == 0:
            continue

        pj += len(observed_cols)

        mean_diff = block[observed_cols].mean() - mu_hat[observed_cols]
        cov_sub = sigma_hat.loc[observed_cols, observed_cols].values

        # Inversija arba pseudo-inversija
        try:
            inv_cov_sub = np.linalg.inv(cov_sub)
        except np.linalg.LinAlgError:
            inv_cov_sub = np.linalg.pinv(cov_sub)

        m_j = len(block)
        d2 += m_j * float(mean_diff.values.T @ inv_cov_sub @ mean_diff.values)

    df_test = pj - n_vars
    p_value = chi2.sf(d2, df_test)

    return d2, df_test, p_value


# ======================================================================
# 3. MCAR rezultatų išvedimas
# ======================================================================

def print_little_mcar_results(X):
    """
    Atliekame MCAR testą ir išspausdiname rezultatus.
    """
    chi2_val, df_val, p_val = little_mcar_test(X)

    print("\n===================== LITTLE MCAR TESTO REZULTATAI =====================")
    print(f"Chi-kvadrato statistika (angl. chi-square statistic) : {chi2_val:.3f}")
    print(f"Laisvės laipsniai:     {df_val}")
    print(f"P-vertė:               {p_val:.3f}")

    if p_val < 0.05:
        print("\nIŠVADA: MCAR HIPOTEZĖ ATMESTA → duomenys NĖRA MCAR.")
        print("Tikėtina MAR struktūra.")
    else:
        print("\nIŠVADA: MCAR hipotezė NEATMETAMA → trūkstamumas gali būti MCAR.")

    print("========================================================================\n")
