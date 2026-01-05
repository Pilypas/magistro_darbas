"""
================================================================================
RANDOM FOREST IMPUTAVIMO MODELIS
================================================================================
Ekonominių rodiklių trūkstamų reikšmių užpildymas naudojant Random Forest algoritmą.

NAUDOJIMAS:
-----------
- Flask aplikacija: fit_and_impute(), get_model_metrics(), get_feature_importance()
- Jupyter Notebook: visi metodai + print_*() funkcijos analizei ir vizualizacijai

PAGRINDINĖS TAISYKLĖS, KURIAS TURI KODAS:
----------------------
1. Struktūriniai nuliai (0) NIEKADA NEIMPUTUOJAMI - imputuojamos tik NaN reikšmės.
2. 0 reikšmės naudojamos kaip TRAIN duomenys (reali informacija), bet NE kaip TEST.
3. Kategoriniai prediktoriai ('geo', 'year') BE trūkumų - tik enkoduojami.
4. Synthetic test be leakage (20% TEST be 0 reikšmių).
5. Prediktorių imputacija (mean) ignoruoja 0 reikšmes.
6. Naudojama SMAPE metrika (veikia su 0 reikšmėmis).

KODO STRUKTŪRA:
---------------
1. IMPORTAI IR PRIKLAUSOMYBĖS
2. PAGALBINĖ KLASĖ: ZeroIgnoringImputer
3. PAGRINDINĖ KLASĖ: RandomForestImputer
   3.1. Inicializacija (__init__)
   3.2. PUBLIC API (Flask + Jupyter Notebook)
   3.3. Duomenų paruošimas
   3.4. Geo statistikų skaičiavimas (angl. Feature Engineering)
   3.5. Stulpelio imputavimas
   3.6. Train/Test padalijimas
   3.7. Modelio treniravimas
   3.8. Feature transformacijos
   3.9. Galutinė imputacija su post-processing (Empirical Bayes Shrinkage metodas)
   3.10. Metrikų skaičiavimas ir saugojimas
================================================================================
"""

# ==============================================================================
# 1. IMPORTAI IR PRIKLAUSOMYBĖS
# ==============================================================================
# Šie importai naudojami tiek Flask aplikacijoje, tiek Jupyter Notebook'e

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from scipy.stats import randint, uniform


# ==============================================================================
# 2. PAGALBINĖ KLASĖ: ZeroIgnoringImputer
# ==============================================================================
# Naudojama: vidiniams skaičiavimams (prediktorių NaN užpildymui)
# Paskirtis: Apskaičiuoja vidurkį ignoruojant 0 ir NaN reikšmes

class ZeroIgnoringImputer:
    """
    Custom imputer, kuris ignoruoja 0 reikšmes skaičiuojant mean.

    Struktūriniai 0 neturėtų įtakoti prediktorių imputacijos,
    nes jie dažnai reiškia "nėra duomenų" arba "neaktualu".

    Metodai:
        fit(X): Apskaičiuoja mean ignoruojant NaN ir 0
        transform(X): Užpildo NaN su apskaičiuotais mean
        fit_transform(X): Fit ir transform vienu žingsniu
    """

    def __init__(self):
        self.statistics_ = None

    # -------------------------------------------------------------------------
    # 2.1. Pagrindiniai metodai
    # -------------------------------------------------------------------------

    def fit(self, X):
        """Apskaičiuoja mean ignoruojant NaN ir 0 reikšmes."""
        X_array = self._to_array(X)
        self.statistics_ = self._compute_nonzero_means(X_array)
        return self

    def transform(self, X):
        """Užpildo NaN reikšmes su apskaičiuotais mean."""
        X_array = self._to_array(X)
        return self._impute_nans(X_array)

    def fit_transform(self, X):
        """Fit ir transform vienu žingsniu."""
        return self.fit(X).transform(X)

    # -------------------------------------------------------------------------
    # 2.2. Vidiniai pagalbiniai metodai
    # -------------------------------------------------------------------------

    @staticmethod
    def _to_array(X):
        """Konvertuoja DataFrame į numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values.copy()
        return np.array(X, copy=True)

    def _compute_nonzero_means(self, X_array):
        """Apskaičiuoja mean kiekvienam stulpeliui ignoruojant 0 ir NaN."""
        means = []
        for col_idx in range(X_array.shape[1]):
            col = X_array[:, col_idx]
            valid_mask = ~np.isnan(col) & (col != 0)

            if np.any(valid_mask):
                means.append(np.mean(col[valid_mask]))
            else:
                # Jei visos reikšmės NaN arba 0, naudojame 0 kaip default
                means.append(0.0)

        return np.array(means)

    def _impute_nans(self, X_array):
        """Užpildo NaN reikšmes su mean."""
        for col_idx in range(X_array.shape[1]):
            nan_mask = np.isnan(X_array[:, col_idx])
            X_array[nan_mask, col_idx] = self.statistics_[col_idx]
        return X_array


# ==============================================================================
# 3. PAGRINDINĖ KLASĖ: RandomForestImputer
# ==============================================================================
# Naudojama: Flask aplikacijoje ir Jupyter Notebook'e
# Paskirtis: Trūkstamų ekonominių rodiklių reikšmių užpildymas

class RandomForestImputer:
    """
    Random Forest imputavimas ekonominiams rodikliams.

    Kiekvienam stulpeliui su trūkstamomis reikšmėmis treniruojamas atskiras RF modelis.
    Synthetic test: 20% indeksų -> TEST (tik iš eilučių, kur target != 0).
    Po vertinimo pertreniruojama ant 100% žinomų taikinių.

    Naudojimas Flask aplikacijoje:
        imputer = RandomForestImputer()
        df_imputed = imputer.fit_and_impute(df)
        metrics = imputer.get_model_metrics()

    Naudojimas Jupyter Notebook'e:
        imputer = RandomForestImputer(cv_folds=5)
        df_imputed = imputer.fit_and_impute(df)
        imputer.print_cv_results()  # Išsami analizė
    """

    # ==========================================================================
    # 3.1. INICIALIZACIJA
    # ==========================================================================
    # Naudojama: Flask + Jupyter Notebook
    # Paskirtis: Nustatyti modelio hiperparametrus ir inicializuoti kintamuosius

    def __init__(
        self,
        n_estimators=100,
        random_state=42,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        categorical_cols=None,
        exclude_columns=None,
        cv_folds=2,
        use_hyperopt=False,
        hyperopt_n_iter=30,
        hyperopt_cv=3,
        use_post_processing=True,
        #shrinkage_k=3.0
    ):
        """
        Inicializuoja Random Forest imputer.

        Args:
            n_estimators: Medžių skaičius Random Forest modelyje
            random_state: Atsitiktinių skaičių generatoriaus seed
            max_depth: Maksimalus medžio gylis
            min_samples_split: Minimalus pavyzdžių skaičius padalijimui
            min_samples_leaf: Minimalus pavyzdžių skaičius lape
            categorical_cols: Kategorinių stulpelių sąrašas
            exclude_columns: Stulpeliai, kurie nebus imputuojami (galima nurodyti jupyter tyrimo kode)
            cv_folds: Kryžminės validacijos fold'ų skaičius
            use_hyperopt: Ar naudoti hiperparametrų optimizavimą (galima nurodyti jupyter tyrimo kode)
            hyperopt_n_iter: Hiperparametrų paieškos iteracijų skaičius
            hyperopt_cv: CV folds hiperparametrų paieškai
            use_post_processing: Ar naudoti post-processing (Empirical Bayes Shrinkage)
        """
        # ----- Baziniai Random Forest parametrai -----
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        # ----- Duomenų konfigūracija -----
        self.categorical_cols = categorical_cols or ['geo', 'year']
        self.exclude_columns = exclude_columns or []

        # ----- Kryžminės validacijos parametrai -----
        self.cv_folds = cv_folds

        # ----- Hiperparametrų optimizavimo nustatymai -----
        self.use_hyperopt = use_hyperopt
        self.hyperopt_n_iter = hyperopt_n_iter
        self.hyperopt_cv = hyperopt_cv

        # ----- Post-processing parametrai (Empirical Bayes Shrinkage) -----
        self.use_post_processing = use_post_processing
        

        # ----- Rezultatų saugojimas -----
        self.models = {}                # Ištreniruoti modeliai kiekvienam stulpeliui
        self.feature_importance = {}    # Feature svarbumo koeficientai
        self.model_metrics = {}         # Test set metrikos (R², nRMSE, nMAE, sMAPE)
        self.test_predictions = {}      # Test predikcijos (Excel failų generavimui)
        self.cv_scores = {}             # Kryžminės validacijos rezultatai
        self.best_params = {}           # Geriausi hiperparametrai (jei hyperopt įjungtas)
        self.geo_stats = {}             # Regiono statistikos kiekvienam rodikliui
        self.shrinkage_applied = {}     # Post-processing informacija

    # ==========================================================================
    # 3.2. PUBLIC API
    # ==========================================================================
    # Šie metodai naudojami Flask aplikacijoje ir Jupyter imputavimui ir rezultatų atvaizdavimui

    def fit_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pagrindinis metodas: treniruoja modelius ir imputuoja trūkstamas reikšmes.

        NAUDOJAMA: Flask aplikacija + Jupyter Notebook

        SVARBU: Naudojame ORIGINALIAS reikšmes kaip features, ne jau imputuotas.
        Tai išvengia "cascade error" problemos, kai vieno stulpelio klaidos
        persiduoda į kitus stulpelius. https://arxiv.org/pdf/1105.0828

        Procesas:
            1. Paruošia DataFrame (kopijuoja, konvertuoja tipus)
            2. Validuoja kategorinius stulpelius
            3. Išsaugo originalias reikšmes features naudojimui
            4. Imputuoja kiekvieną stulpelį su NaN reikšmėmis

        Args:
            df: DataFrame su trūkstamomis reikšmėmis

        Returns:
            DataFrame su imputuotomis reikšmėmis (tik NaN, ne 0)
        """
        # Žingsnis 1: Paruošiame DataFrame
        df_work = self._prepare_dataframe(df)

        # Žingsnis 2: Validuojame kategorinius stulpelius
        self._validate_categorical_columns(df_work)

        # Žingsnis 3: Išsaugome originalias reikšmes features naudojimui
        # Tai užtikrina, kad kiekvienas stulpelis imputuojamas naudojant
        # TIK originalias reikšmes, o ne jau imputuotas iš kitų stulpelių
        self._df_original = df_work.copy()

        # Žingsnis 4: Imputuojame kiekvieną stulpelį su NaN reikšmėmis
        for target_col in df_work.columns:
            if self._should_impute_column(df_work, target_col):
                self._impute_column(df_work, target_col)

        # Žingsnis 5: Atlaisviname atmintį
        del self._df_original

        return df_work

    def get_feature_importance(self):
        """
        Grąžina feature importance kiekvienam stulpeliui.

        NAUDOJAMA: Flask aplikacija + Jupyter Notebook

        Returns:
            dict: {stulpelio_pavadinimas: {feature: importance_value}}
        """
        return self.feature_importance

    def get_model_metrics(self):
        """
        Grąžina modelių metrikos (R², nRMSE, nMAE, sMAPE).

        NAUDOJAMA: Flask aplikacija + Jupyter Notebook

        Returns:
            dict: {stulpelio_pavadinimas: {nrmse, r2, nmae, smape, ...}}
        """
        return self.model_metrics

    def get_test_predictions(self, df=None):
        """
        Grąžina test predikcijas (naudojama Excel failų generavimui).

        NAUDOJAMA: Flask aplikacija + Jupyter Notebook

        Returns:
            dict: {stulpelio_pavadinimas: {y_true, y_pred, test_indices, ...}}
        """
        return self.test_predictions

    def get_cv_scores(self):
        """
        Grąžina cross-validation rezultatus.

        NAUDOJAMA: Flask aplikacija + Jupyter Notebook

        Returns:
            dict: CV rezultatai su vidurkiais ir standartinėmis paklaidomis
        """
        return self.cv_scores

    # ==========================================================================
    # 3.2. PUBLIC API - JUPYTER NOTEBOOK (PAPILDOMI METODAI)
    # ==========================================================================
    # Šie metodai skirti išsamiai analizei ir vizualizacijai Jupyter Notebook'e

    def get_params(self):
        """
        Grąžina modelio hiperparametrus kaip žodyną.

        NAUDOJAMA: Jupyter Notebook (parametrų spausdinimui)

        Returns:
            dict: Visi Random Forest modelio parametrai
        """
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state,
            'cv_folds': self.cv_folds,
            'use_hyperopt': self.use_hyperopt,
            'use_post_processing': self.use_post_processing,   
        }
        if self.use_hyperopt:
            params['hyperopt_n_iter'] = self.hyperopt_n_iter
            params['hyperopt_cv'] = self.hyperopt_cv
        return params

    def get_best_params(self):
        """
        Grąžina geriausius parametrus kiekvienam rodikliui po hiperparametrų optimizavimo.

        NAUDOJAMA: Jupyter Notebook

        Returns:
            dict: Geriausi parametrai pagal rodiklius
        """
        return self.best_params

    def get_best_params_summary(self):
        """
        Grąžina suvestinę apie dažniausiai pasirinktas hiperparametrų reikšmes.

        NAUDOJAMA: Jupyter Notebook

        Returns:
            dict: Statistika apie geriausius parametrus
        """
        if not self.best_params:
            return {}

        summary = {
            'n_estimators': [],
            'max_depth': [],
            'min_samples_split': [],
            'min_samples_leaf': [],
            'max_features': []
        }

        for params in self.best_params.values():
            for key in summary.keys():
                if key in params:
                    summary[key].append(params[key])

        result = {}
        for key, values in summary.items():
            if values:
                if key in ['n_estimators', 'min_samples_split', 'min_samples_leaf']:
                    result[key] = {
                        'min': min(values),
                        'max': max(values),
                        'mean': np.mean(values),
                        'median': np.median(values)
                    }
                elif key == 'max_depth':
                    non_none = [v for v in values if v is not None]
                    result[key] = {
                        'none_count': values.count(None),
                        'min': min(non_none) if non_none else None,
                        'max': max(non_none) if non_none else None,
                        'mean': np.mean(non_none) if non_none else None
                    }
                elif key == 'max_features':
                    from collections import Counter
                    result[key] = dict(Counter(values))

        return result

    def print_params(self):
        """
        Išspausdina modelio parametrus į konsolę.

        NAUDOJAMA: Jupyter Notebook
        """
        params = self.get_params()
        print("Random Forest parametrai:")
        for name, value in params.items():
            print(f"  - {name}: {value}")

        if self.use_hyperopt:
            print("\nHiperparametru optimizavimas IJUNGTAS")
            print(f"  - Iteraciju skaičius: {self.hyperopt_n_iter}")
            print(f"  - CV folds optimizavimui: {self.hyperopt_cv}")

        if self.use_post_processing:
            print("\nEMPIRICAL BAYES SHRINKAGE POST-PROCESSING ĮJUNGTAS:")
            print("  Formules:")
            print("    (23) y_EB = (1 - lambda) * y_ML + lambda * y_region")
            print("    (24) lambda = sigma2_ML / (sigma2_ML + sigma2_region)")
            print("  Kur:")
            print("    - y_ML: RF modelio prognozė (medžių vidurkis)")
            print("    - y_region: regiono istorinis vidurkis (su trend korekcija)")
            print("    - sigma2_ML: RF medžių prognozių dispersija")
            print("    - sigma2_region: regiono istorinių duomenų dispersija")
            print("  Safety bounds:")
            print("    - Ribos ekstremaliu atvejų apsaugai")

    def print_best_params(self):
        """
        Išspausdina geriausius rastus parametrus po hiperparametrų optimizavimo.

        NAUDOJAMA: Jupyter Notebook
        """
        if not self.best_params:
            print("Hiperparametru optimizavimas nebuvo atliktas arba nebaigtas.")
            return

        print("=" * 80)
        print("GERIAUSI RASTI HIPERPARAMETRAI")
        print("=" * 80)

        summary = self.get_best_params_summary()
        if summary:
            print("\nSUVESTINE (per visus rodiklius):")
            print("-" * 50)
            for param, stats in summary.items():
                if param == 'max_features':
                    print(f"  {param}: {stats}")
                elif param == 'max_depth':
                    print(f"  {param}: None kiekis={stats['none_count']}, "
                          f"min={stats['min']}, max={stats['max']}, "
                          f"vidurkis={stats['mean']:.1f}" if stats['mean'] else f"  {param}: visi None")
                else:
                    print(f"  {param}: min={stats['min']}, max={stats['max']}, "
                          f"vidurkis={stats['mean']:.1f}, mediana={stats['median']:.1f}")

        print(f"\nOptimizuotų rodiklių skaičius: {len(self.best_params)}")
        print("=" * 80)

    def get_cv_metrics_df(self):
        """
        Grąžina CV metrikas kaip DataFrame su vidurkiais ir standartinėmis paklaidomis.

        NAUDOJAMA: Jupyter Notebook (lentelių generavimui)

        Returns:
            pd.DataFrame: CV metrikos formatuotos kaip 'vidurkis +/- std'
        """
        if not self.cv_scores:
            return pd.DataFrame()

        metrics_list = []
        for rodiklis, scores in self.cv_scores.items():
            if scores['cv_folds'] > 0:
                metrics_list.append({
                    'Rodiklis': rodiklis,
                    'R²': scores['r2_mean'],
                    'R²_std': scores['r2_std'],
                    'R²_formatted': f"{scores['r2_mean']:.4f} ± {scores['r2_std']:.4f}",
                    'nRMSE': scores['nrmse_mean'],
                    'nRMSE_std': scores['nrmse_std'],
                    'nRMSE_formatted': f"{scores['nrmse_mean']:.4f} ± {scores['nrmse_std']:.4f}",
                    'nMAE': scores['nmae_mean'],
                    'nMAE_std': scores['nmae_std'],
                    'nMAE_formatted': f"{scores['nmae_mean']:.4f} ± {scores['nmae_std']:.4f}",
                    'SMAPE (%)': scores['smape_mean'],
                    'SMAPE_std': scores['smape_std'],
                    'SMAPE_formatted': f"{scores['smape_mean']:.2f} ± {scores['smape_std']:.2f}",
                    'CV_folds': scores['cv_folds'],
                    'Imtis': scores['sample_size']
                })

        return pd.DataFrame(metrics_list)

    def get_cv_summary(self):
        """
        Grąžina bendrą CV suvestinę (vidurkiai per visus rodiklius).

        NAUDOJAMA: Jupyter Notebook

        Returns:
            dict: Bendri CV vidurkiai ir standartinės paklaidos
        """
        if not self.cv_scores:
            return {}

        all_r2 = [s['r2_mean'] for s in self.cv_scores.values()
                  if s['cv_folds'] > 0 and not np.isnan(s.get('r2_mean', float('nan')))]
        all_nrmse = [s['nrmse_mean'] for s in self.cv_scores.values()
                     if s['cv_folds'] > 0 and not np.isnan(s.get('nrmse_mean', float('nan')))]
        all_nmae = [s['nmae_mean'] for s in self.cv_scores.values()
                    if s['cv_folds'] > 0 and not np.isnan(s.get('nmae_mean', float('nan')))]
        all_smape = [s['smape_mean'] for s in self.cv_scores.values()
                     if s['cv_folds'] > 0 and not np.isnan(s.get('smape_mean', float('nan')))]

        return {
            'r2_mean': float(np.mean(all_r2)) if all_r2 else float('nan'),
            'r2_std': float(np.std(all_r2)) if all_r2 else float('nan'),
            'nrmse_mean': float(np.mean(all_nrmse)) if all_nrmse else float('nan'),
            'nrmse_std': float(np.std(all_nrmse)) if all_nrmse else float('nan'),
            'nmae_mean': float(np.mean(all_nmae)) if all_nmae else float('nan'),
            'nmae_std': float(np.std(all_nmae)) if all_nmae else float('nan'),
            'smape_mean': float(np.mean(all_smape)) if all_smape else float('nan'),
            'smape_std': float(np.std(all_smape)) if all_smape else float('nan'),
            'n_indicators': len(all_r2)
        }

    def print_cv_results(self):
        """
        Spausdina CV rezultatus į konsolę (išsami ataskaita).

        NAUDOJAMA: Jupyter Notebook
        """
        cv_df = self.get_cv_metrics_df()
        summary = self.get_cv_summary()

        if cv_df.empty:
            print("CV rezultatų nėra.")
            return

        print("=" * 90)
        print(f"KRYŽMINĖS VALIDACIJOS REZULTATAI ({self.cv_folds}-fold CV)")
        print("=" * 90)

        print("\nBENDRI VIDURKIAI (per visus rodiklius):")
        print("-" * 50)
        print(f"  R²:    {summary['r2_mean']:.4f} ± {summary['r2_std']:.4f}")
        print(f"  nRMSE: {summary['nrmse_mean']:.4f} ± {summary['nrmse_std']:.4f}")
        print(f"  nMAE:  {summary['nmae_mean']:.4f} ± {summary['nmae_std']:.4f}")
        print(f"  sMAPE: {summary['smape_mean']:.2f}% ± {summary['smape_std']:.2f}%")

        print("\nDETALIZUOTI CV REZULTATAI PAGAL RODIKLIUS:")
        print("-" * 90)
        for _, row in cv_df.iterrows():
            print(f"\n{row['Rodiklis']}:")
            print(f"  R²:    {row['R²_formatted']}")
            print(f"  nRMSE: {row['nRMSE_formatted']}")
            print(f"  nMAE:  {row['nMAE_formatted']}")
            print(f"  sMAPE: {row['SMAPE_formatted']}%")
        print("\n" + "=" * 90)

    def get_shrinkage_report(self):
        """
        Grąžina ataskaitą apie pritaikytą Empirical Bayes Shrinkage.

        NAUDOJAMA: Jupyter Notebook

        Returns:
            dict: Informacija apie koreguotas prognozes pagal rodiklius
                  Su σ²_ML, σ²_region, λ reikšmėmis kiekvienai prognozei
        """
        return self.shrinkage_applied

    def print_shrinkage_summary(self):
        """
        Spausdina suvestinę apie Empirical Bayes Shrinkage post-processing.

        Rodome formules (23) ir (24) taikymo rezultatus:
        - Formulė (23): ŷ_EB = (1 - λ) × ŷ_ML + λ × ȳ_region
        - Formulė (24): λ = σ²_ML / (σ²_ML + σ²_region)

        NAUDOJAMA: Jupyter Notebook
        """
        if not self.shrinkage_applied:
            print("Post-processing nebuvo pritaikytas arba nebuvo reikalingas.")
            return

        print("=" * 90)
        print("EMPIRICAL BAYES SHRINKAGE SUVESTINĖ")
        print("Formule (23): y_EB = (1 - lambda) * y_ML + lambda * y_region")
        print("Formule (24): lambda = sigma2_ML / (sigma2_ML + sigma2_region)")
        print("=" * 90)

        total_adjustments = 0
        bounds_applied_count = 0
        all_lambdas = []

        for target_col, adjustments in self.shrinkage_applied.items():
            if adjustments:
                total_adjustments += len(adjustments)
                col_bounds = sum(1 for a in adjustments if a.get('bounds_applied', False))
                bounds_applied_count += col_bounds

                # Lambda statistikos šiam rodikliui
                lambdas = [a.get('lambda', 0) for a in adjustments]
                all_lambdas.extend(lambdas)

                print(f"\n{target_col}:")
                print(f"  Imputuotų reikšmių: {len(adjustments)}")
                print(f"  Lambda (shrinkage koef.) statistika:")
                print(f"    - Vidurkis: {np.mean(lambdas):.4f} ({np.mean(lambdas)*100:.2f}%)")
                print(f"    - Min:      {np.min(lambdas):.4f} ({np.min(lambdas)*100:.2f}%)")
                print(f"    - Max:      {np.max(lambdas):.4f} ({np.max(lambdas)*100:.2f}%)")
                print(f"  Ribos pritaikytos: {col_bounds} atvejams")

                # Rodome kelis pavyzdžius
                examples = adjustments[:3]
                if examples:
                    print("  Pavyzdžiai:")
                    for adj in examples:
                        geo = adj.get('geo', '?')
                        year = adj.get('year', '?')
                        y_ml = adj.get('y_ML', 0)
                        y_eb = adj.get('y_EB', 0)
                        final = adj.get('final_pred', 0)
                        lam = adj.get('lambda', 0)
                        s2_ml = adj.get('sigma2_ML', 0)
                        s2_reg = adj.get('sigma2_region', 0)
                        y_reg = adj.get('year_specific_mean', 0)
                        print(f"    {geo} ({year}):")
                        print(f"      y_ML={y_ml:.2f}, y_region={y_reg:.2f}")
                        print(f"      sigma2_ML={s2_ml:.2f}, sigma2_region={s2_reg:.2f}")
                        print(f"      lambda={lam:.4f} -> y_EB={y_eb:.2f} -> final={final:.2f}")

        print(f"\n" + "-" * 90)
        print(f"BENDRA STATISTIKA:")
        print(f"  Viso imputuotų reikšmių: {total_adjustments}")
        print(f"  Ribos pritaikytos: {bounds_applied_count} atvejams")
        if all_lambdas:
            print(f"  Lambda vidurkis (visi rodikliai): {np.mean(all_lambdas):.4f} ({np.mean(all_lambdas)*100:.2f}%)")
            print(f"  Lambda std:                       {np.std(all_lambdas):.4f}")
        print("=" * 90)

    # ==========================================================================
    # 3.3. DUOMENŲ PARUOŠIMAS (PRIVATE)
    # ==========================================================================
    # Vidiniai metodai duomenų validavimui ir paruošimui

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paruošia DataFrame imputavimui.

        Žingsniai:
            1. Sukuria DataFrame kopiją
            2. Konvertuoja ne-kategorinius 'object' stulpelius į numeric

        Args:
            df: Originalus DataFrame

        Returns:
            pd.DataFrame: Paruoštas DataFrame
        """
        df_work = df.copy()

        # Konvertuojame ne-kategorinius 'object' į numeric
        for col in df_work.columns:
            if (df_work[col].dtype == 'object') and (col not in self.categorical_cols):
                df_work[col] = pd.to_numeric(df_work[col], errors='coerce')

        return df_work

    def _validate_categorical_columns(self, df: pd.DataFrame):
        """
        Patikrina, kad kategoriniai stulpeliai neturi NaN.

        Pagal duomenų struktūrą 'geo' ir 'year' VISADA turi reikšmes.

        Args:
            df: DataFrame validavimui

        Raises:
            ValueError: Jei kategorinis stulpelis turi NaN
        """
        for col in self.categorical_cols:
            if col in df.columns and df[col].isna().any():
                raise ValueError(
                    f"Kategorinis stulpelis '{col}' turi trūkstamų reikšmių, "
                    "pagal duomenis to neturi būti."
                )

    def _should_impute_column(self, df: pd.DataFrame, col: str) -> bool:
        """
        Patikriname, ar stulpelis turėtų būti imputuojamas.

        Stulpelis imputuojamas, jei:
            1. Nėra kategorinis (ne 'geo' ar 'year')
            2. Nėra exclude sąraše
            3. Turi bent vieną NaN reikšmę

        Args:
            df: DataFrame
            col: Stulpelio pavadinimas

        Returns:
            bool: True jei stulpelis turėtų būti imputuojamas
        """
        return (
            col not in self.categorical_cols and
            col not in self.exclude_columns and
            df[col].isna().any()
        )

    # ==========================================================================
    # 3.4. GEO STATISTIKŲ SKAIČIAVIMAS (Feature Engineering)
    # ==========================================================================
    # Regionų statistikos naudojamos kaip papildomi features ir post-processing

    def _compute_geo_stats(self, df: pd.DataFrame, target_col: str) -> dict:
        """
        Apskaičiuoja regiono statistikas target stulpeliui.

        Kiekvienam regionui skaičiuojame:
            - mean: vidurkis (ignoruojant 0 ir NaN)
            - max: maksimumas
            - min: minimumas
            - std: standartinis nuokrypis
            - trend: metinis pokytis (linijinės regresijos koeficientas)

        Args:
            df: DataFrame su duomenimis
            target_col: Stulpelis, kuriam skaičiuojamos statistikos

        Returns:
            dict: {geo_value: {mean, max, min, std, trend}}
        """
        if 'geo' not in df.columns:
            return {}

        geo_stats = {}

        for geo_val in df['geo'].unique():
            geo_data = df[df['geo'] == geo_val][target_col]
            # Filtruojame tik validžias reikšmes (ne NaN ir ne 0)
            valid_data = geo_data[(geo_data.notna()) & (geo_data != 0)]

            if len(valid_data) < 2:
                # Per mažai duomenų - naudojame globalias statistikas
                global_valid = df[target_col][(df[target_col].notna()) & (df[target_col] != 0)]
                if len(global_valid) > 0:
                    geo_stats[geo_val] = {
                        'mean': float(global_valid.mean()),
                        'max': float(global_valid.max()),
                        'min': float(global_valid.min()),
                        'std': float(global_valid.std()) if len(global_valid) > 1 else 0.0,
                        'trend': 0.0
                    }
                else:
                    geo_stats[geo_val] = {
                        'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0, 'trend': 0.0
                    }
                continue

            # Skaičiuojame statistikas
            geo_mean = float(valid_data.mean())
            geo_max = float(valid_data.max())
            geo_min = float(valid_data.min())
            geo_std = float(valid_data.std()) if len(valid_data) > 1 else 0.0

            # Trend: linijinės regresijos koeficientas (year ir value)
            geo_trend = 0.0
            if 'year' in df.columns:
                geo_df = df[(df['geo'] == geo_val) & (df[target_col].notna()) & (df[target_col] != 0)]
                if len(geo_df) >= 2:
                    years = geo_df['year'].values
                    values = geo_df[target_col].values
                    if np.std(years) > 0:
                        geo_trend = float(np.polyfit(years, values, 1)[0])

            geo_stats[geo_val] = {
                'mean': geo_mean,
                'max': geo_max,
                'min': geo_min,
                'std': geo_std,
                'trend': geo_trend
            }

        return geo_stats

    def _add_geo_features(self, df: pd.DataFrame, target_col: str, geo_stats: dict) -> pd.DataFrame:
        """
        Prideda regiono statistikas kaip papildomus features. Čia po shrinkage bandome taikyti

        Pridedami stulpeliai:
            - _geo_mean: regiono vidurkis
            - _geo_max: regiono maksimumas
            - _geo_min: regiono minimumas
            - _geo_std: regiono standartinis nuokrypis
            - _geo_trend: regiono metinis trend

        Args:
            df: DataFrame
            target_col: Target stulpelis
            geo_stats: Apskaičiuotos geo statistikos

        Returns:
            pd.DataFrame: DataFrame su pridėtais geo features
        """
        df_copy = df.copy()

        if 'geo' not in df.columns or not geo_stats:
            return df_copy

        prefix = f"_geo_{target_col}_"
        df_copy[f'{prefix}mean'] = df_copy['geo'].map(lambda x: geo_stats.get(x, {}).get('mean', 0.0))
        df_copy[f'{prefix}max'] = df_copy['geo'].map(lambda x: geo_stats.get(x, {}).get('max', 0.0))
        df_copy[f'{prefix}min'] = df_copy['geo'].map(lambda x: geo_stats.get(x, {}).get('min', 0.0))
        df_copy[f'{prefix}std'] = df_copy['geo'].map(lambda x: geo_stats.get(x, {}).get('std', 0.0))
        df_copy[f'{prefix}trend'] = df_copy['geo'].map(lambda x: geo_stats.get(x, {}).get('trend', 0.0))

        return df_copy

    def _remove_geo_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Pašalina laikinus geo features stulpelius."""
        prefix = f"_geo_{target_col}_"
        cols_to_drop = [c for c in df.columns if c.startswith(prefix)]
        return df.drop(columns=cols_to_drop, errors='ignore')

    # ==========================================================================
    # 3.5. STULPELIO IMPUTAVIMAS
    # ==========================================================================
    # Pagrindinis imputavimo procesas vienam stulpeliui

    def _impute_column(self, df_work: pd.DataFrame, target_col: str):
        """
        Imputuoja vieną stulpelį su trūkstamomis reikšmėmis.

        SVARBU: Naudojame self._df_original (originalias reikšmes) kaip features,
        kad išvengtume "cascade error".

        Procesas:
            1. Apskaičiuoja geo statistikas IŠ ORIGINALIŲ duomenų
            2. Prideda geo statistikas kaip features
            3. Sukuria synthetic test (20%) be 0 reikšmių
            4. Treniruoja ir vertina modelį
            5. Pertreniruoja ant 100% duomenų
            6. Imputuoja NaN reikšmes su post-processing (jei įjungta)

        Args:
            df_work: DataFrame, kuriame bus užpildytos NaN reikšmės
            target_col: Stulpelis imputavimui
        """
        # Naudojame ORIGINALIAS reikšmes features skaičiavimui
        df_features = self._df_original

        # Žingsnis 1: Apskaičiuojame geo statistikas (tik jei post-processing įjungtas)
        if self.use_post_processing:
            geo_stats = self._compute_geo_stats(df_features, target_col)
            self.geo_stats[target_col] = geo_stats
            # Žingsnis 2: Pridedame geo statistikas kaip features
            df_with_geo = self._add_geo_features(df_features, target_col, geo_stats)
        else:
            geo_stats = {}
            self.geo_stats[target_col] = geo_stats
            df_with_geo = df_features.copy()

        feature_cols = [c for c in df_with_geo.columns if c != target_col]
        known_mask = df_with_geo[target_col].notna()

        # Žingsnis 3: Fallback į mean, jei per mažai duomenų
        if known_mask.sum() < 5:
            self._fallback_mean_impute(df_work, target_col, known_mask)
            return

        # Žingsnis 4: Paruošiame duomenis
        X_all_raw = df_with_geo.loc[known_mask, feature_cols].copy()
        y_all = df_with_geo.loc[known_mask, target_col].copy()

        # Žingsnis 5: Train/Test split
        train_mask, test_mask = self._create_train_test_split(y_all)

        # Žingsnis 6: Modelio treniravimas
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            self._train_without_test(X_all_raw, y_all, target_col, feature_cols)
        else:
            self._train_with_test(
                X_all_raw, y_all, train_mask, test_mask,
                target_col, feature_cols
            )

        # Žingsnis 7: Galutinė imputacija
        if self.use_post_processing:
            self._perform_final_imputation_with_bounds(
                df_work, df_with_geo, target_col, feature_cols, geo_stats
            )
        else:
            self._perform_final_imputation(
                df_work, df_with_geo, target_col, feature_cols
            )

        # Žingsnis 8: Išsaugome feature importance
        self._save_feature_importance(target_col)

    def _fallback_mean_impute(self, df: pd.DataFrame, target_col: str, known_mask):
        """
        Fallback: imputuoja su mean + year-based variation, jei per mažai duomenų.

        Args:
            df: DataFrame imputavimui
            target_col: Target stulpelis
            known_mask: Mask su žinomų reikšmių pozicijomis
        """
        known_values = df.loc[known_mask, target_col]
        fill_value = known_values.mean()
        fill_std = known_values.std() if len(known_values) > 1 else abs(fill_value) * 0.1

        if fill_std < 1e-10:
            fill_std = abs(fill_value) * 0.1 if fill_value != 0 else 1.0

        missing_mask = ~known_mask

        # Year-based variation
        if 'year' in df.columns and missing_mask.any():
            years = df.loc[missing_mask, 'year'].values
            mean_year = int(np.median(df['year'].values))

            imputed_values = []
            for i, year in enumerate(years):
                geo = df.loc[missing_mask].iloc[i]['geo'] if 'geo' in df.columns else 'unknown'
                seed = hash(f"{geo}_{year}_{target_col}_fallback_v4") % 100000
                np.random.seed(seed)

                jitter_pct = 0.08 + (year - 2000) * 0.003
                jitter_base = fill_std * min(jitter_pct, 0.15)
                year_offset = (year - mean_year) * fill_std * 0.02
                random_jitter = np.random.normal(0, jitter_base)
                total_variation = random_jitter + year_offset

                imputed_values.append(fill_value + total_variation)

            df.loc[missing_mask, target_col] = imputed_values
        else:
            df.loc[missing_mask, target_col] = fill_value

        self.model_metrics[target_col] = {
            'nrmse': float('nan'),
            'r2': float('nan'),
            'nmae': float('nan'),
            'smape': float('nan'),
            'model_type': 'insufficient_data_mean_impute_v4',
            'sample_size': int(known_mask.sum()),
            'total_samples': int(known_mask.sum())
        }

    # ==========================================================================
    # 3.6. TRAIN/TEST PADALIJIMAS
    # ==========================================================================
    # Synthetic test set sukūrimas modelio vertinimui

    def _create_train_test_split(self, y_all: pd.Series):
        """
        Sukuria train/test split'ą, filtruojant 0 reikšmes iš test'o.

        0 reikšmės grąžinamos į train (reali informacija),
        bet nenaudojamos test'ui (kad išvengtume šališkumo)->  https://arxiv.org/pdf/2106.04525

        Args:
            y_all: Target reikšmės

        Returns:
            tuple: (train_mask, test_mask) - boolean masyvai
        """
        all_indices = y_all.index.to_numpy()

        # Pradinis 80/20 split
        train_mask, test_mask = self._split_indices(all_indices, test_frac=0.2)

        # Filtruojame 0 reikšmes iš test'o
        zero_mask = (y_all.values == 0)
        test_mask = np.logical_and(test_mask, ~zero_mask)
        train_mask = np.logical_or(train_mask, zero_mask)

        return train_mask, test_mask

    def _split_indices(self, indices, test_frac=0.2):
        """
        Atsitiktinai paskirsto indeksus į train/test.

        Args:
            indices: Indeksų masyvas
            test_frac: Test dalies proporcija (default 0.2 = 20%)

        Returns:
            tuple: (train_mask, test_mask)
        """
        rng = np.random.RandomState(self.random_state)

        if len(indices) == 0:
            return np.array([], dtype=bool), np.array([], dtype=bool)

        n_test = max(1, int(len(indices) * test_frac))
        n_test = min(n_test, len(indices))

        test_idx = rng.choice(indices, size=n_test, replace=False)
        test_mask = pd.Index(indices).isin(test_idx)
        train_mask = ~test_mask

        return train_mask, test_mask

    # ==========================================================================
    # 3.7. MODELIO TRENIRAVIMAS
    # ==========================================================================
    # Random Forest modelio treniravimas su/be test set vertinimo

    def _train_without_test(self, X_all_raw, y_all, target_col, feature_cols):
        """
        Treniruoja modelį be test seto vertinimo.

        Naudojama, kai nėra pakankamai ne-nulinių reikšmių test setui.

        Args:
            X_all_raw: Feature DataFrame
            y_all: Target reikšmės
            target_col: Target stulpelio pavadinimas
            feature_cols: Feature stulpelių sąrašas
        """
        num_cols, cat_cols = self._get_column_groups(X_all_raw, feature_cols)

        # Žingsnis 1: Paruošiame transformerius
        num_imputer, encoder = self._fit_transformers(X_all_raw, num_cols, cat_cols)
        X_all = self._apply_transformations(X_all_raw, num_cols, cat_cols, num_imputer, encoder)

        # Žingsnis 2: Hiperparametrų optimizavimas (jei įjungta)
        if self.use_hyperopt:
            best_params = self._perform_hyperopt(X_all, y_all.values, target_col)
        else:
            best_params = None

        # Žingsnis 3: Treniruojame modelį
        model = self._create_model_with_params(best_params)
        model.fit(X_all, y_all.values)

        # Žingsnis 4: Išsaugome modelį
        self._save_model(target_col, model, num_cols, cat_cols, num_imputer, encoder)

        # Žingsnis 5: Metrikos (nėra test seto)
        self.model_metrics[target_col] = {
            'nrmse': float('nan'),
            'r2': float('nan'),
            'nmae': float('nan'),
            'smape': float('nan'),
            'model_type': 'no_test_nonzero',
            'sample_size': int(len(y_all)),
            'total_samples': int(len(y_all))
        }

    def _train_with_test(
        self, X_all_raw, y_all, train_mask, test_mask,
        target_col, feature_cols
    ):
        """
        Treniruoja modelį su test seto vertinimu.

        Procesas:
            1. Padalija duomenis į train/test
            2. Paruošia transformerius ant VISŲ duomenų
            3. Atliekame hiperparametrų optimizavimą (jei įjungta)
            4. Treniruoja vertinimo modelį ant TRAIN
            5. Vertina ant TEST
            6. Atliekame kryžminę validaciją
            7. Treniruoja galutinį modelį ant 100% duomenų

        Args:
            X_all_raw: Visi feature duomenys
            y_all: Visos target reikšmės
            train_mask: Train pozicijų mask
            test_mask: Test pozicijų mask
            target_col: Target stulpelio pavadinimas
            feature_cols: Feature stulpelių sąrašas
        """
        # Žingsnis 1: Padalijame duomenis
        X_train_raw = X_all_raw.loc[train_mask].copy()
        X_test_raw = X_all_raw.loc[test_mask].copy()
        y_train = y_all.loc[train_mask].copy()
        y_test = y_all.loc[test_mask].copy()

        num_cols, cat_cols = self._get_column_groups(X_all_raw, feature_cols)

        # Žingsnis 2: Paruošiame transformerius ant VISŲ duomenų
        num_imputer_full, encoder_full = self._fit_transformers(X_all_raw, num_cols, cat_cols)
        X_all = self._apply_transformations(X_all_raw, num_cols, cat_cols, num_imputer_full, encoder_full)

        # Žingsnis 3: Hiperparametrų optimizavimas (jei įjungta)
        if self.use_hyperopt:
            best_params = self._perform_hyperopt(X_all, y_all.values, target_col)
        else:
            best_params = None

        # Žingsnis 4: Vertinimo transformacijos (fit tik ant TRAIN)
        num_imputer_eval, encoder_eval = self._fit_transformers(X_train_raw, num_cols, cat_cols)
        X_train = self._apply_transformations(X_train_raw, num_cols, cat_cols, num_imputer_eval, encoder_eval)
        X_test = self._apply_transformations(X_test_raw, num_cols, cat_cols, num_imputer_eval, encoder_eval)

        # Žingsnis 5: Treniruojame vertinimo modelį
        model_eval = self._create_model_with_params(best_params)
        model_eval.fit(X_train, y_train.values)
        y_pred = model_eval.predict(X_test)

        # Žingsnis 6: Skaičiuojame metrikos
        self._save_test_metrics(target_col, y_test, y_pred, len(y_all))

        # Žingsnis 7: Kryžminė validacija
        self._perform_cross_validation(X_train, y_train.values, target_col, best_params)

        # Žingsnis 8: Galutinis modelis ant 100% duomenų
        model_full = self._create_model_with_params(best_params)
        model_full.fit(X_all, y_all.values)

        # Žingsnis 9: Išsaugome galutinį modelį
        self._save_model(target_col, model_full, num_cols, cat_cols, num_imputer_full, encoder_full)

    def _perform_cross_validation(self, X, y, target_col, params=None):
        """
        Atlieka kryžminę validaciją ir išsaugo rezultatus.

        Skaičiuoja visas 4 metrikas: R², nRMSE, nMAE, sMAPE.

        Args:
            X: Transformuoti feature duomenys (numpy array)
            y: Target reikšmės (numpy array)
            target_col: Stulpelio pavadinimas
            params: Hiperparametrai (jei None, naudoja default)
        """
        cv_folds = min(self.cv_folds, len(y))
        if cv_folds < 2:
            self.cv_scores[target_col] = {
                'r2_scores': [], 'r2_mean': float('nan'), 'r2_std': float('nan'),
                'nrmse_scores': [], 'nrmse_mean': float('nan'), 'nrmse_std': float('nan'),
                'nmae_scores': [], 'nmae_mean': float('nan'), 'nmae_std': float('nan'),
                'smape_scores': [], 'smape_mean': float('nan'), 'smape_std': float('nan'),
                'cv_folds': 0, 'sample_size': len(y)
            }
            return

        r2_scores, nrmse_scores, nmae_scores, smape_scores = [], [], [], []

        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kfold.split(X):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]

            model_cv = self._create_model_with_params(params)
            model_cv.fit(X_train_cv, y_train_cv)
            y_pred_cv = model_cv.predict(X_val_cv)

            # Metrikos
            r2 = r2_score(y_val_cv, y_pred_cv)
            rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
            mae = np.mean(np.abs(y_val_cv - y_pred_cv))
            smape = self._calculate_smape(y_val_cv, y_pred_cv)

            # Normalizuojame
            y_range = float(np.max(y_val_cv) - np.min(y_val_cv))
            nrmse = rmse / y_range if y_range > 0 else rmse
            nmae = mae / y_range if y_range > 0 else mae

            r2_scores.append(r2)
            nrmse_scores.append(nrmse)
            nmae_scores.append(nmae)
            smape_scores.append(smape)

        self.cv_scores[target_col] = {
            'r2_scores': r2_scores, 'r2_mean': float(np.mean(r2_scores)), 'r2_std': float(np.std(r2_scores)),
            'nrmse_scores': nrmse_scores, 'nrmse_mean': float(np.mean(nrmse_scores)), 'nrmse_std': float(np.std(nrmse_scores)),
            'nmae_scores': nmae_scores, 'nmae_mean': float(np.mean(nmae_scores)), 'nmae_std': float(np.std(nmae_scores)),
            'smape_scores': smape_scores, 'smape_mean': float(np.mean(smape_scores)), 'smape_std': float(np.std(smape_scores)),
            'cv_folds': cv_folds, 'sample_size': len(y)
        }

    def _create_model_with_params(self, params=None):
        """
        Sukuria Random Forest modelį su nurodytais parametrais.

        Args:
            params: dict su parametrais arba None (naudos default)

        Returns:
            RandomForestRegressor: Sukurtas (bet netreniruotas) modelis
        """
        if params is not None:
            return RandomForestRegressor(
                n_estimators=params.get('n_estimators', self.n_estimators),
                random_state=self.random_state,
                max_depth=params.get('max_depth', self.max_depth),
                min_samples_split=params.get('min_samples_split', self.min_samples_split),
                min_samples_leaf=params.get('min_samples_leaf', self.min_samples_leaf),
                max_features=params.get('max_features', 'sqrt'),
                n_jobs=6,
                bootstrap=True,
                oob_score=False
            )
        else:
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features='sqrt',
                n_jobs=6,
                bootstrap=True,
                oob_score=False
            )

    def _get_param_distributions(self):
        """
        Grąžina hiperparametrų paskirstymus Randomized Search CV.

        Returns:
            dict: Parametrų paskirstymai
        """
        return {
            'n_estimators': randint(50, 301),
            'max_depth': [5, 10, 15, 20, 25, 30, None],
            'min_samples_split': randint(2, 21),
            'min_samples_leaf': randint(1, 11),
            'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7]
        }

    def _perform_hyperopt(self, X, y, target_col):
        """
        Atlieka hiperparametrų optimizavimą su RandomizedSearchCV.

        Args:
            X: Feature matrica (numpy array)
            y: Target reikšmės (numpy array)
            target_col: Stulpelio pavadinimas

        Returns:
            dict: Geriausi rasti parametrai
        """
        base_model = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=6,
            bootstrap=True
        )

        param_distributions = self._get_param_distributions()

        cv_folds = min(self.hyperopt_cv, len(y))
        if cv_folds < 2:
            return {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': 'sqrt'
            }

        n_iter = min(self.hyperopt_n_iter, 50) if len(y) < 100 else self.hyperopt_n_iter

        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv_folds,
            scoring='r2',
            random_state=self.random_state,
            n_jobs=6,
            verbose=0,
            return_train_score=False
        )

        random_search.fit(X, y)

        best_params = random_search.best_params_
        self.best_params[target_col] = best_params

        return best_params

    # ==========================================================================
    # 3.8. FEATURE TRANSFORMACIJOS
    # ==========================================================================
    # Skaitinių ir kategorinių feature'ų transformavimas

    def _get_column_groups(self, X: pd.DataFrame, feature_cols):
        """
        Paskirsto stulpelius į skaitmeninius ir kategorinius.

        Args:
            X: Feature DataFrame
            feature_cols: Visų feature stulpelių sąrašas

        Returns:
            tuple: (num_cols, cat_cols)
        """
        num_cols = [
            c for c in feature_cols
            if pd.api.types.is_numeric_dtype(X[c]) and c not in self.categorical_cols
        ]
        cat_cols = [c for c in self.categorical_cols if c in feature_cols]
        return num_cols, cat_cols

    def _fit_transformers(self, X_raw, num_cols, cat_cols):
        """
        Sukuria ir fit'ina transformerius.

        Args:
            X_raw: Raw feature DataFrame
            num_cols: Skaitinių stulpelių sąrašas
            cat_cols: Kategorinių stulpelių sąrašas

        Returns:
            tuple: (num_imputer, encoder)
        """
        # Skaitinių stulpelių imputer (ignoruoja 0)
        num_imputer = None
        if num_cols:
            num_imputer = ZeroIgnoringImputer()
            num_imputer.fit(X_raw[num_cols])

        # Kategorinių stulpelių encoder
        encoder = None
        if cat_cols:
            encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            encoder.fit(X_raw[cat_cols])

        return num_imputer, encoder

    def _apply_transformations(self, X_raw, num_cols, cat_cols, num_imputer, encoder):
        """
        Pritaiko transformacijas ir grąžina numpy array.

        Args:
            X_raw: Raw feature DataFrame
            num_cols: Skaitinių stulpelių sąrašas
            cat_cols: Kategorinių stulpelių sąrašas
            num_imputer: Skaitinių stulpelių imputer
            encoder: Kategorinių stulpelių encoder

        Returns:
            np.ndarray: Transformuota feature matrica
        """
        # Skaitiniai
        if num_cols and num_imputer:
            X_num = num_imputer.transform(X_raw[num_cols])
        else:
            X_num = np.empty((len(X_raw), 0))

        # Kategoriniai
        if cat_cols and encoder:
            X_cat = encoder.transform(X_raw[cat_cols])
        else:
            X_cat = np.empty((len(X_raw), 0))

        return np.hstack([X_num, X_cat])

    # ==========================================================================
    # 3.9. GALUTINĖ IMPUTACIJA SU POST-PROCESSING
    # ==========================================================================
    # NaN reikšmių užpildymas su Empirical Bayes Shrinkage

    def _perform_final_imputation(self, df: pd.DataFrame, df_with_geo: pd.DataFrame,
                                   target_col: str, feature_cols: list):
        """
        Atlieka galutinę imputaciją BE post-processing.

        SVARBU: 0 reikšmės NĖRA imputuojamos - jos lieka 0.

        Args:
            df: DataFrame, kuriame užpildysime NaN
            df_with_geo: DataFrame su geo features
            target_col: Target stulpelis
            feature_cols: Feature stulpelių sąrašas
        """
        missing_mask = df[target_col].isna()

        if not missing_mask.any():
            return

        X_missing_raw = df_with_geo.loc[missing_mask, feature_cols].copy()
        model_info = self.models[target_col]

        X_missing = self._apply_transformations(
            X_missing_raw,
            model_info['num_cols'],
            model_info['cat_cols'],
            model_info['num_imputer'],
            model_info['encoder']
        )

        predictions = model_info['model'].predict(X_missing)
        df.loc[missing_mask, target_col] = predictions

    def _perform_final_imputation_with_bounds(
        self, df: pd.DataFrame, df_with_geo: pd.DataFrame,
        target_col: str, feature_cols: list, geo_stats: dict
    ):
        """
        Atlieka galutinę imputaciją SU EMPIRICAL BAYES SHRINKAGE post-processing.

        Empirical Bayes Shrinkage:
            1. Apskaičiuoja σ²_ML iš individualių RF medžių prognozių
            2. Taiko formulę (24): λ = σ²_ML / (σ²_ML + σ²_region)
            3. Taiko formulę (23): ŷ_EB = (1 - λ) × ŷ_ML + λ × ȳ_region

        SVARBU: 0 reikšmės NĖRA imputuojamos - jos lieka 0.

        Args:
            df: DataFrame, kuriame užpildysime NaN
            df_with_geo: DataFrame su geo features
            target_col: Target stulpelis
            feature_cols: Feature stulpelių sąrašas
            geo_stats: Geo statistikos post-processing
        """
        missing_mask = df[target_col].isna()

        if not missing_mask.any():
            return

        X_missing_raw = df_with_geo.loc[missing_mask, feature_cols].copy()
        model_info = self.models[target_col]

        X_missing = self._apply_transformations(
            X_missing_raw,
            model_info['num_cols'],
            model_info['cat_cols'],
            model_info['num_imputer'],
            model_info['encoder']
        )

        # RF predikcija (vidurkis iš visų medžių)
        predictions = model_info['model'].predict(X_missing)

        # Gauname individualių medžių prognozes σ²_ML skaičiavimui
        tree_predictions = self._get_individual_tree_predictions(
            model_info['model'], X_missing
        )

        # Empirical Bayes Shrinkage post-processing
        if 'geo' in df.columns and geo_stats:
            geo_values = df.loc[missing_mask, 'geo'].values
            year_values = df.loc[missing_mask, 'year'].values if 'year' in df.columns else None

            adjusted_predictions = self._apply_empirical_bayes_shrinkage(
                predictions, tree_predictions, geo_values, year_values, geo_stats, target_col
            )
            df.loc[missing_mask, target_col] = adjusted_predictions
        else:
            df.loc[missing_mask, target_col] = predictions

    def _get_individual_tree_predictions(self, model, X: np.ndarray) -> np.ndarray:
        """
        Gauna individualių RF medžių prognozes.

        Naudojama σ²_ML (modelio prognozių dispersijos) skaičiavimui
        pagal Empirical Bayes Shrinkage formulę (24).

        Args:
            model: Ištreniruotas RandomForestRegressor
            X: Transformuota feature matrica (numpy array)

        Returns:
            np.ndarray: Medžių prognozės, shape (n_trees, n_samples)
        """
        n_trees = len(model.estimators_)
        n_samples = X.shape[0]

        tree_predictions = np.zeros((n_trees, n_samples))

        for i, tree in enumerate(model.estimators_):
            tree_predictions[i, :] = tree.predict(X)

        return tree_predictions

    def _apply_empirical_bayes_shrinkage(
        self, predictions: np.ndarray, tree_predictions: np.ndarray,
        geo_values: np.ndarray, year_values: np.ndarray,
        geo_stats: dict, target_col: str
    ) -> np.ndarray:
        """
        Empirical Bayes Shrinkage pagal formules (23) ir (24).

        Formulė (23): ŷ_EB = (1 - λ) × ŷ_ML + λ × ȳ_region
        Formulė (24): λ = σ²_ML / (σ²_ML + σ²_region)

        Kur:
            - ŷ_ML: RF modelio prognozė (medžių vidurkis)
            - ȳ_region: regiono istorinis vidurkis (su trend korekcija)
            - σ²_ML: RF medžių prognozių dispersija
            - σ²_region: regiono istorinių duomenų dispersija

        Args:
            predictions: RF prognozės (vidurkis iš medžių)
            tree_predictions: Individualių medžių prognozės, shape (n_trees, n_samples)
            geo_values: Regionų reikšmės
            year_values: Metų reikšmės (arba None)
            geo_stats: Geo statistikos
            target_col: Target stulpelio pavadinimas

        Returns:
            np.ndarray: Koreguotos prognozės pagal Empirical Bayes Shrinkage
        """
        adjusted = predictions.copy()
        shrinkage_info = []

        # Mean year skaičiavimui
        mean_year = 2012
        if year_values is not None and len(year_values) > 0:
            mean_year = int(np.median(year_values))

        for i, (pred, geo) in enumerate(zip(predictions, geo_values)):
            stats = geo_stats.get(geo, {})

            if not stats:
                continue

            # Regiono statistikos
            geo_mean = stats.get('mean', pred)
            geo_max = stats.get('max', pred)
            geo_min = stats.get('min', pred)
            geo_std = stats.get('std', 0)
            geo_trend = stats.get('trend', 0)

            # σ²_region - regiono istorinių duomenų dispersija
            sigma2_region = geo_std ** 2 if geo_std > 0 else 1.0

            # σ²_ML - RF medžių prognozių dispersija šiam stebėjimui
            # Skaičiuojama iš individualių medžių prognozių
            sigma2_ML = float(np.var(tree_predictions[:, i], ddof=1))

            # Apsauga nuo per mažos dispersijos
            if sigma2_ML < 1e-10:
                sigma2_ML = sigma2_region * 0.01  # Minimalus 1% nuo regiono dispersijos

            # ========================================
            # FORMULĖ (24): λ = σ²_ML / (σ²_ML + σ²_region)
            # ========================================
            lambda_shrinkage = sigma2_ML / (sigma2_ML + sigma2_region)

            # Year-specific vidurkis (ȳ_region su trend korekcija)
            year = year_values[i] if year_values is not None else mean_year
            year_specific_mean = geo_mean
            if geo_trend != 0:
                year_diff = year - mean_year
                trend_adjustment = geo_trend * year_diff
                year_specific_mean = geo_mean + trend_adjustment

            # ========================================
            # FORMULĖ (23): ŷ_EB = (1 - λ) × ŷ_ML + λ × ȳ_region
            # ========================================
            y_eb = (1 - lambda_shrinkage) * pred + lambda_shrinkage * year_specific_mean

            # Safety bounds (ribos ekstremalių atvejų apsaugai)
            allows_negative = geo_min < 0
            if allows_negative:
                lower_bound = geo_min * 1.2 if geo_min < 0 else geo_min * 0.8
                upper_bound = geo_max * 1.2
            else:
                lower_bound = max(0, geo_min * 0.5)
                upper_bound = geo_max * 1.5

            # Pritaikome ribas
            final_pred = np.clip(y_eb, lower_bound, upper_bound)

            adjusted[i] = final_pred

            # Išsaugome informaciją ataskaitai
            shrinkage_info.append({
                'geo': geo,
                'year': year,
                'y_ML': pred,
                'y_EB': y_eb,
                'final_pred': final_pred,
                'y_region_mean': geo_mean,
                'year_specific_mean': year_specific_mean,
                'sigma2_ML': sigma2_ML,
                'sigma2_region': sigma2_region,
                'lambda': lambda_shrinkage,
                'geo_std': geo_std,
                'trend': geo_trend,
                'bounds': (lower_bound, upper_bound),
                'bounds_applied': final_pred != y_eb
            })

        if shrinkage_info:
            self.shrinkage_applied[target_col] = shrinkage_info

        return adjusted

    # ==========================================================================
    # 3.10. METRIKŲ SKAIČIAVIMAS IR SAUGOJIMAS
    # ==========================================================================
    # Test set metrikų skaičiavimas ir modelių/feature importance saugojimas

    def _save_test_metrics(self, target_col: str, y_test, y_pred, total_samples: int):
        """
        Apskaičiuoja ir išsaugo test metrikos.

        Skaičiuojamos metrikos:
            - R²: Determinacijos koeficientas
            - nRMSE: Normalizuota RMSE
            - nMAE: Normalizuota MAE
            - sMAPE: Simetrinė MAPE

        Args:
            target_col: Target stulpelio pavadinimas
            y_test: Tikrosios test reikšmės
            y_pred: Prognozuotos reikšmės
            total_samples: Bendras pavyzdžių skaičius
        """
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(np.mean(np.abs(y_test.values - y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        smape = self._calculate_smape(y_test.values, y_pred)

        y_range = float(y_test.max() - y_test.min())
        if y_range > 0:
            nrmse = rmse / y_range
            nmae = mae / y_range
        else:
            nrmse = rmse
            nmae = mae

        self.model_metrics[target_col] = {
            'nrmse': nrmse,
            'r2': r2,
            'nmae': nmae,
            'smape': smape,
            'model_type': 'synthetic_test_no_zeros',
            'sample_size': int(len(y_test)),
            'total_samples': int(total_samples)
        }

        self.test_predictions[target_col] = {
            'y_true': y_test.values,
            'y_pred': y_pred,
            'test_indices': y_test.index.tolist(),
            'r2': r2,
            'nmae': nmae,
            'nrmse': nrmse,
            'smape': smape
        }

    @staticmethod
    def _calculate_smape(y_true, y_pred):
        """
        Apskaičiuoja Symmetric MAPE [%].

        Veikia ir su 0 reikšmėmis, nes vardiklis niekada nebūna 0.

        Args:
            y_true: Tikrosios reikšmės
            y_pred: Prognozuotos reikšmės

        Returns:
            float: sMAPE procentais
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        denominator = np.abs(y_true) + np.abs(y_pred) + 1e-9
        return float(100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denominator))

    def _save_model(self, target_col, model, num_cols, cat_cols, num_imputer, encoder):
        """
        Išsaugo modelį ir transformerius.

        Args:
            target_col: Target stulpelio pavadinimas
            model: Ištreniruotas RandomForestRegressor modelis
            num_cols: Skaitinių stulpelių sąrašas
            cat_cols: Kategorinių stulpelių sąrašas
            num_imputer: Skaitinių stulpelių imputer
            encoder: Kategorinių stulpelių encoder
        """
        self.models[target_col] = {
            'model': model,
            'num_cols': num_cols,
            'cat_cols': cat_cols,
            'num_imputer': num_imputer,
            'encoder': encoder
        }

    def _save_feature_importance(self, target_col: str):
        """
        Išsaugo feature importance.

        Args:
            target_col: Target stulpelio pavadinimas
        """
        model_info = self.models[target_col]
        feature_names = (model_info['num_cols'] or []) + (model_info['cat_cols'] or [])
        importances = model_info['model'].feature_importances_

        k = min(len(feature_names), len(importances))
        importance_dict = {
            feature_names[i]: float(importances[i])
            for i in range(k)
        }

        # Rūšiuojame pagal svarbą (didėjimo tvarka)
        self.feature_importance[target_col] = dict(
            sorted(importance_dict.items(), key=lambda kv: kv[1], reverse=True)
        )
