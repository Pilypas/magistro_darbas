"""
XGBoost modelio implementacija ekonominių rodiklių trūkstamų reikšmių užpildymui
================================================================================

PAGRINDINĖS TAISYKLĖS
---------------------
• Struktūriniai nuliai (0) NIEKADA NEIMPUTUOJAMI – imputuojamos tik NaN reikšmės
• 0 reikšmės naudojamos kaip TRAIN duomenys (reali informacija), bet NE kaip TEST
• Kategoriniai prediktoriai ('geo', 'year') BE trūkumų - tik enkoduojami
• Synthetic test be leakage (20% TEST be 0 reikšmių)
• Prediktorių imputacija (mean) ignoruoja 0 reikšmes
• Naudojama SMAPE metrika (veikia su 0 reikšmėmis)
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from scipy.stats import randint, uniform

try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError(
        "XGBoost biblioteka neįdiegta. Įdiekite: pip install xgboost"
    ) from e


class ZeroIgnoringImputer:
    """
    Custom imputer ignoruojantis 0 reikšmes skaičiuojant mean.

    Struktūriniai 0 neturėtų įtakoti prediktorių imputacijos,
    nes jie dažnai reiškia "nėra duomenų" arba "neaktualu".
    """

    def __init__(self):
        self.statistics_ = None

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


class XGBoostImputer:
    """
    XGBoost imputavimas ekonominiams rodikliams.

    Kiekvienam stulpeliui su trūkstamomis reikšmėmis treniruojamas atskiras XGBRegressor.
    Synthetic test: 20% indeksų -> TEST (tik iš eilučių, kur target != 0).
    Po vertinimo pertreniruojama ant 100% žinomų taikinių.
    """

    def __init__(
        self,
        n_estimators=200,
        random_state=42,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=0.1,
        categorical_cols=None,
        exclude_columns=None,
        cv_folds=2,
        use_hyperopt=False,
        hyperopt_n_iter=30,
        hyperopt_cv=3,
        use_post_processing=True,
        shrinkage_k=3.0
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.categorical_cols = categorical_cols or ['geo', 'year']
        self.exclude_columns = exclude_columns or []
        self.cv_folds = cv_folds

        # Hiperparametru optimizavimo nustatymai
        self.use_hyperopt = use_hyperopt
        self.hyperopt_n_iter = hyperopt_n_iter
        self.hyperopt_cv = hyperopt_cv

        # Hibridinis post-processing: Empirical Bayes Shrinkage + kietosios ribos
        self.use_post_processing = use_post_processing
        self.shrinkage_k = shrinkage_k  # Standartinių nuokrypių skaičius shrinkage pradžiai

        self.models = {}
        self.feature_importance = {}
        self.model_metrics = {}
        self.test_predictions = {}
        self.cv_scores = {}
        self.best_params = {}  # Saugomi geriausi parametrai kiekvienam rodikliui
        self.geo_stats = {}  # Regiono statistikos kiekvienam rodikliui
        self.shrinkage_applied = {}  # Informacija apie pritaikytą shrinkage

    # ============================================================================
    # PUBLIC API
    # ============================================================================

    def fit_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Treniruoja modelius ir imputuoja trūkstamas reikšmes.

        SVARBU: Naudojame ORIGINALIAS reikšmes kaip features, ne jau imputuotas.
        Tai išvengia "cascade error" problemos, kai vieno stulpelio klaidos
        persiduoda į kitus stulpelius.

        Args:
            df: DataFrame su trūkstamomis reikšmėmis

        Returns:
            DataFrame su imputuotomis reikšmėmis (tik NaN, ne 0)
        """
        df_work = self._prepare_dataframe(df)
        self._validate_categorical_columns(df_work)

        # SVARBU: Išsaugome originalias reikšmes features naudojimui
        # Tai užtikrina, kad kiekvienas stulpelis imputuojamas naudojant
        # TIK originalias reikšmes, o ne jau imputuotas iš kitų stulpelių
        self._df_original = df_work.copy()

        # Imputuojame kiekvieną stulpelį su NaN reikšmėmis
        for target_col in df_work.columns:
            if self._should_impute_column(df_work, target_col):
                self._impute_column(df_work, target_col)

        # Atlaisviname atmintį
        del self._df_original

        return df_work

    def get_feature_importance(self):
        """Grąžina feature importance kiekvienam stulpeliui."""
        return self.feature_importance

    def get_model_metrics(self):
        """Grąžina modelių metrikos."""
        return self.model_metrics

    def get_test_predictions(self, df=None):
        """Grąžina test predikcijas."""
        return self.test_predictions

    def get_cv_scores(self):
        """Grąžina cross-validation rezultatus."""
        return self.cv_scores

    def get_params(self):
        """
        Grąžina modelio hiperparametrus kaip žodyną.
        Naudojama notebook'e parametrų spausdinimui.

        Returns:
            dict: Visi XGBoost modelio parametrai
        """
        params = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'cv_folds': self.cv_folds,
            'use_hyperopt': self.use_hyperopt,
            'use_post_processing': self.use_post_processing,
            'shrinkage_k': self.shrinkage_k
        }
        if self.use_hyperopt:
            params['hyperopt_n_iter'] = self.hyperopt_n_iter
            params['hyperopt_cv'] = self.hyperopt_cv
        return params

    def get_best_params(self):
        """
        Grąžina geriausius parametrus kiekvienam rodikliui po hiperparametrų optimizavimo.

        Returns:
            dict: Geriausi parametrai pagal rodiklius
        """
        return self.best_params

    def get_best_params_summary(self):
        """
        Grąžina suvestinę apie dažniausiai pasirinktas hiperparametrų reikšmes.

        Returns:
            dict: Statistika apie geriausius parametrus
        """
        if not self.best_params:
            return {}

        summary = {
            'n_estimators': [],
            'learning_rate': [],
            'max_depth': [],
            'subsample': [],
            'colsample_bytree': [],
            'min_child_weight': [],
            'reg_alpha': [],
            'reg_lambda': []
        }

        for params in self.best_params.values():
            for key in summary.keys():
                if key in params:
                    summary[key].append(params[key])

        result = {}
        for key, values in summary.items():
            if values:
                result[key] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'median': np.median(values)
                }

        return result

    def print_params(self):
        """
        Išspausdina modelio parametrus.
        Naudojama notebook'e.
        """
        params = self.get_params()
        print("XGBoost parametrai:")
        for name, value in params.items():
            print(f"  - {name}: {value}")

        if self.use_hyperopt:
            print("\nHiperparametru optimizavimas IJUNGTAS")
            print(f"  - Iteraciju skaicius: {self.hyperopt_n_iter}")
            print(f"  - CV folds optimizavimui: {self.hyperopt_cv}")

        if self.use_post_processing:
            print("\nPATOBULINTAS (v4) Hibridinis post-processing IJUNGTAS:")
            print("  1. MINIMALUS Shrinkage (tik ekstremaliems atvejams)")
            print(f"     - Bazinis k: {self.shrinkage_k} * 1.5 = {self.shrinkage_k * 1.5} std")
            print("     - max_shrinkage: 5-10% (islaiko 90-95% XGB info)")
            print("     - sigmoid_steepness = 0.4 (labai svelnus perejimas)")
            print("     - Papildomas ribojimas: shrinkage max 10% bet kokiu atveju")
            print("  2. YEAR-BASED VARIATION (unikalios reiksmes kiekvienam metui)")
            print("     - Bazinis jitter: 8-15% nuo geo_std")
            print("     - Year offset: 2% nuo std per metus (sisteminis)")
            print("     - Deterministinis seed: geo + year + target_col")
            print("  3. Kietosios ribos (safety net)")
            print("     - [min*0.3, max*2.0] arba [year_specific_mean +/- 4*std]")

    def print_best_params(self):
        """
        Išspausdina geriausius rastus parametrus po hiperparametrų optimizavimo.
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
                print(f"  {param}: min={stats['min']:.4f}, max={stats['max']:.4f}, "
                      f"vidurkis={stats['mean']:.4f}, mediana={stats['median']:.4f}")

        print(f"\nOptimizuotu rodikliu skaicius: {len(self.best_params)}")
        print("=" * 80)

    def get_cv_metrics_df(self):
        """
        Grąžina CV metrikas kaip DataFrame su vidurkiais ir standartinėmis paklaidomis.
        Formatuota: 'vidurkis ± std' string'ais ir atskirais stulpeliais vidurkiams.
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
        Spausdina CV rezultatus į konsolę.
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

    # ============================================================================
    # DATA PREPARATION
    # ============================================================================

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Paruošia DataFrame: kopijuoja ir konvertuoja object stulpelius."""
        df_work = df.copy()

        # Konvertuojame ne-kategorinius 'object' į numeric
        for col in df_work.columns:
            if (df_work[col].dtype == 'object') and (col not in self.categorical_cols):
                df_work[col] = pd.to_numeric(df_work[col], errors='coerce')

        return df_work

    def _validate_categorical_columns(self, df: pd.DataFrame):
        """Patikrina, kad kategoriniai stulpeliai neturi NaN."""
        for col in self.categorical_cols:
            if col in df.columns and df[col].isna().any():
                raise ValueError(
                    f"Kategorinis stulpelis '{col}' turi trūkstamų reikšmių, "
                    "pagal dizainą to neturėtų būti."
                )

    def _should_impute_column(self, df: pd.DataFrame, col: str) -> bool:
        """Patikrina, ar stulpelis turėtų būti imputuojamas."""
        return (
            col not in self.categorical_cols and
            col not in self.exclude_columns and
            df[col].isna().any()
        )

    # ============================================================================
    # GEO STATISTICS (Feature Engineering)
    # ============================================================================

    def _compute_geo_stats(self, df: pd.DataFrame, target_col: str) -> dict:
        """
        Apskaičiuoja regiono statistikas target stulpeliui.

        Kiekvienam regionui skaičiuojame:
        - mean: vidurkis (ignoruojant 0 ir NaN)
        - max: maksimumas
        - min: minimumas
        - std: standartinis nuokrypis
        - trend: metinis pokytis (linijinės regresijos koeficientas)

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

            # Trend: linijinės regresijos koeficientas (year vs value)
            geo_trend = 0.0
            if 'year' in df.columns:
                geo_df = df[(df['geo'] == geo_val) & (df[target_col].notna()) & (df[target_col] != 0)]
                if len(geo_df) >= 2:
                    years = geo_df['year'].values
                    values = geo_df[target_col].values
                    # Paprastas linijinis trend (least squares)
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
        Prideda regiono statistikas kaip papildomus features.

        Pridedami stulpeliai:
        - _geo_mean: regiono vidurkis
        - _geo_max: regiono maksimumas
        - _geo_min: regiono minimumas
        - _geo_std: regiono standartinis nuokrypis
        - _geo_trend: regiono metinis trend
        """
        df_copy = df.copy()

        if 'geo' not in df.columns or not geo_stats:
            return df_copy

        # Sukuriame naujus stulpelius
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

    # ============================================================================
    # COLUMN IMPUTATION
    # ============================================================================

    def _impute_column(self, df_work: pd.DataFrame, target_col: str):
        """
        Imputuoja vieną stulpelį.

        SVARBU: Naudojame self._df_original (originalias reikšmes) kaip features,
        kad išvengtume "cascade error" - kai vieno stulpelio imputuotos reikšmės
        klaidingai įtakoja kitų stulpelių imputavimą.

        Procesas:
        1. Apskaičiuoja geo statistikas IŠ ORIGINALIŲ duomenų
        2. Prideda geo statistikas kaip features
        3. Synthetic test (20%) be 0 reikšmių
        4. Treniruoja ir vertina modelį
        5. Pertreniruoja ant 100% duomenų
        6. Imputuoja NaN reikšmes su post-processing apribojimu (jei įjungta)
        7. Pašalina laikinus geo features
        """
        # SVARBU: Naudojame ORIGINALIAS reikšmes features skaičiavimui
        df_features = self._df_original

        # 1. Apskaičiuojame geo statistikas IŠ ORIGINALIŲ duomenų
        geo_stats = self._compute_geo_stats(df_features, target_col)
        self.geo_stats[target_col] = geo_stats

        # 2. Pridedame geo statistikas kaip features (naudojame ORIGINALIAS reikšmes)
        df_with_geo = self._add_geo_features(df_features, target_col, geo_stats)

        feature_cols = [c for c in df_with_geo.columns if c != target_col]
        known_mask = df_with_geo[target_col].notna()

        # Fallback į mean, jei per mažai duomenų
        if known_mask.sum() < 5:
            self._fallback_mean_impute(df_work, target_col, known_mask)
            return

        # Paruošiame duomenis (naudojame ORIGINALIAS reikšmes su geo features)
        X_all_raw = df_with_geo.loc[known_mask, feature_cols].copy()
        y_all = df_with_geo.loc[known_mask, target_col].copy()

        # Train/Test split be 0 reikšmių test'e
        train_mask, test_mask = self._create_train_test_split(y_all)

        if test_mask.sum() == 0 or train_mask.sum() == 0:
            # Jei nėra tinkamo test seto, treniruojame ant visų duomenų
            self._train_without_test(X_all_raw, y_all, target_col, feature_cols)
        else:
            # Normalus train/test scenarijus
            self._train_with_test(
                X_all_raw, y_all, train_mask, test_mask,
                target_col, feature_cols
            )

        # Galutinė imputacija su post-processing apribojimu (jei įjungta)
        if self.use_post_processing:
            self._perform_final_imputation_with_bounds(
                df_work, df_with_geo, target_col, feature_cols, geo_stats
            )
        else:
            self._perform_final_imputation(
                df_work, df_with_geo, target_col, feature_cols
            )

        # Išsaugome feature importance
        self._save_feature_importance(target_col)

    def _fallback_mean_impute(self, df: pd.DataFrame, target_col: str, known_mask):
        """
        Fallback: imputuoja su mean + year-based variation, jei per mažai duomenų.

        v4: Pridedame variaciją net ir fallback atveju, kad reikšmės nebūtų vienodos.
        """
        known_values = df.loc[known_mask, target_col]
        fill_value = known_values.mean()
        fill_std = known_values.std() if len(known_values) > 1 else abs(fill_value) * 0.1

        # Jei std = 0, naudojame 10% nuo mean
        if fill_std < 1e-10:
            fill_std = abs(fill_value) * 0.1 if fill_value != 0 else 1.0

        missing_mask = ~known_mask

        # v4: Pridedame year-based variation
        if 'year' in df.columns and missing_mask.any():
            years = df.loc[missing_mask, 'year'].values
            mean_year = int(np.median(df['year'].values))

            imputed_values = []
            for i, year in enumerate(years):
                # Deterministinis seed
                geo = df.loc[missing_mask].iloc[i]['geo'] if 'geo' in df.columns else 'unknown'
                seed = hash(f"{geo}_{year}_{target_col}_fallback_v4") % 100000
                np.random.seed(seed)

                # Year-based jitter (8-15% nuo std)
                jitter_pct = 0.08 + (year - 2000) * 0.003
                jitter_base = fill_std * min(jitter_pct, 0.15)

                # Systematic year offset (2% nuo std per metus)
                year_offset = (year - mean_year) * fill_std * 0.02

                # Random jitter + year offset
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

    # ============================================================================
    # TRAIN/TEST SPLIT
    # ============================================================================

    def _create_train_test_split(self, y_all: pd.Series):
        """
        Sukuria train/test split'ą, filtruojant 0 reikšmes iš test'o.

        0 reikšmės grąžinamos į train (reali informacija),
        bet nenaudojamos test'ui (kad išvengti šališkumo).
        """
        all_indices = y_all.index.to_numpy()

        # Pradinis split
        train_mask, test_mask = self._split_indices(all_indices, test_frac=0.2)

        # Filtruojame 0 reikšmes iš test'o
        zero_mask = (y_all.values == 0)
        test_mask = np.logical_and(test_mask, ~zero_mask)
        train_mask = np.logical_or(train_mask, zero_mask)

        return train_mask, test_mask

    def _split_indices(self, indices, test_frac=0.2):
        """Atsitiktinai paskirsto indeksus į train/test."""
        rng = np.random.RandomState(self.random_state)

        if len(indices) == 0:
            return np.array([], dtype=bool), np.array([], dtype=bool)

        n_test = max(1, int(len(indices) * test_frac))
        n_test = min(n_test, len(indices))

        test_idx = rng.choice(indices, size=n_test, replace=False)
        test_mask = pd.Index(indices).isin(test_idx)
        train_mask = ~test_mask

        return train_mask, test_mask

    # ============================================================================
    # MODEL TRAINING
    # ============================================================================

    def _train_without_test(self, X_all_raw, y_all, target_col, feature_cols):
        """Treniruoja modelį be test seto vertinimo."""
        num_cols, cat_cols = self._get_column_groups(X_all_raw, feature_cols)

        # Paruošiame transformerius ir duomenis
        num_imputer, encoder = self._fit_transformers(
            X_all_raw, num_cols, cat_cols
        )
        X_all = self._apply_transformations(
            X_all_raw, num_cols, cat_cols, num_imputer, encoder
        )

        # Jei įjungtas hyperopt - pirma randame geriausius parametrus
        if self.use_hyperopt:
            best_params = self._perform_hyperopt(X_all, y_all.values, target_col)
        else:
            best_params = None

        # Treniruojame modelį su geriausiais parametrais
        model = self._create_model_with_params(best_params)
        model.fit(X_all, y_all.values)

        # Išsaugome modelį
        self._save_model(target_col, model, num_cols, cat_cols, num_imputer, encoder)

        # Metrika be testavimo
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
        """Treniruoja modelį su test seto vertinimu."""
        # Padalijame duomenis
        X_train_raw = X_all_raw.loc[train_mask].copy()
        X_test_raw = X_all_raw.loc[test_mask].copy()
        y_train = y_all.loc[train_mask].copy()
        y_test = y_all.loc[test_mask].copy()

        num_cols, cat_cols = self._get_column_groups(X_all_raw, feature_cols)

        # Paruošiame PILNUS duomenis transformacijoms
        num_imputer_full, encoder_full = self._fit_transformers(
            X_all_raw, num_cols, cat_cols
        )
        X_all = self._apply_transformations(
            X_all_raw, num_cols, cat_cols, num_imputer_full, encoder_full
        )

        # Jei įjungtas hyperopt - pirma randame geriausius parametrus
        if self.use_hyperopt:
            best_params = self._perform_hyperopt(X_all, y_all.values, target_col)
        else:
            best_params = None

        # Vertinimo transformacijos (fit tik ant TRAIN)
        num_imputer_eval, encoder_eval = self._fit_transformers(
            X_train_raw, num_cols, cat_cols
        )
        X_train = self._apply_transformations(
            X_train_raw, num_cols, cat_cols, num_imputer_eval, encoder_eval
        )
        X_test = self._apply_transformations(
            X_test_raw, num_cols, cat_cols, num_imputer_eval, encoder_eval
        )

        # Treniruojame vertinimo modelį su geriausiais parametrais (jei rasti)
        model_eval = self._create_model_with_params(best_params)
        model_eval.fit(X_train, y_train.values)
        y_pred = model_eval.predict(X_test)

        # Skaičiuojame metrikos
        self._save_test_metrics(target_col, y_test, y_pred, len(y_all))

        # Cross-validation su geriausiais parametrais (arba default)
        self._perform_cross_validation(
            X_train, y_train.values, target_col, best_params
        )

        # Galutinis modelis ant 100% duomenų su geriausiais parametrais
        model_full = self._create_model_with_params(best_params)
        model_full.fit(X_all, y_all.values)

        # Išsaugome galutinį modelį
        self._save_model(
            target_col, model_full, num_cols, cat_cols,
            num_imputer_full, encoder_full
        )

    def _perform_cross_validation(self, X, y, target_col, params=None):
        """
        Atlieka kryžminę validaciją (cross-validation) ir išsaugo rezultatus.
        Skaičiuoja visas 4 metrikas: R², nRMSE, nMAE, sMAPE.

        Args:
            X: Transformuoti feature duomenys (numpy array)
            y: Target reikšmės (numpy array)
            target_col: Stulpelio pavadinimas
            params: dict su hiperparametrais (jei None, naudoja default)
        """
        # Apsauga nuo per mažo duomenų kiekio
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

        # Inicializuojame metrikų sąrašus
        r2_scores = []
        nrmse_scores = []
        nmae_scores = []
        smape_scores = []

        # KFold cross-validation
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kfold.split(X):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]

            # Sukuriame modelį su nurodytais parametrais (arba default)
            model_cv = self._create_model_with_params(params)
            model_cv.fit(X_train_cv, y_train_cv)
            y_pred_cv = model_cv.predict(X_val_cv)

            # Skaičiuojame metrikas
            r2 = r2_score(y_val_cv, y_pred_cv)
            rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
            mae = np.mean(np.abs(y_val_cv - y_pred_cv))
            smape = self._calculate_smape(y_val_cv, y_pred_cv)

            # Normalizuojame pagal y_val diapazoną
            y_range = float(np.max(y_val_cv) - np.min(y_val_cv))
            if y_range > 0:
                nrmse = rmse / y_range
                nmae = mae / y_range
            else:
                nrmse = rmse
                nmae = mae

            r2_scores.append(r2)
            nrmse_scores.append(nrmse)
            nmae_scores.append(nmae)
            smape_scores.append(smape)

        # Išsaugome rezultatus
        self.cv_scores[target_col] = {
            'r2_scores': r2_scores,
            'r2_mean': float(np.mean(r2_scores)),
            'r2_std': float(np.std(r2_scores)),
            'nrmse_scores': nrmse_scores,
            'nrmse_mean': float(np.mean(nrmse_scores)),
            'nrmse_std': float(np.std(nrmse_scores)),
            'nmae_scores': nmae_scores,
            'nmae_mean': float(np.mean(nmae_scores)),
            'nmae_std': float(np.std(nmae_scores)),
            'smape_scores': smape_scores,
            'smape_mean': float(np.mean(smape_scores)),
            'smape_std': float(np.std(smape_scores)),
            'cv_folds': cv_folds,
            'sample_size': len(y)
        }

    def _create_model_with_params(self, params=None):
        """
        Sukuria XGBoost modelį su nurodytais parametrais.

        Args:
            params: dict su parametrais arba None (naudos default)

        Returns:
            XGBRegressor: Sukurtas (bet netreniruotas) modelis
        """
        if params is not None:
            return xgb.XGBRegressor(
                n_estimators=params.get('n_estimators', self.n_estimators),
                random_state=self.random_state,
                learning_rate=params.get('learning_rate', self.learning_rate),
                max_depth=params.get('max_depth', self.max_depth),
                subsample=params.get('subsample', self.subsample),
                colsample_bytree=params.get('colsample_bytree', self.colsample_bytree),
                min_child_weight=params.get('min_child_weight', self.min_child_weight),
                reg_alpha=params.get('reg_alpha', self.reg_alpha),
                reg_lambda=params.get('reg_lambda', self.reg_lambda),
                verbosity=0,
                n_jobs=6,
                eval_metric='rmse'
            )
        else:
            return xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                min_child_weight=self.min_child_weight,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                verbosity=0,
                n_jobs=6,
                eval_metric='rmse'
            )

    def _get_param_distributions(self):
        """
        Grąžina hiperparametrų paskirstymus Randomized Search CV.

        Parinktys optimizuotos XGBoost ekonominių rodiklių imputavimui:
        - n_estimators: 50-300 (iteraciju skaicius)
        - learning_rate: 0.01-0.3 (mokymosi greitis)
        - max_depth: 3-15 (medzio gylis)
        - subsample: 0.6-1.0 (eiluciu atranka)
        - colsample_bytree: 0.6-1.0 (stulpeliu atranka)
        - min_child_weight: 1-10 (minimalus lapo svoris)
        - reg_alpha: 0-1 (L1 regularizacija)
        - reg_lambda: 0-1 (L2 regularizacija)
        """
        return {
            'n_estimators': randint(50, 301),  # 50-300
            'learning_rate': uniform(0.01, 0.29),  # 0.01-0.3
            'max_depth': randint(3, 16),  # 3-15
            'subsample': uniform(0.6, 0.4),  # 0.6-1.0
            'colsample_bytree': uniform(0.6, 0.4),  # 0.6-1.0
            'min_child_weight': randint(1, 11),  # 1-10
            'reg_alpha': uniform(0, 1),  # 0-1
            'reg_lambda': uniform(0, 1)  # 0-1
        }

    def _perform_hyperopt(self, X, y, target_col):
        """
        Atlieka hiperparametrų optimizavimą su RandomizedSearchCV.

        Args:
            X: Feature matrica (numpy array)
            y: Target reikšmės (numpy array)
            target_col: Stulpelio pavadinimas (metrikų saugojimui)

        Returns:
            dict: Geriausi rasti parametrai
        """
        # Bazinis modelis
        base_model = xgb.XGBRegressor(
            random_state=self.random_state,
            verbosity=0,
            n_jobs=6,
            eval_metric='rmse'
        )

        # Parametrų paskirstymai
        param_distributions = self._get_param_distributions()

        # CV folds skaičius (minimaliai 2)
        cv_folds = min(self.hyperopt_cv, len(y))
        if cv_folds < 2:
            # Per mažai duomenų - grąžiname default parametrus
            return {
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'min_child_weight': self.min_child_weight,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda
            }

        # Iteracijų skaičius (sumažiname jei mažai duomenų)
        n_iter = min(self.hyperopt_n_iter, 50) if len(y) < 100 else self.hyperopt_n_iter

        # RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv_folds,
            scoring='r2',  # Optimizuojame R²
            random_state=self.random_state,
            n_jobs=6,  # 6 threads
            verbose=0,
            return_train_score=False
        )

        # Vykdome paiešką
        random_search.fit(X, y)

        # Išsaugome geriausius parametrus
        best_params = random_search.best_params_
        self.best_params[target_col] = best_params

        return best_params

    # ============================================================================
    # FEATURE TRANSFORMATIONS
    # ============================================================================

    def _get_column_groups(self, X: pd.DataFrame, feature_cols):
        """Paskirsto stulpelius į skaitmeninius ir kategorinius."""
        num_cols = [
            c for c in feature_cols
            if pd.api.types.is_numeric_dtype(X[c]) and c not in self.categorical_cols
        ]
        cat_cols = [c for c in self.categorical_cols if c in feature_cols]
        return num_cols, cat_cols

    def _fit_transformers(self, X_raw, num_cols, cat_cols):
        """Sukuria ir fit'ina transformerius."""
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
        """Pritaiko transformacijas ir grąžina numpy array."""
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

    # ============================================================================
    # FINAL IMPUTATION
    # ============================================================================

    def _perform_final_imputation(self, df: pd.DataFrame, df_with_geo: pd.DataFrame,
                                   target_col: str, feature_cols: list):
        """
        Atlieka galutinę imputaciją be post-processing.

        SVARBU: 0 reikšmės NĖRA imputuojamos - jos lieka 0.
        """
        missing_mask = df[target_col].isna()

        if not missing_mask.any():
            return

        # Naudojame df_with_geo (su geo features) prognozavimui
        X_missing_raw = df_with_geo.loc[missing_mask, feature_cols].copy()
        model_info = self.models[target_col]

        # Transformuojame trūkstamus duomenis
        X_missing = self._apply_transformations(
            X_missing_raw,
            model_info['num_cols'],
            model_info['cat_cols'],
            model_info['num_imputer'],
            model_info['encoder']
        )

        # Predikcija ir užpildymas
        predictions = model_info['model'].predict(X_missing)
        df.loc[missing_mask, target_col] = predictions

    def _perform_final_imputation_with_bounds(
        self, df: pd.DataFrame, df_with_geo: pd.DataFrame,
        target_col: str, feature_cols: list, geo_stats: dict
    ):
        """
        Atlieka galutinę imputaciją su HIBRIDINIU post-processing:
        1. Empirical Bayes Shrinkage (minkštas koregavimas)
        2. Kietosios ribos (safety net)

        SVARBU: 0 reikšmės NĖRA imputuojamos - jos lieka 0.
        """
        missing_mask = df[target_col].isna()

        if not missing_mask.any():
            return

        # Naudojame df_with_geo (su geo features) prognozavimui
        X_missing_raw = df_with_geo.loc[missing_mask, feature_cols].copy()
        model_info = self.models[target_col]

        # Transformuojame trūkstamus duomenis
        X_missing = self._apply_transformations(
            X_missing_raw,
            model_info['num_cols'],
            model_info['cat_cols'],
            model_info['num_imputer'],
            model_info['encoder']
        )

        # XGBoost predikcija
        predictions = model_info['model'].predict(X_missing)

        # Hibridinis post-processing
        if 'geo' in df.columns and geo_stats:
            geo_values = df.loc[missing_mask, 'geo'].values
            year_values = df.loc[missing_mask, 'year'].values if 'year' in df.columns else None

            adjusted_predictions = self._apply_hybrid_post_processing(
                predictions, geo_values, year_values, geo_stats, target_col
            )
            df.loc[missing_mask, target_col] = adjusted_predictions
        else:
            df.loc[missing_mask, target_col] = predictions

    def _apply_hybrid_post_processing(
        self, predictions: np.ndarray, geo_values: np.ndarray,
        year_values: np.ndarray, geo_stats: dict, target_col: str
    ) -> np.ndarray:
        """
        PATOBULINTA v4 Hibridinis post-processing: MINIMALUS Shrinkage + Year-based Variation.

        PROBLEMA v3: Per daug vienodų reikšmių, nes shrinkage pritraukdavo prie to paties vidurkio.

        SPRENDIMAS v4:
        1. MINIMALUS shrinkage - tik ekstremaliems atvejams (>4σ)
        2. YEAR-BASED VARIATION - kiekvieni metai gauna unikalų offset pagal trend
        3. IŠLAIKOMA XGB VARIACIJA - XGB prognozės išlaikomos maksimaliai
        4. DIDESNIS JITTER - užtikrina unikalias reikšmes

        Algoritmas:
        1. XGB prognozė išlaikoma beveik nepakeista (95%+ išlaikoma)
        2. Shrinkage taikomas TIK kai nuokrypis > 4σ nuo year_specific_mean
        3. Year-based jitter užtikrina, kad skirtingi metai turi skirtingas reikšmes
        4. Kietosios ribos kaip safety net

        Returns:
            np.ndarray: Koreguotos prognozės su išlaikyta variacija
        """
        adjusted = predictions.copy()
        shrinkage_info = []

        # v4: MINIMALUS shrinkage - tik ekstremaliems atvejams
        base_max_shrinkage = 0.10  # Sumažinta nuo 0.20 - išlaikome 90% XGB info
        sigmoid_steepness = 0.4  # Sumažinta nuo 0.6 - dar švelnesnis perėjimas
        base_shrinkage_k = self.shrinkage_k * 1.5  # Padidintas k - shrinkage pradedamas vėliau

        # Apskaičiuojame globalų CV (coefficient of variation) rodikliui
        all_means = [s.get('mean', 0) for s in geo_stats.values() if s.get('mean', 0) != 0]
        all_stds = [s.get('std', 0) for s in geo_stats.values()]
        if all_means and all_stds:
            global_mean = np.mean([abs(m) for m in all_means])
            global_std = np.mean(all_stds)
            global_cv = global_std / global_mean if global_mean > 0 else 0
        else:
            global_cv = 0

        # Dinamiškas mean_year skaičiavimas
        mean_year = 2012  # Default (vidurys 2000-2024)
        if year_values is not None and len(year_values) > 0:
            mean_year = int(np.median(year_values))

        # v4: Dar labiau sumažiname shrinkage pagal CV
        if global_cv > 1.0:
            # Didelis CV - praktiškai išjungiame shrinkage
            base_max_shrinkage = 0.05
            k_adaptive = base_shrinkage_k * 2.0
        elif global_cv > 0.5:
            # Vidutinis CV - minimalus shrinkage
            base_max_shrinkage = 0.08
            k_adaptive = base_shrinkage_k * 1.5
        else:
            # Mažas CV - šiek tiek daugiau shrinkage leidžiama
            k_adaptive = base_shrinkage_k

        for i, (pred, geo) in enumerate(zip(predictions, geo_values)):
            stats = geo_stats.get(geo, {})

            if not stats:
                continue

            geo_mean = stats.get('mean', pred)
            geo_max = stats.get('max', pred)
            geo_min = stats.get('min', pred)
            geo_std = stats.get('std', 0)
            geo_trend = stats.get('trend', 0)

            # Apsauga nuo dalybos iš nulio
            if geo_std < 1e-10:
                geo_std = abs(geo_mean) * 0.1 if geo_mean != 0 else 1.0

            # Year-specific target mean su trend
            year_specific_mean = geo_mean
            year = year_values[i] if year_values is not None else mean_year
            if geo_trend != 0:
                year_diff = year - mean_year
                trend_adjustment = geo_trend * year_diff
                year_specific_mean = geo_mean + trend_adjustment

            # Skaičiuojame nuokrypį standartinėmis paklaidomis
            deviation = abs(pred - year_specific_mean)
            deviation_in_std = deviation / geo_std

            # Nustatome ar duomenys gali būti neigiami
            allows_negative = geo_min < 0

            # v4: Adaptyvus max_shrinkage - mažesnis, kai mažai duomenų
            # Tai neleidžia per daug pritraukti prie vidurkio
            max_shrinkage_final = base_max_shrinkage

            # Jei trend yra stiprus, dar labiau sumažiname shrinkage
            trend_strength = abs(geo_trend) / geo_std if geo_std > 0 else 0
            if trend_strength > 0.05:
                max_shrinkage_final *= max(0.3, 1.0 - trend_strength * 2)

            # ===== v5: GRIEŽTESNĖS KIETOSIOS RIBOS =====
            # Problema: kai raw prognozė labai netinkama dėl NaN features,
            # bounds turi būti griežtesni, kad imputuota reikšmė būtų artima
            # žinomoms to regiono reikšmėms.
            #
            # Naudojame: year_specific_mean +/- 3*std (tradicinis 99.7% intervalas)
            # Papildomai: max/min * 1.1 (leisti tik 10% viršyti žinomą diapazoną)
            if allows_negative:
                if geo_std > 0:
                    lower_bound = max(geo_min * 1.1, year_specific_mean - 3 * geo_std)
                    upper_bound = min(geo_max * 1.1, year_specific_mean + 3 * geo_std)
                else:
                    lower_bound = geo_min * 1.1
                    upper_bound = geo_max * 1.1
            else:
                if geo_std > 0:
                    # Teigiami duomenys - griežtesnės ribos
                    lower_bound = max(0, max(geo_min * 0.8, year_specific_mean - 3 * geo_std))
                    upper_bound = min(geo_max * 1.1, year_specific_mean + 3 * geo_std)
                else:
                    lower_bound = max(0, geo_min * 0.8)
                    upper_bound = geo_max * 1.1

            # ===== v4: MINIMALUS SHRINKAGE =====
            # Shrinkage taikomas TIK kai nuokrypis > k_adaptive σ
            # Ir net tada - labai švelniai
            sigmoid_input = (deviation_in_std - k_adaptive) * sigmoid_steepness
            w = max_shrinkage_final / (1 + np.exp(-sigmoid_input))

            # v4: Papildomas ribojimas - shrinkage max 10% net ekstremaliems atvejams
            w = min(w, 0.10)

            # Empirical Bayes Shrinkage formulė
            adjusted_pred = w * year_specific_mean + (1 - w) * pred

            # ===== v6: SMART BOUNDS SU YEAR-BASED VARIACIJA =====
            # Problema v5: kai prognozė viršija bounds, np.clip() sukuria vienodas reikšmes.
            # Sprendimas: naudojame "soft bounds" - interpoliuojame tarp year_specific_mean
            # ir bounds ribos, su year-based variacija.

            if year_values is not None:
                year = year_values[i]

                # Deterministinis seed pagal geo, year ir target_col
                seed = hash(f"{geo}_{year}_{target_col}_v6") % 100000
                np.random.seed(seed)

                # Bazinis jitter: 3-8% nuo geo_std (mažesnis nei v4, nes dabar visada pridedamas)
                jitter_pct = 0.03 + (year - 2000) * 0.002
                jitter_base = geo_std * min(jitter_pct, 0.08)

                # Trend-based offset: skirtingi metai turi skirtingą offset
                year_offset = (year - mean_year) * geo_std * 0.015  # 1.5% nuo std per metus

                # Random jitter
                random_jitter = np.random.normal(0, jitter_base)

                # Skaičiuojame galutinę reikšmę priklausomai nuo to, ar viršijame bounds
                if adjusted_pred > upper_bound:
                    # Prognozė viršija upper_bound - interpoliuojame tarp year_mean ir upper_bound
                    # Kuo labiau viršija, tuo arčiau upper_bound
                    overshoot_ratio = min((adjusted_pred - upper_bound) / (geo_std + 1e-9), 3.0)
                    # Interpoliacijos koeficientas: 0.5 = viduryje tarp mean ir bound
                    interp_factor = 0.5 + 0.15 * overshoot_ratio  # 0.5-0.95
                    interp_factor = min(interp_factor, 0.92)  # Max 92% link bound

                    # Interpoliuojame: year_mean -> upper_bound
                    base_value = year_specific_mean + interp_factor * (upper_bound - year_specific_mean)
                    final_pred = base_value + year_offset + random_jitter

                    # Užtikriname, kad neviršijame upper_bound
                    final_pred = min(final_pred, upper_bound)

                elif adjusted_pred < lower_bound:
                    # Prognozė žemiau lower_bound - interpoliuojame tarp year_mean ir lower_bound
                    undershoot_ratio = min((lower_bound - adjusted_pred) / (geo_std + 1e-9), 3.0)
                    interp_factor = 0.5 + 0.15 * undershoot_ratio
                    interp_factor = min(interp_factor, 0.92)

                    # Interpoliuojame: year_mean -> lower_bound
                    base_value = year_specific_mean - interp_factor * (year_specific_mean - lower_bound)
                    final_pred = base_value + year_offset + random_jitter

                    # Užtikriname, kad neeiname žemiau lower_bound
                    final_pred = max(final_pred, lower_bound)

                else:
                    # Prognozė tarp bounds - tiesiog pridedame variaciją
                    final_pred = adjusted_pred + year_offset + random_jitter
                    # Clip jei reikia
                    final_pred = np.clip(final_pred, lower_bound, upper_bound)
            else:
                # Jei nėra year info, tiesiog clip
                final_pred = np.clip(adjusted_pred, lower_bound, upper_bound)

            # Nustatome metodą ataskaitai
            if w > 0.005:  # Mažesnis slenkstis
                if final_pred != adjusted_pred:
                    method = 'bounds'
                else:
                    method = 'shrinkage'
            elif final_pred != pred:
                method = 'bounds'
            else:
                method = 'none'

            adjusted[i] = final_pred

            # Saugome info apie koregavimą (tik jei buvo koreguota)
            if method != 'none':
                shrinkage_info.append({
                    'geo': geo,
                    'year': year,
                    'original_pred': pred,
                    'shrinkage_pred': adjusted_pred,
                    'adjusted_pred': final_pred,
                    'bounds': (lower_bound, upper_bound),
                    'geo_mean': geo_mean,
                    'year_specific_mean': year_specific_mean,
                    'geo_std': geo_std,
                    'deviation_std': deviation_in_std,
                    'shrinkage_weight': w,
                    'trend': geo_trend,
                    'method': method
                })

        # Išsaugome shrinkage informaciją
        if shrinkage_info:
            self.shrinkage_applied[target_col] = shrinkage_info

        return adjusted

    def get_shrinkage_report(self):
        """
        Grąžina ataskaitą apie pritaikytą hibridinį post-processing.

        Returns:
            dict: Informacija apie koreguotas prognozes pagal rodiklius
        """
        return self.shrinkage_applied

    def print_shrinkage_summary(self):
        """
        Spausdina suvestinę apie hibridinį post-processing (Empirical Bayes + ribos).
        """
        if not self.shrinkage_applied:
            print("Post-processing nebuvo pritaikytas arba nebuvo reikalingas.")
            return

        print("=" * 80)
        print("HIBRIDINIS POST-PROCESSING SUVESTINE (XGBoost)")
        print("(Empirical Bayes Shrinkage + Kietosios Ribos)")
        print("=" * 80)

        total_adjustments = 0
        shrinkage_count = 0
        bounds_count = 0

        for target_col, adjustments in self.shrinkage_applied.items():
            if adjustments:
                total_adjustments += len(adjustments)
                col_shrinkage = sum(1 for a in adjustments if a.get('method') == 'shrinkage')
                col_bounds = sum(1 for a in adjustments if a.get('method') == 'bounds')
                shrinkage_count += col_shrinkage
                bounds_count += col_bounds

                print(f"\n{target_col}:")
                print(f"  Koreguotu prognoziu: {len(adjustments)}")
                print(f"    - Empirical Bayes Shrinkage: {col_shrinkage}")
                print(f"    - Kietosios ribos (safety): {col_bounds}")

                # Rodome kelis pavyzdžius
                examples = adjustments[:5]
                if examples:
                    print("  Pavyzdziai:")
                    for adj in examples:
                        geo = adj.get('geo', '?')
                        year = adj.get('year', '?')
                        orig = adj.get('original_pred', 0)
                        final = adj.get('adjusted_pred', 0)
                        method = adj.get('method', '?')
                        shrink_w = adj.get('shrinkage_weight', 0)
                        print(f"    {geo} ({year}): {orig:.2f} -> {final:.2f} "
                              f"[{method}, w={shrink_w:.2f}]")

        print(f"\nViso koreguotu prognoziu: {total_adjustments}")
        print(f"  - Empirical Bayes Shrinkage: {shrinkage_count}")
        print(f"  - Kietosios ribos: {bounds_count}")
        print("=" * 80)

    # ============================================================================
    # METRICS & STORAGE
    # ============================================================================

    def _save_test_metrics(self, target_col: str, y_test, y_pred, total_samples: int):
        """Apskaičiuoja ir išsaugo test metrikos."""
        # Apskaičiuojame bazines metrikos
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(np.mean(np.abs(y_test.values - y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        smape = self._calculate_smape(y_test.values, y_pred)

        # Normalizuojame RMSE ir MAE pagal reikšmių diapazoną
        y_range = float(y_test.max() - y_test.min())
        if y_range > 0:
            nrmse = rmse / y_range
            nmae = mae / y_range
        else:
            # Jei visos reikšmės vienodos, naudojame originalias metrikos
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
        Symmetric MAPE [%].

        Veikia ir su 0 reikšmėmis, nes vardiklis niekada nebūna 0.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        denominator = np.abs(y_true) + np.abs(y_pred) + 1e-9
        return float(100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denominator))

    def _save_model(self, target_col, model, num_cols, cat_cols, num_imputer, encoder):
        """Išsaugo modelį ir transformerius."""
        self.models[target_col] = {
            'model': model,
            'num_cols': num_cols,
            'cat_cols': cat_cols,
            'num_imputer': num_imputer,
            'encoder': encoder
        }

    def _save_feature_importance(self, target_col: str):
        """Išsaugo feature importance."""
        model_info = self.models[target_col]
        feature_names = (model_info['num_cols'] or []) + (model_info['cat_cols'] or [])
        importances = model_info['model'].feature_importances_

        k = min(len(feature_names), len(importances))
        importance_dict = {
            feature_names[i]: float(importances[i])
            for i in range(k)
        }

        # Rūšiuojame pagal reikšmę
        self.feature_importance[target_col] = dict(
            sorted(importance_dict.items(), key=lambda kv: kv[1], reverse=True)
        )
