"""
Random Forest modelio implementacija ekonominių rodiklių trūkstamų reikšmių užpildymui
=======================================================================================

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

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score


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


class RandomForestImputer:
    """
    Random Forest imputavimas ekonominiams rodikliams.

    Kiekvienam stulpeliui su trūkstamomis reikšmėmis treniruojamas atskiras RF.
    Synthetic test: 20% indeksų -> TEST (tik iš eilučių, kur target != 0).
    Po vertinimo pertreniruojama ant 100% žinomų taikinių.
    """

    def __init__(
        self,
        n_estimators=100,
        random_state=42,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        categorical_cols=None,
        exclude_columns=None
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.categorical_cols = categorical_cols or ['geo', 'year']
        self.exclude_columns = exclude_columns or []

        self.models = {}
        self.feature_importance = {}
        self.model_metrics = {}
        self.test_predictions = {}

    # ============================================================================
    # PUBLIC API
    # ============================================================================

    def fit_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Treniruoja modelius ir imputuoja trūkstamas reikšmes.

        Args:
            df: DataFrame su trūkstamomis reikšmėmis

        Returns:
            DataFrame su imputuotomis reikšmėmis (tik NaN, ne 0)
        """
        df_work = self._prepare_dataframe(df)
        self._validate_categorical_columns(df_work)

        # Imputuojame kiekvieną stulpelį su NaN reikšmėmis
        for target_col in df_work.columns:
            if self._should_impute_column(df_work, target_col):
                self._impute_column(df_work, target_col)

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
                    "pagal duomenis to neturi būti."
                )

    def _should_impute_column(self, df: pd.DataFrame, col: str) -> bool:
        """Patikriname, ar stulpelis turėtų būti imputuojamas."""
        return (
            col not in self.categorical_cols and
            col not in self.exclude_columns and
            df[col].isna().any()
        )

    # ============================================================================
    # COLUMN IMPUTATION
    # ============================================================================

    def _impute_column(self, df_work: pd.DataFrame, target_col: str):
        """
        Imputuoja vieną stulpelį.

        Procesas:
        1. Synthetic test (20%) be 0 reikšmių
        2. Treniruoja ir vertina modelį
        3. Pertreniruoja ant 100% duomenų
        4. Imputuoja NaN reikšmes (0 neliečiami)
        """
        feature_cols = [c for c in df_work.columns if c != target_col]
        known_mask = df_work[target_col].notna()

        # Fallback į mean, jei per mažai duomenų
        if known_mask.sum() < 5:
            self._fallback_mean_impute(df_work, target_col, known_mask)
            return

        # Paruošiame duomenis
        X_all_raw = df_work.loc[known_mask, feature_cols].copy()
        y_all = df_work.loc[known_mask, target_col].copy()

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

        # Galutinė imputacija: užpildome tik NaN reikšmes
        self._perform_final_imputation(df_work, target_col, feature_cols)

        # Išsaugome feature importance
        self._save_feature_importance(target_col)

    def _fallback_mean_impute(self, df: pd.DataFrame, target_col: str, known_mask):
        """Fallback: imputuoja su mean, jei per mažai duomenų."""
        fill_value = df.loc[known_mask, target_col].mean()
        df.loc[~known_mask, target_col] = fill_value

        self.model_metrics[target_col] = {
            'nrmse': float('nan'),
            'r2': float('nan'),
            'nmae': float('nan'),
            'smape': float('nan'),
            'model_type': 'insufficient_data_mean_impute',
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
        bet nenaudojamos test'ui (kad išvengtume šališkumo).
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

        # Treniruojame modelį
        model = self._create_and_train_model(X_all, y_all.values)

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

        # Treniruojame ir vertiname
        model_eval = self._create_and_train_model(X_train, y_train.values)
        y_pred = model_eval.predict(X_test)

        # Skaičiuojame metrikos
        self._save_test_metrics(target_col, y_test, y_pred, len(y_all))

        # Pertreniruojame ant 100% duomenų
        num_imputer_full, encoder_full = self._fit_transformers(
            X_all_raw, num_cols, cat_cols
        )
        X_all = self._apply_transformations(
            X_all_raw, num_cols, cat_cols, num_imputer_full, encoder_full
        )
        model_full = self._create_and_train_model(X_all, y_all.values)

        # Išsaugome galutinį modelį
        self._save_model(
            target_col, model_full, num_cols, cat_cols,
            num_imputer_full, encoder_full
        )

    def _create_and_train_model(self, X, y):
        """Sukuria ir treniruoja Random Forest modelį."""
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features='sqrt',
            n_jobs=1,
            bootstrap=True,
            oob_score=False
        )
        model.fit(X, y)
        return model

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

    def _perform_final_imputation(self, df: pd.DataFrame, target_col: str, feature_cols):
        """
        Atlieka galutinę imputaciją: užpildo tik NaN reikšmes.

        SVARBU: 0 reikšmės NĖRA imputuojamos - jos lieka 0.
        """
        missing_mask = df[target_col].isna()

        if not missing_mask.any():
            return

        X_missing_raw = df.loc[missing_mask, feature_cols].copy()
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
            # Jei visos reikšmės vienodos, naudojame originalias metrikas
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
