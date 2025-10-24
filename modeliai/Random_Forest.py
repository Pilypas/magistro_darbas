import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

class RandomForestImputer:
    """
    Random Forest imputavimas ekonominiams rodikliams.

    • Tikslai su trūkstamomis reikšmėmis: kiekvienam stulpeliui treniruojamas atskiras RF.
    • Prediktoriai: kiti rodikliai + kategoriniai 'geo', 'time_period' (be trūkumų).
    • Synthetic test be leakage:
        - 20% indeksų -> TEST (pagal known target eilutes),
        - Skaitinių prediktorių imputacija (mean) FIT tik ant TRAIN,
        - 'geo'/'time_period' – tik OrdinalEncoder (be imputacijos), FIT tik ant TRAIN.
    • Po vertinimo pertreniruojama ant 100% žinomų taikinių realiai imputacijai.
    """

    def __init__(
        self,
        n_estimators=100,
        random_state=42,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        categorical_cols=None
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        # Kategoriniai prediktoriai be trūkumų
        self.categorical_cols = categorical_cols if categorical_cols else ['geo', 'time_period']

        self.models = {}
        self.feature_importance = {}
        self.model_metrics = {}
        self.test_predictions = {}

    # ---------- Pagalbinės ----------

    def _split_train_test_indices(self, indices, test_frac=0.2, seed=42):
        rng = np.random.RandomState(seed)
        n_test = max(1, int(len(indices) * test_frac))
        test_idx = rng.choice(indices, size=n_test, replace=False)
        test_mask = indices.isin(test_idx)
        return (~test_mask), test_mask  # train_mask, test_mask

    def _column_groups(self, X, feature_cols):
        # Skaitiniai = visi neskategoriniai, kurie yra numeric dtype
        num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c]) and c not in self.categorical_cols]
        # Kategoriniai = būtent nurodyti 'geo' ir 'time_period', jei egzistuoja kadre
        cat_cols = [c for c in self.categorical_cols if c in feature_cols]
        return num_cols, cat_cols

    def _create_model(self):
        return RandomForestRegressor(
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

    # ---------- API ----------

    def fit_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df_work = df.copy()

        # Konvertuojame ne-kategorinius 'object' į skaičius, jei įmanoma
        for col in df_work.columns:
            if (df_work[col].dtype == 'object') and (col not in self.categorical_cols):
                df_work[col] = pd.to_numeric(df_work[col], errors='coerce')

        # Saugumo patikra: garantuojame, kad cat stulpeliuose nėra NA
        for c in self.categorical_cols:
            if c in df_work.columns:
                if df_work[c].isna().any():
                    raise ValueError(f"Kategorinis stulpelis '{c}' turi trūkstamų reikšmių, "
                                     "pagal dizainą to neturėtų būti.")

        for target_col in df_work.columns:
            if target_col in self.categorical_cols:
                continue  # niekada neimputuojame kategorinių
            if df_work[target_col].isna().any():
                self._impute_column(df_work, target_col)

        return df_work

    def _impute_column(self, df_work: pd.DataFrame, target_col: str):
        feature_cols = [c for c in df_work.columns if c != target_col]
        not_missing_mask = df_work[target_col].notna()

        if not_missing_mask.sum() < 5:
            fill_value = df_work[target_col].mean()
            df_work.loc[~not_missing_mask, target_col] = fill_value
            self.model_metrics[target_col] = {
                'model_type': 'insufficient_data',
                'sample_size': int(not_missing_mask.sum())
            }
            return

        # Visi žinomi (vertinimui)
        X_all_raw = df_work.loc[not_missing_mask, feature_cols].copy()
        y_all = df_work.loc[not_missing_mask, target_col].copy()

        # Train/Test indeksai (synthetic test)
        train_mask, test_mask = self._split_train_test_indices(y_all.index, test_frac=0.2, seed=self.random_state)
        X_train_raw = X_all_raw.loc[train_mask].copy()
        X_test_raw  = X_all_raw.loc[test_mask].copy()
        y_train = y_all.loc[train_mask].copy()
        y_test  = y_all.loc[test_mask].copy()

        # Grupės
        num_cols, cat_cols = self._column_groups(X_all_raw, feature_cols)

        # --- TRANSFORMACIJOS VERTINIMUI (fit TIK ant TRAIN) ---

        # 1) Skaitiniai prediktoriai – gali turėti NA -> imputacija tik iš TRAIN
        if num_cols:
            num_imp_eval = SimpleImputer(strategy='mean')
            X_train_num = num_imp_eval.fit_transform(X_train_raw[num_cols])
            X_test_num  = num_imp_eval.transform(X_test_raw[num_cols])
        else:
            X_train_num = np.empty((len(X_train_raw), 0))
            X_test_num  = np.empty((len(X_test_raw), 0))

        # 2) Kategoriniai prediktoriai (geo, time_period) – be NA, tik kodavimas
        if cat_cols:
            enc_eval = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_train_cat = enc_eval.fit_transform(X_train_raw[cat_cols])
            X_test_cat  = enc_eval.transform(X_test_raw[cat_cols])
        else:
            X_train_cat = np.empty((len(X_train_raw), 0))
            X_test_cat  = np.empty((len(X_test_raw), 0))

        X_train_eval = np.hstack([X_train_num, X_train_cat])
        X_test_eval  = np.hstack([X_test_num, X_test_cat])

        # --- Treniruotė ir metrikos ---
        model_eval = self._create_model()
        model_eval.fit(X_train_eval, y_train.values)

        y_pred = model_eval.predict(X_test_eval)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2   = float(r2_score(y_test, y_pred))
        mae  = float(np.mean(np.abs(y_test.values - y_pred)))
        if not np.any(y_test.values == 0):
            try:
                mape = float(mean_absolute_percentage_error(y_test, y_pred) * 100)
            except Exception:
                mape = float(np.mean(np.abs((y_test.values - y_pred) /
                                            np.where(y_test.values != 0, y_test.values, 1))) * 100)
        else:
            mape = 0.0

        self.model_metrics[target_col] = {
            'rmse': rmse, 'r2': r2, 'mae': mae, 'mape': mape,
            'model_type': 'synthetic_test',
            'sample_size': int(len(y_test)),
            'total_samples': int(len(y_all))
        }
        self.test_predictions[target_col] = {
            'y_test': y_test.values, 'y_pred': y_pred,
            'r2': r2, 'mae': mae, 'rmse': rmse
        }

        # --- PERMOKYMAS ant 100% žinomų taikinių realiai imputacijai ---
        # Skaitinių prediktorių imputacija ir kategorijų kodavimas – FIT ant visų žinomų
        if num_cols:
            num_imp_full = SimpleImputer(strategy='mean')
            X_all_num = num_imp_full.fit_transform(X_all_raw[num_cols])
        else:
            num_imp_full = None
            X_all_num = np.empty((len(X_all_raw), 0))

        if cat_cols:
            enc_full = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_all_cat = enc_full.fit_transform(X_all_raw[cat_cols])
        else:
            enc_full = None
            X_all_cat = np.empty((len(X_all_raw), 0))

        X_all_full = np.hstack([X_all_num, X_all_cat])

        model_full = self._create_model()
        model_full.fit(X_all_full, y_all.values)

        # Išsaugome viską, ko reikės realiai imputacijai
        self.models[target_col] = {
            'model': model_full,
            'num_cols': num_cols,
            'cat_cols': cat_cols,
            'num_imputer': num_imp_full,
            'encoder': enc_full
        }

        # --- REALI IMPUTACIJA: užpildome target NA ---
        missing_mask = df_work[target_col].isna()
        if missing_mask.any():
            X_missing_raw = df_work.loc[missing_mask, feature_cols].copy()

            if num_cols:
                X_missing_num = num_imp_full.transform(X_missing_raw[num_cols])
            else:
                X_missing_num = np.empty((len(X_missing_raw), 0))

            if cat_cols:
                X_missing_cat = enc_full.transform(X_missing_raw[cat_cols])
            else:
                X_missing_cat = np.empty((len(X_missing_raw), 0))

            X_missing = np.hstack([X_missing_num, X_missing_cat])
            preds = model_full.predict(X_missing)
            df_work.loc[missing_mask, target_col] = preds

        # --- Feature importance (vienas-į-vieną pagal pradinius požymius) ---
        feature_names = (self.models[target_col]['num_cols'] or []) + (self.models[target_col]['cat_cols'] or [])
        importances = model_full.feature_importances_
        k = min(len(feature_names), len(importances))
        importance = {feature_names[i]: float(importances[i]) for i in range(k)}
        importance = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))
        self.feature_importance[target_col] = importance

    # ---------- Getteriai ----------

    def get_feature_importance(self):
        return self.feature_importance

    def get_model_metrics(self):
        return self.model_metrics

    def get_test_predictions(self, df=None):
        return self.test_predictions
