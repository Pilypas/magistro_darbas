"""
Random Forest modelio implementacija trūkstamų reikšmių užpildymui
Magistro darbas - 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder

class RandomForestImputer:
    """
    Random Forest modelio klasė trūkstamų reikšmių užpildymui
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        """
        Inicijuoja Random Forest imputavimo modelį
        
        Parameters:
        -----------
        n_estimators : int
            Medžių skaičius miškai
        random_state : int
            Atsitiktinumo kontrolės parametras
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = {}
        self.feature_importance = {}
        self.label_encoders = {}
        self.encoded_columns = set()
        self.model_metrics = {}
        
    def _encode_categorical_features(self, df):
        """Koduoja kategorinio tipo požymius Random Forest modeliui"""
        df_encoded = df.copy()
        
        for col in df.columns:
            if df[col].dtype == 'object':
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    non_null_values = df_encoded[col].dropna()
                    if len(non_null_values) > 0:
                        self.label_encoders[col].fit(non_null_values)
                        self.encoded_columns.add(col)
                
                if col in self.encoded_columns:
                    non_null_mask = df_encoded[col].notna()
                    if non_null_mask.sum() > 0:
                        df_encoded.loc[non_null_mask, col] = self.label_encoders[col].transform(
                            df_encoded.loc[non_null_mask, col]
                        )
        
        return df_encoded
    
    def _create_model(self, is_categorical=False):
        """Sukuria Random Forest modelį pagal duomenų tipą"""
        if is_categorical:
            return RandomForestClassifier(
                n_estimators=self.n_estimators, 
                random_state=self.random_state,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt'
            )
        else:
            return RandomForestRegressor(
                n_estimators=self.n_estimators, 
                random_state=self.random_state,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt'
            )
    
    def fit_and_impute(self, df):
        """
        Apmoko Random Forest modelius ir užpildo trūkstamas reikšmes
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Duomenų rinkinys su trūkstamomis reikšmėmis
            
        Returns:
        --------
        pandas.DataFrame
            Duomenų rinkinys su užpildytomis reikšmėmis
        """
        # Koduojame kategorinio tipo požymius
        df_encoded = self._encode_categorical_features(df)
        
        # Konvertuojame į skaičius
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                try:
                    df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
                except:
                    pass
        
        # Imputuojame kiekvieną stulpelį su trūkstamomis reikšmėmis
        for col in df_encoded.columns:
            if df_encoded[col].isna().any():
                self._impute_column(df_encoded, col)
        
        # Dekoduojame kategorinius požymius atgal
        df_final = df_encoded.copy()
        for col in self.encoded_columns:
            if col in df_final.columns:
                try:
                    # Užtikriname, kad reikšmės yra teisingame diapazone
                    valid_range = range(len(self.label_encoders[col].classes_))
                    df_final[col] = df_final[col].round().astype(int)
                    df_final[col] = df_final[col].clip(
                        lower=valid_range[0], 
                        upper=valid_range[-1]
                    )
                    df_final[col] = self.label_encoders[col].inverse_transform(df_final[col])
                except Exception as e:
                    print(f"Įspėjimas dekodavime {col}: {e}")
                    pass
        
        return df_final
    
    def _impute_column(self, df_encoded, col):
        """Užpildo vieną stulpelį naudojant Random Forest"""
        # Paruošiame duomenis
        feature_cols = [c for c in df_encoded.columns if c != col]
        
        # Filtruojame eilutes, kuriose tikslinė reikšmė nėra tuščia
        mask_not_missing = df_encoded[col].notna()
        
        if mask_not_missing.sum() < 5:
            # Per mažai duomenų - naudojame paprastą užpildymą
            if col in self.encoded_columns:
                fill_value = int(df_encoded[col].mode().iloc[0]) if not df_encoded[col].mode().empty else 0
            else:
                fill_value = df_encoded[col].mean()
            df_encoded[col].fillna(fill_value, inplace=True)
            return
        
        # Paruošiame mokymo duomenis
        X_train = df_encoded.loc[mask_not_missing, feature_cols].copy()
        y_train = df_encoded.loc[mask_not_missing, col].copy()
        
        # Užpildome trūkstamas reikšmes požymiuose
        for feature_col in X_train.columns:
            if X_train[feature_col].isna().any():
                fill_val = X_train[feature_col].mean() if X_train[feature_col].dtype in ['float64', 'int64'] else X_train[feature_col].mode().iloc[0] if not X_train[feature_col].mode().empty else 0
                X_train[feature_col].fillna(fill_val, inplace=True)
        
        # Sukuriame ir apmokome modelį
        is_categorical = col in self.encoded_columns
        model = self._create_model(is_categorical)
        
        # Vertinimo duomenų padalijimas
        if len(y_train) > 10:
            eval_random_state = self.random_state + hash('random_forest') % 1000
            X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
                X_train, y_train, test_size=0.2, random_state=eval_random_state
            )
            model.fit(X_train_eval, y_train_eval)
            
            # Skaičiuojame metrikas
            y_pred = model.predict(X_test_eval)
            metrics = {}
            
            if is_categorical:
                metrics['accuracy'] = accuracy_score(y_test_eval, np.round(y_pred))
                metrics['model_type'] = 'classification'
            else:
                rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred))
                r2 = r2_score(y_test_eval, y_pred)
                
                # MAPE skaičiavimas
                mape = 0
                if not np.any(y_test_eval == 0):
                    try:
                        mape = mean_absolute_percentage_error(y_test_eval, y_pred) * 100
                    except:
                        mape = np.mean(np.abs((y_test_eval - y_pred) / np.where(y_test_eval != 0, y_test_eval, 1))) * 100
                
                metrics = {
                    'rmse': float(rmse),
                    'r2': float(r2),
                    'mape': float(mape),
                    'model_type': 'regression',
                    'mae': float(np.mean(np.abs(y_test_eval - y_pred))),
                    'sample_size': len(y_test_eval)
                }
            
            self.model_metrics[col] = metrics
            
            # Iš naujo apmokome visais duomenimis
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
            self.model_metrics[col] = {'model_type': 'insufficient_data', 'sample_size': len(y_train)}
        
        self.models[col] = model
        
        # Išsaugome požymių svarbą
        importance_dict = dict(zip(X_train.columns, model.feature_importances_))
        self.feature_importance[col] = importance_dict
        
        # Prognozuojame trūkstamas reikšmes
        missing_mask = df_encoded[col].isna()
        if missing_mask.any():
            X_missing = df_encoded.loc[missing_mask, feature_cols].copy()
            
            # Užpildome trūkstamas reikšmes požymiuose
            for feature_col in X_missing.columns:
                if X_missing[feature_col].isna().any():
                    fill_val = X_train[feature_col].mean() if X_train[feature_col].dtype in ['float64', 'int64'] else X_train[feature_col].mode().iloc[0] if not X_train[feature_col].mode().empty else 0
                    X_missing[feature_col].fillna(fill_val, inplace=True)
            
            # Prognozuojame
            predictions = model.predict(X_missing)
            
            # Užpildome trūkstamas reikšmes
            if is_categorical:
                predictions = np.round(predictions).astype(int)
            
            df_encoded.loc[missing_mask, col] = predictions
    
    def get_feature_importance(self):
        """Grąžina požymių svarbos informaciją"""
        return self.feature_importance
    
    def get_model_metrics(self):
        """Grąžina modelio metrikas"""
        return self.model_metrics