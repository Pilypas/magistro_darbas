from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
from modeliai import RandomForestImputer
try:
    from modeliai import XGBoostImputer
    XGBOOST_IMPUTER_AVAILABLE = True
except ImportError:
    XGBOOST_IMPUTER_AVAILABLE = False
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.utils
import json
import os
from werkzeug.utils import secure_filename
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import uuid
import json as json_module

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# MySQL Configuration - only works in production with environment variables
# No default values for security - requires proper environment setup
DB_CONFIG = {
    'host': os.environ.get('MYSQL_HOST'),
    'user': os.environ.get('MYSQL_USER'),
    'password': os.environ.get('MYSQL_PASSWORD'),
    'database': os.environ.get('MYSQL_DATABASE'),
    'connect_timeout': 10,
    'ssl_disabled': False,
    'autocommit': True,
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_unicode_ci'
}

# Check if all required MySQL environment variables are set
MYSQL_ENABLED = all([
    os.environ.get('MYSQL_HOST'),
    os.environ.get('MYSQL_USER'),
    os.environ.get('MYSQL_PASSWORD'),
    os.environ.get('MYSQL_DATABASE')
])

print(f"MySQL enabled: {MYSQL_ENABLED}")
if not MYSQL_ENABLED:
    print("MySQL environment variables not set - comments feature will be disabled")
    print("Required: MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE")

# Database connection manager
class DatabaseManager:
    def __init__(self, config, enabled=True):
        self.config = config
        self.connection = None
        self.enabled = enabled

    def get_connection(self):
        if not self.enabled:
            print("MySQL is disabled - environment variables not configured")
            return None

        try:
            if self.connection is None or not self.connection.is_connected():
                self.connection = mysql.connector.connect(**self.config)
            return self.connection
        except Error as e:
            print(f"Database connection error: {e}")
            return None

    def close_connection(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()

# Initialize database manager - only if MySQL is properly configured
db_manager = DatabaseManager(DB_CONFIG, enabled=MYSQL_ENABLED)

# Database initialization functions
def init_database_tables():
    """Initialize all required database tables"""
    if not MYSQL_ENABLED:
        return

    try:
        connection = db_manager.get_connection()
        if not connection:
            return

        cursor = connection.cursor()

        # Create imputacijos_rezultatai table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS imputacijos_rezultatai (
                id INT AUTO_INCREMENT PRIMARY KEY,
                rezultato_id VARCHAR(36) UNIQUE NOT NULL,
                originalus_failas VARCHAR(255) NOT NULL,
                imputuotas_failas VARCHAR(255) NOT NULL,
                modelio_tipas ENUM('random_forest', 'xgboost') NOT NULL,
                n_estimators INT NOT NULL,
                random_state INT NOT NULL,
                originalus_trukstamu_kiekis INT NOT NULL,
                imputuotas_trukstamu_kiekis INT NOT NULL,
                apdorotu_stulpeliu_kiekis INT NOT NULL,
                modelio_metrikos JSON,
                pozymiu_svarba JSON,
                grafikai JSON,
                sukurimo_data DATETIME DEFAULT CURRENT_TIMESTAMP,
                busena ENUM('vykdomas', 'baigtas', 'klaida') DEFAULT 'vykdomas'
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)

        cursor.close()
        connection.commit()
        print("Database tables initialized successfully")

    except Error as e:
        print(f"Error initializing database tables: {e}")

# Initialize tables on startup
init_database_tables()

# MLImputer wrapper klasė, kuri naudoja atskirų modelių failus
class MLImputer:
    """Wrapper klasė, kuri pasirenka tinkamą modelį pagal model_type"""
    
    def __init__(self, model_type='random_forest', n_estimators=100, random_state=42):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        # Inicijuojame tinkamą modelį
        if model_type == 'xgboost' and XGBOOST_IMPUTER_AVAILABLE:
            try:
                self.imputer = XGBoostImputer(n_estimators=n_estimators, random_state=random_state)
            except Exception as e:
                # Bet kokios klaidos - naudojame Random Forest
                self.imputer = RandomForestImputer(n_estimators=n_estimators, random_state=random_state)
                self.model_type = 'random_forest'
        else:
            # Naudojame Random Forest (arba jei XGBoost neprieinamas)
            if model_type == 'xgboost' and not XGBOOST_IMPUTER_AVAILABLE:
                self.model_type = 'random_forest'
            self.imputer = RandomForestImputer(n_estimators=n_estimators, random_state=random_state)
    
    def fit_transform(self, df):
        """Užpildo trūkstamas reikšmes naudojant pasirinktą modelį"""
        return self.imputer.fit_and_impute(df)
    
    def get_feature_importance(self):
        """Grąžina požymių svarbos informaciją"""
        return self.imputer.get_feature_importance()
    
    def get_model_metrics(self):
        """Grąžina modelio metrikas"""
        return self.imputer.get_model_metrics()

@app.route('/')
def index():
    return send_file('templates/index.html')

@app.route('/imputacija')
def imputacija():
    return send_file('templates/imputacija.html')

@app.route('/apie')
def apie():
    return send_file('templates/apie.html')

@app.route('/komentarai')
def komentarai():
    return send_file('templates/komentarai.html')

@app.route('/rezultatai')
def rezultatai():
    return send_file('templates/rezultatai.html')

@app.route('/rezultatai/<result_id>')
def rezultatas_detali(result_id):
    return send_file('templates/rezultatas_detali.html')

@app.route('/palyginimas')
def palyginimas():
    return send_file('templates/palyginimas.html')

@app.route('/api/rezultatai', methods=['GET'])
def get_rezultatai():
    """Get all imputation results"""
    if not MYSQL_ENABLED:
        return jsonify({
            "rezultatai": [],
            "message": "Rezultatų funkcija neprieinama - duomenų bazė nesukonfigūruota"
        })

    try:
        connection = db_manager.get_connection()
        if not connection:
            return jsonify({"error": "Nepavyko prisijungti prie duomenų bazės"}), 500

        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT rezultato_id, originalus_failas, imputuotas_failas,
                   modelio_tipas, n_estimators, random_state,
                   originalus_trukstamu_kiekis, imputuotas_trukstamu_kiekis,
                   apdorotu_stulpeliu_kiekis, sukurimo_data, busena
            FROM imputacijos_rezultatai
            ORDER BY sukurimo_data DESC
        """)
        rezultatai = cursor.fetchall()
        cursor.close()

        return jsonify({"rezultatai": rezultatai})

    except Error as e:
        return jsonify({"error": f"Duomenų bazės klaida: {str(e)}"}), 500

@app.route('/api/rezultatai/<result_id>', methods=['GET'])
def get_rezultatas(result_id):
    """Get detailed imputation result"""
    if not MYSQL_ENABLED:
        return jsonify({"error": "Rezultatų funkcija neprieinama - duomenų bazė nesukonfigūruota"}), 503

    try:
        connection = db_manager.get_connection()
        if not connection:
            return jsonify({"error": "Nepavyko prisijungti prie duomenų bazės"}), 500

        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM imputacijos_rezultatai
            WHERE rezultato_id = %s
        """, (result_id,))

        rezultatas = cursor.fetchone()
        cursor.close()

        if not rezultatas:
            return jsonify({"error": "Rezultatas nerastas"}), 404

        # Parse JSON fields
        if rezultatas['modelio_metrikos']:
            rezultatas['modelio_metrikos'] = json_module.loads(rezultatas['modelio_metrikos'])
        if rezultatas['pozymiu_svarba']:
            rezultatas['pozymiu_svarba'] = json_module.loads(rezultatas['pozymiu_svarba'])
        if rezultatas['grafikai']:
            rezultatas['grafikai'] = json_module.loads(rezultatas['grafikai'])

        return jsonify({"rezultatas": rezultatas})

    except Error as e:
        return jsonify({"error": f"Duomenų bazės klaida: {str(e)}"}), 500

@app.route('/api/palyginimas', methods=['POST'])
def compare_results():
    """Compare multiple imputation results"""
    if not MYSQL_ENABLED:
        return jsonify({"error": "Palyginimo funkcija neprieinama - duomenų bazė nesukonfigūruota"}), 503

    try:
        data = request.get_json()
        result_ids = data.get('result_ids', [])

        if len(result_ids) < 2:
            return jsonify({"error": "Pasirinkite bent 2 rezultatus palyginimui"}), 400

        connection = db_manager.get_connection()
        if not connection:
            return jsonify({"error": "Nepavyko prisijungti prie duomenų bazės"}), 500

        cursor = connection.cursor(dictionary=True)

        # Get selected results
        placeholders = ','.join(['%s'] * len(result_ids))
        cursor.execute(f"""
            SELECT * FROM imputacijos_rezultatai
            WHERE rezultato_id IN ({placeholders})
            ORDER BY sukurimo_data DESC
        """, result_ids)

        rezultatai = cursor.fetchall()
        cursor.close()

        if len(rezultatai) != len(result_ids):
            return jsonify({"error": "Kai kurie rezultatai nerasti"}), 404

        # Parse JSON fields for each result (skip heavy graphics data for comparison)
        for rezultatas in rezultatai:
            if rezultatas['modelio_metrikos']:
                rezultatas['modelio_metrikos'] = json_module.loads(rezultatas['modelio_metrikos'])
            if rezultatas['pozymiu_svarba']:
                rezultatas['pozymiu_svarba'] = json_module.loads(rezultatas['pozymiu_svarba'])
            # Skip graphics for comparison to reduce data size
            rezultatas['grafikai'] = None

        # Generate comparison data
        comparison_data = generate_comparison_data(rezultatai)

        return jsonify({
            "rezultatai": rezultatai,
            "palyginimo_duomenys": comparison_data
        })

    except Error as e:
        return jsonify({"error": f"Duomenų bazės klaida: {str(e)}"}), 500

def generate_comparison_data(results):
    """Generate comparison analysis data"""
    comparison = {
        "bendroji_statistika": {},
        "modeliu_palyginimas": {},
        "efektyvumo_rodikliai": {},
        "rekomendacijos": []
    }

    # Basic statistics comparison
    comparison["bendroji_statistika"] = {
        "rezultatu_skaicius": len(results),
        "modeliu_tipai": list(set([r['modelio_tipas'] for r in results])),
        "bendras_trukstamu_uzpildymas": sum([r['originalus_trukstamu_kiekis'] - r['imputuotas_trukstamu_kiekis'] for r in results])
    }

    # Model comparison
    model_stats = {}
    for result in results:
        model_type = result['modelio_tipas']
        if model_type not in model_stats:
            model_stats[model_type] = {
                "kiekis": 0,
                "vidutinis_uzpildymas": 0,
                "vidutinis_rmse": 0,
                "vidutinis_r2": 0,
                "vidutinis_laikas": 0
            }

        model_stats[model_type]["kiekis"] += 1
        uzpildymas = result['originalus_trukstamu_kiekis'] - result['imputuotas_trukstamu_kiekis']
        model_stats[model_type]["vidutinis_uzpildymas"] += uzpildymas

        # Calculate average metrics from model_metrics
        if result['modelio_metrikos']:
            rmse_values = []
            r2_values = []
            for col_metrics in result['modelio_metrikos'].values():
                if col_metrics.get('rmse'):
                    rmse_values.append(col_metrics['rmse'])
                if col_metrics.get('r2'):
                    r2_values.append(col_metrics['r2'])

            if rmse_values:
                model_stats[model_type]["vidutinis_rmse"] += sum(rmse_values) / len(rmse_values)
            if r2_values:
                model_stats[model_type]["vidutinis_r2"] += sum(r2_values) / len(r2_values)

    # Calculate averages
    for model_type, stats in model_stats.items():
        if stats["kiekis"] > 0:
            stats["vidutinis_uzpildymas"] /= stats["kiekis"]
            stats["vidutinis_rmse"] /= stats["kiekis"]
            stats["vidutinis_r2"] /= stats["kiekis"]

    comparison["modeliu_palyginimas"] = model_stats

    # Efficiency indicators
    comparison["efektyvumo_rodikliai"] = calculate_efficiency_indicators(results)

    # Generate recommendations
    comparison["rekomendacijos"] = generate_recommendations(model_stats, results)

    return comparison

def calculate_efficiency_indicators(results):
    """Calculate efficiency indicators for comparison"""
    indicators = {
        "geriausias_uzpildymas": None,
        "geriausias_rmse": None,
        "geriausias_r2": None,
        "greičiausias": None
    }

    best_filling = max(results, key=lambda r: r['originalus_trukstamu_kiekis'] - r['imputuotas_trukstamu_kiekis'])
    indicators["geriausias_uzpildymas"] = {
        "rezultato_id": best_filling['rezultato_id'],
        "modelis": best_filling['modelio_tipas'],
        "uzpildyta": best_filling['originalus_trukstamu_kiekis'] - best_filling['imputuotas_trukstamu_kiekis']
    }

    return indicators

def generate_recommendations(model_stats, results):
    """Generate recommendations based on comparison"""
    recommendations = []

    if len(model_stats) >= 2:
        rf_stats = model_stats.get('random_forest', {})
        xgb_stats = model_stats.get('xgboost', {})

        if rf_stats and xgb_stats:
            if rf_stats.get('vidutinis_r2', 0) > xgb_stats.get('vidutinis_r2', 0):
                recommendations.append({
                    "tipas": "modelio_pasirinkimas",
                    "tekstas": "Random Forest modelis rodo geresnį R² rezultatą ir yra rekomenduojamas tikslumui.",
                    "prioritetas": "aukštas"
                })
            else:
                recommendations.append({
                    "tipas": "modelio_pasirinkimas",
                    "tekstas": "XGBoost modelis rodo geresnį R² rezultatą ir yra rekomenduojamas tikslumui.",
                    "prioritetas": "aukštas"
                })

    if len(results) > 3:
        recommendations.append({
            "tipas": "duomenu_kokybe",
            "tekstas": "Rekomenduojama patikrinti duomenų kokybę - atlikta daug bandymų.",
            "prioritetas": "vidutinis"
        })

    return recommendations

@app.route('/api/komentarai', methods=['GET'])
def get_komentarai():
    # Check if MySQL is enabled
    if not MYSQL_ENABLED:
        return jsonify({
            "komentarai": [],
            "message": "Komentarų funkcija neprieinama - duomenų bazė nesukonfigūruota"
        })

    try:
        connection = db_manager.get_connection()
        if not connection:
            return jsonify({"error": "Nepavyko prisijungti prie duomenų bazės"}), 500

        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, vardas, el_pastas, komentaras, sukurimo_data
            FROM komentarai
            ORDER BY sukurimo_data DESC
        """)
        komentarai = cursor.fetchall()
        cursor.close()

        return jsonify({"komentarai": komentarai})
    except Error as e:
        return jsonify({"error": f"Duomenų bazės klaida: {str(e)}"}), 500

@app.route('/api/komentarai', methods=['POST'])
def add_komentaras():
    # Check if MySQL is enabled
    if not MYSQL_ENABLED:
        return jsonify({
            "error": "Komentarų funkcija neprieinama - duomenų bazė nesukonfigūruota. Susisiekite su administratoriumi."
        }), 503

    try:
        data = request.get_json()
        vardas = data.get('vardas', '').strip()
        el_pastas = data.get('el_pastas', '').strip()
        komentaras = data.get('komentaras', '').strip()

        if not vardas or not komentaras:
            return jsonify({"error": "Vardas ir komentaras yra privalomi"}), 400

        connection = db_manager.get_connection()
        if not connection:
            return jsonify({"error": "Nepavyko prisijungti prie duomenų bazės"}), 500

        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS komentarai (
                id INT AUTO_INCREMENT PRIMARY KEY,
                vardas VARCHAR(100) NOT NULL,
                el_pastas VARCHAR(255),
                komentaras TEXT NOT NULL,
                sukurimo_data DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)

        # Fix existing table charset if it exists
        cursor.execute("""
            ALTER TABLE komentarai
            CONVERT TO CHARACTER SET utf8mb4
            COLLATE utf8mb4_unicode_ci
        """)

        cursor.execute("""
            INSERT INTO komentarai (vardas, el_pastas, komentaras)
            VALUES (%s, %s, %s)
        """, (vardas, el_pastas if el_pastas else None, komentaras))

        connection.commit()
        cursor.close()

        return jsonify({"message": "Komentaras sėkmingai pridėtas"})
    except Error as e:
        return jsonify({"error": f"Duomenų bazės klaida: {str(e)}"}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Failas neįkeltas"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Failas nepasirinktas"}), 400
        
        if file and file.filename.lower().endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Read and analyze the CSV
            df = pd.read_csv(filepath)
            
            # Basic statistics
            stats = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict()
            }
            
            return jsonify({
                "message": "Failas sėkmingai įkeltas",
                "filename": filename,
                "stats": stats
            })
        
        return jsonify({"error": "Prašome įkelti CSV failą"}), 400
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze/<filename>')
def analyze_data(filename):
    try:
        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
        if not os.path.exists(filepath):
            return jsonify({"error": "Failas nerastas"}), 404
        
        df = pd.read_csv(filepath)
        
        # Generate visualizations
        plots = {}
        
        # Missing values heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
        plt.title('Trūkstamų reikšmių šilumos žemėlapis')
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plots['missing_heatmap'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Missing values bar chart
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]
        
        if len(missing_counts) > 0:
            plt.figure(figsize=(10, 6))
            missing_counts.plot(kind='bar')
            plt.title('Trūkstamų reikšmių kiekis pagal stulpelį')
            plt.xlabel('Stulpeliai')
            plt.ylabel('Trūkstamų reikšmių skaičius')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plots['missing_bar'] = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
        
        # Correlation matrix for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('Koreliacijos matrica')
            plt.tight_layout()
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plots['correlation'] = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
        
        # Summary statistics
        summary_stats = df.describe().to_dict()
        
        # Missing values percentage by column
        missing_by_column = []
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percent = round((missing_count / len(df)) * 100, 2)
            if missing_count > 0:  # Only include columns with missing values
                missing_by_column.append({
                    "column": col,
                    "missing_count": int(missing_count),
                    "missing_percentage": missing_percent,
                    "data_type": str(df[col].dtype)
                })
        
        # Sort by missing percentage (highest first)
        missing_by_column.sort(key=lambda x: x['missing_percentage'], reverse=True)
        
        return jsonify({
            "plots": plots,
            "summary_stats": summary_stats,
            "missing_summary": {
                "total_missing": int(df.isnull().sum().sum()),
                "columns_with_missing": len(df.columns[df.isnull().any()]),
                "complete_rows": len(df.dropna()),
                "missing_percentage": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2)
            },
            "missing_by_column": missing_by_column
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def save_imputation_result(filename, imputed_filename, model_type, n_estimators, random_state,
                          original_missing, imputed_missing, plots, feature_importance, model_metrics):
    """Save imputation result to database"""
    if not MYSQL_ENABLED:
        return None

    try:
        connection = db_manager.get_connection()
        if not connection:
            return None

        result_id = str(uuid.uuid4())

        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO imputacijos_rezultatai (
                rezultato_id, originalus_failas, imputuotas_failas,
                modelio_tipas, n_estimators, random_state,
                originalus_trukstamu_kiekis, imputuotas_trukstamu_kiekis,
                apdorotu_stulpeliu_kiekis, modelio_metrikos,
                pozymiu_svarba, grafikai, busena
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            result_id, filename, imputed_filename, model_type,
            n_estimators, random_state, original_missing, imputed_missing,
            len(feature_importance) if feature_importance else 0,
            json_module.dumps(model_metrics) if model_metrics else None,
            json_module.dumps(feature_importance) if feature_importance else None,
            json_module.dumps(plots) if plots else None,
            'baigtas'
        ))

        connection.commit()
        cursor.close()

        return result_id

    except Error as e:
        print(f"Error saving imputation result: {e}")
        return None

@app.route('/impute/<filename>', methods=['POST'])
def impute_missing_values(filename):
    try:
        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
        if not os.path.exists(filepath):
            return jsonify({"error": "Failas nerastas"}), 404

        df = pd.read_csv(filepath)
        original_missing = df.isnull().sum().sum()

        if original_missing == 0:
            return jsonify({"message": "Nėra trūkstamų reikšmių užpildymui"}), 400

        # Get parameters from request
        data = request.get_json() or {}
        model_type = data.get('model_type', 'random_forest')
        n_estimators = data.get('n_estimators', 100)
        random_state = data.get('random_state', 42)

        # Perform imputation
        imputer = MLImputer(model_type=model_type, n_estimators=n_estimators, random_state=random_state)
        df_imputed = imputer.fit_transform(df)

        # Generate unique filename for imputed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]  # Short unique ID

        # Remove extension from original filename
        base_name = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1]

        # Create unique imputed filename
        imputed_filename = f"{base_name}_imputed_{model_type}_{timestamp}_{unique_id}{extension}"
        imputed_filepath = os.path.join(UPLOAD_FOLDER, imputed_filename)
        df_imputed.to_csv(imputed_filepath, index=False)

        # Generate comparison plots
        plots = {}

        # Before/After missing values comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Original missing values
        missing_original = df.isnull().sum()
        missing_original = missing_original[missing_original > 0]
        if len(missing_original) > 0:
            missing_original.plot(kind='bar', ax=ax1)
            ax1.set_title('Trūkstamos reikšmės - originalūs duomenys')
            ax1.set_xlabel('Stulpeliai')
            ax1.set_ylabel('Trūkstamų kiekis')
            ax1.tick_params(axis='x', rotation=45)

        # After imputation (should be zero)
        missing_imputed = df_imputed.isnull().sum()
        missing_imputed = missing_imputed[missing_imputed > 0]
        if len(missing_imputed) > 0:
            missing_imputed.plot(kind='bar', ax=ax2)
        else:
            ax2.bar([], [])
        ax2.set_title('Trūkstamos reikšmės - po užpildymo')
        ax2.set_xlabel('Stulpeliai')
        ax2.set_ylabel('Trūkstamų kiekis')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plots['comparison'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()

        # Feature importance plot
        feature_importance = imputer.get_feature_importance()
        if feature_importance:
            fig, axes = plt.subplots(len(feature_importance), 1,
                                   figsize=(12, 4 * len(feature_importance)))
            if len(feature_importance) == 1:
                axes = [axes]

            for idx, (col, importance) in enumerate(feature_importance.items()):
                if importance:
                    features = list(importance.keys())
                    importances = list(importance.values())

                    axes[idx].barh(features, importances)
                    axes[idx].set_title(f'Požymių svarba užpildant: {col}')
                    axes[idx].set_xlabel('Svarba')

            plt.tight_layout()
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plots['feature_importance'] = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()

        # Generate model performance plots
        model_metrics = imputer.get_model_metrics()
        performance_plots = generate_model_performance_plots(model_metrics)
        plots.update(performance_plots)

        # Save result to database
        result_id = save_imputation_result(
            filename, imputed_filename, model_type, n_estimators, random_state,
            int(original_missing), int(df_imputed.isnull().sum().sum()),
            plots, feature_importance, model_metrics
        )

        response_data = {
            "message": "Užpildymas sėkmingai baigtas",
            "imputed_filename": imputed_filename,
            "original_missing": int(original_missing),
            "imputed_missing": int(df_imputed.isnull().sum().sum()),
            "plots": plots,
            "feature_importance": feature_importance,
            "model_metrics": model_metrics,
            "model_type": model_type
        }

        if result_id:
            response_data["result_id"] = result_id

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    try:
        # Use secure_filename but preserve the structure
        secure_name = secure_filename(filename)
        filepath = os.path.join(UPLOAD_FOLDER, secure_name)

        # Debug: print what we're looking for
        print(f"Looking for file: {filepath}")
        print(f"Original filename: {filename}")
        print(f"Secure filename: {secure_name}")

        if not os.path.exists(filepath):
            # Try without secure_filename in case it's mangling the name
            direct_path = os.path.join(UPLOAD_FOLDER, filename)
            print(f"Trying direct path: {direct_path}")

            if os.path.exists(direct_path):
                filepath = direct_path
            else:
                # List all files in upload folder for debugging
                import glob
                all_files = glob.glob(os.path.join(UPLOAD_FOLDER, "*"))
                print(f"Available files: {[os.path.basename(f) for f in all_files]}")
                return jsonify({"error": f"Failas nerastas: {filename}"}), 404

        return send_file(filepath, as_attachment=True)

    except Exception as e:
        print(f"Download error: {e}")
        return jsonify({"error": str(e)}), 500

def generate_model_performance_plots(model_metrics):
    """Generate comprehensive model performance visualization plots"""
    plots = {}
    
    # Filter only regression models with metrics
    regression_metrics = {col: metrics for col, metrics in model_metrics.items() 
                         if metrics.get('model_type') == 'regression' and 'rmse' in metrics}
    
    if not regression_metrics:
        return plots
    
    # 1. Model Performance Summary Table Plot
    fig, ax = plt.subplots(figsize=(14, max(6, len(regression_metrics) * 0.8)))
    
    # Prepare data for table
    columns = ['Stulpelis', 'RMSE', 'R²', 'MAPE (%)', 'MAE', 'Imties dydis']
    table_data = []
    
    for col, metrics in regression_metrics.items():
        table_data.append([
            col,
            f"{metrics['rmse']:.4f}",
            f"{metrics['r2']:.4f}",
            f"{metrics['mape']:.2f}",
            f"{metrics['mae']:.4f}",
            str(metrics['sample_size'])
        ])
    
    # Create table
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data,
                    colLabels=columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header styling
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F2F2F2')
    
    plt.title('Modelių efektyvumo vertinimo metrikos', fontsize=14, fontweight='bold', pad=20)
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plots['performance_table'] = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    # 2. RMSE Comparison Bar Chart
    if len(regression_metrics) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        columns = list(regression_metrics.keys())
        rmse_values = [regression_metrics[col]['rmse'] for col in columns]
        
        bars = ax.bar(columns, rmse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(columns)])
        
        ax.set_xlabel('Užpildyti stulpeliai', fontsize=12)
        ax.set_ylabel('RMSE vertė', fontsize=12)
        ax.set_title('Root Mean Square Error (RMSE) palyginimas', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, rmse_values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plots['rmse_comparison'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
    
    # 3. R² Score Comparison
    if len(regression_metrics) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        r2_values = [regression_metrics[col]['r2'] for col in columns]
        
        bars = ax.bar(columns, r2_values, color=['#2E8B57', '#FF6347', '#4682B4', '#DAA520', '#8A2BE2'][:len(columns)])
        
        ax.set_xlabel('Užpildyti stulpeliai', fontsize=12)
        ax.set_ylabel('R² koeficientas', fontsize=12)
        ax.set_title('Determinacijos koeficiento (R²) palyginimas', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, r2_values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add interpretation lines
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Geras modelis (R² ≥ 0.7)')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Vidutinis modelis (R² ≥ 0.5)')
        ax.legend(loc='lower right')
        
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plots['r2_comparison'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
    
    # 4. MAPE Comparison
    if len(regression_metrics) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        mape_values = [regression_metrics[col]['mape'] for col in columns]
        
        bars = ax.bar(columns, mape_values, color=['#DC143C', '#FF8C00', '#32CD32', '#4169E1', '#8B008B'][:len(columns)])
        
        ax.set_xlabel('Užpildyti stulpeliai', fontsize=12)
        ax.set_ylabel('MAPE (%)', fontsize=12)
        ax.set_title('Mean Absolute Percentage Error (MAPE) palyginimas', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, mape_values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mape_values)*0.02,
                   f'{value:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add interpretation lines
        ax.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='Labai geras modelis (MAPE ≤ 10%)')
        ax.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Geras modelis (MAPE ≤ 20%)')
        ax.legend(loc='upper right')
        
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plots['mape_comparison'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
    
    # 5. Combined Metrics Radar Chart (if multiple models)
    if len(regression_metrics) > 1:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Normalize metrics for radar chart (0-1 scale)
        metrics_names = ['RMSE', 'R²', 'MAPE', 'MAE']
        
        for i, (col, metrics) in enumerate(regression_metrics.items()):
            # Normalize values (invert RMSE, MAPE, MAE so higher is better)
            max_rmse = max([m['rmse'] for m in regression_metrics.values()])
            max_mape = max([m['mape'] for m in regression_metrics.values()])
            max_mae = max([m['mae'] for m in regression_metrics.values()])
            
            normalized_values = [
                1 - (metrics['rmse'] / max_rmse) if max_rmse > 0 else 1,  # Inverted RMSE
                metrics['r2'],  # R² (already 0-1)
                1 - (metrics['mape'] / max_mape) if max_mape > 0 else 1,  # Inverted MAPE
                1 - (metrics['mae'] / max_mae) if max_mae > 0 else 1  # Inverted MAE
            ]
            
            angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
            normalized_values += normalized_values[:1]  # Complete the circle
            angles += angles[:1]
            
            color = plt.cm.Set1(i / len(regression_metrics))
            ax.plot(angles, normalized_values, 'o-', linewidth=2, label=col, color=color)
            ax.fill(angles, normalized_values, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1)
        ax.set_title('Modelių efektyvumo palyginimas\n(Didesnis plotas = geresnis modelis)', y=1.08, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plots['radar_comparison'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
    
    return plots

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)