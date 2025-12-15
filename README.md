# Mašininio mokymosi modelių taikymas prognozuojant regionų ekonominius rodiklius NUTS 2 lygmeniu

**Pagrindinis magistro darbo tyrimo failas:** `Tyrimas_Ekonominiu_Rodikliu_Imputacija.ipynb` - Jupyter Notebook su visa tyrimo metodologija, eksperimentais ir rezultatų analize.

**Flask WEB aplikacija** veikia kaip ekspertinė sistema ir duomenų vizualizacijos įrankis, leidžiantis interaktyviai tyrinėti imputuotus duomenis, generuoti dinamines vizualizacijas bei atlikti trūkstamų reikšmių užpildymą naudojant mašininio mokymosi algoritmus (Random Forest ir XGBoost).

## Flask WEB aplikacijos pagrindinės funkcijos

### Duomenų įkėlimas ir analizė
- **CSV/Excel failų įkėlimas**: Drag & drop arba failo naršymo funkcija
- **Automatinė duomenų analizė**: Statistikų apskaičiavimas duomenims, trūkstamų reikšmių identifikavimas
- **Little MCAR testas**: Statistinis testas trūkstamų duomenų mechanizmo nustatymui (Jupyter Notebook faile)

### Vizualizacijos
- Trūkstamų reikšmių heatmap
- Koreliacijos matrica
- Feature importance grafikai
- KDE (Kernel Density Estimation) pasiskirstymo grafikai
- Originalių ir imputuotų reikšmių palyginimas

### Imputacijos modeliai
- **Random Forest**: Ensemble metodas su medžių balsavimu
- **XGBoost**: Gradient boosting algoritmas su regularizacija

### Modelių vertinimas
- Sintetinis testavimas (train/test split)
- sMAPE, nRMSE, nMAE ir R² metrikos
- Cross-validation
- Hiperparametrų optimizavimas (RandomizedSearchCV)

### Papildomos funkcijos
- **Modelių palyginimas**: Random Forest ir XGBoost rezultatų palyginimo analizė
- **Rezultatų saugojimas**: MySQL duomenų bazėje
- **PDF ataskaitų generavimas**: Detalios ataskaitos su grafikais
- **El. pašto siuntimas**: Rezultatų išsiuntimas el. paštu
- **Komentarų sistema**: Naudotojų atsiliepimai

## Sistemos reikalavimai

- Python 3.8+
- Flask web framework
- Pandas, NumPy, SciPy, Scikit-learn
- XGBoost (1.7.6)
- scikit-learn
- Matplotlib, Seaborn, Plotly
- MySQL (komentarams, rezultatų išsaugojimui)
- ReportLab (PDF generavimui)

## Instaliavimas ir paleidimas

### 1. Priklausomybių instaliavimas

```bash
pip install -r requirements.txt
```

### 2. Aplinkos kintamųjų nustatymas (pasirinktinai)

Sukurkite `.env` failą su šiais kintamaisiais:

```env
# MySQL konfigūracija (komentarams, rezultatų išsaugojimui)
MYSQL_HOST=your_host
MYSQL_USER=your_user
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=your_database

# El. pašto konfigūracija
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=your_email
SMTP_PASSWORD=your_password
SMTP_FROM_EMAIL=noreply@example.com
SMTP_FROM_NAME=Duomenų Analizės Sistema



```

### 3. Aplikacijos paleidimas

**Flask aplikacijos paleidimas lokalioje komapiuterio aplinkoje:**
```bash
python app.py
```






Aplikacija bus prieinama adresu: http://localhost:5000

## Naudojimas

1. **Atidarykite aplikaciją** naršyklėje (http://localhost:5000)
2. **Įkelkite CSV failą** su ekonominiais rodikliais
3. **Peržiūrėkite duomenų statistikas** ir trūkstamų reikšmių informaciją
4. **Pasirinkite imputacijos modelį** (Random Forest arba XGBoost)
5. **Konfigūruokite modelio parametrus** (n_estimators, max_depth ir kt.)
6. **Paleiskite imputaciją** ir peržiūrėkite rezultatus
7. **Analizuokite metrikas** ir požymių svarbą
8. **Atsisiųskite rezultatus** (CSV, Excel arba PDF formatu)

## Algoritmų aprašymas

### Random Forest Imputer
- Kiekvienam stulpeliui su trūkstamomis reikšmėmis treniruojamas atskiras RF modelis
- Struktūriniai nuliai (0) naudojami kaip mokymo duomenys, bet neimputuojami
- Kategoriniai prediktoriai ('geo', 'year') tik enkoduojami
- Sintetinis testavimas be duomenų nutekėjimo (20 % TEST)

### XGBoost Imputer
- Gradient boosting su XGBRegressor
- Regularizacijos parametrai (reg_alpha, reg_lambda)
- Ankstyvasis sustabdymas (early stopping)


### Vertinimo metrikos
- **SMAPE** (Symmetric Mean Absolute Percentage Error)
- **nRMSE** (Normalized Root Mean Squared Error)
- **nMAE** (Normalized Mean Absolute Error)
- **R²** (Determination Coefficient)

## Projekto struktūra

```
magistro_darbas/
├── app.py                     # Pagrindinis Flask aplikacijos failas
├── requirements.txt           # Python priklausomybės
├── gunicorn_config.py         # Gunicorn konfigūracija produkcijai
├── render.yaml                # Render.com diegimo konfigūracija
├── runtime.txt                # Python versija
├── little_mcar_test.py        # Little MCAR testo implementacija
├── modeliai/
│   ├── __init__.py            # Modelių paketo inicializacija
│   ├── Random_Forest.py       # Random Forest imputer'io klasė
│   └── XGBoost.py             # XGBoost imputer'io klasė
├── irankiai/
│   ├── __init__.py
│   └── siusti.py              # El. pašto siuntimo įrankiai
├── templates/
│   ├── base.html              # Bazinis šablonas
│   ├── index.html             # Pagrindinis puslapis
│   ├── imputacija.html        # Imputacijos puslapis
│   ├── ikelti_duomenys.html   # Duomenų įkėlimo puslapis
│   ├── rezultatai.html        # Rezultatų sąrašas
│   ├── rezultatas_detali.html # Detalūs rezultatai
│   ├── palyginimas.html       # Modelių palyginimas
│   ├── komentarai.html        # Komentarų puslapis
│   └── apie.html              # Informacija apie sistemą
├── static/                    # CSS, paveiksliukai
├── staticjs/                  # JavaScript failai
├── uploads/                   # Įkeltų failų (t.y NUTS 2 regionų duomenų) direktorija
└── README.md                  # Projekto dokumentacija
```

## API endpoint'ai

### Puslapiai
- `GET /` - Pagrindinis puslapis
- `GET /imputacija` - Imputacijos sąsaja
- `GET /ikelti-duomenys` - Duomenų įkėlimas
- `GET /rezultatai` - Rezultatų sąrašas
- `GET /rezultatai/<result_id>` - Konkretūs rezultatai
- `GET /palyginimas` - Modelių palyginimas
- `GET /komentarai` - Komentarai
- `GET /apie` - Apie sistemą

### API
- `POST /upload` - CSV failo įkėlimas
- `GET /analyze/<filename>` - Duomenų analizė
- `POST /impute/<filename>` - Trūkstamų reikšmių užpildymas
- `GET /download/<filename>` - Failo atsisiuntimas
- `POST /api/palyginimas` - Modelių palyginimas
- `GET /api/rezultatai` - Rezultatų sąrašas (JSON)
- `GET /api/rezultatai/<result_id>` - Rezultato detalės (JSON)
- `GET /api/rezultatai/<result_id>/kde/<indicator>` - KDE grafikas
- `GET /api/rezultatai/<result_id>/koreliacijos-analize` - Koreliacijos analizė
- `POST /api/send-result-email` - Rezultatų siuntimas el. paštu
- `GET /api/comments/<result_id>` - Rezultato komentarai
- `POST /api/comments` - Naujas komentaras
- `GET /api/system-status` - Sistemos būsena

## Modelių parametrai

### Random Forest
| Parametras | Aprašymas | Numatyta reikšmė |
|------------|-----------|------------------|
| n_estimators | Medžių skaičius | 100 |
| max_depth | Maksimalus medžio gylis | 15 |
| min_samples_split | Minimalus pavyzdžių skaičius padalinimui | 5 |
| min_samples_leaf | Minimalus pavyzdžių skaičius lape | 2 |
| random_state | Atsitiktinumo sėkla | 42 |

### XGBoost
| Parametras | Aprašymas | Numatyta reikšmė |
|------------|-----------|------------------|
| n_estimators | Boosting'o iteracijų skaičius | 200 |
| learning_rate | Mokymosi greitis | 0.1 |
| max_depth | Maksimalus medžio gylis | 6 |
| reg_alpha | L1 regularizacija | 0 |
| reg_lambda | L2 regularizacija | 1 |
| random_state | Atsitiktinumo sėkla | 42 |

## Palaikomi duomenų formatai

- CSV failai su skaitinėmis ir kategorinėmis reikšmėmis
- Excel failai (.xlsx, .xls)
- Automatinis duomenų tipų atpažinimas
- Maksimalus failo dydis: 16MB
- Palaikomi kategoriniai stulpeliai: 'geo', 'year'

## Diegimas į Render.com

Projektas sukonfigūruotas diegimui į Render.com platformą:

1. Reikia sukurti naują Web Service Render.com
2. Prijungti GitHub repozitoriją
3. Nustatyti aplinkos kintamuosius
4. Ir diegimas vyks automatiškai pagal `render.yaml` konfigūraciją

## Autorius

Magistro darbo autorius: Irmantas Pilypas ITVM24.<br>
Vilniaus universitetas, Šiaulių akademija 2025
