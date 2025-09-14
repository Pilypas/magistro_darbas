# Random Forest trūkstamų reikšmių užpildymo WEB aplikacija

Magistro darbo WEB aplikacija, skirta CSV duomenų failų analizei ir trūkstamų reikšmių užpildymui naudojant Random Forest algoritmą.

## Funkcionalumas

- **CSV failų įkėlimas**: Drag & drop arba failo naršymo funkcija
- **Duomenų analizė**: Automatinis duomenų struktūros analizavimas
- **Vizualizacijos**: 
  - Trūkstamų reikšmių heatmap
  - Koreliacijos matrica
  - Feature importance grafikai
- **Random Forest imputation**: Trūkstamų reikšmių užpildymas naudojant mašininio mokymosi algoritmą
- **Rezultatų eksportas**: Galimybė atsisiųsti apdorotus duomenis

## Sistemos reikalavimai

- Python 3.8+
- Flask web framework
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn, Plotly

## Instaliavimas ir paleidimas

### 1. Priklausomybių instaliavimas

```bash
pip install -r requirements.txt
```

### 2. Aplikacijos paleidimas

```bash
python app.py
```

Aplikacija bus prieinama adresu: http://localhost:5000

## Naudojimas

1. **Atidarykite aplikaciją** naršyklėje (http://localhost:5000)
2. **Įkelkite CSV failą** vilkdami į žymėtą sritį arba spausdami "browse"
3. **Peržiūrėkite duomenų statistikas** - matysite eilučių/stulpelių skaičių ir trūkstamų reikšmių informaciją
4. **Atlikite detalią analizę** - gaukite vizualizacijas ir koreliacijos matricą
5. **Paleiskite trūkstamų reikšmių užpildymą** - konfigūruokite Random Forest parametrus
6. **Atsisiųskite rezultatus** - gausite apdorotą CSV failą su užpildytomis reikšmėmis

## Algoritmo aprašymas

Aplikacija naudoja **Random Forest** algoritmą trūkstamoms reikšmėms užpildyti:

- **Regresiniai modeliai** - skaitinėms reikšmėms
- **Klasifikavimo modeliai** - kategorinėms reikšmėms
- **Feature importance analizė** - rodo, kurie kintamieji svarbiausi prognozėje
- **Cross-validation** - užtikrina modelio stabilumą

## Projekto struktūra

```
magistro_darbas/
├── app.py                 # Pagrindinis Flask aplikacijos failas
├── requirements.txt       # Python priklausomybės
├── templates/
│   └── index.html        # Frontend sąsaja
├── uploads/              # Įkeltų failų direktorija (sukuriama automatiškai)
└── README.md            # Projekto dokumentacija
```

## API endpoint'ai

- `GET /` - Pagrindinis puslapis
- `POST /upload` - CSV failo įkėlimas
- `GET /analyze/<filename>` - Duomenų analizės rezultatai
- `POST /impute/<filename>` - Trūkstamų reikšmių užpildymas
- `GET /download/<filename>` - Apdoroto failo atsisiuntimas

## Parametrų konfigūravimas

**Random Forest parametrai:**
- `n_estimators` - medžių skaičius (numatyta: 100)
- `random_state` - atsitiktinumo kontrolė reprodukcijos tikslais (numatyta: 42)

## Palaikomos duomenų struktūros

- CSV failai su skaitinėmis ir kategorinėmis reikšmėmis
- Automatinis duomenų tipų atpažinimas
- Maksimalus failo dydis: 16MB

## Autorius

Magistro darbo autorius
Vilniaus universitetas