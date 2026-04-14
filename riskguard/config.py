from pathlib import Path

# Se centraliza rutas compartidas para no repetir paths literales en la app

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

TRAINING_DATASET_PATH = DATA_DIR / "CSV_TFG.csv"
RAW_DATASET_PATH = DATA_DIR / "project_risk_raw_dataset.csv"
STYLES_PATH = ASSETS_DIR / "styles.css"
