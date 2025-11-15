# config.py
from pathlib import Path

# Пути к файлам
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "data_raw.xlsx"
PROCESSED_DIR = DATA_DIR / "processed"

# Создаем директории
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Имена выходных файлов
PROCESSED_JSON = PROCESSED_DIR / "dataset.json"
PROCESSED_CSV = PROCESSED_DIR / "dataset.csv"
FEATURES_CSV = PROCESSED_DIR / "features.csv"

# Настройки обработки данных
CURRENT_DATE = "2025-12-01"
KPI_COLUMNS = ['июнь', 'июль', 'август', 'сентябрь', 'октябрь']
TARGET_COLUMN = "Состояние выгорания"

# Маппинги для кодирования
BINARY_MAPPING = {
    'да': 1, 'нет': 0, 
    'прошел': 1, 'не прошел': 0, 
    'не проходил': 0, 'нет аттестации': 0
}

BURNOUT_MAPPING = {
    'все хорошо': 0, 
    'усталость': 1, 
    'выгорел': 2
}