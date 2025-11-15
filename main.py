# main.py
from data_processor import DataProcessor
from data_manager import DataManager
from config import *

def main():
    # Обработка данных
    print("1. Обработка данных...")
    processor = DataProcessor()
    processor.process_all()
    
    # Сохранение в разных форматах
    print("\n2. Сохранение данных...")
    DataManager.save_processed_data(processor.df)
    
    # Информация о датасете
    print("\n3. Информация о датасете:")
    info = DataManager.get_dataset_info()
    if info:
        print(f"   Всего записей: {info['total_records']}")
        print(f"   Колонок: {len(info['columns'])}")
        print(f"   Пропусков в целевой переменной: {info['target_missing']}")
        print(f"   Распределение целевой переменной: {info['target_distribution']}")
    
    print("\n=== Обработка завершена ===")

if __name__ == "__main__":
    main()