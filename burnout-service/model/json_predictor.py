import pandas as pd
import numpy as np
import json
import joblib
from sklearn.preprocessing import LabelEncoder

class JSONPredictor:
    def __init__(self, model_path='svm_model.pkl'):
        """Инициализация предсказателя для JSON данных"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.metrics = model_data.get('metrics', {})
            print(f"✅ Модель загружена из {model_path}")
            
            # Загружаем ожидаемые признаки из обучающих данных
            self.expected_features = self._get_expected_features()
            
        except FileNotFoundError:
            print(f"❌ Модель {model_path} не найдена")
            self.model = None
            self.scaler = None
            self.expected_features = None
        
        # Инициализируем кодировщики для категориальных признаков
        self.label_encoders = {}
    
    def _get_expected_features(self):
        """Получаем ожидаемые признаки из обучающих данных"""
        try:
            # Пытаемся загрузить обучающие данные чтобы узнать ожидаемые признаки
            from data_loader import DataLoader
            splits = DataLoader.load_splits()
            if splits:
                X_train, _, _, _, _, _ = splits
                return list(X_train.columns)
        except:
            pass
        
        # Если не удалось загрузить, возвращаем None
        return None
    
    def load_json_data(self, json_path):
        """Загрузка данных из JSON файла"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ JSON данные загружены из {json_path}")
            return data
        except FileNotFoundError:
            print(f"❌ JSON файл {json_path} не найден")
            return None
        except json.JSONDecodeError:
            print(f"❌ Ошибка декодирования JSON файла {json_path}")
            return None
    
    def transform_to_model_features(self, employee_data):
        """Преобразование сырых данных в формат модели"""
        # Создаем копию данных
        data = employee_data.copy()
        
        # Создаем DataFrame с правильными признаками
        processed_data = {}
        
        # 1. Базовые демографические признаки
        if 'возраст' in data:
            processed_data['возраст'] = float(data['возраст'])
        else:
            processed_data['возраст'] = 30.0  # значение по умолчанию
        
        # 2. Преобразование стажа в месяцы
        if 'Стаж' in data:
            processed_data['Стаж_месяцы'] = float(self._experience_to_months(data['Стаж']))
        else:
            processed_data['Стаж_месяцы'] = 24.0  # 2 года по умолчанию
        
        # 3. KPI показатели
        kpi_months = ['июнь', 'июль', 'август', 'сентябрь', 'октябрь']
        kpi_values = []
        
        for month in kpi_months:
            if month in data:
                kpi_value = data[month]
                # Преобразуем в число если нужно
                if isinstance(kpi_value, str):
                    try:
                        kpi_value = float(kpi_value) if kpi_value.replace('.', '').replace(',', '').isdigit() else 0.8
                    except:
                        kpi_value = 0.8
                processed_data[f'KPI_{month}'] = float(kpi_value)
                kpi_values.append(float(kpi_value))
            else:
                processed_data[f'KPI_{month}'] = 0.8  # среднее значение по умолчанию
                kpi_values.append(0.8)
        
        # 4. Статистики по KPI
        if kpi_values:
            kpi_array = np.array(kpi_values)
            processed_data['KPI_заполнено_показателей'] = float(len([x for x in kpi_values if x > 0]))
            processed_data['KPI_стабильность'] = float(kpi_array.std() if len(kpi_values) > 1 else 0.1)
            processed_data['KPI_мин'] = float(kpi_array.min())
            processed_data['KPI_макс'] = float(kpi_array.max())
            processed_data['KPI_размах'] = float(kpi_array.max() - kpi_array.min() if len(kpi_values) > 0 else 0.2)
            
            # Расчет тренда KPI
            if len(kpi_values) >= 2:
                x = np.arange(len(kpi_values))
                try:
                    trend = np.polyfit(x, kpi_values, 1)[0]
                    processed_data['KPI_тренд'] = float(trend)
                except:
                    processed_data['KPI_тренд'] = 0.0
            else:
                processed_data['KPI_тренд'] = 0.0
            
            # Последний KPI
            processed_data['KPI_последний'] = float(kpi_values[-1] if kpi_values else 0.8)
        
        # 5. Обработка отпуска
        if 'Отпуск (когда ходил в последний раз)' in data:
            vacation_date = data['Отпуск (когда ходил в последний раз)']
            processed_data['Отпуск_месяцев_назад'] = float(self._vacation_months_ago(vacation_date))
        else:
            processed_data['Отпуск_месяцев_назад'] = 6.0  # 6 месяцев по умолчанию
        
        # 6. Бинарные признаки
        binary_mapping = {
            'да': 1, 'нет': 0, 'yes': 1, 'no': 0,
            'прошел': 1, 'не прошел': 0, 'нет аттестации': 0,
            'завершена': 1, 'в процессе': 0,
            'Руководитель': 1, 'Сотрудник': 0, 'Сотрутник': 0
        }
        
        # Больничный
        if 'Больничный (брал или нет в 2025 году)' in data:
            sick_key = data['Больничный (брал или нет в 2025 году)']
            processed_data['Больничный'] = float(binary_mapping.get(sick_key, 0))
        else:
            processed_data['Больничный'] = 0.0
        
        # Выговор
        if 'Выговор (да/нет)' in data:
            reprimand_key = data['Выговор (да/нет)']
            processed_data['Выговор'] = float(binary_mapping.get(reprimand_key, 0))
        else:
            processed_data['Выговор'] = 0.0
        
        # Аттестация
        if 'Прохождение аттестации (прошел/не прошел/нет аттестации)' in data:
            attestation_key = data['Прохождение аттестации (прошел/не прошел/нет аттестации)']
            processed_data['Прохождение аттестации'] = float(binary_mapping.get(attestation_key, 0))
        else:
            processed_data['Прохождение аттестации'] = 1.0  # по умолчанию прошел
        
        # Участие в активностях
        if 'Участие в активностях корпоративных' in data:
            activities_key = data['Участие в активностях корпоративных']
            processed_data['Участие в активностях'] = float(binary_mapping.get(activities_key, 0))
        else:
            processed_data['Участие в активностях'] = 1.0  # по умолчанию участвует
        
        # Обучение
        if 'Обучение' in data:
            training_key = data['Обучение']
            processed_data['Обучение'] = float(binary_mapping.get(training_key, 0))
        else:
            processed_data['Обучение'] = 1.0  # по умолчанию завершено
        
        # Руководитель
        if 'В подчиненнии сотрудники' in data:
            manager_key = data['В подчиненнии сотрудники']
            processed_data['Руководитель'] = float(binary_mapping.get(manager_key, 0))
        else:
            processed_data['Руководитель'] = 0.0  # по умолчанию сотрудник
        
        # 7. One-Hot Encoding для города и должности
        cities = ['Москва', 'Санкт-Петербург', 'Новосибирск', 'Самара', 'Красноярск', 
                 'Казань', 'Омск', 'Екатеринбург', 'Кемерово']
        for city_name in cities:
            processed_data[f'Город_{city_name}'] = 0.0
        
        if 'Город' in data:
            city = data['Город']
            for city_name in cities:
                if city == city_name:
                    processed_data[f'Город_{city_name}'] = 1.0
                    break
            else:
                # Если город не найден, ставим Москву по умолчанию
                processed_data['Город_Москва'] = 1.0
        else:
            processed_data['Город_Москва'] = 1.0
        
        positions = [
            'Менеджер по работе с клиентами', 'Старший менеджер по работе с клиентами',
            'Курьер', 'Кладовщик', 'Бригадир', 'Юрист', 'Бухгалтер', 'Кассир',
            'Логист', 'Менеджер по территориальному развити.', 'Разработчик бэкенд',
            'Дизайнер', 'Тестировщик', 'Разработчик фронт', 'Руководитель проекта',
            'Руководитель отдела продаж', 'Руководитель клиентсокго отдела',
            'Главный бухгалтер', 'Руководитель склада', 'Руководитель контактного-центра 1 линии',
            'Директор филиала', 'Менеджер по продажам'
        ]
        for pos_name in positions:
            processed_data[f'Должность_{pos_name}'] = 0.0
        
        if 'Должность' in data:
            position = data['Должность']
            for pos_name in positions:
                if position == pos_name:
                    processed_data[f'Должность_{pos_name}'] = 1.0
                    break
            else:
                # Если должность не найдена, ставим менеджера по умолчанию
                processed_data['Должность_Менеджер по работе с клиентами'] = 1.0
        else:
            processed_data['Должность_Менеджер по работе с клиентами'] = 1.0
        
        # 8. Пол (определяем по ФИО если не указан)
        if 'пол' in data:
            processed_data['пол'] = 1.0 if data['пол'] == 'муж' else 0.0
        elif 'ФИО' in data:
            # Автоматическое определение пола по ФИО
            fio = data['ФИО']
            if any(ending in fio.split()[0] for ending in ['вна', 'ова', 'ева', 'ина', 'ская']):
                processed_data['пол'] = 0.0  # женский
            else:
                processed_data['пол'] = 1.0  # мужской
        else:
            processed_data['пол'] = 1.0  # по умолчанию мужской
        
        # Создаем DataFrame
        df = pd.DataFrame([processed_data])
        
        # Выравниваем с ожидаемыми признаками
        if self.expected_features:
            # Добавляем отсутствующие признаки
            for feature in self.expected_features:
                if feature not in df.columns:
                    df[feature] = 0.0
            
            # Убираем лишние признаки
            df = df[self.expected_features]
        
        return df
    
    def _experience_to_months(self, experience_str):
        """Преобразование стажа в месяцы"""
        if isinstance(experience_str, (int, float)):
            return int(experience_str)
        
        if not isinstance(experience_str, str):
            return 24  # 2 года по умолчанию
        
        total_months = 0
        try:
            # Ищем годы
            if 'год' in experience_str:
                years_match = experience_str.split('год')[0].strip()
                if years_match.isdigit():
                    total_months += int(years_match) * 12
                elif 'лет' in experience_str:
                    years_match = experience_str.split('лет')[0].strip()
                    if years_match.isdigit():
                        total_months += int(years_match) * 12
            
            # Ищем месяцы
            if 'месяц' in experience_str:
                months_part = experience_str.split('месяц')[0]
                months_match = months_part.split()[-1]
                if months_match.isdigit():
                    total_months += int(months_match)
        except:
            pass
        
        return total_months if total_months > 0 else 24
    
    def _vacation_months_ago(self, vacation_date):
        """Расчет сколько месяцев назад был отпуск"""
        if not vacation_date or vacation_date == 'нет':
            return 12  # год назад по умолчанию
        
        try:
            from datetime import datetime
            
            if isinstance(vacation_date, str):
                # Пробуем разные форматы дат
                for fmt in ['%Y-%m-%d', '%d.%m.%Y', '%Y-%m-%d %H:%M:%S']:
                    try:
                        vacation_dt = datetime.strptime(vacation_date, fmt)
                        break
                    except:
                        continue
                else:
                    return 12
            
            current_date = datetime(2025, 1, 1)  # Используем дату из конфига
            months_ago = (current_date.year - vacation_dt.year) * 12 + (current_date.month - vacation_dt.month)
            return max(1, months_ago)
        except:
            return 12
    
    def process_single_employee(self, employee_data):
        """Обработка данных одного сотрудника"""
        # Преобразуем в формат модели
        processed_data = self.transform_to_model_features(employee_data)
        return processed_data
    
    def predict_burnout(self, processed_data):
        """Предсказание выгорания для обработанных данных"""
        if self.model is None:
            print("❌ Модель не загружена")
            return None
        
        # Масштабируем данные
        try:
            scaled_data = self.scaler.transform(processed_data)
        except ValueError as e:
            print(f"❌ Ошибка при масштабировании данных: {e}")
            return None
        
        # Предсказание
        prediction = self.model.predict(scaled_data)
        probability = self.model.predict_proba(scaled_data)
        
        return {
            'prediction': int(prediction[0]),  # Преобразуем в int для JSON
            'burnout_probability': float(probability[0, 1]),  # Вероятность выгорания
            'no_burnout_probability': float(probability[0, 0]),  # Вероятность отсутствия выгорания
            'confidence': float(max(probability[0]))  # Уверенность предсказания
        }
    
    def interpret_prediction(self, prediction_result):
        """Интерпретация результатов предсказания"""
        prediction = prediction_result['prediction']
        burnout_prob = prediction_result['burnout_probability']
        confidence = prediction_result['confidence']
        
        if prediction == 1:
            status = "ВЫГОРАНИЕ"
            if burnout_prob > 0.7:
                recommendation = "❗ Высокий риск выгорания. Требуется немедленное внимание"
            elif burnout_prob > 0.5:
                recommendation = "⚠️  Средний риск выгорания. Рекомендуется профилактика"
            else:
                recommendation = "⚠️  Возможное выгорание. Рекомендуется наблюдение"
            color = "🔴"
        else:
            status = "НОРМА"
            if burnout_prob < 0.2:
                recommendation = "✅ Отличное состояние. Продолжать текущие практики"
            elif burnout_prob < 0.4:
                recommendation = "✅ Хорошее состояние. Рекомендуется профилактика"
            else:
                recommendation = "🟡 Нормальное состояние. Рекомендуется наблюдение"
            color = "🟢"
        
        interpretation = {
            'status': status,
            'probability': burnout_prob,
            'confidence': confidence,
            'recommendation': recommendation,
            'color': color
        }
        
        return interpretation
    
    def process_json_file(self, json_path):
        """Обработка всего JSON файла"""
        data = self.load_json_data(json_path)
        if data is None:
            return None
        
        results = []
        
        # Проверяем структуру JSON
        if isinstance(data, list):
            # Массив сотрудников
            employees = data
        elif isinstance(data, dict):
            # Один сотрудник или объект с данными
            if 'employees' in data:
                employees = data['employees']
            else:
                employees = [data]  # Один сотрудник
        else:
            print("❌ Неподдерживаемый формат JSON")
            return None
        
        print(f"🔍 Обработка {len(employees)} сотрудников...")
        
        for i, employee_data in enumerate(employees):
            print(f"\n👤 Сотрудник {i+1}:")
            
            # Извлекаем идентификатор сотрудника
            employee_id = employee_data.get('ФИО', f'Сотрудник_{i+1}')
            print(f"   ID: {employee_id}")
            
            # Обрабатываем данные
            processed_data = self.process_single_employee(employee_data)
            
            # Проверяем, что данные корректны
            if processed_data.empty:
                print(f"   ❌ Не удалось обработать данные сотрудника")
                continue
            
            # Предсказание
            prediction_result = self.predict_burnout(processed_data)
            if prediction_result is None:
                continue
            
            # Интерпретация
            interpretation = self.interpret_prediction(prediction_result)
            
            # Формируем результат
            result = {
                'employee_id': employee_id,
                'prediction': prediction_result['prediction'],
                'burnout_probability': round(prediction_result['burnout_probability'], 4),
                'no_burnout_probability': round(prediction_result['no_burnout_probability'], 4),
                'confidence': round(prediction_result['confidence'], 4),
                'status': interpretation['status'],
                'recommendation': interpretation['recommendation'],
                'color': interpretation['color']
            }
            
            results.append(result)
            
            # Вывод результата
            print(f"   {interpretation['color']} Статус: {interpretation['status']}")
            print(f"   📊 Вероятность выгорания: {prediction_result['burnout_probability']:.1%}")
            print(f"   🎯 Уверенность: {prediction_result['confidence']:.1%}")
            print(f"   💡 Рекомендация: {interpretation['recommendation']}")
        
        return results
    
    def save_results(self, results, output_path='prediction_results.json'):
        """Сохранение результатов в JSON файл"""
        try:
            # Преобразуем все значения в JSON-сериализуемые типы
            serializable_results = []
            for result in results:
                serializable_result = {}
                for key, value in result.items():
                    if isinstance(value, (np.integer, np.int64)):
                        serializable_result[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64)):
                        serializable_result[key] = float(value)
                    else:
                        serializable_result[key] = value
                serializable_results.append(serializable_result)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            print(f"💾 Результаты сохранены в {output_path}")
        except Exception as e:
            print(f"❌ Ошибка при сохранении результатов: {e}")

def main():
    """Главная функция для обработки JSON файлов"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Прогнозирование выгорания из JSON файлов')
    parser.add_argument('json_file', help='Путь к JSON файлу с данными сотрудников')
    parser.add_argument('--output', '-o', default='prediction_results.json', 
                       help='Путь для сохранения результатов')
    parser.add_argument('--model', '-m', default='svm_model.pkl', 
                       help='Путь к файлу модели')
    
    args = parser.parse_args()
    
    print("🎯 ЗАПУСК ПРОГНОЗИРОВАНИЯ ВЫГОРАНИЯ ИЗ JSON")
    print("=" * 50)
    
    # Инициализация предсказателя
    predictor = JSONPredictor(args.model)
    
    if predictor.model is None:
        return
    
    # Обработка JSON файла
    results = predictor.process_json_file(args.json_file)
    
    if results:
        # Сохранение результатов
        predictor.save_results(results, args.output)
        
        # Статистика
        burnout_count = sum(1 for r in results if r['prediction'] == 1)
        total_count = len(results)
        
        print(f"\n📈 СТАТИСТИКА:")
        print(f"   Всего сотрудников: {total_count}")
        print(f"   С выгоранием: {burnout_count}")
        print(f"   Без выгорания: {total_count - burnout_count}")
        print(f"   Процент выгорания: {burnout_count/total_count*100:.1f}%")

if __name__ == "__main__":
    main()