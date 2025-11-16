import pandas as pd
import numpy as np
import json
import joblib
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

class JSONPredictor:
    def __init__(self, model_path='svm_model.pkl'):
        """Инициализация предсказателя для JSON данных"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.metrics = model_data.get('metrics', {})
            print(f"✅ Модель загружена из {model_path}")
            
            # Получаем реальные признаки из модели или scaler
            self.expected_features = self._get_real_expected_features()
            print(f"📊 Реальные признаки из модели: {len(self.expected_features)}")
            
        except FileNotFoundError:
            print(f"❌ Модель {model_path} не найдена")
            self.model = None
            self.scaler = None
            self.expected_features = None

    def _get_real_expected_features(self):
        """Получаем реальные признаки из модели или scaler"""
        if hasattr(self.scaler, 'feature_names_in_'):
            return list(self.scaler.feature_names_in_)
        elif hasattr(self.model, 'feature_names_in_'):
            return list(self.model.feature_names_in_)
        else:
            print("⚠️  Признаки не найдены в модели, создаем на основе CSV структуры")
            return self._get_fallback_features()

    def _get_fallback_features(self):
        """Резервный метод для создания признаков на основе CSV структуры"""
        expected_features = [
            'возраст', 'KPI_июнь', 'KPI_июль', 'KPI_август', 'KPI_сентябрь', 'KPI_октябрь',
            'Обучение', 'Стаж_месяцы', 'KPI_заполнено_показателей', 'KPI_стабильность',
            'KPI_мин', 'KPI_макс', 'KPI_размах', 'KPI_тренд', 'KPI_последний',
            'Отпуск_месяцев_назад', 'Руководитель'
        ]
        
        positions = [
            'Бригадир', 'Бухгалтер', 'Главный бухгалтер', 'Дизайнер', 'Директор филиала',
            'Кассир', 'Кладовщик', 'Курьер', 'Логист', 'Менеджер по продажам',
            'Менеджер по работе с клиентами', 'Менеджер по территориальному развити.',
            'Разработчик бэкенд', 'Разработчик фронт', 'Руководитель клиентсокго отдела',
            'Руководитель контактного-центра 1 линии', 'Руководитель отдела продаж',
            'Руководитель проекта', 'Руководитель склада',
            'Старший менеджер группы регионального развития по клиентскому сервису',
            'Старший менеджер по работе с клиентами', 'Тестировщик', 'Юрист'
        ]
        
        for pos in positions:
            expected_features.append(f'Должность_{pos}')
        
        return expected_features

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
        data = employee_data.copy()
        all_possible_features = {}
        
        # 1. Базовые демографические признаки - нормализуем возраст
        age = data.get('возраст', 30.0)
        all_possible_features['возраст'] = float(age) / 100.0  # Нормализуем возраст
        
        # 2. Преобразование стажа в месяцы - нормализуем
        experience_months = self._experience_to_months(data.get('Стаж', '2 года'))
        all_possible_features['Стаж_месяцы'] = float(experience_months) / 120.0  # Нормализуем к 10 годам
        
        # 3. KPI показатели - используем реальные значения от 0 до 1
        kpi_months = ['июнь', 'июль', 'август', 'сентябрь', 'октябрь']
        kpi_values = []
        
        for month in kpi_months:
            kpi_value = data.get(month, 0.8)
            if isinstance(kpi_value, str):
                try:
                    kpi_value = float(kpi_value) 
                    # Ограничиваем значения KPI от 0 до 1
                    kpi_value = max(0.0, min(1.0, kpi_value))
                except:
                    kpi_value = 0.8
            all_possible_features[f'KPI_{month}'] = float(kpi_value)
            kpi_values.append(float(kpi_value))
        
        # 4. Статистики по KPI - нормализуем все метрики
        kpi_array = np.array(kpi_values)
        
        # Количество заполненных KPI (от 0 до 5)
        all_possible_features['KPI_заполнено_показателей'] = float(len([x for x in kpi_values if x > 0])) / 5.0
        
        # Стандартное отклонение KPI (нормализуем к 0.5)
        std_value = kpi_array.std() if len(kpi_values) > 1 else 0.1
        all_possible_features['KPI_стабильность'] = float(std_value) / 0.5
        
        # Минимальное и максимальное значение KPI (уже нормализованы от 0 до 1)
        all_possible_features['KPI_мин'] = float(kpi_array.min())
        all_possible_features['KPI_макс'] = float(kpi_array.max())
        
        # Размах KPI (нормализуем к 1.0)
        kpi_range = kpi_array.max() - kpi_array.min()
        all_possible_features['KPI_размах'] = float(kpi_range)
        
        # Тренд KPI (нормализуем к 0.1)
        if len(kpi_values) >= 2:
            x = np.arange(len(kpi_values))
            try:
                trend = np.polyfit(x, kpi_values, 1)[0]
                all_possible_features['KPI_тренд'] = float(trend) / 0.1
            except:
                all_possible_features['KPI_тренд'] = 0.0
        else:
            all_possible_features['KPI_тренд'] = 0.0
        
        # Последний KPI (уже нормализован)
        all_possible_features['KPI_последний'] = float(kpi_values[-1] if kpi_values else 0.8)
        
        # 5. Обработка отпуска - нормализуем к 24 месяцам
        vacation_date = data.get('Отпуск (когда ходил в последний раз)', 'нет')
        vacation_months = self._vacation_months_ago(vacation_date)
        all_possible_features['Отпуск_месяцев_назад'] = float(vacation_months) / 24.0
        
        # 6. Обучение - бинарный признак
        training_key = data.get('Обучение', 'в процессе')
        binary_mapping = {
            'да': 1, 'нет': 0, 'yes': 1, 'no': 0,
            'прошел': 1, 'не прошел': 0, 'нет аттестации': 0,
            'завершена': 1, 'в процессе': 0, 'завершено': 1
        }
        all_possible_features['Обучение'] = float(binary_mapping.get(training_key, 0))
        
        # 7. Руководитель - бинарный признак
        manager_key = data.get('В подчиненнии сотрудники', 'Сотрудник')
        manager_mapping = {
            'Руководитель': 1, 'Сотрудник': 0, 'да': 1, 'нет': 0
        }
        all_possible_features['Руководитель'] = float(manager_mapping.get(manager_key, 0))
        
        # 8. Должности - one-hot encoding
        positions = [
            'Бригадир', 'Бухгалтер', 'Главный бухгалтер', 'Дизайнер', 'Директор филиала',
            'Кассир', 'Кладовщик', 'Курьер', 'Логист', 'Менеджер по продажам',
            'Менеджер по работе с клиентами', 'Менеджер по территориальному развити.',
            'Разработчик бэкенд', 'Разработчик фронт', 'Руководитель клиентсокго отдела',
            'Руководитель контактного-центра 1 линии', 'Руководитель отдела продаж',
            'Руководитель проекта', 'Руководитель склада',
            'Старший менеджер группы регионального развития по клиентскому сервису',
            'Старший менеджер по работе с клиентами', 'Тестировщик', 'Юрист'
        ]
        
        for pos_name in positions:
            all_possible_features[f'Должность_{pos_name}'] = 0.0
        
        current_position = data.get('Должность', 'Менеджер по работе с клиентами')
        for pos_name in positions:
            if current_position == pos_name:
                all_possible_features[f'Должность_{pos_name}'] = 1.0
                break
        else:
            if 'менеджер' in current_position.lower() or 'руководитель' in current_position.lower():
                all_possible_features['Должность_Менеджер по работе с клиентами'] = 1.0
            else:
                all_possible_features['Должность_Менеджер по работе с клиентами'] = 1.0
        
        # Создаем DataFrame с правильным порядком признаков
        df_all = pd.DataFrame([all_possible_features])
        processed_data = {}
        
        if self.expected_features:
            for feature in self.expected_features:
                if feature in df_all.columns:
                    processed_data[feature] = df_all[feature].iloc[0]
                else:
                    processed_data[feature] = 0.0
        else:
            processed_data = all_possible_features
        
        df = pd.DataFrame([processed_data])
        
        if self.expected_features:
            df = df[self.expected_features]
        
        # Диагностика данных
        print(f"🔍 Создано признаков: {len(df.columns)}")
        print(f"📊 Диапазон значений: [{df.min().min():.3f}, {df.max().max():.3f}]")
        print(f"📈 Пример KPI: {[df[f'KPI_{month}'].iloc[0] for month in ['июнь', 'июль', 'август']]}")
        
        return df

    def _experience_to_months(self, experience_str):
        """Преобразование стажа в месяцы"""
        if isinstance(experience_str, (int, float)):
            return min(int(experience_str), 120)  # Ограничиваем 10 годами
        
        if not isinstance(experience_str, str):
            return 24
        
        total_months = 0
        try:
            if 'год' in experience_str:
                years_match = experience_str.split('год')[0].strip()
                if years_match.isdigit():
                    total_months += min(int(years_match) * 12, 120)
            elif 'лет' in experience_str:
                years_match = experience_str.split('лет')[0].strip()
                if years_match.isdigit():
                    total_months += min(int(years_match) * 12, 120)
            
            if 'месяц' in experience_str:
                months_part = experience_str.split('месяц')[0]
                months_match = months_part.split()[-1]
                if months_match.isdigit():
                    total_months += min(int(months_match), 11)
        except:
            pass
        
        return total_months if total_months > 0 else 24

    def _vacation_months_ago(self, vacation_date):
        """Расчет сколько месяцев назад был отпуск"""
        if not vacation_date or vacation_date == 'нет':
            return 12
        
        try:
            if isinstance(vacation_date, str):
                for fmt in ['%Y-%m-%d', '%d.%m.%Y', '%Y-%m-%d %H:%M:%S']:
                    try:
                        vacation_dt = datetime.strptime(vacation_date, fmt)
                        break
                    except:
                        continue
                else:
                    return 12
            
            current_date = datetime(2025, 1, 1)
            months_ago = (current_date.year - vacation_dt.year) * 12 + (current_date.month - vacation_dt.month)
            return min(max(1, months_ago), 24)  # Ограничиваем 2 годами
        except:
            return 12

    def process_single_employee(self, employee_data):
        """Обработка данных одного сотрудника"""
        processed_data = self.transform_to_model_features(employee_data)
        print(f"🔢 Форма данных: {processed_data.shape}")
        return processed_data

    def predict_burnout(self, processed_data):
        """Предсказание выгорания для обработанных данных"""
        if self.model is None:
            print("❌ Модель не загружена")
            return None
        
        print(f"🎯 Данные для предсказания: {processed_data.shape}")
        
        try:
            scaled_data = self.scaler.transform(processed_data)
            print(f"📐 Диапазон масштабированных данных: [{scaled_data.min():.3f}, {scaled_data.max():.3f}]")
        except ValueError as e:
            print(f"❌ Ошибка при масштабировании данных: {e}")
            return None
        
        prediction = self.model.predict(scaled_data)
        probability = self.model.predict_proba(scaled_data)
        
        print(f"🎯 Результат предсказания: {prediction[0]}")
        print(f"📊 Вероятности: [Норма: {probability[0, 0]:.3f}, Выгорание: {probability[0, 1]:.3f}]")
        
        return {
            'prediction': int(prediction[0]),
            'burnout_probability': float(probability[0, 1]),
            'no_burnout_probability': float(probability[0, 0]),
            'confidence': float(max(probability[0]))
        }

    def interpret_prediction(self, prediction_result):
        """Интерпретация результатов предсказания"""
        prediction = prediction_result['prediction']
        burnout_prob = prediction_result['burnout_probability']
        
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
        
        return {
            'status': status,
            'probability': burnout_prob,
            'recommendation': recommendation,
            'color': color
        }

    def process_json_file(self, json_path):
        """Обработка всего JSON файла"""
        data = self.load_json_data(json_path)
        if data is None:
            return None
        
        results = []
        
        if isinstance(data, list):
            employees = data
        elif isinstance(data, dict):
            if 'employees' in data:
                employees = data['employees']
            else:
                employees = [data]
        else:
            print("❌ Неподдерживаемый формат JSON")
            return None
        
        print(f"🔍 Обработка {len(employees)} сотрудников...")
        
        for i, employee_data in enumerate(employees):
            print(f"\n👤 Сотрудник {i+1}:")
            employee_id = employee_data.get('ФИО', f'Сотрудник_{i+1}')
            print(f"   ID: {employee_id}")
            
            processed_data = self.process_single_employee(employee_data)
            
            if processed_data.empty:
                print(f"   ❌ Не удалось обработать данные сотрудника")
                continue
            
            prediction_result = self.predict_burnout(processed_data)
            if prediction_result is None:
                continue
            
            interpretation = self.interpret_prediction(prediction_result)
            
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
            
            print(f"   {interpretation['color']} Статус: {interpretation['status']}")
            print(f"   📊 Вероятность выгорания: {prediction_result['burnout_probability']:.1%}")
            print(f"   💡 Рекомендация: {interpretation['recommendation']}")
        
        return results

    def save_results(self, results, output_path='prediction_results.json'):
        """Сохранение результатов в JSON файл"""
        try:
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
    parser.add_argument('--output', '-o', default='prediction_results.json', help='Путь для сохранения результатов')
    parser.add_argument('--model', '-m', default='svm_model.pkl', help='Путь к файлу модели')
    
    args = parser.parse_args()
    
    print("🎯 ЗАПУСК ПРОГНОЗИРОВАНИЯ ВЫГОРАНИЯ ИЗ JSON")
    print("=" * 50)
    
    predictor = JSONPredictor(args.model)
    
    if predictor.model is None:
        return
    
    results = predictor.process_json_file(args.json_file)
    
    if results:
        predictor.save_results(results, args.output)
        
        burnout_count = sum(1 for r in results if r['prediction'] == 1)
        total_count = len(results)
        
        print(f"\n📈 СТАТИСТИКА:")
        print(f"   Всего сотрудников: {total_count}")
        print(f"   С выгоранием: {burnout_count}")
        print(f"   Без выгорания: {total_count - burnout_count}")
        print(f"   Процент выгорания: {burnout_count/total_count*100:.1f}%")

if __name__ == "__main__":
    main()