"""check_data.py - Быстрая диагностика датасета Titanic"""
import os
import pandas as pd

# Автоматически определяем путь относительно этого скрипта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../lab_6_titanic/src
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Идём на 1 уровень вверх → .../lab_6_titanic
CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'titanic.csv')

print("🔍 ДИАГНОСТИКА ЗАГРУЗКИ ДАННЫХ")
print("-" * 50)
print(f"📁 Абсолютный путь: {CSV_PATH}")
print(f"📂 Файл существует: {os.path.exists(CSV_PATH)}")

if not os.path.exists(CSV_PATH):
    print("\n❌ Файл не найден! Проверьте структуру папок:")
    print(f"   Ожидается: {CSV_PATH}")
    exit(1)

try:
    df = pd.read_csv(CSV_PATH)
    print(f"\n✅ УСПЕШНО ЗАГРУЖЕНО:")
    print(f"   Строк: {df.shape[0]}")
    print(f"   Столбцов: {df.shape[1]}")
    print(f"\n📋 Колонки: {df.columns.tolist()}")
    print(f"\n📊 Survived (выживаемость):")
    print(df['Survived'].value_counts())
    print(f"   Средний % выживших: {df['Survived'].mean()*100:.1f}%")

    print(f"\n👀 Первые 3 строки:")
    print(df.head(3).to_string())

except Exception as e:
    print(f"\n❌ Ошибка при чтении CSV: {e}")
    print("💡 Попробуйте добавить encoding='utf-8-sig' или 'cp1251' в pd.read_csv()")