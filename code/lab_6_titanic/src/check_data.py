import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'titanic.csv')

if not os.path.exists(CSV_PATH):
    print("\nФайл не найден")
    exit(1)

try:
    df = pd.read_csv(CSV_PATH)
    print(f"Кол-во строк: {df.shape[0]}")
    print(f"Кол-во столбцов: {df.shape[1]}")
    print(f"\nКолонки: {df.columns.tolist()}")
    print(f"Кол-во выживших: {df['Survived'].value_counts()}")
    print(f"Средний % выживших: {df['Survived'].mean() * 100:.1f}%")

    print(f"\nПервые 3 строки:")
    print(df.head(3).to_string())

except Exception as e:
    print(f"\nОшибка при чтении CSV: {e}")
