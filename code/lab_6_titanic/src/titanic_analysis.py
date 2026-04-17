import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


class TitanicDataProcessor:
    """Класс для обработки датасета"""

    def __init__(self, filepath: str, output_dir: str = 'output'):
        """Инициализация процессора данных."""
        self.filepath = filepath
        self.output_dir = output_dir
        self.df: Optional[pd.DataFrame] = None
        self._create_output_dir()

    def _create_output_dir(self):
        """Создаёт директорию для вывода, если она не существует."""
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output: {self.output_dir}")

    def load_data(self) -> pd.DataFrame:
        """
        Загрузить данные из CSV-файла

        Returns:
        pd.DataFrame
        """
        print("№1: ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ")

        self.df = pd.read_csv(self.filepath)
        print(f"Данные загружены: {self.df.shape[0]} строк, {self.df.shape[1]} столбцов")
        return self.df

    def show_first_rows(self, n: int = 10):
        """Вывести первые 10 строк дата-фрейма"""
        print(f"\nПервые {n} строк датафрейма:")
        print(self.df.head(n).to_string())

    def check_data_types(self):
        """Проверить типы данных каждого столбца"""
        print(f"\nТипы данных столбцов:")
        for col, dtype in self.df.dtypes.items():
            print(f"  {col:15} {dtype}")

    def check_missing_values(self) -> pd.Series:
        """
        Определить количество пропусков в каждом столбце

        Returns:
        pd.Series
        """
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df) * 100).round(2)

        print(f"\nПропуcки:")
        for col in missing[missing > 0].index:
            print(f"  {col:15} {missing[col]:3} ({missing_percent[col]}%)")

        return missing

    def get_statistics(self):
        """Получить статистические характеристики числовых признаков"""
        print(f"\nСтатистические характеристики числовых признаков:")
        print(self.df.describe().to_string())

    def plot_histograms(self, save: bool = True):
        """Построить гистограммы распределения для числовых признаков"""
        df = self.df.copy()
        df['Status'] = df['Survived'].map({0: 'Погиб', 1: 'Выжил'})

        colors = {'Погиб': '#E74C3C', 'Выжил': '#2ECC71'}

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Анализ факторов выживаемости Titanic', fontsize=16, fontweight='bold', y=0.98)

        sns.countplot(
            data=df, x='Sex', hue='Status',
            palette=colors, ax=axes[0, 0]
        )
        axes[0, 0].set_title('1. Пол и Выживаемость', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Пол пассажира')
        axes[0, 0].set_ylabel('Количество людей')
        axes[0, 0].legend(title='Статус')

        pclass_col = 'Pclass_str' if 'Pclass_str' in df.columns else 'Pclass'

        class_counts = df.groupby([pclass_col, 'Status']).size().unstack(fill_value=0)
        class_counts.plot(
            kind='bar', stacked=True, ax=axes[0, 1],
            color=[colors['Погиб'], colors['Выжил']]
        )
        axes[0, 1].set_title('2. Класс билета и Выживаемость', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Класс билета')
        axes[0, 1].set_ylabel('Количество пассажиров')
        axes[0, 1].tick_params(axis='x', rotation=0)
        axes[0, 1].legend(title='Статус', loc='upper right')

        sns.histplot(
            data=df, x='Age', hue='Status', bins=30, alpha=0.6, multiple='layer',
            palette=colors, ax=axes[1, 0]
        )
        axes[1, 0].set_title('3. Возраст и Выживаемость', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Возраст')
        axes[1, 0].set_ylabel('Частота')
        axes[1, 0].legend(title='Статус')

        sns.violinplot(
            data=df, x=pclass_col, y='Age', hue='Status',
            split=True, palette=colors, ax=axes[1, 1], inner='quartile'
        )
        axes[1, 1].set_title('4. Возраст и Класс', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Класс билета')
        axes[1, 1].set_ylabel('Возраст')
        axes[1, 1].legend(title='Статус')

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'survival_dashboard.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    def handle_missing_values(self):
        """Обработка пропусков значений в столбцах Age, Embarked, Cabin."""
        print("№2: ОБРАБОТКА ПРОПУСКОВ")

        if 'Age' in self.df.columns:
            age_median = self.df['Age'].median()
            age_mean = self.df['Age'].mean()
            print(f"\nСтолбец 'Age':")
            print(f"   Среднее: {age_mean:.2f}, Медиана: {age_median:.2f}")
            print(f"   Пропусков: {self.df['Age'].isnull().sum()}")

            self.df['Age'] = self.df['Age'].fillna(age_median)
            print(f"Пропуски заполнены медианой")

            self.df['Age_group'] = pd.cut(
                self.df['Age'],
                bins=[0, 12, 18, 35, 60, 100],
                labels=['Child', 'Teen', 'Young', 'Adult', 'Senior']
            )
            print(f"Создан признак 'Age_group'")

        if 'Embarked' in self.df.columns:
            embarked_mode = self.df['Embarked'].mode()[0]
            print(f"\nСтолбец 'Embarked':")
            print(f"Наиболее частое: '{embarked_mode}'")
            print(f"Пропусков: {self.df['Embarked'].isnull().sum()}")

            self.df['Embarked'] = self.df['Embarked'].fillna(embarked_mode)
            print(f"Пропуски заполнены на наиболее частое")

        if 'Cabin' in self.df.columns:
            print(f"\nСтолбец 'Cabin':")
            print(f"Пропусков: {self.df['Cabin'].isnull().sum()} "
                  f"({self.df['Cabin'].isnull().sum() / len(self.df) * 100:.1f}%)")

            self.df['Cabin_deck'] = self.df['Cabin'].apply(
                lambda x: str(x)[0] if pd.notna(x) and str(x) != 'nan' else 'U'
            )
            print(f"Создан признак 'Cabin_deck' (по первой букве каюты)")

        return self.df

    def transform_features(self):
        """
        Преобразовать Pclass в категориальный тип строкового значения (из числового - в строковое: 1 - F, 2 - S, 3 - T )
        Создать новый признак Title из столбца Name (мистер, миссис и т.д.)
        Преобразовать Sex в числовой формат (0/1)
        Создать признак FamilySize = SibSp + Parch + 1
        Создать признак IsAlone (1 если FamilySize = 1, иначе 0)
        """
        print("№3: ПРЕОБРАЗОВАНИЕ ПРИЗНАКОВ")

        if 'Pclass' in self.df.columns:
            pclass_mapping = {1: 'F', 2: 'S', 3: 'T'}  # First, Second, Third
            self.df['Pclass_str'] = self.df['Pclass'].map(pclass_mapping).astype('category')
            print(f"\nPclass преобразован: {pclass_mapping}")
            print(f"Уникальные значения: {self.df['Pclass_str'].unique().tolist()}")

        if 'Name' in self.df.columns:
            self.df['Title'] = self.df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
            self.df['Title'] = self.df['Title'].replace({
                'Mlle': 'Miss',
                'Ms': 'Miss',
                'Mme': 'Mrs'
            })
            self.df['Title'] = self.df['Title'].str.strip().fillna('Unknown')

            print(f"\nСоздан признак 'Title'. Всего уникальных категорий: {self.df['Title'].nunique()}")
            print(f"   Категории: {sorted(self.df['Title'].unique().tolist())}")

        if 'Sex' in self.df.columns:
            sex_mapping = {'male': 0, 'female': 1}
            self.df['Sex_num'] = self.df['Sex'].map(sex_mapping)
            print(f"\nSex преобразован: {sex_mapping}")

        if all(col in self.df.columns for col in ['SibSp', 'Parch']):
            self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch'] + 1
            self.df['IsAlone'] = (self.df['FamilySize'] == 1).astype(int)
            print(f"\nСозданы признаки:")
            print(f"FamilySize = SibSp + Parch + 1")
            print(f"IsAlone = 1 если FamilySize==1, иначе 0")
            print(f"Распределение IsAlone: {self.df['IsAlone'].value_counts().to_dict()}")

        return self.df

    def handle_outliers(self, fare_winsor_percentile: float = 0.95,
                        age_winsor_percentile: float = 0.95):
        """
        Обрабатывает выбросы с помощью IQR-метода

        fare_winsor_percentile : float
            Перцентиль для ограничения значений Fare (по умолчанию 0.95)
        age_winsor_percentile : float
            Перцентиль для ограничения значений Age (по умолчанию 0.95)
        """
        print("№4: ОБРАБОТКА ВЫБРОСОВ")

        if 'Fare' in self.df.columns:
            print(f"\nАнализ столбца 'Fare':")

            Q1 = self.df['Fare'].quantile(0.25)
            Q3 = self.df['Fare'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound_iqr = Q3 + 1.5 * IQR

            upper_bound_95 = self.df['Fare'].quantile(0.95)

            print(f"   • IQR = {IQR:.2f}")
            print(f"   • Граница выбросов (IQR): {upper_bound_iqr:.2f}")
            print(f"   • 95-й перцентиль (для обрезки): {upper_bound_95:.2f}")

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].boxplot(
                self.df['Fare'].dropna(),
                vert=True,
                patch_artist=True,
                boxprops=dict(facecolor='steelblue', edgecolor='black', linewidth=1.5),
                medianprops=dict(color='red', linewidth=2.5),
                meanprops=dict(marker='D', markerfacecolor='green',
                               markeredgecolor='black', markersize=8),
                flierprops=dict(marker='o', markerfacecolor='crimson',
                                markeredgecolor='black', markersize=5, alpha=0.6),
                showmeans=True
            )
            axes[0].set_title('Boxplot: Fare\n(формальные выбросы по IQR)', fontsize=11, fontweight='bold')
            axes[0].set_ylabel('Стоимость билета (£)', fontsize=10)
            axes[0].tick_params(axis='x', labelbottom=False)  # Убираем метку "1" внизу
            axes[0].grid(axis='y', alpha=0.3, linestyle='--')

            # Добавляем подписи ключевых значений
            axes[0].axhline(Q1, color='gray', linestyle=':', alpha=0.5)
            axes[0].axhline(Q3, color='gray', linestyle=':', alpha=0.5)
            axes[0].text(1.2, Q1, f'Q1={Q1:.1f}', fontsize=8, va='center')
            axes[0].text(1.2, Q3, f'Q3={Q3:.1f}', fontsize=8, va='center')
            axes[0].text(1.2, upper_bound_iqr, f'IQR-граница={upper_bound_iqr:.1f}',
                         fontsize=8, va='center', color='red', fontweight='bold')

            # ========================================================
            # ГРАФИК 2: LOG-ГИСТОГРАММА — реальное распределение
            # ========================================================
            # log1p = log(x + 1) — позволяет работать с нулевыми значениями
            fare_log = np.log1p(self.df['Fare'])

            axes[1].hist(fare_log, bins=40, edgecolor='black',
                         color='lightcoral', alpha=0.8)
            axes[1].axvline(np.log1p(upper_bound_95), color='darkred',
                            linestyle='-', linewidth=2, label=f'95% перцентиль')
            axes[1].axvline(np.log1p(upper_bound_iqr), color='red',
                            linestyle='--', linewidth=1.5, label=f'IQR-граница')
            axes[1].set_title('Распределение: log(Fare + 1)\n(нормализованный вид)',
                              fontsize=11, fontweight='bold')
            axes[1].set_xlabel('log(Стоимость + 1)', fontsize=10)
            axes[1].set_ylabel('Частота', fontsize=10)
            axes[1].legend(fontsize=9)
            axes[1].grid(axis='y', alpha=0.3, linestyle='--')

            # Подпись: как интерпретировать ось X
            axes[1].text(0.5, 0.02,
                         'Ось X: log(£+1) → 0=£0, 2.3=£9, 3.0=£19, 4.6=£99, 6.2=£499',
                         fontsize=8, style='italic', ha='center', transform=axes[1].transAxes)

            plt.tight_layout()
            filepath = os.path.join(self.output_dir, 'fare_analysis.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"   ✓ График сохранён: {filepath}")
            plt.close()

            # Применяем Winsorization (обрезка по 95-му перцентилю)
            self.df['Fare_winsorized'] = self.df['Fare'].clip(upper=upper_bound_95)
            print(f"   ✓ Winsorization: значения > £{upper_bound_95:.2f} заменены на {upper_bound_95:.2f}")

    def aggregate_data(self):
        """
        Агрегация данных
        Посчитать среднее выживание по классам
        Группировать данные по Pclass и Sex
        Посчитать медианный возраст по портам посадки
        Создать сводную таблицу выживаемости по новым признакам
        Сохранить очищенные данные в новый CSV-файл
        """
        print("№5: АГРЕГАЦИЯ ДАННЫХ")

        if all(col in self.df.columns for col in ['Pclass', 'Survived']):
            survival_by_class = self.df.groupby('Pclass')['Survived'].mean()
            print(f"\nСредний процент выживших по классам билетов:")
            for pclass, rate in survival_by_class.items():
                print(f"   Класс {pclass}: {rate * 100:.1f}%")

        if all(col in self.df.columns for col in ['Pclass', 'Sex', 'Survived']):
            survival_by_class_sex = self.df.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack()
            print(f"\nВыживаемость по классу и полу:")
            print(survival_by_class_sex.to_string())

        if all(col in self.df.columns for col in ['Embarked', 'Age']):
            age_by_port = self.df.groupby('Embarked')['Age'].median()
            print(f"\nМедианный возраст по портам посадки:")
            for port, age in age_by_port.items():
                print(f"   Порт '{port}': {age:.1f} лет")

        pivot_cols = [col for col in ['Age_group', 'Title', 'IsAlone', 'Cabin_deck']
                      if col in self.df.columns]
        if 'Survived' in self.df.columns and pivot_cols:
            for col in pivot_cols:
                if self.df[col].nunique() <= 10:
                    pivot = pd.crosstab(self.df[col], self.df['Survived'],
                                        margins=True, margins_name='Total')
                    pivot['Survival_Rate_%'] = (pivot[1] / pivot['Total'] * 100).round(1)
                    print(f"\nВыживаемость по признаку '{col}':")
                    print(pivot.to_string())

        return self.df

    def save_cleaned_data(self, filename: str = 'titanic_cleaned.csv'):
        """Сохраняет обработанные данные в новый CSV-файл."""
        filepath = os.path.join(self.output_dir, filename)
        self.df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\nОчищенные данные сохранены: {filepath}")
        print(f"   Размер файла: {os.path.getsize(filepath) / 1024:.1f} KB")
        return filepath

    def run_full_pipeline(self):
        """Запускает полный пайплайн обработки данных."""
        self.load_data()
        self.show_first_rows()
        self.check_data_types()
        self.check_missing_values()
        self.get_statistics()
        self.plot_histograms(save=True)
        self.handle_missing_values()
        self.transform_features()
        self.handle_outliers()
        self.aggregate_data()
        output_path = self.save_cleaned_data()
        return output_path


def main():
    current_script_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_script_path)
    project_root = os.path.dirname(src_dir)
    INPUT_FILE = os.path.join(project_root, 'data', 'titanic.csv')
    OUTPUT_DIR = os.path.join(project_root, 'output')
    if not os.path.exists(INPUT_FILE):
        print(f"\nОшибка")
        return
    try:
        processor = TitanicDataProcessor(filepath=INPUT_FILE, output_dir=OUTPUT_DIR)

        output_path = processor.run_full_pipeline()
        print(f"Файл сохранен: {output_path}")

    except Exception as e:
        print(f"\nОшибка")
        print(f"   {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
