import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
import self

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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
        print("№1: Первичный анализ данных")

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
        axes[0, 0].set_ylabel('Кол-во людей')
        axes[0, 0].legend(title='Статус')

        pclass_col = 'Pclass_str' if 'Pclass_str' in df.columns else 'Pclass'

        class_counts = df.groupby([pclass_col, 'Status']).size().unstack(fill_value=0)
        class_counts.plot(
            kind='bar', stacked=True, ax=axes[0, 1],
            color=[colors['Выжил'], colors['Погиб']]
        )
        axes[0, 1].set_title('2. Класс билета и Выживаемость', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Класс билета')
        axes[0, 1].set_ylabel('Кол-во пассажиров')
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
        print("№2: Обработка пропусков")

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
        print("№3: Преобразование признаков")

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
        Обрабатывает выбросы с помощью IQR-метода и winsorization.
        """
        print("№4: Обработка выбросов")

        COLOR_BAR = '#D5D8DC'
        COLOR_LINE = '#566573'
        COLOR_MEDIAN = '#922B21'

        if 'Fare' in self.df.columns:
            print("Анализ столбца 'Fare':")

            Q1 = self.df['Fare'].quantile(0.25)
            Q3 = self.df['Fare'].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound_iqr = Q3 + 1.5 * IQR
            upper_bound_95 = self.df['Fare'].quantile(fare_winsor_percentile)

            print(f"Граница выбросов (IQR): {upper_bound_iqr:.2f} £")
            print(f"{int(fare_winsor_percentile * 100)}-й перцентиль: {upper_bound_95:.2f} £")

            # Копия для seaborn
            df_fare = self.df.copy()
            df_fare['log_fare'] = np.log1p(df_fare['Fare'])

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            #Гистограмма 1
            axes[0].boxplot(df_fare['Fare'].dropna(), vert=True, patch_artist=True,
                            showmeans=True,
                            boxprops=dict(facecolor=COLOR_BAR, edgecolor='#566573', linewidth=1.5),
                            medianprops=dict(color=COLOR_MEDIAN, linewidth=2.5),
                            meanprops=dict(marker='D', markerfacecolor='#2E86C1', markeredgecolor='black',
                                           markersize=8),
                            flierprops=dict(marker='o', markerfacecolor=COLOR_MEDIAN, markeredgecolor='black',
                                            markersize=6, alpha=0.6))

            axes[0].set_title('Коробчатая диаграмма: Стоимость билета\n(формальные выбросы по IQR)', fontsize=12,
                              fontweight='bold', pad=12)
            axes[0].set_ylabel('Стоимость билета (£)', fontsize=11)
            axes[0].tick_params(axis='x', labelbottom=False)
            axes[0].grid(axis='y', alpha=0.3, linestyle='--')

            axes[0].axhline(Q1, color='gray', linestyle=':', alpha=0.4)
            axes[0].axhline(Q3, color='gray', linestyle=':', alpha=0.4)
            axes[0].text(1.08, Q1, f'Q1 = {Q1:.1f}', fontsize=9, va='center', ha='left', color='#555')
            axes[0].text(1.08, Q3, f'Q3 = {Q3:.1f}', fontsize=9, va='center', ha='left', color='#555')
            axes[0].text(1.08, upper_bound_iqr, f'IQR-граница = {upper_bound_iqr:.1f}',
                         fontsize=9, va='center', ha='left', color=COLOR_MEDIAN, fontweight='bold')
            axes[0].set_xlim(0.7, 1.4)

            #Гистограмма 2
            sns.histplot(data=df_fare, x='log_fare', bins=35, kde=True, stat='count',
                         color=COLOR_BAR, edgecolor='black', alpha=0.8, linewidth=1.2,
                         line_kws={'color': COLOR_LINE, 'linewidth': 2.5}, ax=axes[1])

            axes[1].axvline(np.log1p(upper_bound_iqr), color=COLOR_MEDIAN, linestyle='--', linewidth=2,
                            label=f'IQR-граница ({upper_bound_iqr:.0f}£)')
            axes[1].axvline(np.log1p(upper_bound_95), color=COLOR_LINE, linestyle='-', linewidth=2.5,
                            label=f'{int(fare_winsor_percentile * 100)}% перцентиль ({upper_bound_95:.0f}£)')

            axes[1].set_title('Распределение: log(Стоимость + 1)\n(сглаженный вид с KDE)', fontsize=12,
                              fontweight='bold', pad=12)
            axes[1].set_xlabel('log(Стоимость + 1)', fontsize=11)
            axes[1].set_ylabel('Частота', fontsize=11)
            axes[1].legend(fontsize=9, loc='upper right', framealpha=0.9)
            axes[1].grid(axis='y', alpha=0.3, linestyle='--')

            log_ticks = np.arange(0, 6.5, 1)
            real_vals = np.expm1(log_ticks).astype(int)
            axes[1].set_xticks(log_ticks)
            axes[1].set_xticklabels([f'log={int(t)}\n({int(v)}£)' if t > 0 else '0\n(0£)'
                                     for t, v in zip(log_ticks, real_vals)], fontsize=9, ha='center')

            plt.tight_layout()
            filepath = os.path.join(self.output_dir, 'fare_analysis.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"График сохранён: {filepath}")
            plt.close()

            self.df['Fare_winsorized'] = self.df['Fare'].clip(upper=upper_bound_95)
            print(f"Winsorization: значения > {upper_bound_95:.2f} £ обрезаны.")

        #Визуал
        if 'Age' in self.df.columns:
            print("Анализ столбца 'Age':")

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            data_age = self.df['Age'].dropna()

            # Гистограмма
            axes[0].hist(data_age, bins=30, edgecolor='black', alpha=0.7, color=COLOR_BAR)
            median_age = data_age.median()
            axes[0].axvline(median_age, color=COLOR_MEDIAN, linestyle='--', linewidth=2,
                            label=f'Медиана: {median_age:.1f}')
            axes[0].set_title('Распределение возраста', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Возраст', fontsize=11)
            axes[0].set_ylabel('Частота', fontsize=11)
            axes[0].legend(fontsize=9)
            axes[0].grid(axis='y', alpha=0.3, linestyle='--')

            axes[1].boxplot(data_age, vert=True, patch_artist=True,
                            boxprops=dict(facecolor=COLOR_BAR, edgecolor='#566573'),
                            medianprops=dict(color=COLOR_MEDIAN, linewidth=2),
                            showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='#2E86C1', markersize=8))
            axes[1].set_title('Коробчатая диаграмма: Возраст', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Возраст (лет)', fontsize=11)
            axes[1].tick_params(axis='x', labelbottom=False)  # Убираем цифру "1"
            axes[1].grid(axis='y', alpha=0.3, linestyle='--')

            plt.tight_layout()
            filepath = os.path.join(self.output_dir, 'age_distribution.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"График сохранён: {filepath}")
            plt.close()

            age_cutoff = self.df['Age'].quantile(age_winsor_percentile)
            self.df['Age_winsorized'] = self.df['Age'].clip(upper=age_cutoff)
            print(
                f"Winsorization Age: значения > {age_winsor_percentile * 100}% перцентили ({age_cutoff:.1f} лет) заменены")

        cols_to_drop = [col for col in self.df.columns if col.endswith('_outlier')]
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)

        return self.df

    def aggregate_data(self):
        """
        Агрегация данных
        Посчитать среднее выживание по классам
        Группировать данные по Pclass и Sex
        Посчитать медианный возраст по портам посадки
        Создать сводную таблицу выживаемости по новым признакам
        Сохранить очищенные данные в новый CSV-файл
        """
        print("№5: Агрегация данных")

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
