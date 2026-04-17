"""
Модуль тестов для лабораторной работы по NumPy.

Содержит тесты для проверки:
- Создания и обработки массивов
- Векторных и матричных операций
- Статистического анализа
- Визуализации данных
"""

import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from main import *


# ===========================================================
# ========================== ТЕСТЫ ===========================
# ===========================================================

def test_create_vector() -> None:
    """
    Проверка функции create_vector.

    Тестирует:
    - Тип возвращаемого значения (np.ndarray)
    - Форму массива (10,)
    - Значения элементов (0-9)
    """
    v: np.ndarray = create_vector()
    assert isinstance(v, np.ndarray)
    assert v.shape == (10,)
    assert np.array_equal(v, np.arange(10))


def test_create_matrix() -> None:
    """
    Проверка функции create_matrix.

    Тестирует:
    - Тип возвращаемого значения (np.ndarray)
    - Форму матрицы (5, 5)
    - Диапазон значений [0, 1)
    """
    m: np.ndarray = create_matrix()
    assert isinstance(m, np.ndarray)
    assert m.shape == (5, 5)
    assert np.all((m >= 0) & (m < 1))


def test_reshape_vector() -> None:
    """
    Проверка функции reshape_vector.

    Тестирует:
    - Преобразование формы (10,) -> (2, 5)
    - Сохранение значений после преобразования
    """
    v: np.ndarray = np.arange(10)
    reshaped: np.ndarray = reshape_vector(v)
    assert reshaped.shape == (2, 5)
    assert reshaped[0, 0] == 0
    assert reshaped[1, 4] == 9


def test_vector_add() -> None:
    """
    Проверка функции vector_add.

    Тестирует:
    - Поэлементное сложение векторов
    - Работу с разными размерами векторов
    """
    assert np.array_equal(
        vector_add(np.array([1, 2, 3]), np.array([4, 5, 6])),
        np.array([5, 7, 9])
    )
    assert np.array_equal(
        vector_add(np.array([0, 1]), np.array([1, 1])),
        np.array([1, 2])
    )


def test_scalar_multiply() -> None:
    """
    Проверка функции scalar_multiply.

    Тестирует:
    - Умножение вектора на скаляр
    - Корректность результатов
    """
    assert np.array_equal(
        scalar_multiply(np.array([1, 2, 3]), 2),
        np.array([2, 4, 6])
    )


def test_elementwise_multiply() -> None:
    """
    Проверка функции elementwise_multiply.

    Тестирует:
    - Поэлементное умножение векторов
    - Корректность результатов
    """
    assert np.array_equal(
        elementwise_multiply(np.array([1, 2, 3]), np.array([4, 5, 6])),
        np.array([4, 10, 18])
    )


def test_dot_product() -> None:
    """
    Проверка функции dot_product.

    Тестирует:
    - Скалярное произведение векторов
    - Корректность вычислений
    """
    assert dot_product(np.array([1, 2, 3]), np.array([4, 5, 6])) == 32
    assert dot_product(np.array([2, 0]), np.array([3, 5])) == 6


def test_matrix_multiply() -> None:
    """
    Проверка функции matrix_multiply.

    Тестирует:
    - Умножение матриц
    - Сравнение с оператором @
    """
    A: np.ndarray = np.array([[1, 2], [3, 4]])
    B: np.ndarray = np.array([[2, 0], [1, 2]])
    assert np.array_equal(matrix_multiply(A, B), A @ B)


def test_matrix_determinant() -> None:
    """
    Проверка функции matrix_determinant.

    Тестирует:
    - Вычисление определителя матрицы 2x2
    - Точность вычислений (округление до 5 знаков)
    """
    A: np.ndarray = np.array([[1, 2], [3, 4]])
    assert round(matrix_determinant(A), 5) == -2.0


def test_matrix_inverse() -> None:
    """
    Проверка функции matrix_inverse.

    Тестирует:
    - Вычисление обратной матрицы
    - Проверка: A @ A_inv = I
    """
    A: np.ndarray = np.array([[1, 2], [3, 4]])
    invA: np.ndarray = matrix_inverse(A)
    assert np.allclose(A @ invA, np.eye(2))


def test_solve_linear_system() -> None:
    """
    Проверка функции solve_linear_system.

    Тестирует:
    - Решение системы линейных уравнений Ax = b
    - Проверка: A @ x = b
    """
    A: np.ndarray = np.array([[2, 1], [1, 3]])
    b: np.ndarray = np.array([1, 2])
    x: np.ndarray = solve_linear_system(A, b)
    assert np.allclose(A @ x, b)


def test_load_dataset() -> None:
    """
    Проверка функции load_dataset.

    Тестирует:
    - Загрузку CSV файла
    - Преобразование в NumPy массив
    - Корректность данных
    """
    test_data: str = "math,physics,informatics\n78,81,90\n85,89,88"
    with open("test_data.csv", "w") as f:
        f.write(test_data)
    try:
        data: np.ndarray = load_dataset("test_data.csv")
        assert data.shape == (2, 3)
        assert np.array_equal(data[0], [78, 81, 90])
    finally:
        if os.path.exists("test_data.csv"):
            os.remove("test_data.csv")


def test_statistical_analysis() -> None:
    """
    Проверка функции statistical_analysis.

    Тестирует:
    - Вычисление среднего значения
    - Вычисление минимума и максимума
    """
    data: np.ndarray = np.array([10, 20, 30])
    result: Dict[str, float] = statistical_analysis(data)
    assert result["mean"] == 20
    assert result["min"] == 10
    assert result["max"] == 30


def test_normalization() -> None:
    """
    Проверка функции normalize_data.

    Тестирует:
    - Min-Max нормализацию
    - Диапазон значений [0, 1]
    """
    data: np.ndarray = np.array([0, 5, 10])
    norm: np.ndarray = normalize_data(data)
    assert np.allclose(norm, np.array([0, 0.5, 1]))


def test_plot_histogram() -> None:
    """
    Проверка функции plot_histogram.

    Тестирует:
    - Создание файла гистограммы
    - Отсутствие ошибок при выполнении
    """
    data: np.ndarray = np.array([1, 2, 3, 4, 5])
    plot_histogram(data)
    assert os.path.exists("code/numpy_lab/plots/histogram.png")


def test_plot_heatmap() -> None:
    """
    Проверка функции plot_heatmap.

    Тестирует:
    - Создание файла тепловой карты
    - Отсутствие ошибок при выполнении
    """
    matrix: np.ndarray = np.array([[1, 0.5], [0.5, 1]])
    plot_heatmap(matrix)
    assert os.path.exists("code/numpy_lab/plots/heatmap.png")


def test_plot_line() -> None:
    """
    Проверка функции plot_line.

    Тестирует:
    - Создание файла линейного графика
    - Отсутствие ошибок при выполнении
    """
    x: np.ndarray = np.array([1, 2, 3])
    y: np.ndarray = np.array([4, 5, 6])
    plot_line(x, y)
    assert os.path.exists("code/numpy_lab/plots/line_plot.png")


if __name__ == "__main__":
    print("Запустите pytest code/numpy_lab/tests.py -v для проверки лабораторной работы.")
