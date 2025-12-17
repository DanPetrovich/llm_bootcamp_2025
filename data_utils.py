from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


def load_description(path: str | Path = "description.json") -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Загружает описание данных из description.json и преобразует в формат,
    ожидаемый промптами LLM.

    Parameters
    ----------
    path : str | Path, optional
        Путь к файлу description.json. По умолчанию "description.json"

    Returns
    -------
    structure_info : Dict[str, str]
        Словарь с типами колонок: {column_name: dtype_string}
    description_info : Dict[str, Any]
        Полное описание датасета из JSON файла
    """
    path = Path(path)
    
    with path.open("r", encoding="utf-8") as f:
        description_data = json.load(f)
    
    # Преобразуем fields в словарь structure_info
    structure_info = {}
    type_mapping = {
        "integer": "int64",
        "string": "object",
        "boolean": "bool",
        "float": "float64",
        "array<string>": "object",  # массив строк в pandas это object
        "date": "object",
    }
    
    for field in description_data.get("fields", []):
        field_name = field.get("name")
        field_type = field.get("type", "string")
        # Маппим типы из JSON в pandas dtypes
        pandas_dtype = type_mapping.get(field_type, "object")
        structure_info[field_name] = pandas_dtype
    
    return structure_info, description_data
import numpy as np


def load_vacancies(
    path: str | Path = "vacancies.json",
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Dict[str, Any]]]:
    """
    Преобразует данные из vacancies.json в pandas.DataFrame и
    возвращает вместе со структурой и описанием данных.

    Parameters
    ----------
    path : str | Path, optional
        Путь к файлу vacancies.json. По умолчанию "vacancies.json"
        в корне проекта.

    Returns
    -------
    df : pd.DataFrame
        Датафрейм с вакансиями. Данные уже в плоском формате.
    structure : pd.Series
        Структура датафрейма: типы данных по каждому столбцу (df.dtypes).
    description : Dict[str, Dict[str, Any]]
        Описание датафрейма: словарь, где ключи - названия колонок,
        значения - словари с метриками для каждой колонки (без nan).
    """
    path = Path(path)

    # Читаем JSON файл - теперь это плоский массив объектов
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # Создаем DataFrame напрямую из списка словарей
    df = pd.DataFrame(raw)

    # Структура датафрейма
    structure = df.dtypes

    # Улучшенное описание: убираем nan и делаем более структурированным
    description = df.describe()

    return df, structure, description
