from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


def load_vacancies(
    path: str | Path = "vacancies.json",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
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
        Датафрейм с вакансиями. Поле "data" разворачивается в столбцы,
        а верхнеуровневый "id" добавляется как отдельный столбец.
    structure : pd.Series
        Структура датафрейма: типы данных по каждому столбцу (df.dtypes).
    description : pd.DataFrame
        Описание датафрейма: базовые статистики и частоты (df.describe(include="all")).
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # Ожидаем структуру: список объектов вида {"id": ..., "data": {...}}
    records: list[Dict[str, Any]] = []
    for item in raw:
        base: Dict[str, Any] = {}

        # Верхнеуровневый id
        if isinstance(item, dict):
            if "id" in item:
                base["id"] = item["id"]

            data_block = item.get("data")
            if isinstance(data_block, dict):
                base.update(data_block)

        if base:
            records.append(base)

    df = pd.DataFrame.from_records(records)

    # Структура и описание датафрейма
    structure = df.dtypes
    description = df.describe(include="all")

    return df, structure, description


