import os
import re
from typing import Any, Dict

import requests


def get_system_prompt() -> str:
    """
    System prompt for the LLM that enforces:
    - pure Python code output (no explanations, no markdown),
    - local executability,
    - reading data from vacancies.json,
    - saving plots to disk and returning paths to them in a dictionary.
    """
    return (
        "You are an expert Python data analyst and engineer. "
        "You write **only** pure executable Python 3 code (no markdown, "
        "no backticks, no shell or bash commands, no comments outside of Python code). "
        "Your code will be executed in a local environment that already has "
        "pandas, matplotlib, seaborn, numpy and the standard library installed.\n\n"
        "You receive a natural-language user task and a description of a dataset "
        "that has already been preprocessed in the caller. The caller uses a helper "
        "function 'load_vacancies' from the local module 'data_utils' to load the "
        "data from 'vacancies.json' and flatten the nested 'data' field into a "
        "convenient DataFrame.\n\n"
        "Your job is to:\n"
        "1. Import and use the helper function from 'data_utils' instead of reading "
        "the JSON file directly. Your typical pattern should be:\n"
        "   from data_utils import load_vacancies\n"
        "   df, structure, description = load_vacancies()\n"
        "2. Compute exactly the analytics requested by the user (aggregations, "
        "statistics, segments, time series, etc.).\n"
        "3. If the user asks for plots, create them using matplotlib or seaborn, "
        "save them to PNG files on disk, and include their file paths in the result.\n"
        "4. At the end of the script, construct a Python dictionary named "
        "'ANALYTICS_RESULT' that contains:\n"
        "   - a key 'metrics' with a nested dict of all numeric and other metrics you computed;\n"
        "   - a key 'plots' with a list of dicts, each containing at least 'name' and 'path' "
        "for every saved plot.\n"
        "5. Do not print huge dataframes; if needed, aggregate or sample them. Focus on "
        "meaningful metrics.\n\n"
        "Important constraints:\n"
        "- Do not import or use any network libraries or call remote APIs.\n"
        "- Do not read from or write to any files other than 'vacancies.json' (through "
        "the provided helper function) and image files for plots.\n"
        "- Do not install packages. Use only pandas, matplotlib, seaborn, numpy, os, pathlib, datetime.\n"
        "- Never output any bash/shell snippets or commands (such as rm, cd, ls, chmod, curl, wget, etc.).\n"
        "- The ONLY thing you must return to the caller is valid Python code. "
        "No explanations, no comments outside the code block, no additional text."
    )


def get_user_prompt(
    user_task: str,
    structure: Dict[str, Any],
    df_description: Dict[str, Any],
) -> str:
    """
    Build the user prompt that describes:
    - the user's analytic request,
    - the high-level structure of the DataFrame,
    - precomputed description/statistics.
    """
    return (
        "User task (in natural language):\n"
        f"{user_task}\n\n"
        "Current DataFrame structure (column -> dtype as string):\n"
        f"{structure}\n\n"
        "Precomputed description/statistics of the dataset (per-column stats):\n"
        f"{df_description}\n\n"
        "Write Python code that, when executed locally, performs the requested analysis "
        "on 'vacancies.json', builds any requested plots, saves them as PNG files, and "
        "populates the 'ANALYTICS_RESULT' dictionary as specified in the system prompt."
    )


def _extract_code_block(text: str) -> str:
    """
    Извлекает чистый Python-код из ответа модели:
    - убирает ```python ... ``` или ``` ... ``` обёртки,
    - обрезает лишние пробелы.
    """
    # Ищем тройные кавычки ```...```
    code_fence_pattern = re.compile(r"```(?:python)?(.*)```", re.DOTALL | re.IGNORECASE)
    match = code_fence_pattern.search(text)
    if match:
        code = match.group(1)
    else:
        code = text

    # Убираем возможные префиксы типа "Code:" и ведущие/хвостовые пробелы
    code = code.strip()
    if code.lower().startswith("code:"):
        code = code[5:].strip()

    return code


def llm_request(user_prompt: str, system_prompt: str) -> str:
    """
    Make a request to the Groq LLM and return the cleaned code string.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['API_KEY']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    raw_content = data["choices"][0]["message"]["content"]
    return _extract_code_block(raw_content)

