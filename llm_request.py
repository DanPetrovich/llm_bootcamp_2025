import os
import re
import time
from threading import Lock
from typing import Any, Dict

import requests


# Глобальная переменная для отслеживания времени последнего запроса
_last_request_time = 0
_request_lock = Lock()
API_REQUEST_DELAY = 1.0  # Минимальная задержка между запросами в секундах
MAX_RETRY_WAIT_TIME = 30  # Максимальное время ожидания при 429 ошибке (секунды)


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
    previous_error: str | None = None,
    previous_code: str | None = None,
) -> str:
    """
    Build the user prompt that describes:
    - the user's analytic request,
    - the high-level structure of the DataFrame,
    - detailed field descriptions from description.json.
    - optionally: previous error and code for retry attempts.
    """
    # Формируем описание полей из description.json
    fields_description = ""
    if isinstance(df_description, dict) and "fields" in df_description:
        fields_description = "\nDetailed field descriptions:\n"
        for field in df_description["fields"]:
            name = field.get("name", "")
            field_type = field.get("type", "")
            desc = field.get("description", "")
            example = field.get("example", "")
            hint = field.get("aggregation_hint", "")
            
            fields_description += f"- {name} ({field_type}): {desc}"
            if example:
                fields_description += f" Example: {example}"
            if hint:
                fields_description += f" Hint: {hint}"
            fields_description += "\n"
        
        if "notes" in df_description:
            fields_description += "\nDataset notes:\n"
            for note in df_description["notes"]:
                fields_description += f"- {note}\n"
    
    base_prompt = (
        "User task (in natural language):\n"
        f"{user_task}\n\n"
        "DataFrame structure (column -> dtype):\n"
        f"{structure}\n\n"
        f"{fields_description}\n"
    )
    
    if previous_error and previous_code:
        retry_section = (
            "\n"
            "⚠️ IMPORTANT: The previous code attempt failed with an error. "
            "Please fix the code based on the error message below.\n\n"
            f"Previous error: {previous_error}\n\n"
            "Previous code (for reference, fix the issues):\n"
            f"```python\n{previous_code[:2000]}\n```\n\n"
            "Please generate corrected Python code that addresses the error above. "
            "Make sure to follow all constraints from the system prompt.\n"
        )
        return base_prompt + retry_section
    else:
        return (
            base_prompt
            + "Write Python code that, when executed locally, performs the requested analysis "
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


def get_relevance_check_system_prompt() -> str:
    """
    System prompt for checking if user's question is relevant to the vacancies dataset.
    """
    return (
        "You are a data analyst assistant. Your task is to determine if a user's question "
        "can be answered using the vacancies.json dataset.\n\n"
        "The dataset contains information about job vacancies with fields like: "
        "vacancy_id, published_at, position, specialization, seniority, salary_from, "
        "salary_to, stack (technologies), company_name, location, etc.\n\n"
        "Respond with ONLY one word:\n"
        "- 'YES' if the question can be answered using this dataset\n"
        "- 'NO' if the question is not related to job vacancies data or cannot be answered with this dataset\n\n"
        "Do not provide any explanation, only 'YES' or 'NO'."
    )


def _wait_before_request() -> None:
    """Добавляет задержку перед запросом к API для предотвращения rate limiting."""
    global _last_request_time
    
    with _request_lock:
        current_time = time.time()
        time_since_last = current_time - _last_request_time
        
        if time_since_last < API_REQUEST_DELAY:
            sleep_time = API_REQUEST_DELAY - time_since_last
            time.sleep(sleep_time)
        
        _last_request_time = time.time()


def _make_api_request_with_retry(url: str, headers: Dict[str, str], payload: Dict[str, Any], max_retries: int = 3) -> requests.Response:
    """
    Выполняет запрос к API с обработкой 429 ошибок и повторными попытками.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    for attempt in range(max_retries):
        _wait_before_request()
        
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if r.status_code == 429:
            # Rate limit exceeded - используем Retry-After или экспоненциальную задержку
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                try:
                    wait_time = int(retry_after)
                    # Если Retry-After слишком большой - сразу выбрасываем ошибку
                    if wait_time > MAX_RETRY_WAIT_TIME:
                        error_msg = (
                            f"Rate limit exceeded (429). API requires waiting {wait_time}s ({wait_time // 60} minutes), "
                            f"which exceeds maximum wait time of {MAX_RETRY_WAIT_TIME}s. "
                            f"Please wait and try again later, or check your Groq API quota."
                        )
                        logger.error(error_msg)
                        raise requests.exceptions.HTTPError(error_msg, response=r)
                    
                    logger.warning(f"Rate limit exceeded (429). Waiting {wait_time}s (from Retry-After header) before retry {attempt + 1}/{max_retries}...")
                except (ValueError, TypeError):
                    wait_time = MAX_RETRY_WAIT_TIME  # Фиксированная задержка
                    logger.warning(f"Rate limit exceeded (429). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
            else:
                # Если Retry-After нет, используем фиксированную задержку
                wait_time = MAX_RETRY_WAIT_TIME  # Фиксированная задержка (30 секунд)
                logger.warning(f"Rate limit exceeded (429). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
            
            if attempt < max_retries - 1:
                time.sleep(wait_time)
                continue
            else:
                # После всех попыток все еще 429 - выбрасываем понятную ошибку
                error_msg = (
                    f"Rate limit exceeded after {max_retries} attempts. "
                    f"Groq API rate limit has been reached. "
                    f"Please wait a few minutes before trying again, or check your API quota."
                )
                logger.error(error_msg)
                raise requests.exceptions.HTTPError(error_msg, response=r)
        else:
            r.raise_for_status()
            return r
    
    # Не должно сюда дойти, но на всякий случай
    r.raise_for_status()
    return r


def check_query_relevance(user_task: str) -> bool:
    """
    Проверяет, связан ли запрос пользователя с данными вакансий.
    Возвращает True, если запрос релевантен, False - если нет.
    """
    system_prompt = get_relevance_check_system_prompt()
    user_prompt = f"User question: {user_task}\n\nCan this question be answered using the vacancies dataset? Answer YES or NO only."
    
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
        "temperature": 0.1,  # Низкая температура для более детерминированного ответа
    }

    r = _make_api_request_with_retry(url, headers, payload)
    data = r.json()
    
    response = data["choices"][0]["message"]["content"].strip().upper()
    return response.startswith("YES")


def get_report_system_prompt() -> str:
    """
    System prompt for generating a human-readable analytical report.
    """
    return (
        "You are an experienced data analyst preparing a brief analytical report. "
        "Your task is to write a clear, concise, and professional summary of the analysis results "
        "in natural language, as if you were presenting findings to a stakeholder.\n\n"
        "Guidelines:\n"
        "- Write in Russian (if the user's question was in Russian) or English (if in English).\n"
        "- Be extremely concise: write ONLY 1 paragraph maximum (3-5 sentences).\n"
        "- Highlight the most important findings and insights from the metrics.\n"
        "- Reference the generated plots by their ORDER NUMBER (1st, 2nd, 3rd, etc.) when discussing visualizations.\n"
        "- NEVER mention file names or paths of plots. Use only order numbers like 'первый график', 'second chart', etc.\n"
        "- Use specific numbers from the metrics to support your conclusions.\n"
        "- Write in a professional but accessible tone.\n"
        "- Do not include technical jargon unless necessary.\n"
        "- Combine context, key findings, and conclusions in a single paragraph.\n\n"
        "Important: When referring to plots, use their order number (1st, 2nd, 3rd) instead of file names. "
        "For example: 'Как видно на первом графике...' or 'The second chart shows that...'"
    )


def _convert_to_json_serializable(obj: Any) -> Any:
    """
    Преобразует numpy/pandas типы в нативные Python типы для JSON сериализации.
    """
    import numpy as np
    import pandas as pd
    
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def get_report_user_prompt(user_task: str, analytics_result: Dict[str, Any]) -> str:
    """
    Build the user prompt for report generation that includes:
    - the original user's question/task,
    - the computed analytics results (metrics and plots).
    """
    import json
    
    # Преобразуем метрики в JSON-сериализуемый формат
    metrics = analytics_result.get("metrics", {})
    metrics_serializable = _convert_to_json_serializable(metrics)
    
    # Форматируем метрики для читаемости
    metrics_str = json.dumps(metrics_serializable, ensure_ascii=False, indent=2)
    
    # Форматируем информацию о графиках с номерами по порядку
    plots_info = analytics_result.get("plots", [])
    plots_str = ""
    if plots_info:
        plots_str = "\nGenerated plots (refer to them by order number: 1st, 2nd, 3rd, etc.):\n"
        for idx, plot in enumerate(plots_info, 1):
            plot_name = plot.get("name", "unnamed")
            plots_str += f"{idx}. {plot_name}\n"
    else:
        plots_str = "\nNo plots were generated for this analysis.\n"
    
    return (
        f"Original user question/task:\n{user_task}\n\n"
        f"Analytics results:\n"
        f"Metrics:\n{metrics_str}\n"
        f"{plots_str}\n"
        "Please write a very brief analytical report (1 paragraph, 3-5 sentences) that answers the user's question, "
        "highlights the most important findings from the metrics, and references the plots by their ORDER NUMBER "
        "(1st, 2nd, 3rd, etc.) when discussing visualizations. "
        "DO NOT mention file names or paths. Use only order numbers like 'первый график', 'second chart', etc. "
        "Write as if you are a data analyst presenting findings to a stakeholder. Be concise and to the point."
    )


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

    r = _make_api_request_with_retry(url, headers, payload)
    data = r.json()

    raw_content = data["choices"][0]["message"]["content"]
    return _extract_code_block(raw_content)


def llm_request_report(user_prompt: str, system_prompt: str) -> str:
    """
    Make a request to the Groq LLM for generating a text report (not code).
    Returns the raw text response without code extraction.
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
        "temperature": 0.7,  # Немного выше для более естественного текста
    }

    r = _make_api_request_with_retry(url, headers, payload)
    data = r.json()

    return data["choices"][0]["message"]["content"].strip()

