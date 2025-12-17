import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from config import API_KEY, DEBUG
from data_utils import load_description
from llm_request import (
    get_system_prompt,
    get_user_prompt,
    llm_request,
    get_report_system_prompt,
    get_report_user_prompt,
    llm_request_report,
    check_query_relevance,
)
from code_executor import execute_generated_code, CodeValidationError


GENERATED_CODE_DIR = Path("generated_code")


def _save_generated_code(code: str, user_task: str) -> Path:
    """Сохраняет сгенерированный код в файл с уникальным именем."""
    # Создаем директорию, если её нет
    GENERATED_CODE_DIR.mkdir(exist_ok=True)
    
    # Генерируем уникальное имя файла на основе задачи и времени
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_hash = hashlib.md5(user_task.encode()).hexdigest()[:8]
    filename = f"code_{timestamp}_{task_hash}.py"
    filepath = GENERATED_CODE_DIR / filename
    
    # Сохраняем код
    with filepath.open("w", encoding="utf-8") as f:
        f.write(f"# Generated code for task: {user_task}\n")
        f.write(f"# Generated at: {datetime.now().isoformat()}\n\n")
        f.write(code)
    
    print(f"Generated code saved to: {filepath}")
    return filepath


def _cleanup_plots(plots: List[Dict[str, Any]]) -> None:
    """
    Удаляет файлы графиков после отправки пользователю.
    Безопасно для параллельных запросов - удаляет только указанные файлы.
    """
    for plot in plots:
        path = plot.get("path")
        if path:
            try:
                plot_path = Path(path)
                if plot_path.exists() and plot_path.is_file():
                    plot_path.unlink()
                    print(f"Deleted plot file: {path}")
            except Exception as e:  # noqa: BLE001
                # Логируем ошибку, но не прерываем выполнение
                print(f"Warning: Failed to delete plot file {path}: {e}")


def main() -> None:
    print(f"DEBUG={DEBUG}")
    print(f"API_KEY={'set' if API_KEY else 'not set'}")

    # TODO: здесь нужно получать реальный запрос пользователя (например, из Телеграма)
    user_task = "посчитай средние зарплаты по годам"

    if not user_task:
        print("Please provide a user task.")
        return

    # Проверяем релевантность запроса
    print("Checking query relevance...")
    if not check_query_relevance(user_task):
        print(
            "❌ Ваш вопрос не связан с данными о вакансиях.\n"
            "Я могу отвечать только на вопросы, которые можно решить с помощью анализа файла vacancies.json, "
            "содержащего информацию о вакансиях (зарплаты, технологии, компании, локации и т.д.)."
        )
        return

    print("✅ Query is relevant to vacancies data. Proceeding with analysis...")

    # Загружаем описание данных из description.json
    structure_info, description_info = load_description()

    # Формируем system prompt (он не меняется между попытками)
    system_prompt = get_system_prompt()

    # Механизм повторных попыток
    max_attempts = 3
    previous_error = None
    previous_code = None

    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Attempt {attempt}/{max_attempts} ---")
        
        # Формируем user prompt (с ошибкой, если это повторная попытка)
        user_prompt = get_user_prompt(
            user_task, 
            structure_info, 
            description_info,
            previous_error=previous_error,
            previous_code=previous_code,
        )

        # Запрашиваем код у LLM
        code = llm_request(user_prompt=user_prompt, system_prompt=system_prompt)

        try:
            analytics_result = execute_generated_code(code)
            # Успех!
            print("\n✅ Code executed successfully!")
            # Сохраняем успешно выполненный код
            _save_generated_code(code, user_task)
            print("ANALYTICS_RESULT:")
            print(analytics_result)
            
            # Генерируем краткий отчет на основе результатов
            print("\n--- Generating analytical report ---")
            report_system_prompt = get_report_system_prompt()
            report_user_prompt = get_report_user_prompt(user_task, analytics_result)
            report = llm_request_report(report_user_prompt, report_system_prompt)
            
            print("\n" + "=" * 80)
            print("ANALYTICAL REPORT:")
            print("=" * 80)
            print(report)
            print("=" * 80)
            
            # Удаляем графики после успешного выполнения
            plots = analytics_result.get("plots") or []
            if plots:
                _cleanup_plots(plots)
                print(f"\nCleaned up {len(plots)} plot file(s)")
            
            return
        except CodeValidationError as e:
            error_msg = str(e)
            print(f"❌ Generated code was rejected as unsafe: {error_msg}")
            print(f"Code snippet: {code[:500]}...")
            
            if attempt < max_attempts:
                previous_error = error_msg
                previous_code = code
                print(f"Retrying with error feedback...")
            else:
                print(f"\n❌ Failed after {max_attempts} attempts. Giving up.")
                return
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"❌ Code execution failed: {error_msg}")
            print(f"Code snippet: {code[:500]}...")
            
            if attempt < max_attempts:
                previous_error = error_msg
                previous_code = code
                print(f"Retrying with error feedback...")
            else:
                print(f"\n❌ Failed after {max_attempts} attempts. Giving up.")
                return


if __name__ == "__main__":
    main()


