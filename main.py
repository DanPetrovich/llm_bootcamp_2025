from config import API_KEY, DEBUG
from data_utils import load_vacancies
from llm_request import get_system_prompt, get_user_prompt, llm_request
from code_executor import execute_generated_code, CodeValidationError


def main() -> None:
    print(f"DEBUG={DEBUG}")
    print(f"API_KEY={'set' if API_KEY else 'not set'}")

    # TODO: здесь нужно получать реальный запрос пользователя (например, из Телеграма)
    user_task = "Посчитай квантили зарплаты за последние 3 года"

    # Загружаем данные
    df, structure, description = load_vacancies()

    # Готовим структурированное описание для промпта
    structure_info = {col: str(dtype) for col, dtype in structure.items()}
    description_info = description.to_dict()

    # Формируем промпты для LLM
    system_prompt = get_system_prompt()
    user_prompt = get_user_prompt(user_task, structure_info, description_info)

    # Запрашиваем код у LLM
    code = llm_request(user_prompt=user_prompt, system_prompt=system_prompt)

    try:
        analytics_result = execute_generated_code(code)
    except CodeValidationError as e:
        print(f"Generated code was rejected as unsafe: {e}")
        print(f"Code: {code[:500]}")
        return

    print("ANALYTICS_RESULT:")
    print(analytics_result)


if __name__ == "__main__":
    main()


