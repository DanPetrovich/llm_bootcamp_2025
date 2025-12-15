from __future__ import annotations

import ast
from typing import Any, Dict


class CodeValidationError(Exception):
    """Raised when generated code is considered unsafe."""


ALLOWED_MODULES = {
    "pandas",
    "matplotlib",
    "matplotlib.pyplot",
    "seaborn",
    "numpy",
    "os",
    "pathlib",
    "data_utils",
    "datetime",
}

FORBIDDEN_CALLS = {
    "exec",
    "eval",
    "compile",
    "open",
    "system",
    "popen",
}

FORBIDDEN_MODULE_PREFIXES = (
    "subprocess",
    "sys",
    "shlex",
    "socket",
    "requests",
    "http",
    "urllib",
)


def _validate_code_safety(code: str) -> None:
    """
    Best-effort валидация Python-кода перед exec.

    - Разрешаем только ограниченный набор модулей для import.
    - Запрещаем некоторые опасные вызовы (exec, eval, open, subprocess.* и т.п.).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeValidationError(f"Generated code has syntax error: {e}") from e

    for node in ast.walk(tree):
        # Ограничиваем импортируемые модули
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                # Разрешаем только ALLOWED_MODULES и явно блокируем модули из FORBIDDEN_MODULE_PREFIXES
                if any(name.startswith(prefix) for prefix in FORBIDDEN_MODULE_PREFIXES):
                    raise CodeValidationError(f"Import of module '{name}' is not allowed")
                if name not in ALLOWED_MODULES:
                    raise CodeValidationError(f"Import of module '{name}' is not allowed")

        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module not in ALLOWED_MODULES and not any(
                module.startswith(prefix) for prefix in FORBIDDEN_MODULE_PREFIXES
            ):
                # Разрешаем from pandas / matplotlib / seaborn / numpy / os / pathlib
                if module.split(".")[0] not in {m.split(".")[0] for m in ALLOWED_MODULES}:
                    raise CodeValidationError(f"Import from module '{module}' is not allowed")

        # Запрещаем прямые вызовы опасных функций
        if isinstance(node, ast.Call):
            # Вызовы по имени: open(...), eval(...), exec(...)
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALLS:
                    raise CodeValidationError(f"Call to '{node.func.id}' is not allowed")

            # Вызовы через атрибут: os.system, subprocess.run и т.п.
            if isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr
                if attr_name in FORBIDDEN_CALLS:
                    raise CodeValidationError(f"Call to attribute '{attr_name}' is not allowed")


def execute_generated_code(code: str) -> Dict[str, Any]:
    """
    Валидирует и выполняет сгенерированный ЛЛМ код.

    Возвращает словарь ANALYTICS_RESULT из выполненного скрипта.
    """
    _validate_code_safety(code)

    # Выполняем код в отдельном namespace
    exec_namespace: Dict[str, Any] = {}
    exec(code, exec_namespace, exec_namespace)

    if "ANALYTICS_RESULT" not in exec_namespace:
        raise CodeValidationError("Generated code did not define 'ANALYTICS_RESULT'")

    result = exec_namespace["ANALYTICS_RESULT"]
    if not isinstance(result, dict):
        raise CodeValidationError("'ANALYTICS_RESULT' must be a dict")

    return result


