from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message

from config import API_KEY, TELEGRAM_BOT_TOKEN, DEBUG
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
GENERATED_CODE_DIR = Path("generated_code")


def _require_token() -> str:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env file")
    return TELEGRAM_BOT_TOKEN


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
    
    logger.info(f"Generated code saved to: {filepath}")
    return filepath


def _run_analysis(user_task: str) -> Dict[str, Any]:
    """Запускает анализ данных на основе запроса пользователя."""
    # Проверяем релевантность запроса (синхронный вызов)
    is_relevant = check_query_relevance(user_task)
    
    if not is_relevant:
        raise ValueError(
            "Ваш вопрос не связан с данными о вакансиях. "
            "Я могу отвечать только на вопросы, которые можно решить с помощью анализа файла vacancies.json, "
            "содержащего информацию о вакансиях (зарплаты, технологии, компании, локации и т.д.)."
        )
    
    # Загружаем описание данных из description.json
    structure_info, description_info = load_description()

    system_prompt = get_system_prompt()

    previous_error = None
    previous_code = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        user_prompt = get_user_prompt(
            user_task,
            structure_info,
            description_info,
            previous_error=previous_error,
            previous_code=previous_code,
        )

        code = llm_request(user_prompt=user_prompt, system_prompt=system_prompt)

        try:
            analytics_result = execute_generated_code(code)
            # Сохраняем успешно выполненный код
            _save_generated_code(code, user_task)
            return analytics_result
        except CodeValidationError as e:
            previous_error = str(e)
            previous_code = code
            if attempt == MAX_ATTEMPTS:
                raise
        except Exception as e:  # noqa: BLE001
            previous_error = f"{type(e).__name__}: {e}"
            previous_code = code
            if attempt == MAX_ATTEMPTS:
                raise

    raise RuntimeError("Failed to generate analytics after retries")


def _build_report(user_task: str, analytics_result: Dict[str, Any]) -> str:
    """Генерирует краткий отчет на основе результатов аналитики."""
    report_system_prompt = get_report_system_prompt()
    report_user_prompt = get_report_user_prompt(user_task, analytics_result)
    return llm_request_report(report_user_prompt, report_system_prompt)


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
                    logger.debug(f"Deleted plot file: {path}")
            except Exception as e:  # noqa: BLE001
                # Логируем ошибку, но не прерываем выполнение
                logger.warning(f"Failed to delete plot file {path}: {e}")


# Инициализация бота и диспетчера
bot = Bot(token=_require_token())
dp = Dispatcher()


@dp.message(Command("start"))
async def cmd_start(message: Message) -> None:
    """Обработчик команды /start."""
    await message.answer(
        "Привет! Я бот для аналитики вакансий.\n"
        "Отправь мне запрос на анализ данных, и я выполню его и пришлю результаты с графиками.\n\n"
        "Пример: 'Посчитай средние зарплаты по seniority'"
    )


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """Обработчик команды /help."""
    await message.answer(
        "Я могу выполнить аналитику по данным вакансий.\n\n"
        "Просто напиши свой запрос на естественном языке, например:\n"
        "- 'Посчитай средние зарплаты по seniority'\n"
        "- 'Построй график распределения зарплат'\n"
        "- 'Сравни зарплаты в Москве и Санкт-Петербурге'\n\n"
        "Я сгенерирую код, выполню анализ и пришлю результаты с графиками."
    )


@dp.message()
async def handle_message(message: Message) -> None:
    """Обработчик всех текстовых сообщений."""
    user_text = message.text.strip()
    
    if not user_text:
        await message.answer("Пожалуйста, отправь текстовый запрос.")
        return

    # Отправляем сообщение о начале работы
    status_msg = await message.answer("Принял запрос, считаю аналитику...")
    print(f"message: {user_text}")

    try:
        # Запускаем анализ (синхронная функция в отдельном потоке)
        loop = asyncio.get_event_loop()
        analytics_result = await loop.run_in_executor(None, _run_analysis, user_text)
        
        # Генерируем отчет
        report = await loop.run_in_executor(None, _build_report, user_text, analytics_result)

        # Удаляем статусное сообщение
        await status_msg.delete()

        # Отправляем текстовый отчет
        await message.answer(report)

        # Отправляем графики, если есть
        plots = analytics_result.get("plots") or []
        plots_sent = []  # Отслеживаем успешно отправленные графики для последующего удаления
        
        for plot in plots:
            path = plot.get("path")
            name = plot.get("name", "plot")
            if path and Path(path).exists():
                try:
                    with open(path, "rb") as photo_file:
                        await message.answer_photo(
                            types.BufferedInputFile(photo_file.read(), filename=name),
                            caption=name
                        )
                    plots_sent.append(plot)  # Добавляем в список успешно отправленных
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to send photo {path}: {e}")
                    await message.answer(f"Не удалось отправить график '{name}': {e}")
        
        # Удаляем графики после успешной отправки всех сообщений
        if plots_sent:
            _cleanup_plots(plots_sent)
            logger.info(f"Cleaned up {len(plots_sent)} plot file(s) for user {message.from_user.id}")

    except ValueError as e:
        # Ошибка релевантности запроса
        await status_msg.delete()
        await message.answer(str(e))
    except CodeValidationError as e:
        await status_msg.delete()
        await message.answer(f"❌ Код не прошел проверку безопасности: {e}")
    except Exception as e:  # noqa: BLE001
        await status_msg.delete()
        logger.exception("Error during analysis")
        await message.answer(f"❌ Не удалось выполнить анализ: {e}")


async def main() -> None:
    """Главная функция для запуска бота."""
    print(f"DEBUG={DEBUG}")
    print(f"API_KEY={'set' if API_KEY else 'not set'}")
    print("Telegram bot started. Press Ctrl+C to stop.")
    
    # Удаляем вебхук, если он был установлен
    await bot.delete_webhook(drop_pending_updates=True)
    
    # Запускаем polling
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped by user.")
