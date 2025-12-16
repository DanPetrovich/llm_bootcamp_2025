from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message

from config import API_KEY, TELEGRAM_BOT_TOKEN, DEBUG
from data_utils import load_vacancies
from llm_request import (
    get_system_prompt,
    get_user_prompt,
    llm_request,
    get_report_system_prompt,
    get_report_user_prompt,
    llm_request_report,
)
from code_executor import execute_generated_code, CodeValidationError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


def _require_token() -> str:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env file")
    return TELEGRAM_BOT_TOKEN


def _run_analysis(user_task: str) -> Dict[str, Any]:
    """Запускает анализ данных на основе запроса пользователя."""
    df, structure, description = load_vacancies()
    structure_info = {col: str(dtype) for col, dtype in structure.items()}
    description_info = description

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
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to send photo {path}: {e}")
                    await message.answer(f"Не удалось отправить график '{name}': {e}")

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
