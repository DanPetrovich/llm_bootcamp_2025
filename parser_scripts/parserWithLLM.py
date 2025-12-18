import argparse
import json
import os
import re
import threading
from html import unescape
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ----------------------------
# HTML -> text
# ----------------------------

class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []

    def handle_data(self, d: str) -> None:
        self._chunks.append(d)

    def handle_entityref(self, name: str) -> None:
        self._chunks.append(unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        self._chunks.append(unescape(f"&#{name};"))

    def get_text(self) -> str:
        return " ".join("".join(self._chunks).split())


def html_to_text(s: Any) -> Optional[str]:
    if not s or not isinstance(s, str):
        return None
    s = unescape(s)
    stripper = _HTMLStripper()
    stripper.feed(s)
    t = stripper.get_text()
    return t if t else None


def norm_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        return v if v else None
    return str(v)


def clamp_text(s: Optional[str], max_chars: int) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    if len(s) <= max_chars:
        return s
    # аккуратно отрезаем по границе предложения/строки, если получится
    cut = s[:max_chars]
    m = re.search(r"(.{0," + str(max_chars) + r"}[.!?])\s", cut[::-1])
    return cut if not m else cut  # упрощённо: просто режем


# ----------------------------
# JSON extraction from model output
# ----------------------------

def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Достаём первый JSON-object из произвольного текста.
    Работает по балансу фигурных скобок.
    """
    if not text:
        return None

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


def coerce_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    return None


def ensure_list_of_str(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        out = []
        for x in v:
            if x is None:
                continue
            if isinstance(x, str):
                xs = x.strip()
                if xs:
                    out.append(xs)
            else:
                out.append(str(x))
        return out
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    return [str(v)]


def normalize_extracted(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Нормализуем выход модели в стабильный для агрегаций формат:
    - списки -> list[str]
    - булевы -> bool
    - missing -> null/[] по смыслу
    """
    scores = obj.get("scores") if isinstance(obj.get("scores"), dict) else {}

    relocation = obj.get("relocation_support")
    if not isinstance(relocation, dict):
        relocation = {"available": None, "details": []}

    out = {
        "responsibilities": ensure_list_of_str(obj.get("responsibilities")),
        "must_have_skills": ensure_list_of_str(obj.get("must_have_skills")),
        "nice_to_have_skills": ensure_list_of_str(obj.get("nice_to_have_skills")),
        "soft_skills": ensure_list_of_str(obj.get("soft_skills")),

        "english_required": coerce_bool(obj.get("english_required")),
        "english_level_text": norm_str(obj.get("english_level_text")),

        "remote_policy_text": norm_str(obj.get("remote_policy_text")),
        "schedule": ensure_list_of_str(obj.get("schedule")),

        "relocation_support": {
            "available": coerce_bool(relocation.get("available")),
            "details": ensure_list_of_str(relocation.get("details")),
        },

        "comp_components": ensure_list_of_str(obj.get("comp_components")),
        "benefits": ensure_list_of_str(obj.get("benefits")),

        "hiring_steps_count": obj.get("hiring_steps_count") if isinstance(obj.get("hiring_steps_count"), int) else None,
        "test_task_present": coerce_bool(obj.get("test_task_present")),

        "company_domain": norm_str(obj.get("company_domain")),
        "product_type": ensure_list_of_str(obj.get("product_type")),
        "revenue_model": ensure_list_of_str(obj.get("revenue_model")),
        "engineering_practices": ensure_list_of_str(obj.get("engineering_practices")),

        "scores": {
            "clarity_score": scores.get("clarity_score") if isinstance(scores.get("clarity_score"), (int, float)) else None,
            "remote_clarity_score": scores.get("remote_clarity_score") if isinstance(scores.get("remote_clarity_score"), (int, float)) else None,
            "process_transparency_score": scores.get("process_transparency_score") if isinstance(scores.get("process_transparency_score"), (int, float)) else None,
        }
    }
    return out


# ----------------------------
# Ollama client
# ----------------------------

def ollama_generate(
    model: str,
    prompt: str,
    host: str,
    temperature: float = 0.0,
    num_ctx: int = 4096,
    timeout_s: int = 300,
) -> str:
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        }
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


# ----------------------------
# Prompt
# ----------------------------

ALLOWED_HINTS = """
Требования к формату:
- Ответ ТОЛЬКО в виде валидного JSON-объекта (без markdown, без ```).
- Если информации нет: используй null для строк/чисел/булевых, и [] для списков.
- Списки должны быть массивами строк.
- english_required: true/false/null.
- test_task_present: true/false/null.
- hiring_steps_count: integer или null.
- scores: числа 0..5 или null.
"""

SCHEMA_KEYS = """
Ключи JSON (строго эти; лишних ключей не добавляй):
responsibilities (array<string>)
must_have_skills (array<string>)
nice_to_have_skills (array<string>)
soft_skills (array<string>)
english_required (boolean|null)
english_level_text (string|null)
remote_policy_text (string|null)
schedule (array<string>)
relocation_support (object {available:boolean|null, details:array<string>})
comp_components (array<string>)
benefits (array<string>)
hiring_steps_count (integer|null)
test_task_present (boolean|null)
company_domain (string|null)
product_type (array<string>)
revenue_model (array<string>)
engineering_practices (array<string>)
scores (object {clarity_score:number|null, remote_clarity_score:number|null, process_transparency_score:number|null})
"""

def build_prompt(offer_text: Optional[str], company_text: Optional[str]) -> str:
    offer_text = offer_text or ""
    company_text = company_text or ""

    return f"""Ты — движок извлечения структурированных признаков из текста вакансии и описания компании.
Извлеки данные для аналитики по заданной схеме.

{ALLOWED_HINTS}
{SCHEMA_KEYS}

Текст оффера (вакансии):
\"\"\"{offer_text}\"\"\"

Текст о компании:
\"\"\"{company_text}\"\"\"

Верни JSON.
"""


# ----------------------------
# IO + processing
# ----------------------------

def get_texts_from_record(rec: Dict[str, Any], max_offer_chars: int, max_company_chars: int) -> Tuple[Optional[str], Optional[str]]:
    d = rec.get("data") if isinstance(rec.get("data"), dict) else {}

    # оффер: приоритет offer_description, затем description, затем short_description
    offer_raw = d.get("offer_description") or d.get("description") or d.get("short_description")
    offer_text = html_to_text(offer_raw) if isinstance(offer_raw, str) else html_to_text(str(offer_raw)) if offer_raw else None
    offer_text = clamp_text(offer_text, max_offer_chars)

    # компания: company.short_description
    company = d.get("company") if isinstance(d.get("company"), dict) else {}
    company_raw = company.get("short_description") or company.get("description") or company.get("name")
    company_text = html_to_text(company_raw) if isinstance(company_raw, str) else html_to_text(str(company_raw)) if company_raw else None
    company_text = clamp_text(company_text, max_company_chars)

    return offer_text, company_text


def process_one(
    rec: Dict[str, Any],
    model: str,
    host: str,
    max_offer_chars: int,
    max_company_chars: int,
    num_ctx: int,
    timeout_s: int,
) -> Dict[str, Any]:
    vacancy_id = rec.get("id")

    offer_text, company_text = get_texts_from_record(rec, max_offer_chars, max_company_chars)
    prompt = build_prompt(offer_text, company_text)

    try:
        raw = ollama_generate(
            model=model,
            prompt=prompt,
            host=host,
            temperature=0.0,
            num_ctx=num_ctx,
            timeout_s=timeout_s,
        )
        obj = extract_first_json_object(raw)
        if not isinstance(obj, dict):
            return {
                "vacancy_id": vacancy_id,
                "llm_model": model,
                "ok": False,
                "error": "Could not parse JSON from model output",
                "raw_response_preview": (raw[:800] if isinstance(raw, str) else None),
            }

        norm = normalize_extracted(obj)
        return {
            "vacancy_id": vacancy_id,
            "llm_model": model,
            "ok": True,
            **norm,
        }

    except Exception as e:
        return {
            "vacancy_id": vacancy_id,
            "llm_model": model,
            "ok": False,
            "error": str(e),
        }


def load_done_ids_from_jsonl(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                vid = obj.get("vacancy_id")
                if vid is not None:
                    done.add(vid)
            except Exception:
                continue
    return done


def jsonl_to_json_array(jsonl_path: str, json_path: str) -> None:
    arr = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                arr.append(json.loads(line))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input vacancies JSON (list of {id, data})")
    ap.add_argument("--output", required=True, help="Output path (.jsonl recommended)")
    ap.add_argument("--out-format", choices=["jsonl", "json"], default="jsonl")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--model", default="gemma3")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=300)

    ap.add_argument("--max-offer-chars", type=int, default=5000)
    ap.add_argument("--max-company-chars", type=int, default=2000)
    ap.add_argument("--num-ctx", type=int, default=4096)

    ap.add_argument("--resume", action="store_true", help="Skip already processed vacancy_id in output")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Input JSON must be a list")

    out_jsonl = args.output if args.out_format == "jsonl" else args.output + ".jsonl.tmp"

    done_ids = load_done_ids_from_jsonl(out_jsonl) if args.resume else set()

    # фильтруем записи, если resume
    todo = [r for r in raw if r.get("id") not in done_ids]

    lock = threading.Lock()

    def write_jsonl(obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        with lock:
            with open(out_jsonl, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    iterator = todo
    if tqdm is not None:
        iterator = tqdm(total=len(todo), desc="LLM extract")  # type: ignore

    futures = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for rec in todo:
            futures.append(ex.submit(
                process_one,
                rec,
                args.model,
                args.ollama_host,
                args.max_offer_chars,
                args.max_company_chars,
                args.num_ctx,
                args.timeout,
            ))

        completed_iter = as_completed(futures)
        if tqdm is not None:
            completed_iter = tqdm(completed_iter, total=len(futures), desc="Processing")  # type: ignore

        for fut in completed_iter:
            res = fut.result()
            write_jsonl(res)

    # если нужен JSON массив
    if args.out_format == "json":
        jsonl_to_json_array(out_jsonl, args.output)
        # можно удалить tmp, если хотите:
        # os.remove(out_jsonl)

    print(f"Done. Written to: {args.output if args.out_format=='json' else out_jsonl}")
    if args.resume:
        print(f"Resume mode ON. Skipped already processed: {len(done_ids)}")


if __name__ == "__main__":
    main()
