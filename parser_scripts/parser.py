# clean_and_select.py
# Usage:
#   python clean_and_select.py --input vacancies_100.json --output vacancies_min.json

import argparse
import json
import re
from datetime import date
from html import unescape
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Helpers (same as before)
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
    text = stripper.get_text()
    return text or None


def norm_str(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        return v if v else None
    return v


def norm_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        out: List[Any] = []
        for item in v:
            item2 = norm_str(item)
            if item2 is not None:
                out.append(item2)
        return out
    x = norm_str(v)
    return [x] if x is not None else []


def safe_date_iso(s: Any) -> Optional[str]:
    s = norm_str(s)
    if not s:
        return None
    try:
        return date.fromisoformat(s).isoformat()
    except Exception:
        return None


def parse_range_str(raw: Any) -> Tuple[Optional[int], Optional[int]]:
    s = norm_str(raw)
    if not s:
        return (None, None)

    m = re.match(r"^\s*(\d+)\s*\+\s*$", s)
    if m:
        return (int(m.group(1)), None)

    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    return (None, None)


def parse_money_int(token: str) -> Optional[int]:
    token = token.replace("\u202f", " ").replace("\u00a0", " ")
    digits = re.sub(r"[^\d]", "", token)
    return int(digits) if digits else None


def extract_salary_from_text(desc: Any) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
    s = norm_str(desc)
    if not s:
        return (None, None, None, None)

    sl = s.lower()

    currency = None
    if "₽" in s or "руб" in sl:
        currency = "RUB"
    elif "$" in s:
        currency = "USD"
    elif "€" in s:
        currency = "EUR"

    period = None
    if "мес" in sl:
        period = "month"
    elif "год" in sl or "year" in sl:
        period = "year"
    elif "час" in sl or "hour" in sl:
        period = "hour"

    nums = re.findall(r"\d[\d\s\u00a0\u202f]*", s)
    ints = [parse_money_int(n) for n in nums]
    ints = [x for x in ints if x is not None]
    if not ints:
        return (None, None, currency, period)

    if re.search(r"\bот\b", sl):
        return (ints[0], None, currency, period)

    if re.search(r"\bдо\b", sl) and len(ints) == 1:
        return (None, ints[0], currency, period)

    if len(ints) >= 2:
        return (ints[0], ints[1], currency, period)

    return (ints[0], None, currency, period)


def normalize_salary(d: Dict[str, Any]) -> Dict[str, Any]:
    desc = d.get("salary_description")
    from_ = d.get("salary_display_from")
    to_ = d.get("salary_display_to")
    cur_sym = d.get("salary_currency")
    taxes = d.get("salary_taxes")
    is_total = d.get("salary_is_total")
    hidden = d.get("salary_hidden")

    from_ = from_ if isinstance(from_, (int, float)) else None
    to_ = to_ if isinstance(to_, (int, float)) else None

    currency_code = None
    if cur_sym:
        if cur_sym == "₽":
            currency_code = "RUB"
        elif cur_sym == "$":
            currency_code = "USD"
        elif cur_sym == "€":
            currency_code = "EUR"
        else:
            currency_code = str(cur_sym)

    period = None

    f2, t2, cur2, p2 = extract_salary_from_text(desc)
    if from_ is None:
        from_ = f2
    if to_ is None:
        to_ = t2
    if currency_code is None:
        currency_code = cur2
    if period is None:
        period = p2

    salary_mid = None
    if isinstance(from_, (int, float)) and isinstance(to_, (int, float)):
        salary_mid = int((from_ + to_) / 2)
    elif isinstance(from_, (int, float)):
        salary_mid = int(from_)
    elif isinstance(to_, (int, float)):
        salary_mid = int(to_)

    taxes = norm_str(taxes)
    if not taxes and desc:
        dl = str(desc).lower()
        if "на руки" in dl:
            taxes = "net"
        elif "до налог" in dl:
            taxes = "gross"

    return {
        "salary_from": from_,
        "salary_to": to_,
        "salary_mid": salary_mid,
        "salary_currency": currency_code,
        "salary_period": period,
        "salary_taxes": taxes,
        "salary_is_total_comp": bool(is_total) if is_total is not None else None,
        "salary_hidden": bool(hidden) if hidden is not None else None,
    }


def normalize_locations(d: Dict[str, Any]) -> Dict[str, Any]:
    location_codes = norm_list(d.get("locations"))

    cities: List[str] = []
    countries: List[str] = []

    top_city = norm_str(d.get("city"))
    top_country = norm_str(d.get("country"))
    if top_city:
        cities.append(top_city)
    if top_country:
        countries.append(top_country)

    display_locs_raw = d.get("display_locations") or []
    if isinstance(display_locs_raw, list):
        for item in display_locs_raw:
            if not isinstance(item, dict):
                continue
            c = norm_str(item.get("city"))
            k = norm_str(item.get("country"))
            if c:
                # иногда там "Amsterdam, The Hague"
                cities.extend([x.strip() for x in c.split(",") if x.strip()])
            if k:
                countries.extend([x.strip() for x in k.split(",") if x.strip()])

    # uniq preserving order
    def uniq(seq: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in seq:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    cities_u = uniq(cities)
    countries_u = uniq(countries)

    remote_options = norm_str(d.get("remote_options"))
    relocation_options = norm_list(d.get("relocation_options"))

    is_remote = ("remote" in location_codes) or (remote_options is not None)
    is_relocate = ("relocate" in location_codes) or bool(relocation_options)

    return {
        "primary_city": cities_u[0] if cities_u else None,
        "primary_country": countries_u[0] if countries_u else None,
        "is_remote": is_remote,
        "is_relocate": is_relocate,
    }


def normalize_company(c: Any) -> Dict[str, Any]:
    if not isinstance(c, dict):
        c = {}
    size_raw = norm_str(c.get("size"))
    size_min, size_max = parse_range_str(size_raw)
    return {
        "company_name": norm_str(c.get("name")),
        "company_industry": norm_str(c.get("industry")),
        "company_size_min": size_min,
        "company_size_max": size_max,
    }


def normalize_record_min(rec: Dict[str, Any]) -> Dict[str, Any]:
    d = rec.get("data") if isinstance(rec.get("data"), dict) else {}

    published_at = safe_date_iso(d.get("published_at"))
    published_year = int(published_at[:4]) if published_at else None
    published_month = published_at[:7] if published_at else None  # YYYY-MM

    vacancy_path = norm_str(d.get("url"))
    vacancy_url = None
    if isinstance(vacancy_path, str) and vacancy_path.startswith("/"):
        vacancy_url = "https://getmatch.ru" + vacancy_path
    else:
        vacancy_url = vacancy_path

    out: Dict[str, Any] = {
        "vacancy_id": rec.get("id"),
        "published_at": published_at,
        "published_year": published_year,
        "published_month": published_month,

        "is_active": d.get("is_active") if isinstance(d.get("is_active"), bool) else None,
        "offer_type": norm_str(d.get("offer_type")),
        "source_type": norm_str(d.get("type")),

        "position": norm_str(d.get("position")),
        "specialization": norm_str(d.get("specialization")),
        "seniority": norm_str(d.get("seniority")),
        "position_level": norm_str(d.get("position_level")),
        "required_years_experience": d.get("required_years_of_experience") if isinstance(d.get("required_years_of_experience"), int) else None,

        "stack": norm_list(d.get("stack")),
        "stack_count": len(norm_list(d.get("stack"))),

        "vacancy_url": vacancy_url,
    }

    out.update(normalize_salary(d))
    out.update(normalize_locations(d))
    out.update(normalize_company(d.get("company")))

    # добиваем пустые строки -> None (на всякий)
    for k, v in list(out.items()):
        out[k] = norm_str(v) if isinstance(v, str) else v

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Input JSON must be a list of records")

    minimal = [normalize_record_min(r) for r in raw]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(minimal, f, ensure_ascii=False, indent=args.indent)

    print(f"OK: {len(minimal)} records -> {args.output}")


if __name__ == "__main__":
    main()
