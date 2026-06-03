"""
validate_clinic_data.py
========================
Валидатор ретроспективных данных клиники перед калибровкой.

Запуск:
    python validate_clinic_data.py --input clinic_intake_template.csv
    python validate_clinic_data.py --input data.csv --output validated.csv --report report.txt

Что проверяет:
  1. Наличие обязательных колонок
  2. Типы и диапазоны значений
  3. Логическую консистентность (MII ≤ OKK, PN2 ≤ MII, ...)
  4. Категориальные значения
  5. Строки без исходов (фильтруются, не блокируют)

Выход:
  validated_{input}.csv   — очищенный датасет (только валидные строки)
  validation_report.txt   — детальный отчёт по ошибкам
"""

from __future__ import annotations
import argparse
import csv
import sys
import os
from datetime import datetime
from typing import Optional

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[WARN] pandas не установлен. Установите: pip install pandas")
    sys.exit(1)


# ─── Правила валидации ────────────────────────────────────────────────────────

REQUIRED_COLS = [
    "cycle_id", "patient_id", "age", "amh", "afc", "bmi",
    "attempt_number", "sperm_source", "follicles_14mm",
    "okk", "mii", "pn2", "blasts_total", "blasts_good",
    "outcome", "outcome_known",
]

OPTIONAL_COLS = [
    "diagnosis", "protocol_type", "fsh_start_iu", "rlh_used", "rlh_dose_iu",
    "stim_days", "e2_trigger_pmol", "cleavage_d3", "emb_frozen",
    "emb_transferred", "euploid", "ohss_grade", "cycle_cancelled", "outcome_date",
]

RANGE_RULES: dict[str, tuple] = {
    "age":             (18,    55,    float),
    "amh":             (0.01,  30.0,  float),
    "afc":             (0,     70,    int),
    "bmi":             (14.0,  55.0,  float),
    "attempt_number":  (1,     15,    int),
    "follicles_14mm":  (0,     55,    int),
    "e2_trigger_pmol": (100,   90000, float),
    "fsh_start_iu":    (50,    900,   float),
    "rlh_dose_iu":     (0,     450,   float),
    "stim_days":       (3,     28,    int),
    "okk":             (0,     65,    int),
    "mii":             (0,     65,    int),
    "pn2":             (0,     65,    int),
    "cleavage_d3":     (0,     65,    int),
    "blasts_total":    (0,     45,    int),
    "blasts_good":     (0,     45,    int),
    "emb_frozen":      (0,     45,    int),
    "emb_transferred": (0,     8,     int),
    "euploid":         (0,     40,    int),
    "ohss_grade":      (0,     3,     int),
    "rlh_used":        (0,     1,     int),
    "cycle_cancelled": (0,     1,     int),
    "outcome_known":   (0,     1,     int),
}

CATEGORICAL_RULES: dict[str, list] = {
    "sperm_source":  ["partner", "donor", "surgical"],
    "diagnosis":     ["normal", "pcos", "dor", "mfi", "unexplained", "other"],
    "protocol_type": ["long_agonist", "short_agonist", "antagonist", "mini", "other"],
    "outcome":       ["no_transfer", "biochemical", "clinical",
                      "ongoing", "delivery", "miscarriage", "unknown"],
}

# Бинарный целевой признак: эти значения outcome = 1
POSITIVE_OUTCOMES = {"clinical", "ongoing", "delivery"}
# Эти значения outcome = 0
NEGATIVE_OUTCOMES = {"no_transfer", "biochemical", "miscarriage"}
# Эти outcome пропускаются при калибровке (но строка остаётся)
UNKNOWN_OUTCOMES  = {"unknown"}

# Логические правила: (меньший, <=, больший) — если оба не NaN
CONSISTENCY_RULES = [
    ("mii",         "<=", "okk"),
    ("pn2",         "<=", "mii"),
    ("blasts_total","<=", "pn2"),
    ("blasts_good", "<=", "blasts_total"),
    ("emb_frozen",  "<=", "blasts_total"),
    ("euploid",     "<=", "blasts_good"),
]


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    try:
        v = float(str(val).strip().replace(",", "."))
        return None if pd.isna(v) else v
    except (ValueError, TypeError):
        return None

def _safe_int(val) -> Optional[int]:
    f = _safe_float(val)
    return None if f is None else int(round(f))

def _normalize_cat(val: str) -> str:
    return str(val).strip().lower()


# ─── Основной класс ──────────────────────────────────────────────────────────

class ClinicDataValidator:

    def __init__(self, input_path: str):
        self.input_path = input_path
        self.errors: list[dict] = []      # {row, col, level, message}
        self.warnings: list[dict] = []
        self.df: Optional[pd.DataFrame] = None
        self.df_valid: Optional[pd.DataFrame] = None

    # ── Загрузка ──────────────────────────────────────────────────────────────
    def load(self) -> bool:
        try:
            ext = os.path.splitext(self.input_path)[1].lower()

            if ext in (".xlsx", ".xls", ".xlsm"):
                raw = pd.read_excel(self.input_path, sheet_name=0,
                                    header=None, dtype=str)
                raw = raw.fillna("")

                ALL_KNOWN = set(REQUIRED_COLS + OPTIONAL_COLS + ["cycle_id", "patient_id"])

                # ── Стратегия 1: шаблон Digital Twin ─────────────────────────
                # Строка 0 скрытая с машинными именами (cycle_id, age, amh…)
                # Строки 1-3 = группы / русские названия / подсказки
                # Строка 4 = пример; строки 5+ = данные
                row0_vals = [str(v).strip().lower() for v in raw.iloc[0].values
                             if str(v).strip()]
                if len(row0_vals) >= 5 and sum(v in ALL_KNOWN for v in row0_vals) >= 5:
                    headers = [str(v).strip().lower() for v in raw.iloc[0].values]
                    data    = raw.iloc[5:].copy()   # пропускаем группы/подсказки/пример
                    data.columns = headers
                    print("[OK] Обнаружен шаблон Digital Twin (машинные имена в строке 1)")

                else:
                    # ── Стратегия 2: произвольный Excel от клиники ────────────
                    # Ищем первую строку где ≥5 ячеек совпадают с известными именами
                    header_row = None
                    for i in range(min(8, len(raw))):
                        vals = [str(v).strip().lower() for v in raw.iloc[i].values]
                        matches = sum(v in ALL_KNOWN for v in vals)
                        if matches >= 5:
                            header_row = i
                            break

                    if header_row is not None:
                        headers = [str(v).strip().lower()
                                   for v in raw.iloc[header_row].values]
                        # Пропускаем 1–2 строки после заголовка (подсказки/пример)
                        skip = 1
                        next_row = raw.iloc[header_row + 1].values
                        next_vals = [str(v).strip().lower() for v in next_row if str(v).strip()]
                        # Если строка после заголовка не числовая — это подсказки, пропускаем
                        numeric_count = sum(
                            True for v in next_vals
                            if any(c.isdigit() for c in str(v))
                        )
                        if numeric_count < 3:
                            skip = 2   # пропустить подсказки + пример
                        data = raw.iloc[header_row + skip:].copy()
                        data.columns = headers
                        print(f"[OK] Найден заголовок в строке {header_row + 1}")
                    else:
                        # Последний fallback: первая строка как заголовок
                        headers = [str(v).strip().lower()
                                   for v in raw.iloc[0].values]
                        data    = raw.iloc[1:].copy()
                        data.columns = headers
                        print("[WARN] Заголовок не распознан автоматически. "
                              "Используется первая строка.")

                self.df = data.replace("", pd.NA).dropna(how="all").reset_index(drop=True)
                # Убираем строки где все обязательные числовые поля — не числа
                # (защита от случайно попавших строк-подсказок)
                def _is_data_row(row):
                    age_val = str(row.get("age", "")).strip()
                    return bool(age_val) and any(c.isdigit() for c in age_val)
                mask = self.df.apply(_is_data_row, axis=1)
                n_dropped = (~mask).sum()
                self.df = self.df[mask].reset_index(drop=True)
                if n_dropped:
                    print(f"[INFO] Пропущено {n_dropped} нечисловых строк (подсказки/пример)")
                print(f"[OK] Загружен Excel: {self.input_path}  ({len(self.df)} строк данных)")

            else:
                # CSV
                self.df = pd.read_csv(self.input_path, dtype=str, keep_default_na=False)
                self.df.columns = [c.strip().lower() for c in self.df.columns]
                if self.df.shape[0] > 0:
                    first_age = self.df.iloc[0].get("age", "")
                    if first_age and not any(c.isdigit() for c in str(first_age)):
                        self.df = self.df.iloc[1:].reset_index(drop=True)
                self.df = self.df.replace("", pd.NA).dropna(how="all").reset_index(drop=True)
                print(f"[OK] Загружен CSV: {len(self.df)} строк")

            return True
        except Exception as e:
            print(f"[ERR] Не удалось загрузить файл: {e}")
            import traceback; traceback.print_exc()
            return False

    # ── Проверка колонок ──────────────────────────────────────────────────────
    def _check_columns(self):
        present = set(self.df.columns)
        missing_req = [c for c in REQUIRED_COLS if c not in present]
        missing_opt = [c for c in OPTIONAL_COLS if c not in present]

        for c in missing_req:
            self.errors.append({"row": "header", "col": c,
                                 "level": "ERROR",
                                 "message": f"Обязательная колонка '{c}' отсутствует"})
        for c in missing_opt:
            self.warnings.append({"row": "header", "col": c,
                                   "level": "WARN",
                                   "message": f"Опциональная колонка '{c}' отсутствует (будет заполнена NaN)"})
            self.df[c] = pd.NA   # добавляем пустую колонку

        unknown_cols = present - set(REQUIRED_COLS) - set(OPTIONAL_COLS) - {"cycle_id", "patient_id"}
        if unknown_cols:
            self.warnings.append({"row": "header", "col": str(unknown_cols),
                                   "level": "WARN",
                                   "message": f"Неизвестные колонки (игнорируются): {unknown_cols}"})

        return len(missing_req) == 0

    # ── Проверка диапазонов ───────────────────────────────────────────────────
    def _check_ranges(self, idx: int, row: pd.Series):
        for col, (lo, hi, dtype) in RANGE_RULES.items():
            val_str = str(row.get(col, "")).strip()
            if not val_str or val_str == "nan":
                if col in REQUIRED_COLS:
                    self.errors.append({"row": idx + 2, "col": col,
                                         "level": "ERROR",
                                         "message": f"Обязательное поле '{col}' пустое"})
                continue

            val = _safe_float(val_str)
            if val is None:
                self.errors.append({"row": idx + 2, "col": col,
                                     "level": "ERROR",
                                     "message": f"Не-числовое значение: '{val_str}'"})
                continue

            if not (lo <= val <= hi):
                level = "ERROR" if col in REQUIRED_COLS else "WARN"
                self.errors.append({"row": idx + 2, "col": col,
                                     "level": level,
                                     "message": f"Значение {val} вне диапазона [{lo}, {hi}]"})

    # ── Проверка категорий ────────────────────────────────────────────────────
    def _check_categoricals(self, idx: int, row: pd.Series):
        for col, valid_vals in CATEGORICAL_RULES.items():
            val_str = str(row.get(col, "")).strip()
            if not val_str or val_str == "nan":
                if col in REQUIRED_COLS:
                    self.errors.append({"row": idx + 2, "col": col,
                                         "level": "ERROR",
                                         "message": f"Обязательное поле '{col}' пустое"})
                continue
            norm = _normalize_cat(val_str)
            if norm not in valid_vals:
                self.errors.append({"row": idx + 2, "col": col,
                                     "level": "ERROR",
                                     "message": f"Недопустимое значение '{val_str}'. "
                                                f"Допустимые: {valid_vals}"})

    # ── Логическая консистентность ────────────────────────────────────────────
    def _check_consistency(self, idx: int, row: pd.Series):
        def _get(col):
            v = _safe_float(str(row.get(col, "")))
            return v

        for col_a, op, col_b in CONSISTENCY_RULES:
            a = _get(col_a)
            b = _get(col_b)
            if a is None or b is None:
                continue
            if op == "<=" and not (a <= b + 0.5):   # +0.5 допуск на округление
                self.errors.append({"row": idx + 2, "col": f"{col_a},{col_b}",
                                     "level": "ERROR",
                                     "message": f"Нарушена логика: {col_a}={int(a)} > {col_b}={int(b)}"})

    # ── Флаг outcome для калибровки ───────────────────────────────────────────
    def _add_calibration_flag(self):
        """Добавляет колонку outcome_binary: 1/0/NaN."""
        def _map(row):
            known = str(row.get("outcome_known", "0")).strip()
            if known not in ("1", "1.0"):
                return pd.NA
            outcome = _normalize_cat(str(row.get("outcome", "")))
            if outcome in POSITIVE_OUTCOMES:
                return 1
            elif outcome in NEGATIVE_OUTCOMES:
                return 0
            else:
                return pd.NA  # unknown — не используем

        self.df["outcome_binary"] = self.df.apply(_map, axis=1)
        n_calibration = self.df["outcome_binary"].notna().sum()
        print(f"[OK] Строк с известным исходом (для калибровки): {n_calibration}")
        if n_calibration < 50:
            self.warnings.append({"row": "summary", "col": "outcome",
                                   "level": "WARN",
                                   "message": f"Только {n_calibration} строк с исходом. "
                                              f"Для надёжной калибровки нужно ≥100, "
                                              f"для GBDT meta-learner ≥200."})

    # ── Основной прогон ───────────────────────────────────────────────────────
    def validate(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("Сначала вызовите load()")

        if not self._check_columns():
            print("[ERR] Критические ошибки в структуре файла. Исправьте колонки.")
            return pd.DataFrame()

        row_errors: set[int] = set()

        for idx, row in self.df.iterrows():
            before = len(self.errors)
            self._check_ranges(idx, row)
            self._check_categoricals(idx, row)
            self._check_consistency(idx, row)
            if len(self.errors) > before:
                row_errors.add(idx)

        self._add_calibration_flag()

        # Фильтрация: исключаем строки с ERROR-уровнем
        self.df_valid = self.df.drop(index=list(row_errors)).reset_index(drop=True)

        n_total   = len(self.df)
        n_invalid = len(row_errors)
        n_valid   = len(self.df_valid)

        print(f"\n{'='*55}")
        print(f"ИТОГ ВАЛИДАЦИИ")
        print(f"{'='*55}")
        print(f"  Всего строк   : {n_total}")
        print(f"  С ошибками    : {n_invalid}  (исключены)")
        print(f"  Валидных      : {n_valid}")
        print(f"  Ошибок        : {len([e for e in self.errors if e['level']=='ERROR'])}")
        print(f"  Предупреждений: {len(self.warnings)}")
        print(f"{'='*55}\n")

        return self.df_valid

    # ── Сохранение результатов ────────────────────────────────────────────────
    def save(self, out_data: str = None, out_report: str = None):
        base = os.path.splitext(os.path.basename(self.input_path))[0]
        out_data   = out_data   or f"validated_{base}.csv"
        out_report = out_report or f"validation_report_{base}.txt"

        if self.df_valid is not None and len(self.df_valid) > 0:
            self.df_valid.to_csv(out_data, index=False, encoding="utf-8-sig")
            print(f"[OK] Валидные данные: {out_data}  ({len(self.df_valid)} строк)")

        with open(out_report, "w", encoding="utf-8") as f:
            f.write(f"ОТЧЁТ ВАЛИДАЦИИ — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Входной файл: {self.input_path}\n")
            f.write("=" * 70 + "\n\n")

            errors_only = [e for e in self.errors if e["level"] == "ERROR"]
            if errors_only:
                f.write(f"ОШИБКИ ({len(errors_only)}):\n")
                f.write("-" * 50 + "\n")
                for e in errors_only:
                    f.write(f"  Строка {e['row']:>5} | {e['col']:<25} | {e['message']}\n")
                f.write("\n")

            if self.warnings:
                f.write(f"ПРЕДУПРЕЖДЕНИЯ ({len(self.warnings)}):\n")
                f.write("-" * 50 + "\n")
                for w in self.warnings:
                    f.write(f"  {str(w['row']):>5} | {w['col']:<25} | {w['message']}\n")
                f.write("\n")

            if self.df_valid is not None:
                f.write("СТАТИСТИКА ПО ВАЛИДНОМУ ДАТАСЕТУ:\n")
                f.write("-" * 50 + "\n")
                for col in ["age", "amh", "afc", "bmi", "okk", "mii", "pn2",
                            "blasts_total", "blasts_good"]:
                    if col in self.df_valid.columns:
                        vals = pd.to_numeric(self.df_valid[col], errors="coerce").dropna()
                        if len(vals):
                            f.write(f"  {col:<20} n={len(vals):>4}  "
                                    f"mean={vals.mean():.2f}  "
                                    f"sd={vals.std():.2f}  "
                                    f"[{vals.min():.1f}–{vals.max():.1f}]\n")

                if "outcome_binary" in self.df_valid.columns:
                    n_pos = (self.df_valid["outcome_binary"] == 1).sum()
                    n_neg = (self.df_valid["outcome_binary"] == 0).sum()
                    n_unk = self.df_valid["outcome_binary"].isna().sum()
                    f.write(f"\n  Исходы для калибровки:\n")
                    f.write(f"    Положительных (clinical+): {n_pos}\n")
                    f.write(f"    Отрицательных             : {n_neg}\n")
                    f.write(f"    Неизвестных (не используются): {n_unk}\n")
                    if n_pos + n_neg > 0:
                        rate = n_pos / (n_pos + n_neg)
                        f.write(f"    Клиническая беременность  : {rate*100:.1f}%\n")

        print(f"[OK] Отчёт:          {out_report}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Валидатор данных клиники для IVF Digital Twin"
    )
    parser.add_argument("--input",  required=True, help="Входной CSV (выгрузка МИС)")
    parser.add_argument("--output", default=None,  help="Выходной CSV (валидные строки)")
    parser.add_argument("--report", default=None,  help="Текстовый отчёт об ошибках")
    args = parser.parse_args()

    validator = ClinicDataValidator(args.input)
    if not validator.load():
        sys.exit(1)

    df_valid = validator.validate()
    validator.save(args.output, args.report)

    if len(df_valid) == 0:
        print("[ERR] Нет валидных строк. Проверьте отчёт.")
        sys.exit(1)

    print("\nСледующий шаг:")
    print(f"  python calibrate_for_clinic.py --data {args.output or 'validated_*.csv'} --clinic НазваниеКлиники")


if __name__ == "__main__":
    main()
