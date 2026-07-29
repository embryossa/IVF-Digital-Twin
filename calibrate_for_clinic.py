# Copyright 2025-2026 Sergei Sergeev
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Commercial use requires a separate license: see COMMERCIAL-LICENSE.md
"""
calibrate_for_clinic.py
========================
Адаптация IVF Digital Twin под конкретную клинику.

Запуск (из корня проекта, где лежат модели):
    python calibrate_for_clinic.py --data validated_data.csv --clinic "ClinicName"

Что делает:
  1. Прогоняет каждый цикл через модели (L1 prior, KAT L3, GAT L6, CSDI L5)
  2. Вычисляет ECE до калибровки для каждой модели
  3. Подбирает T_i (температурное масштабирование) per-model через NLL
  4. Вычисляет ECE после калибровки — показывает улучшение
  5. Строит OOD baseline (Mahalanobis) по клиническому и лабораторному подпространствам
  6. Если ≥200 исходов — обучает GBDT meta-learner для динамического τ(X)
  7. Сохраняет clinic_adaptation_{clinic}_{date}.json

Требования:
  pip install pandas numpy scipy scikit-learn tqdm
  (для GBDT): pip install xgboost  (или используется sklearn GBT)
  (для инференса): torch, torch-geometric, joblib — как в основном проекте
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")

# ─── Путь к проекту ───────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()
for _candidate in [_HERE, _HERE / "src"]:
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))


# ─── Прогресс-бар (опциональный) ─────────────────────────────────────────────
try:
    from tqdm import tqdm
    def _iter(iterable, desc=""):
        return tqdm(iterable, desc=desc, ncols=80)
except ImportError:
    def _iter(iterable, desc=""):
        print(f"  {desc}...")
        return iterable


# ══════════════════════════════════════════════════════════════════════════════
#  УТИЛИТЫ КАЛИБРОВКИ
# ══════════════════════════════════════════════════════════════════════════════

_EPS = 1e-7

def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, _EPS, 1.0 - _EPS)

def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1.0 - p))

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def temperature_scale(p_raw: np.ndarray, T: float) -> np.ndarray:
    """Применить температурное масштабирование: p_cal = σ(logit(p) / T)."""
    return _sigmoid(_logit(p_raw) / max(T, 1e-3))

def nll_loss(T: float, p_raw: np.ndarray, y_true: np.ndarray) -> float:
    """NLL для оптимизации T."""
    p_cal = _clip(temperature_scale(p_raw, T))
    return -float(np.mean(y_true * np.log(p_cal) + (1 - y_true) * np.log(1 - p_cal)))

def fit_temperature(p_raw: np.ndarray, y_true: np.ndarray) -> float:
    """Найти оптимальное T ∈ [0.2, 8.0] методом bounded minimization."""
    if len(p_raw) < 10:
        return 1.0
    result = minimize_scalar(
        lambda T: nll_loss(T, p_raw, y_true),
        bounds=(0.2, 8.0),
        method="bounded",
    )
    return float(result.x)

def compute_ece(p_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (равномерные бины по уверенности)."""
    p_pred = np.asarray(p_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n   = len(y_true)
    for i in range(n_bins):
        mask = (p_pred >= bins[i]) & (p_pred < bins[i + 1])
        if mask.sum() == 0:
            continue
        conf = float(np.mean(p_pred[mask]))
        acc  = float(np.mean(y_true[mask]))
        ece += (mask.sum() / n) * abs(conf - acc)
    return round(ece, 5)

def compute_brier(p_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((p_pred - y_true) ** 2))


# ══════════════════════════════════════════════════════════════════════════════
#  OOD BASELINE (Mahalanobis)
# ══════════════════════════════════════════════════════════════════════════════

def fit_mahalanobis_baseline(X: np.ndarray, reg: float = 1e-3) -> dict:
    """Подобрать mean и cov_inv для одного подпространства."""
    X    = np.asarray(X, dtype=float)
    good = ~np.isnan(X).any(axis=1)
    X    = X[good]
    if len(X) < 5:
        return {"mean": None, "cov_inv": None, "n": 0}
    mu   = X.mean(axis=0)
    cov  = np.cov(X, rowvar=False)
    cov  = np.atleast_2d(cov) + reg * np.eye(X.shape[1])
    cov_inv = np.linalg.pinv(cov)
    return {
        "mean":    mu.tolist(),
        "cov_inv": cov_inv.tolist(),
        "n":       int(len(X)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GBDT META-LEARNER для динамического τ(X)
# ══════════════════════════════════════════════════════════════════════════════

def fit_gbdt_tau_learner(X_feat: np.ndarray,
                          p_kat: np.ndarray,
                          p_gat: np.ndarray,
                          y_true: np.ndarray) -> Optional[object]:
    """
    Обучить GBDT предсказывать преимущество KAT над GAT.
    target_i = BS(p_gat_i, y_i) - BS(p_kat_i, y_i)
      > 0 → KAT лучше для этого пациента → τ_kat нужно повысить
      < 0 → GAT лучше → снизить τ_kat

    Возвращает обученную модель или None.
    """
    if len(y_true) < 200:
        print(f"[SKIP] GBDT meta-learner: только {len(y_true)} исходов. "
              f"Нужно ≥200. Будет использован фиксированный τ_KAT=2.4.")
        return None

    bs_kat = (y_true - p_kat) ** 2
    bs_gat = (y_true - p_gat) ** 2
    target = bs_gat - bs_kat     # > 0 → KAT лучше

    # Удаляем NaN
    valid = ~(np.isnan(X_feat).any(axis=1) |
              np.isnan(target) | np.isnan(p_kat) | np.isnan(p_gat))
    X_feat = X_feat[valid]
    target = target[valid]

    if len(target) < 150:
        print(f"[SKIP] После удаления NaN осталось только {len(target)} строк для GBDT.")
        return None

    # Пробуем XGBoost, иначе sklearn
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0,
        )
        print("[GBDT] Используется XGBoost")
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.08,
            subsample=0.8, random_state=42,
        )
        print("[GBDT] XGBoost не установлен, используется sklearn GBT")

    model.fit(X_feat, target)

    # Корреляция предсказания с целевой
    pred = model.predict(X_feat)
    corr = float(np.corrcoef(pred, target)[0, 1])
    print(f"[GBDT] Обучено на {len(target)} примерах. Corr(pred, target) = {corr:.3f}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
#  ИНФЕРЕНС ЧЕРЕЗ СУЩЕСТВУЮЩИЕ МОДЕЛИ
# ══════════════════════════════════════════════════════════════════════════════

def run_inference_on_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Прогнать каждую строку датасета через predict_single_patient() из ivf_core.
    Добавляет колонки: p_kat_raw, p_gat_raw, p_L1, p_csdi.
    Требует наличия модели в проекте (Prediction_KAN.pth и т.д.).
    """
    # Импорт ядра — должен быть доступен из текущей директории
    try:
        from ivf_core import predict_single_patient
        print("[OK] ivf_core загружен")
    except ImportError as e:
        print(f"[ERR] Не удалось импортировать ivf_core: {e}")
        print("      Убедитесь, что скрипт запущен из корня проекта (где лежат модели).")
        sys.exit(1)

    def _f(col, row, default=None):
        v = str(row.get(col, "")).strip()
        if not v or v == "nan":
            return default
        try:
            return float(v.replace(",", "."))
        except ValueError:
            return default

    results = []
    n_ok = 0
    n_err = 0

    for idx, row in _iter(df.iterrows(), desc="Прогон инференса"):
        try:
            age        = _f("age", row, 35.0)
            amh        = _f("amh", row, 1.5)
            afc        = int(_f("afc", row, 10))
            bmi        = _f("bmi", row, 23.0)
            attempt    = int(_f("attempt_number", row, 1))
            follicles  = int(_f("follicles_14mm", row, afc))

            # Известные лабораторные значения (mid-cycle / post-retrieval)
            okk    = _f("okk", row)
            mii    = _f("mii", row)
            pn2    = _f("pn2", row)
            blasts = _f("blasts_total", row)
            good   = _f("blasts_good", row)

            result = predict_single_patient(
                age=age, amh=amh, afc=afc, bmi=bmi,
                attempt=attempt, follicles=follicles,
                known_okk=int(okk) if okk is not None else None,
                known_mii=int(mii) if mii is not None else None,
                known_pn2=int(pn2) if pn2 is not None else None,
                known_blasts=int(blasts) if blasts is not None else None,
                known_good=int(good) if good is not None else None,
            )

            res        = result.get("res", {})
            nn_pred    = res.get("nn_prediction", {})
            gnn_result = result.get("gnn_result", {})

            p_kat  = nn_pred.get("base_prob_mean")
            p_L1   = res.get("p_per_transfer")
            p_csdi_val = result.get("p_csdi")
            p_gat  = gnn_result.get("gnn_prob") if gnn_result else None

            results.append({
                "p_kat_raw": p_kat,
                "p_gat_raw": p_gat,
                "p_L1":      p_L1,
                "p_csdi":    p_csdi_val,
            })
            n_ok += 1

        except Exception as e:
            results.append({"p_kat_raw": None, "p_gat_raw": None,
                             "p_L1": None, "p_csdi": None})
            n_err += 1

    print(f"\n[OK] Инференс завершён: {n_ok} успешно, {n_err} ошибок")

    for col in ["p_kat_raw", "p_gat_raw", "p_L1", "p_csdi"]:
        df[col] = [r[col] for r in results]

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  ОСНОВНОЙ КЛАСС КАЛИБРОВКИ
# ══════════════════════════════════════════════════════════════════════════════

class ClinicCalibrator:

    # Полный набор: сырые ковариаты + все 6 производных признаков
    # (производные вычисляются через add_derived_features() до вызова calibrate)
    GBDT_FEATURE_NAMES = [
        "age", "amh", "afc", "bmi", "attempt_number",
        "okk", "mii", "pn2", "blasts_total", "blasts_good",
        "fert_rate", "cleav_rate", "blast_rate", "good_blast_rate",
        "occ_rate", "kpi_score",
    ]

    def __init__(self, clinic_name: str):
        self.clinic_name = clinic_name
        self.results: dict = {
            "clinic_name":        clinic_name,
            "calibration_date":   datetime.now().isoformat(timespec="seconds"),
            "n_cycles_total":     0,
            "n_cycles_outcomes":  0,
            "pregnancy_rate":     None,

            "temperature": {
                "T_kat":  1.0,
                "T_gat":  1.0,
                "T_L1":   1.0,
                "T_csdi": 1.0,
            },

            "ece": {
                "before": {},
                "after":  {},
            },

            "brier": {
                "before": {},
                "after":  {},
            },

            "ood": {
                "clinical":   {"features": ["age", "amh", "afc", "bmi"],
                                "mean": None, "cov_inv": None, "n": 0},
                "embryology": {"features": ["okk", "mii", "pn2", "blasts_total"],
                                "mean": None, "cov_inv": None, "n": 0},
            },

            "gbdt_tau_available": False,
            "gbdt_tau_model_path": None,

            "notes": [],
        }

    def _note(self, msg: str):
        print(f"  [NOTE] {msg}")
        self.results["notes"].append(msg)

    # ── Подготовка признаков для GBDT ─────────────────────────────────────────
    def _build_gbdt_features(self, df: pd.DataFrame) -> np.ndarray:
        """Требует, чтобы df уже содержал производные колонки (add_derived_features)."""
        X = np.zeros((len(df), len(self.GBDT_FEATURE_NAMES)), dtype=float)
        for j, col in enumerate(self.GBDT_FEATURE_NAMES):
            if col in df.columns:
                X[:, j] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
        return X

    # ── Основная калибровка ───────────────────────────────────────────────────
    def calibrate(self, df: pd.DataFrame):
        # Вычисляем производные признаки (частоты + KPIScore) из сырых колонок
        df = add_derived_features(df)
        self.results["n_cycles_total"] = len(df)

        # Маска строк с известным исходом
        y_all    = pd.to_numeric(df.get("outcome_binary", pd.NA), errors="coerce")
        mask_out = y_all.notna()
        df_cal   = df[mask_out].copy()
        y        = y_all[mask_out].astype(float).values

        n_out = len(y)
        self.results["n_cycles_outcomes"] = n_out
        self.results["pregnancy_rate"]    = round(float(y.mean()), 4) if n_out > 0 else None

        print(f"\n[CALIB] Клиника: {self.clinic_name}")
        print(f"[CALIB] Строк с исходом: {n_out}  (беременность {y.mean()*100:.1f}%)")

        if n_out < 30:
            self._note("Слишком мало исходов для калибровки. Минимум 30, рекомендуется ≥100.")
            print("[WARN] Недостаточно данных. Температуры T_i=1.0 (без изменения).")
            return

        # ── Температурная калибровка per-model ────────────────────────────────
        print("\n[CALIB] Температурное масштабирование (per-model)...")
        model_cols = {
            "T_kat":  "p_kat_raw",
            "T_gat":  "p_gat_raw",
            "T_L1":   "p_L1",
            "T_csdi": "p_csdi",
        }

        for T_key, p_col in model_cols.items():
            if p_col not in df_cal.columns:
                self._note(f"{p_col} отсутствует — T=1.0")
                continue

            p_raw = pd.to_numeric(df_cal[p_col], errors="coerce").values
            mask  = ~np.isnan(p_raw)
            p_raw = p_raw[mask]
            y_sub = y[mask]

            if len(p_raw) < 20:
                self._note(f"{T_key}: только {len(p_raw)} точек — пропуск")
                continue

            # ECE до
            ece_before   = compute_ece(p_raw, y_sub)
            brier_before = compute_brier(p_raw, y_sub)

            # Подбор T
            T_opt = fit_temperature(p_raw, y_sub)
            self.results["temperature"][T_key] = round(T_opt, 4)

            # ECE после
            p_cal        = temperature_scale(p_raw, T_opt)
            ece_after    = compute_ece(p_cal, y_sub)
            brier_after  = compute_brier(p_cal, y_sub)

            self.results["ece"]["before"][T_key]   = ece_before
            self.results["ece"]["after"][T_key]    = ece_after
            self.results["brier"]["before"][T_key] = round(brier_before, 5)
            self.results["brier"]["after"][T_key]  = round(brier_after, 5)

            improvement = (ece_before - ece_after) / max(ece_before, 1e-9) * 100
            print(f"  {T_key:<8}  T={T_opt:.3f}  "
                  f"ECE {ece_before:.4f} → {ece_after:.4f}  "
                  f"({improvement:+.1f}%)  "
                  f"n={len(p_raw)}")

        # ── OOD baseline ──────────────────────────────────────────────────────
        print("\n[CALIB] OOD baseline (Mahalanobis)...")

        def _to_num(col):
            return pd.to_numeric(df[col], errors="coerce").values if col in df.columns \
                   else np.full(len(df), np.nan)

        X_clin = np.column_stack([
            _to_num("age"), _to_num("amh"), _to_num("afc"), _to_num("bmi")
        ])
        X_emb  = np.column_stack([
            _to_num("okk"), _to_num("mii"), _to_num("pn2"), _to_num("blasts_total")
        ])

        clin_stats = fit_mahalanobis_baseline(X_clin)
        emb_stats  = fit_mahalanobis_baseline(X_emb)

        self.results["ood"]["clinical"].update(clin_stats)
        self.results["ood"]["embryology"].update(emb_stats)

        print(f"  Клиническое подпространство : n={clin_stats['n']}")
        print(f"  Лабораторное подпространство: n={emb_stats['n']}")

        # ── GBDT meta-learner для динамического τ(X) ─────────────────────────
        print("\n[CALIB] GBDT meta-learner (динамический τ)...")

        p_kat_arr = pd.to_numeric(df_cal.get("p_kat_raw", pd.NA), errors="coerce").values
        p_gat_arr = pd.to_numeric(df_cal.get("p_gat_raw", pd.NA), errors="coerce").values
        both_ok   = ~(np.isnan(p_kat_arr) | np.isnan(p_gat_arr))

        if both_ok.sum() >= 200:
            X_feat = self._build_gbdt_features(df_cal[both_ok])
            gbdt   = fit_gbdt_tau_learner(
                X_feat, p_kat_arr[both_ok], p_gat_arr[both_ok], y[both_ok]
            )
            if gbdt is not None:
                try:
                    import joblib
                    model_path = f"gbdt_tau_{self.clinic_name.replace(' ', '_')}.joblib"
                    joblib.dump(gbdt, model_path)
                    self.results["gbdt_tau_available"]   = True
                    self.results["gbdt_tau_model_path"]  = model_path
                    self.results["gbdt_feature_names"]   = self.GBDT_FEATURE_NAMES
                    print(f"[OK] GBDT сохранён: {model_path}")
                except Exception as e:
                    self._note(f"Не удалось сохранить GBDT: {e}")
        else:
            print(f"  Строк с обеими моделями: {both_ok.sum()} < 200. GBDT не обучается.")

    # ── Вывод итогов ──────────────────────────────────────────────────────────
    def print_summary(self):
        print("\n" + "=" * 60)
        print("ИТОГИ КАЛИБРОВКИ")
        print("=" * 60)
        print(f"Клиника          : {self.clinic_name}")
        print(f"Дата             : {self.results['calibration_date']}")
        print(f"Циклов в данных  : {self.results['n_cycles_total']}")
        print(f"С известным исходом: {self.results['n_cycles_outcomes']}")
        if self.results["pregnancy_rate"] is not None:
            print(f"Беременность     : {self.results['pregnancy_rate']*100:.1f}%")
        print()
        print("Температурные параметры L7:")
        for k, v in self.results["temperature"].items():
            ece_b = self.results["ece"]["before"].get(k, "—")
            ece_a = self.results["ece"]["after"].get(k, "—")
            ece_str = f"ECE {ece_b:.4f} → {ece_a:.4f}" if ece_b != "—" else "нет данных"
            print(f"  {k:<10}: T = {v:.4f}   {ece_str}")
        print()
        print(f"OOD baseline     : clinical n={self.results['ood']['clinical']['n']}, "
              f"embryology n={self.results['ood']['embryology']['n']}")
        print(f"GBDT meta-learner: {'ДА → ' + self.results['gbdt_tau_model_path'] if self.results['gbdt_tau_available'] else 'НЕТ (мало данных)'}")
        if self.results["notes"]:
            print("\nПримечания:")
            for note in self.results["notes"]:
                print(f"  • {note}")

    # ── Сохранение ────────────────────────────────────────────────────────────
    def save(self, out_path: str = None) -> str:
        date_str  = datetime.now().strftime("%Y%m%d")
        name_safe = self.clinic_name.replace(" ", "_").replace("/", "-")
        out_path  = out_path or f"clinic_adaptation_{name_safe}_{date_str}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        print(f"\n[OK] Адаптация сохранена: {out_path}")
        print("     Передайте этот файл в BEFE через:")
        print(f"     befe = BayesianEvidenceFusionEngine.from_clinic_adaptation('{out_path}')")
        return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  ЗАГРУЗКА АДАПТАЦИИ В BEFE  (вспомогательная функция)
# ══════════════════════════════════════════════════════════════════════════════

def load_clinic_adaptation(path: str) -> dict:
    """
    Загрузить файл адаптации и вернуть параметры для использования в BEFE.

    Пример использования в app.py:
        adaptation = load_clinic_adaptation("clinic_adaptation_MyClinic_20240115.json")
        T_kat  = adaptation["temperature"]["T_kat"]
        T_gat  = adaptation["temperature"]["T_gat"]
        tau_kat_dynamic = adaptation.get("gbdt_tau_available", False)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def apply_clinic_calibration(p_raw: float, model: str, adaptation: dict) -> float:
    """
    Применить температурную калибровку из adaptation к одному скалярному p_raw.

    Args:
        p_raw      : сырая вероятность [0,1]
        model      : "T_kat" | "T_gat" | "T_L1" | "T_csdi"
        adaptation : словарь из load_clinic_adaptation()

    Returns:
        откалиброванная вероятность [0,1]
    """
    T = adaptation.get("temperature", {}).get(model, 1.0)
    p = float(np.clip(p_raw, _EPS, 1.0 - _EPS))
    logit_p = float(np.log(p / (1.0 - p)))
    return float(1.0 / (1.0 + np.exp(-logit_p / max(T, 1e-3))))


def compute_dynamic_tau_kat(patient_features: dict,
                             adaptation: dict,
                             tau_base: float = 2.4,
                             tau_min: float = 0.5,
                             tau_max: float = 4.5) -> float:
    """
    Вычислить динамический τ_KAT через GBDT meta-learner.
    Если GBDT недоступен — вернуть tau_base.

    Args:
        patient_features : dict с ключами из GBDT_FEATURE_NAMES
        adaptation       : словарь из load_clinic_adaptation()
        tau_base         : базовый τ_KAT (из befe.py, обычно 2.4)

    Returns:
        τ_KAT ∈ [tau_min, tau_max]
    """
    if not adaptation.get("gbdt_tau_available", False):
        return tau_base

    model_path = adaptation.get("gbdt_tau_model_path")
    if not model_path or not os.path.exists(model_path):
        return tau_base

    try:
        import joblib
        gbdt = joblib.load(model_path)
        feat_names = adaptation.get("gbdt_feature_names",
                                     ClinicCalibrator.GBDT_FEATURE_NAMES)
        X = np.array([[patient_features.get(f, 0.0) for f in feat_names]],
                     dtype=float)
        advantage = float(gbdt.predict(X)[0])

        # Переводим advantage в τ через мягкое масштабирование
        # advantage > 0 → KAT лучше → τ_kat выше tau_base
        # advantage < 0 → GAT лучше → τ_kat ниже tau_base
        k = 2.0   # крутизна sigmoid (настраивается)
        scale_factor = float(1.0 / (1.0 + np.exp(-k * advantage)))  # ∈ (0,1)
        tau_raw = tau_min + (tau_max - tau_min) * scale_factor
        return float(np.clip(tau_raw, tau_min, tau_max))

    except Exception as e:
        print(f"[WARN] GBDT τ-inference error: {e}. Использован tau_base={tau_base}")
        return tau_base


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Калибровка IVF Digital Twin под клинику"
    )
    parser.add_argument("--data",    required=True,
                        help="Валидированный CSV (выход validate_clinic_data.py)")
    parser.add_argument("--clinic",  required=True,
                        help="Название клиники (используется в имени выходного файла)")
    parser.add_argument("--output",  default=None,
                        help="Путь к выходному JSON (по умолчанию автоматическое имя)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Пропустить прогон инференса (если p_kat_raw уже есть в CSV)")
    args = parser.parse_args()

    # Загрузка данных
    print(f"\nЗагрузка данных: {args.data}")
    df = pd.read_csv(args.data, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.replace("", pd.NA)
    print(f"Загружено строк: {len(df)}")

    # Инференс (если нужен)
    model_cols_present = all(c in df.columns for c in ["p_kat_raw", "p_gat_raw", "p_L1"])
    if not model_cols_present and not args.skip_inference:
        print("\nКолонки p_kat_raw / p_gat_raw отсутствуют — запуск инференса...")
        df = run_inference_on_dataset(df)
        # Сохраняем промежуточный файл с предсказаниями
        inference_path = args.data.replace(".csv", "_with_predictions.csv")
        df.to_csv(inference_path, index=False, encoding="utf-8-sig")
        print(f"[OK] Данные с предсказаниями сохранены: {inference_path}")
    elif args.skip_inference:
        print("[SKIP] Инференс пропущен (--skip-inference)")
    else:
        print("[OK] Колонки предсказаний уже присутствуют в CSV")

    # Калибровка
    calibrator = ClinicCalibrator(clinic_name=args.clinic)
    calibrator.calibrate(df)
    calibrator.print_summary()
    calibrator.save(args.output)

    print("\nГотово. Передайте JSON-файл в BEFE через load_clinic_adaptation().")


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
#  ПРОИЗВОДНЫЕ ПРИЗНАКИ — патч (добавлен post-hoc)
# ══════════════════════════════════════════════════════════════════════════════

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет 6 производных колонок, точно совпадающих с формулами pipeline.

    fert_rate        = pn2 / max(mii,1)
    cleav_rate       = cleavage_d3 / max(pn2,1)  [proxy=1.0 если нет cleavage_d3]
    blast_rate       = blasts_total / max(pn2,1)
    good_blast_rate  = blasts_good / max(pn2,1)   # = blasts_good / 2PN (как в KAT build_nn_features)
    occ_rate         = okk / max(follicles_14mm,1)
    kpi_score        = A+B+C+D+E  (5–25), формула из ivf_digital_twin.py
    """
    df = df.copy()
    def _n(col, default=0.0):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default).values
        return np.full(len(df), default, dtype=float)

    mii  = np.maximum(_n("mii",          0.1), 0.1)
    pn2  = _n("pn2",          0.0)
    bl   = _n("blasts_total", 0.0)
    good = _n("blasts_good",  0.0)
    okk  = _n("okk",          0.0)
    foll = np.maximum(_n("follicles_14mm", 1.0), 1.0)
    age  = _n("age",          35.0)
    # cleavage_d3 опционально; proxy — pn2 (all 2PN cleave)
    if "cleavage_d3" in df.columns:
        cl = pd.to_numeric(df["cleavage_d3"], errors="coerce")
        cl_vals = np.where(cl.isna(), pn2, cl.fillna(0).values)
    else:
        cl_vals = pn2

    df["fert_rate"]       = np.clip(pn2 / np.maximum(mii, 1), 0, 1)
    df["cleav_rate"]      = np.clip(cl_vals / np.maximum(pn2, 1), 0, 1)
    df["blast_rate"]      = np.clip(bl   / np.maximum(pn2, 1), 0, 1)
    df["good_blast_rate"] = np.clip(good / np.maximum(pn2, 1), 0, 1)   # blasts_good / 2PN
    df["occ_rate"]        = np.clip(okk  / np.maximum(foll,1), 0, 1)

    # KPIScore — точная копия calculate_nn_kpi_score()
    a = np.where(age >= 40, 1, np.where(age <= 36, 5, 3))
    b = np.where(foll > 15, 5, np.where(foll >= 8, 3, 1))
    c = np.where(np.round(mii) <= 3, 1, np.where(np.round(mii) <= 7, 3, 5))
    fr = df["fert_rate"].values
    d = np.where(fr < 0.50, 1, np.where(fr <= 0.65, 3, 5))
    e = np.where(np.round(good) == 0, 1, np.where(np.round(good) <= 2, 3, 5))
    df["kpi_score"] = (a + b + c + d + e).astype(int)

    return df
