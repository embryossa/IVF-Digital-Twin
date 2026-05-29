# ============================================================
# EMBRYO OUTCOME — Hybrid CSDI + LightGBM v3.0
#
# ┌─────────────────────────────────────────────────────────┐
# │  Изменения v2 → v3                                      │
# │                                                         │
# │  [FIX-1] COUNT-only генерация                           │
# │    CSDI генерирует только 2 счётные переменные:         │
# │      • Число Bl (total)                                 │
# │      • Число Bl хор.кач-ва (good)                       │
# │    Частоты вычисляются аналитически:                    │
# │      • blast_rate = Bl / 2pN    (из COND)              │
# │      • good_rate  = good / Bl   (из сгенерированных)   │
# │    Зачем: диффузия плохо генерирует дроби — особенно   │
# │    когда знаменатель сам является случайным. Вычисляя   │
# │    дробь из пар (total, good) мы автоматически:         │
# │      а) получаем биологически корректное good ≤ total   │
# │      б) устраняем независимое дрейфование числителя     │
# │         и знаменателя в разные стороны                  │
# │      в) улучшаем KS-статистику для частот               │
# │                                                         │
# │  [FIX-2] Данных для CSDI больше (85% вместо 70%)        │
# │    Разбивка: 85% диффузия | 7.5% LGB | 7.5% conformal  │
# │    Основная причина ухудшения KS в v2 — меньше данных   │
# │                                                         │
# │  [FIX-3] Conformal с clip к биологическим границам      │
# │    Counts: max(0, lo), max(0, hi)                       │
# │    Rates: clip([0,1])                                   │
# │                                                         │
# │  [FIX-4] Сохранение в директорию (pipeline-ready)       │
# │    embryo_v3_model/                                     │
# │      config.json          — гиперпараметры + фичи       │
# │      csdi_weights.pt      — веса нейросети              │
# │      normalizer.pt       — QuantileNormalizer           │
# │      lgb_model.txt        — LightGBM native format      │
# │      platt_calibrator.pt — Platt scaling               │
# │      conformal.pt        — конформальные радиусы       │
# └─────────────────────────────────────────────────────────┘
#
# Pipeline:
#   COND(7) ──► CSDI ──► Bl, good_Bl (count, 2)
#                             │
#                             ▼  derive_rates(Bl, good_Bl, 2pN)
#                        blast_rate, good_rate
#                             │
#   COND(7) + Bl + good_Bl ──►  LightGBM + Platt ──► P(pregnancy)
# ============================================================

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, roc_curve)
from scipy import stats
from scipy.stats import wasserstein_distance
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# КОНСТАНТЫ И СХЕМА ПРИЗНАКОВ
# ═══════════════════════════════════════════════════════════

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

COND_FEATURES = [
    "Количество фолликулов",
    "Число ОКК",
    "Число инсеминированных",
    "2 pN",
    "Частота получения ОКК",
    "Частота оплодотворения",
    "KPIScore",
]
COND_DIM = len(COND_FEATURES)   # 7

# [FIX-1] CSDI генерирует только счётные переменные
COUNT_FEATURES = [
    "Число Bl",
    "Число Bl хор.кач-ва",
]
COUNT_DIM = len(COUNT_FEATURES)  # 2

# Частоты вычисляются аналитически из COUNT + COND
RATE_FEATURES = [
    "Частота формирования бластоцист",
    "Частота формирования бластоцист хорошего качества",
]

# Полный набор выходных признаков (для пользователя)
OUTPUT_FEATURES = COUNT_FEATURES + RATE_FEATURES   # 4

# Индекс "2 pN" в COND_FEATURES — знаменатель для blast_rate
PN2_IDX = COND_FEATURES.index("2 pN")

# Столбцы во входном Excel (обязательные)
ALL_DB_FEATURES = COND_FEATURES + COUNT_FEATURES + ["Исход переноса"]


# ═══════════════════════════════════════════════════════════
# ДАННЫЕ
# ═══════════════════════════════════════════════════════════

def load_data(path: str = 'all_df_with_KPI.xlsx') -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df[ALL_DB_FEATURES].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df.dropna(inplace=True)
    print(f"[DATA] {len(df)} циклов | "
          f"Беременность: {df['Исход переноса'].mean():.1%}")
    return df


# ═══════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════

def derive_rates(count_arr: np.ndarray,
                 pn2_values: np.ndarray) -> np.ndarray:
    """
    [FIX-1] Аналитическое вычисление частот из пар (Bl, good_Bl).

    count_arr:  [N, 2]  — (Число Bl, Число Bl хор.кач-ва)
    pn2_values: [N]     — значение "2 pN" для каждого образца
                          (одно значение для одного пациента,
                           или массив при батчевой генерации)

    Формулы:
      blast_rate = Bl / 2pN
        → частота формирования бластоцист = bl / fertilized
        → знаменатель 2pN из COND, известен точно
      good_rate  = good_Bl / Bl
        → доля хорошего качества = good / total
        → знаменатель сам сгенерирован, поэтому good_rate
          моделирует совместное распределение (total, good)

    Ограничения:
      • good_Bl  ≤ Bl  (биологически: хороших ≤ всего)
      • blast_rate ∈ [0, 1]
      • good_rate  ∈ [0, 1]

    Returns: [N, 2] — (blast_rate, good_rate)
    """
    Bl   = count_arr[:, 0]
    gBl  = np.minimum(count_arr[:, 1], Bl)   # enforce good ≤ total

    pn2  = np.maximum(pn2_values, 1)          # protect from zero division
    blast_rate = np.clip(Bl / pn2, 0.0, 1.0)

    good_rate  = np.where(Bl > 0,
                          np.clip(gBl / Bl, 0.0, 1.0),
                          0.0)
    return np.stack([blast_rate, good_rate], axis=1)


def post_process_counts(arr: np.ndarray) -> np.ndarray:
    """
    Биологические ограничения для 2 счётных признаков.
    arr: [N, COUNT_DIM] в оригинальной шкале (после inverse_transform)
    Returns: [N, COUNT_DIM] — целые неотрицательные числа
    """
    Bl  = np.maximum(0, np.round(arr[:, 0])).astype(np.float32)
    gBl = np.minimum(
        np.maximum(0, np.round(arr[:, 1])),
        Bl
    ).astype(np.float32)
    return np.stack([Bl, gBl], axis=1)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray,
                n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / n) * abs(
                y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


# ═══════════════════════════════════════════════════════════
# [1] QUANTILE NORMALIZER (только для COUNT)
# ═══════════════════════════════════════════════════════════

class QuantileNormalizer:
    """
    Квантильная нормализация: x → Φ⁻¹(F̂(x)) ~ N(0,1).
    Применяется к COND и COUNT. Частоты не нормализуются —
    они вычисляются аналитически после inverse transform COUNT.
    """
    def __init__(self, n_quantiles: int = 1000):
        self.n_quantiles = n_quantiles
        self.cond_qt = QuantileTransformer(
            n_quantiles=n_quantiles, output_distribution='normal',
            random_state=SEED)
        self.count_qt = QuantileTransformer(
            n_quantiles=n_quantiles, output_distribution='normal',
            random_state=SEED)

    def fit_transform(self, df: pd.DataFrame):
        Xc = self.cond_qt.fit_transform(
            df[COND_FEATURES].values).astype(np.float32)
        Xt = self.count_qt.fit_transform(
            df[COUNT_FEATURES].values).astype(np.float32)
        return Xc, Xt

    def transform_cond(self, arr: np.ndarray) -> np.ndarray:
        return self.cond_qt.transform(arr).astype(np.float32)

    def inverse_count(self, arr: np.ndarray) -> np.ndarray:
        """N(0,1) → оригинальная шкала COUNT."""
        return self.count_qt.inverse_transform(arr)


# ═══════════════════════════════════════════════════════════
# [2] CSDI-TRANSFORMER (COUNT_DIM = 2 токена)
# ═══════════════════════════════════════════════════════════

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -np.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        emb  = t.float().unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class CSDILayer(nn.Module):
    """
    Self-attention (COUNT токены ↔ COUNT токены) +
    Cross-attention (COUNT токены → COND токены) +
    FFN (GELU, 4×).
    Всё с pre-norm для стабильности.
    """
    def __init__(self, hidden: int, n_heads: int, dropout: float):
        super().__init__()
        self.sa_norm = nn.LayerNorm(hidden)
        self.sa      = nn.MultiheadAttention(
            hidden, n_heads, dropout=dropout, batch_first=True)
        self.ca_norm = nn.LayerNorm(hidden)
        self.ca      = nn.MultiheadAttention(
            hidden, n_heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(hidden)
        self.ff      = nn.Sequential(
            nn.Linear(hidden, hidden * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor,
                cond_tokens: torch.Tensor) -> torch.Tensor:
        h = self.sa_norm(x)
        h, _ = self.sa(h, h, h)
        x = x + h
        h = self.ca_norm(x)
        h, _ = self.ca(h, cond_tokens, cond_tokens)
        x = x + h
        h = self.ff_norm(x)
        return x + self.ff(h)


class CSDIDenoiser(nn.Module):
    """
    CSDI-Transformer для COUNT_DIM = 2 выходных токенов.
    Каждый токен = один счётный признак (Bl, good_Bl).
    Кондиционирование через cross-attention к COND_DIM = 7 токенам.

    Меньше токенов → задача проще → лучшее качество каждого токена.
    """
    def __init__(self, count_dim: int = COUNT_DIM, cond_dim: int = COND_DIM,
                 hidden: int = 128, n_heads: int = 4, n_layers: int = 6,
                 time_emb_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.count_dim  = count_dim
        self.cond_dim   = cond_dim
        self.time_emb   = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_proj  = nn.Linear(time_emb_dim, hidden)
        self.input_proj = nn.Linear(1, hidden)   # каждый COUNT-токен
        self.cond_proj  = nn.Linear(1, hidden)   # каждый COND-токен
        self.count_pos  = nn.Embedding(count_dim, hidden)
        self.cond_pos   = nn.Embedding(cond_dim,  hidden)
        self.layers     = nn.ModuleList([
            CSDILayer(hidden, n_heads, dropout) for _ in range(n_layers)])
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                x_cond: torch.Tensor) -> torch.Tensor:
        # t: [B], x_t: [B, count_dim], x_cond: [B, cond_dim]
        t_h   = self.time_proj(self.time_emb(t))              # [B, hidden]
        x_tok = self.input_proj(x_t.unsqueeze(-1))            # [B, 2, hidden]
        x_tok = x_tok + self.count_pos(
            torch.arange(self.count_dim, device=x_t.device))
        x_tok = x_tok + t_h.unsqueeze(1)
        c_tok = self.cond_proj(x_cond.unsqueeze(-1))          # [B, 7, hidden]
        c_tok = c_tok + self.cond_pos(
            torch.arange(self.cond_dim, device=x_cond.device))
        c_tok = c_tok + t_h.unsqueeze(1)
        h = x_tok
        for layer in self.layers:
            h = layer(h, c_tok)
        return self.output_head(h).squeeze(-1)                 # [B, 2]


# ═══════════════════════════════════════════════════════════
# [3] ДИФФУЗИОННЫЙ ПРОЦЕСС
# ═══════════════════════════════════════════════════════════

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    f     = torch.cos(((steps / T) + s) / (1 + s) * np.pi / 2) ** 2
    acp   = f / f[0]
    return (1 - acp[1:] / acp[:-1]).clamp(1e-5, 0.9999).float()


class GaussianDiffusion:
    def __init__(self, T: int = 1000, device: str = 'cpu'):
        self.T      = T
        self.device = device
        betas       = cosine_beta_schedule(T).to(device)
        alphas      = 1.0 - betas
        acp         = torch.cumprod(alphas, dim=0)
        acp_prev    = F.pad(acp[:-1], (1, 0), value=1.0)
        self.alphas_cp      = acp
        self.sqrt_acp       = acp.sqrt()
        self.sqrt_one_m_acp = (1 - acp).sqrt()
        self.posterior_var  = (
                betas * (1 - acp_prev) / (1 - acp)).clamp(min=1e-20)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return (self.sqrt_acp[t].unsqueeze(1) * x0 +
                self.sqrt_one_m_acp[t].unsqueeze(1) * noise, noise)

    def p_losses(self, model, x0, x_cond) -> torch.Tensor:
        B   = x0.shape[0]
        t   = torch.randint(0, self.T, (B,), device=self.device)
        x_t, noise = self.q_sample(x0, t)
        return F.mse_loss(model(x_t, t, x_cond), noise)

    @torch.no_grad()
    def ddim_sample(self, model, x_cond: torch.Tensor,
                    n_samples: int, count_dim: int,
                    ddim_steps: int = 50) -> torch.Tensor:
        """DDIM (Song et al. 2021): детерминированный, 50 шагов."""
        model.eval()
        x    = torch.randn(n_samples, count_dim, device=self.device)
        cond = (x_cond.expand(n_samples, -1)
                if x_cond.shape[0] == 1 else x_cond)
        ts   = torch.linspace(self.T - 1, 0, ddim_steps,
                              dtype=torch.long, device=self.device)
        for i, t_idx in enumerate(ts):
            t_vec      = torch.full((n_samples,), t_idx.item(),
                                    device=self.device, dtype=torch.long)
            pred_noise = model(x, t_vec, cond)
            acp_t      = self.alphas_cp[t_idx]
            x0_pred    = (x - (1 - acp_t).sqrt() * pred_noise) / acp_t.sqrt()
            x0_pred    = x0_pred.clamp(-4, 4)
            if i == len(ts) - 1:
                x = x0_pred
            else:
                acp_next = self.alphas_cp[ts[i + 1]]
                x = acp_next.sqrt() * x0_pred + (1 - acp_next).sqrt() * pred_noise
        return x   # [n_samples, count_dim]


# ═══════════════════════════════════════════════════════════
# [4] LIGHTGBM + PLATT CLASSIFIER
# ═══════════════════════════════════════════════════════════

class OutcomeClassifier:
    """
    LightGBM + Platt scaling для P(беременность).

    Вход: COND(7) + COUNT-медианы(2) = 9 признаков.
    Частоты (RATE) не передаются в LGB — они функционально
    зависят от COUNT и COND, поэтому дублируют информацию.

    LightGBM:  is_unbalance=True,  boosting='dart'
    Platt:     LogisticRegression на удержанной части
    """
    LGB_FEATURE_NAMES = COND_FEATURES + COUNT_FEATURES   # 9 фичей

    def __init__(self, n_estimators: int = 600,
                 learning_rate: float = 0.04,
                 num_leaves: int = 31,
                 min_child_samples: int = 20,
                 random_state: int = SEED):
        self.lgb_params = dict(
            n_estimators      = n_estimators,
            learning_rate     = learning_rate,
            num_leaves        = num_leaves,
            min_child_samples = min_child_samples,
            is_unbalance      = True,
            boosting_type     = 'dart',
            drop_rate         = 0.1,
            max_drop          = 50,
            colsample_bytree  = 0.8,
            subsample         = 0.8,
            subsample_freq    = 5,
            reg_alpha         = 0.1,
            reg_lambda        = 0.1,
            random_state      = random_state,
            n_jobs            = -1,
            verbose           = -1,
        )
        self.clf       = lgb.LGBMClassifier(**self.lgb_params)
        self.calibrator = LogisticRegression()
        self._fitted   = False
        # После fit хранит бустер нативно (для сохранения .txt)
        self._booster  = None

    def _build_X(self, df_cond: pd.DataFrame,
                 count_medians: np.ndarray) -> np.ndarray:
        """df_cond: [n, COND_DIM], count_medians: [n, 2] → [n, 9]."""
        return np.hstack([df_cond[COND_FEATURES].values, count_medians])

    def fit(self, df: pd.DataFrame, count_medians: np.ndarray):
        X = self._build_X(df, count_medians)
        y = df["Исход переноса"].values.astype(int)

        # 80/20: обучение / калибровка Platt
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X, y, test_size=0.20, random_state=SEED, stratify=y)

        print(f"[LGB] Обучение: {len(X_tr)} | Калибровка: {len(X_cal)}")
        self.clf.fit(
            X_tr, y_tr,
            eval_set=[(X_cal, y_cal)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)])

        self._booster = self.clf.booster_

        # Platt scaling
        raw_cal = self.clf.predict_proba(X_cal)[:, 1].reshape(-1, 1)
        self.calibrator.fit(raw_cal, y_cal)
        self._fitted = True

        p_cal = self.calibrator.predict_proba(raw_cal)[:, 1]
        print(f"[LGB] AUC (cal): {roc_auc_score(y_cal, p_cal):.4f} | "
              f"ECE (cal): {compute_ece(y_cal, p_cal):.4f}")

        imp     = self.clf.feature_importances_
        max_imp = max(imp.max(), 1)
        pairs   = sorted(zip(self.LGB_FEATURE_NAMES, imp),
                         key=lambda x: x[1], reverse=True)
        print("[LGB] Feature importance:")
        for name, val in pairs:
            bar = '█' * max(1, int(val / max_imp * 20))
            print(f"      {name:<50} {bar} {val:6.0f}")

    def predict_proba(self, df_cond: pd.DataFrame,
                      count_medians: np.ndarray) -> np.ndarray:
        assert self._fitted
        X   = self._build_X(df_cond, count_medians)
        raw = self.clf.predict_proba(X)[:, 1].reshape(-1, 1)
        return self.calibrator.predict_proba(raw)[:, 1]

    def save_lgb(self, path: str):
        """Сохраняет LGB бустер в нативном текстовом формате."""
        self._booster.save_model(path)

    @staticmethod
    def load_lgb(path: str) -> lgb.Booster:
        return lgb.Booster(model_file=path)


# ═══════════════════════════════════════════════════════════
# [5] CONFORMALIZATION (для COUNT; с clip ≥ 0)
# ═══════════════════════════════════════════════════════════

class ConformalizationLayer:
    """
    [FIX-3] Split Conformal Prediction для COUNT переменных.

    Квантиль остатков вычисляется отдельно для каждого признака.
    Интервал симметричный: [med - q, med + q], но нижняя граница
    клиппируется к 0 (Bl не может быть отрицательным).

    Частоты (RATE) не конформализуются — они автоматически ∈ [0,1]
    по конструкции derive_rates().
    """
    def __init__(self):
        self.quantiles: dict = {}   # float → ndarray[COUNT_DIM]

    def fit(self, actual: np.ndarray, pred_median: np.ndarray,
            levels: list = [0.50, 0.90]):
        """
        actual:      [n_cal, COUNT_DIM]
        pred_median: [n_cal, COUNT_DIM]
        """
        residuals = np.abs(actual - pred_median)
        n = residuals.shape[0]
        for alpha in levels:
            q_level = min(np.ceil((n + 1) * alpha) / n, 1.0)
            self.quantiles[alpha] = np.quantile(residuals, q_level, axis=0)
        print("[CONFORMAL] Поправочные радиусы (COUNT):")
        for alpha, q in sorted(self.quantiles.items()):
            parts = "  ".join(
                f"{COUNT_FEATURES[j][:10]}={q[j]:.2f}"
                for j in range(COUNT_DIM))
            print(f"  {int(alpha*100)}%-PI: {parts}")

    def get_intervals(self, pred_medians: np.ndarray,
                      level: float = 0.90) -> tuple:
        """
        pred_medians: [n, COUNT_DIM] или [COUNT_DIM]
        Returns: (lo, hi) — оба [n, COUNT_DIM] или [COUNT_DIM]
                 lo ≥ 0  (биологическое ограничение)
        """
        if level not in self.quantiles:
            raise KeyError(f"Уровень {level} не калиброван")
        q  = self.quantiles[level]
        lo = np.maximum(0, pred_medians - q)
        hi = pred_medians + q
        return lo, hi



# ── Портируемая сериализация sklearn-объектов ──────────────────
# Сохраняем только массивы numpy (без зависимости от имени модуля)

def _serialize_qt(qt):
    """QuantileTransformer → dict numpy-массивов."""
    return {
        'quantiles_':     qt.quantiles_,
        'references_':    qt.references_,
        'n_quantiles_':   int(qt.n_quantiles_),
        'n_features_in_': int(qt.n_features_in_),
        'output_distribution': qt.output_distribution,
        'random_state':   qt.random_state,
    }

def _deserialize_qt(d):
    """dict numpy-массивов → готовый QuantileTransformer."""
    qt = QuantileTransformer(
        n_quantiles=d['n_quantiles_'],
        output_distribution=str(d['output_distribution']),
        random_state=d['random_state'])
    qt.quantiles_     = np.array(d['quantiles_'])
    qt.references_    = np.array(d['references_'])
    qt.n_quantiles_   = int(d['n_quantiles_'])
    qt.n_features_in_ = int(d['n_features_in_'])
    return qt

# ═══════════════════════════════════════════════════════════
# [6] ГЛАВНЫЙ КЛАСС — ГИБРИДНАЯ МОДЕЛЬ v3
# ═══════════════════════════════════════════════════════════

class EmbryoHybridV3:
    """
    Трёхстадийная генеративная модель:
      Stage 1: CSDI генерирует (Bl, good_Bl) — COUNT
      Stage 2: derive_rates() вычисляет (blast_rate, good_rate)
      Stage 3: LightGBM + Platt → P(беременность)
      Calibration: ConformalizationLayer → откалиброванные PI для COUNT

    API для пайплайна:
      model = EmbryoHybridV3.load('embryo_v3_model')
      result = model.mc_sample(patient_dict)
      # result['P_pregnancy'], result['PI_90'], result['samples']
    """

    DEFAULT_SAVE_DIR = 'embryo_v3_model'

    def __init__(
            self,
            # CSDI
            T: int             = 1000,
            hidden: int        = 128,
            n_heads: int       = 4,
            n_layers: int      = 6,      # +2 слоя vs v2 (задача проще)
            time_emb_dim: int  = 64,
            dropout: float     = 0.1,
            lr: float          = 3e-4,
            epochs: int        = 200,
            batch_size: int    = 128,
            warmup_epochs: int = 15,
            ddim_steps: int    = 50,
            n_quantiles: int   = 1000,
            # LightGBM
            lgb_n_estimators: int    = 600,
            lgb_learning_rate: float = 0.04,
            lgb_num_leaves: int      = 31,
    ):
        self.T             = T
        self.hidden        = hidden
        self.n_heads       = n_heads
        self.n_layers      = n_layers
        self.time_emb_dim  = time_emb_dim
        self.dropout       = dropout
        self.lr            = lr
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.warmup_epochs = warmup_epochs
        self.ddim_steps    = ddim_steps
        self.n_quantiles   = n_quantiles
        self.lgb_n_estimators   = lgb_n_estimators
        self.lgb_learning_rate  = lgb_learning_rate
        self.lgb_num_leaves     = lgb_num_leaves
        self.device        = DEVICE
        self.best_threshold = 0.5   # обновляется после evaluate

        self.normalizer  = QuantileNormalizer(n_quantiles)
        self.diffusion   = None
        self.denoiser    = None
        self.classifier  = OutcomeClassifier(
            lgb_n_estimators, lgb_learning_rate, lgb_num_leaves)
        self.conformal   = ConformalizationLayer()
        self.history     = {'loss': [], 'val_loss': []}

    # ── Stage 1: CSDI ─────────────────────────────────────────────
    def _fit_diffusion(self, df_diff: pd.DataFrame,
                       df_holdout: pd.DataFrame) -> np.ndarray:
        """
        Обучает CSDI на df_diff.
        Возвращает COUNT-медианы для df_holdout (нужны Stage 2+3).
        """
        Xc, Xt = self.normalizer.fit_transform(df_diff)
        Xc_tr, Xc_val, Xt_tr, Xt_val = train_test_split(
            Xc, Xt, test_size=0.15, random_state=SEED)

        def t(a): return torch.tensor(a, device=self.device)
        dl = DataLoader(TensorDataset(t(Xc_tr), t(Xt_tr)),
                        batch_size=self.batch_size, shuffle=True,
                        drop_last=False)
        Xc_vt, Xt_vt = t(Xc_val), t(Xt_val)

        self.diffusion = GaussianDiffusion(T=self.T, device=str(self.device))
        self.denoiser  = CSDIDenoiser(
            count_dim=COUNT_DIM, cond_dim=COND_DIM,
            hidden=self.hidden, n_heads=self.n_heads,
            n_layers=self.n_layers, time_emb_dim=self.time_emb_dim,
            dropout=self.dropout).to(self.device)

        n_p = sum(p.numel() for p in self.denoiser.parameters())
        print(f"[CSDI] {n_p:,} параметров | {self.device}")
        print(f"[CSDI] COUNT_DIM={COUNT_DIM} | T={self.T} | "
              f"epochs={self.epochs}\n")

        opt = torch.optim.AdamW(
            self.denoiser.parameters(), lr=self.lr, weight_decay=1e-4)

        def lr_lambda(ep):
            if ep < self.warmup_epochs:
                return ep / max(self.warmup_epochs, 1)
            prog = (ep - self.warmup_epochs) / max(
                self.epochs - self.warmup_epochs, 1)
            return 0.5 * (1 + np.cos(np.pi * prog))

        scheduler  = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        best_val   = float('inf')
        best_state = None

        for epoch in range(self.epochs):
            self.denoiser.train()
            bl = []
            for xc_b, xt_b in dl:
                loss = self.diffusion.p_losses(self.denoiser, xt_b, xc_b)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
                opt.step()
                bl.append(loss.item())
            self.denoiser.eval()
            with torch.no_grad():
                vl = self.diffusion.p_losses(
                    self.denoiser, Xt_vt, Xc_vt).item()
            tr = float(np.mean(bl))
            self.history['loss'].append(tr)
            self.history['val_loss'].append(vl)
            scheduler.step()
            if vl < best_val:
                best_val   = vl
                best_state = {k: v.cpu().clone()
                              for k, v in self.denoiser.state_dict().items()}
            if (epoch + 1) % 25 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:4d}/{self.epochs} | "
                      f"train={tr:.4f} | val={vl:.4f} | "
                      f"lr={opt.param_groups[0]['lr']:.2e}")

        self.denoiser.load_state_dict(
            {k: v.to(self.device) for k, v in best_state.items()})
        print(f"\n[CSDI] Лучший val_loss: {best_val:.4f}")
        return self._batch_count_medians(df_holdout, n_samples=100)

    # ── Батчевая генерация медиан COUNT ───────────────────────────
    def _batch_count_medians(self, df: pd.DataFrame,
                             n_samples: int = 100,
                             patient_batch: int = 256) -> np.ndarray:
        """Генерирует медианы COUNT для каждого пациента. [n, 2]"""
        Xc = self.normalizer.transform_cond(df[COND_FEATURES].values)
        n  = len(df)
        out = []
        for s in range(0, n, patient_batch):
            e   = min(s + patient_batch, n)
            B, N = e - s, n_samples
            rep = np.repeat(Xc[s:e], N, axis=0)
            raw = self.diffusion.ddim_sample(
                self.denoiser,
                torch.tensor(rep, device=self.device),
                B * N, COUNT_DIM, self.ddim_steps)
            inv = post_process_counts(
                self.normalizer.inverse_count(raw.cpu().numpy()))
            out.append(np.median(inv.reshape(B, N, COUNT_DIM), axis=1))
        return np.vstack(out)   # [n, 2]

    # ── Основной метод обучения ───────────────────────────────────
    def fit(self, df: pd.DataFrame) -> 'EmbryoHybridV3':
        """
        [FIX-2] Разбивка: 85% диффузия | 7.5% LGB | 7.5% conformal
        Больше данных для CSDI → лучшие маргинальные распределения.
        """
        df_diff, df_rest = train_test_split(
            df, test_size=0.15, random_state=SEED)
        df_lgb, df_conf  = train_test_split(
            df_rest, test_size=0.50, random_state=SEED)

        print(f"[FIT] Диффузия: {len(df_diff)} | "
              f"LGB: {len(df_lgb)} | Conformal: {len(df_conf)}")
        print(f"      P-train: {df_diff['Исход переноса'].mean():.1%}\n")

        # Stage 1
        print("─" * 52)
        print("  Stage 1/3: CSDI-Transformer (COUNT: Bl, good_Bl)")
        print("─" * 52)
        count_med_rest = self._fit_diffusion(df_diff, df_rest)
        count_med_lgb  = count_med_rest[:len(df_lgb)]
        count_med_conf = count_med_rest[len(df_lgb):]

        # Stage 2
        print("\n" + "─" * 52)
        print("  Stage 2/3: LightGBM + Platt (Исход переноса)")
        print("─" * 52)
        self.classifier.fit(df_lgb, count_med_lgb)

        # Stage 3
        print("\n" + "─" * 52)
        print("  Stage 3/3: Conformalization (покрытие PI для COUNT)")
        print("─" * 52)
        actual_conf = df_conf[COUNT_FEATURES].values
        self.conformal.fit(actual_conf, count_med_conf)

        return self

    # ── Генерация для одного пациента ────────────────────────────
    def generate(self, patient: dict,
                 n_samples: int = 2000) -> pd.DataFrame:
        """
        Возвращает DataFrame с OUTPUT_FEATURES (4 столбца):
          Число Bl | Число Bl хор.кач-ва |
          Частота бластоцист | Частота бластоцист хор.кач-ва
        """
        assert self.denoiser is not None, "Сначала вызови fit()"
        cond_arr = np.array([[patient[f] for f in COND_FEATURES]],
                            dtype=np.float32)
        cond_t   = torch.tensor(
            self.normalizer.transform_cond(cond_arr), device=self.device)
        raw      = self.diffusion.ddim_sample(
            self.denoiser, cond_t, n_samples, COUNT_DIM, self.ddim_steps)
        counts   = post_process_counts(
            self.normalizer.inverse_count(raw.cpu().numpy()))
        pn2_val  = np.full(n_samples, float(patient["2 pN"]))
        rates    = derive_rates(counts, pn2_val)
        return pd.DataFrame(
            np.hstack([counts, rates]), columns=OUTPUT_FEATURES)

    # ── MC-сводка ─────────────────────────────────────────────────
    def mc_sample(self, patient: dict, n_samples: int = 2000) -> dict:
        df_gen   = self.generate(patient, n_samples)
        counts   = df_gen[COUNT_FEATURES].values   # [N, 2]
        med_cnt  = np.median(counts, axis=0)

        # LGB prediction
        cond_df = pd.DataFrame([patient])
        p_preg  = float(self.classifier.predict_proba(
            cond_df, med_cnt.reshape(1, -1))[0])

        # Wilson CI
        n_pos, n, z = int(p_preg * 1000), 1000, 1.96
        denom = n + z**2
        ctr   = (n_pos + z**2 / 2) / denom
        mrg   = z * np.sqrt(n_pos * (n - n_pos) / n + z**2 / 4) / denom

        # Conformal PI для COUNT (lo ≥ 0)
        lo90, hi90 = self.conformal.get_intervals(
            med_cnt.reshape(1, -1), level=0.90)
        lo50, hi50 = self.conformal.get_intervals(
            med_cnt.reshape(1, -1), level=0.50)

        return {
            'P_pregnancy':       p_preg,
            'CI_95':             (max(0, ctr - mrg), min(1, ctr + mrg)),
            'blast_total_median': float(np.median(counts[:, 0])),
            'good_blast_median':  float(np.median(counts[:, 1])),
            'blast_rate_median':  float(df_gen[RATE_FEATURES[0]].median()),
            'good_rate_median':   float(df_gen[RATE_FEATURES[1]].median()),
            'PI_90_counts': {
                f: (float(lo90[0, j]), float(hi90[0, j]))
                for j, f in enumerate(COUNT_FEATURES)},
            'PI_50_counts': {
                f: (float(lo50[0, j]), float(hi50[0, j]))
                for j, f in enumerate(COUNT_FEATURES)},
            'samples': df_gen,
        }

    # ── Батчевая генерация для оценки ────────────────────────────
    def generate_for_evaluation(
            self,
            df_eval: pd.DataFrame,
            n_samples_per_patient: int = 200,
            patient_batch_size: int    = 100,
    ) -> tuple:
        """
        Returns:
          count_all: [n_pat, N, COUNT_DIM] — сгенерированные COUNT
          rate_all:  [n_pat, N, 2]         — производные RATE
          p_preg:    [n_pat]               — P(беременность) из LGB
        """
        assert self.denoiser is not None
        Xc   = self.normalizer.transform_cond(df_eval[COND_FEATURES].values)
        pn2  = df_eval["2 pN"].values
        n    = len(df_eval)
        cnt_list, rat_list = [], []
        med_list = []

        print(f"[EVAL-GEN] {n} пациентов × {n_samples_per_patient} образцов...")

        for s in range(0, n, patient_batch_size):
            e    = min(s + patient_batch_size, n)
            B, N = e - s, n_samples_per_patient
            rep  = np.repeat(Xc[s:e], N, axis=0)
            raw  = self.diffusion.ddim_sample(
                self.denoiser,
                torch.tensor(rep, device=self.device),
                B * N, COUNT_DIM, self.ddim_steps)
            cnt  = post_process_counts(
                self.normalizer.inverse_count(raw.cpu().numpy()))
            cnt_r = cnt.reshape(B, N, COUNT_DIM)

            pn2_rep = np.repeat(pn2[s:e], N)
            rat   = derive_rates(cnt, pn2_rep)
            rat_r = rat.reshape(B, N, 2)

            cnt_list.append(cnt_r)
            rat_list.append(rat_r)
            med_list.append(np.median(cnt_r, axis=1))
            print(f"  {e:5d}/{n}", end='\r')

        print()
        count_all = np.concatenate(cnt_list, axis=0)   # [n, N, 2]
        rate_all  = np.concatenate(rat_list, axis=0)   # [n, N, 2]
        medians   = np.vstack(med_list)                # [n, 2]
        p_preg    = self.classifier.predict_proba(df_eval, medians)
        return count_all, rate_all, p_preg

    # ── Сохранение (директория) ───────────────────────────────────
    def save(self, save_dir: str = DEFAULT_SAVE_DIR):
        """
        [FIX-4] Сохраняет все компоненты в директорию.

        embryo_v3_model/
          config.json          — гиперпараметры, имена признаков
          csdi_weights.pt      — state_dict денойзера
          normalizer.pt       — QuantileNormalizer
          lgb_model.txt        — LightGBM бустер (native)
          platt_calibrator.pt — Platt LogisticRegression
          conformal.pt        — ConformalizationLayer
          training_history.json — кривые обучения
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1. Config (json-serializable)
        config = dict(
            version='3.0',
            T=self.T, hidden=self.hidden, n_heads=self.n_heads,
            n_layers=self.n_layers, time_emb_dim=self.time_emb_dim,
            dropout=self.dropout, ddim_steps=self.ddim_steps,
            n_quantiles=self.n_quantiles,
            lgb_n_estimators=self.lgb_n_estimators,
            lgb_learning_rate=self.lgb_learning_rate,
            lgb_num_leaves=self.lgb_num_leaves,
            best_threshold=self.best_threshold,
            cond_features=COND_FEATURES,
            count_features=COUNT_FEATURES,
            rate_features=RATE_FEATURES,
            output_features=OUTPUT_FEATURES,
            pn2_denominator_col="2 pN",
        )
        with open(f'{save_dir}/config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # 2. CSDI weights
        torch.save(self.denoiser.state_dict(),
                   f'{save_dir}/csdi_weights.pt')

        # 3. Normalizer — только numpy-массивы (без зависимости от имени модуля)
        torch.save({
            'n_quantiles':  self.normalizer.n_quantiles,
            'cond_qt':      _serialize_qt(self.normalizer.cond_qt),
            'count_qt':     _serialize_qt(self.normalizer.count_qt),
        }, f'{save_dir}/normalizer.pt')

        # 4. LightGBM — сохраняем через __getstate__ (portable, не зависит от __main__)
        torch.save(self.classifier.clf.__getstate__(), f'{save_dir}/lgb_state.pt')

        # 5. Platt — только веса LogisticRegression
        torch.save({
            'coef_':      self.classifier.calibrator.coef_,
            'intercept_': self.classifier.calibrator.intercept_,
            'classes_':   self.classifier.calibrator.classes_,
        }, f'{save_dir}/platt_calibrator.pt')

        # 6. Conformal — только квантильные радиусы (float → ndarray)
        torch.save(
            {str(k): v for k, v in self.conformal.quantiles.items()},
            f'{save_dir}/conformal.pt')

        # 7. Training history
        with open(f'{save_dir}/training_history.json', 'w') as f:
            json.dump(self.history, f)

        print(f"[SAVE] {save_dir}/")
        for fname in ['config.json', 'csdi_weights.pt', 'normalizer.pt',
                      'lgb_state.pt', 'platt_calibrator.pt',
                      'conformal.pt', 'training_history.json']:
            sz = os.path.getsize(f'{save_dir}/{fname}')
            print(f"  {fname:<28} {sz/1024:7.1f} KB")

    @classmethod
    def load(cls, save_dir: str = DEFAULT_SAVE_DIR) -> 'EmbryoHybridV3':
        """
        Загружает все компоненты из директории.
        Используйте для встраивания в пайплайн:
            model = EmbryoHybridV3.load('embryo_v3_model')
            result = model.mc_sample(patient_dict)
        """
        with open(f'{save_dir}/config.json', encoding='utf-8') as f:
            cfg = json.load(f)

        obj = cls(
            T=cfg['T'], hidden=cfg['hidden'], n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers'], time_emb_dim=cfg['time_emb_dim'],
            dropout=cfg['dropout'], ddim_steps=cfg['ddim_steps'],
            n_quantiles=cfg['n_quantiles'],
            lgb_n_estimators=cfg['lgb_n_estimators'],
            lgb_learning_rate=cfg['lgb_learning_rate'],
            lgb_num_leaves=cfg['lgb_num_leaves'],
        )
        obj.best_threshold = cfg.get('best_threshold', 0.5)

        # CSDI
        obj.diffusion = GaussianDiffusion(T=cfg['T'], device=str(DEVICE))
        obj.denoiser  = CSDIDenoiser(
            count_dim=COUNT_DIM, cond_dim=COND_DIM,
            hidden=cfg['hidden'], n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers'], time_emb_dim=cfg['time_emb_dim'],
            dropout=cfg['dropout'],
        ).to(DEVICE)
        obj.denoiser.load_state_dict(
            torch.load(f'{save_dir}/csdi_weights.pt',
                       map_location=DEVICE, weights_only=True))
        obj.denoiser.eval()

        # Normalizer — реконструируем из numpy-массивов
        norm_state = torch.load(f'{save_dir}/normalizer.pt',
                                map_location='cpu', weights_only=False)
        obj.normalizer = QuantileNormalizer(norm_state['n_quantiles'])
        obj.normalizer.cond_qt  = _deserialize_qt(norm_state['cond_qt'])
        obj.normalizer.count_qt = _deserialize_qt(norm_state['count_qt'])

        # LightGBM — восстанавливаем через __setstate__
        lgb_state = torch.load(f'{save_dir}/lgb_state.pt',
                               map_location='cpu', weights_only=False)
        obj.classifier.clf = lgb.LGBMClassifier()
        obj.classifier.clf.__setstate__(lgb_state)

        platt_state = torch.load(f'{save_dir}/platt_calibrator.pt',
                                 map_location='cpu', weights_only=False)
        lr = LogisticRegression()
        lr.coef_      = np.array(platt_state['coef_'])
        lr.intercept_ = np.array(platt_state['intercept_'])
        lr.classes_   = np.array(platt_state['classes_'])
        obj.classifier.calibrator = lr
        obj.classifier._fitted = True

        # Conformal — реконструируем из квантильных радиусов
        conf_state = torch.load(f'{save_dir}/conformal.pt',
                                map_location='cpu', weights_only=False)
        obj.conformal = ConformalizationLayer()
        obj.conformal.quantiles = {
            float(k): np.array(v) for k, v in conf_state.items()}

        # History
        hist_path = f'{save_dir}/training_history.json'
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                obj.history = json.load(f)

        print(f"[LOAD] {save_dir}/  (threshold={obj.best_threshold:.3f})")
        return obj


# ═══════════════════════════════════════════════════════════
# [7] КОМПЛЕКСНАЯ ОЦЕНКА
# ═══════════════════════════════════════════════════════════

def evaluate_model(
        model: EmbryoHybridV3,
        df_test: pd.DataFrame,
        n_samples_per_patient: int = 200,
        patient_batch_size: int    = 100,
) -> dict:
    """
    Метрики для COUNT и RATE признаков + бинарного исхода.
    """
    print(f"\n{'='*60}")
    print(f"  EVALUATION | test_n={len(df_test)} | "
          f"n_samples={n_samples_per_patient}")
    print(f"{'='*60}")

    count_all, rate_all, p_preg = model.generate_for_evaluation(
        df_test, n_samples_per_patient, patient_batch_size)

    actual_cnt  = df_test[COUNT_FEATURES].values   # [n, 2]
    y_true      = df_test["Исход переноса"].values.astype(int)

    # Актуальные RATE из реальных данных (для сравнения)
    actual_rate = derive_rates(actual_cnt, df_test["2 pN"].values)

    # ── Исход ────────────────────────────────────────────────────
    auroc = roc_auc_score(y_true, p_preg)
    auprc = average_precision_score(y_true, p_preg)
    brier = brier_score_loss(y_true, p_preg)
    ece   = compute_ece(y_true, p_preg)

    y_pred = (p_preg >= 0.5).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())

    fpr, tpr, thresholds = roc_curve(y_true, p_preg)
    f1   = 2 * tpr * (1 - fpr) / (tpr + (1 - fpr) + 1e-9)
    best = float(thresholds[np.argmax(f1)])
    # Сохраняем оптимальный порог в модель
    model.best_threshold = best

    y_opt = (p_preg >= best).astype(int)
    tp2 = int(((y_opt == 1) & (y_true == 1)).sum())
    fn2 = int(((y_opt == 0) & (y_true == 1)).sum())
    tn2 = int(((y_opt == 0) & (y_true == 0)).sum())
    fp2 = int(((y_opt == 1) & (y_true == 0)).sum())

    # ── COUNT-метрики ─────────────────────────────────────────────
    ks_cnt, wass_cnt, c90_cnt, c50_cnt = {}, {}, {}, {}
    mae_cnt, rmse_cnt = {}, {}

    for j, feat in enumerate(COUNT_FEATURES):
        real_v   = actual_cnt[:, j]
        gen_v    = count_all[:, :, j].flatten()
        pred_med = np.median(count_all[:, :, j], axis=1)

        ks_s, ks_p     = stats.ks_2samp(real_v, gen_v)
        ks_cnt[feat]   = {'stat': float(ks_s), 'p': float(ks_p)}
        wass_cnt[feat] = float(wasserstein_distance(real_v, gen_v))
        mae_cnt[feat]  = float(np.abs(pred_med - real_v).mean())
        rmse_cnt[feat] = float(np.sqrt(((pred_med - real_v) ** 2).mean()))

        q90 = float(model.conformal.quantiles.get(0.90, np.zeros(COUNT_DIM))[j])
        q50 = float(model.conformal.quantiles.get(0.50, np.zeros(COUNT_DIM))[j])
        lo90 = np.maximum(0, pred_med - q90)
        hi90 = pred_med + q90
        lo50 = np.maximum(0, pred_med - q50)
        hi50 = pred_med + q50
        c90_cnt[feat] = float(((real_v >= lo90) & (real_v <= hi90)).mean())
        c50_cnt[feat] = float(((real_v >= lo50) & (real_v <= hi50)).mean())

    # ── RATE-метрики ──────────────────────────────────────────────
    ks_rat, wass_rat, mae_rat, rmse_rat = {}, {}, {}, {}
    c90_rat, c50_rat = {}, {}

    for j, feat in enumerate(RATE_FEATURES):
        real_v   = actual_rate[:, j]
        gen_v    = rate_all[:, :, j].flatten()
        pred_med = np.median(rate_all[:, :, j], axis=1)

        ks_s, ks_p     = stats.ks_2samp(real_v, gen_v)
        ks_rat[feat]   = {'stat': float(ks_s), 'p': float(ks_p)}
        wass_rat[feat] = float(wasserstein_distance(real_v, gen_v))
        mae_rat[feat]  = float(np.abs(pred_med - real_v).mean())
        rmse_rat[feat] = float(np.sqrt(((pred_med - real_v) ** 2).mean()))

        # Rate PI — перцентильные (rate ∈ [0,1] по конструкции)
        lo90 = np.quantile(rate_all[:, :, j], 0.05, axis=1)
        hi90 = np.quantile(rate_all[:, :, j], 0.95, axis=1)
        lo50 = np.quantile(rate_all[:, :, j], 0.25, axis=1)
        hi50 = np.quantile(rate_all[:, :, j], 0.75, axis=1)
        c90_rat[feat] = float(((real_v >= lo90) & (real_v <= hi90)).mean())
        c50_rat[feat] = float(((real_v >= lo50) & (real_v <= hi50)).mean())

    return {
        'AUROC': auroc, 'AUPRC': auprc, 'Brier': brier, 'ECE': ece,
        'Sensitivity': tp / max(tp + fn, 1),
        'Specificity': tn / max(tn + fp, 1),
        'Best_threshold': best,
        'Sensitivity_opt': tp2 / max(tp2 + fn2, 1),
        'Specificity_opt': tn2 / max(tn2 + fp2, 1),
        'Pred_prev':   float(p_preg.mean()),
        'Actual_prev': float(y_true.mean()),
        # COUNT
        'KS_count': ks_cnt, 'Wasserstein_count': wass_cnt,
        'Coverage_90_count': c90_cnt, 'Coverage_50_count': c50_cnt,
        'MAE_count': mae_cnt, 'RMSE_count': rmse_cnt,
        # RATE
        'KS_rate': ks_rat, 'Wasserstein_rate': wass_rat,
        'Coverage_90_rate': c90_rat, 'Coverage_50_rate': c50_rat,
        'MAE_rate': mae_rat, 'RMSE_rate': rmse_rat,
        # Raw
        '_p_preg': p_preg, '_y_true': y_true,
        '_count_all': count_all, '_rate_all': rate_all,
        '_actual_cnt': actual_cnt, '_actual_rate': actual_rate,
    }


def print_metrics(metrics: dict):
    print(f"\n{'='*70}")
    print(f"  РЕЗУЛЬТАТЫ — Hybrid CSDI + LightGBM v3  (COUNT-only generation)")
    print(f"{'='*70}")

    print(f"\n── ИСХОД БЕРЕМЕННОСТИ {'─'*47}")
    print(f"  AUROC:                  {metrics['AUROC']:.4f}")
    print(f"  AUPRC:                  {metrics['AUPRC']:.4f}")
    print(f"  Brier Score:            {metrics['Brier']:.4f}")
    print(f"  ECE:                    {metrics['ECE']:.4f}")
    print(f"  Sensitivity @0.50:      {metrics['Sensitivity']:.3f}")
    print(f"  Specificity @0.50:      {metrics['Specificity']:.3f}")
    print(f"  Оптим. порог:           {metrics['Best_threshold']:.3f}")
    print(f"  Sensitivity @opt:       {metrics['Sensitivity_opt']:.3f}")
    print(f"  Specificity @opt:       {metrics['Specificity_opt']:.3f}")
    print(f"  Предсказанная P:        {metrics['Pred_prev']:.1%}")
    print(f"  Фактическая P:          {metrics['Actual_prev']:.1%}")
    print(f"  Δ (смещение):           {metrics['Pred_prev'] - metrics['Actual_prev']:+.1%}")

    def _block(title, ks_d, wass_d, c90_d, c50_d, mae_d, rmse_d, feats):
        print(f"\n── {title} {'─'*(65-len(title))}")
        hdr = (f"  {'Признак':<46} {'KS':>5} {'Wass':>6} "
               f"{'C90':>6} {'C50':>6} {'MAE':>6} {'RMSE':>6}")
        print(hdr)
        print("  " + "─" * 84)
        for feat in feats:
            ok  = "✓" if ks_d[feat]['p'] > 0.05 else "✗"
            print(f"  {feat:<46} {ks_d[feat]['stat']:>5.3f} "
                  f"{wass_d[feat]:>6.3f} {c90_d[feat]:>5.1%} "
                  f"{c50_d[feat]:>5.1%} {mae_d[feat]:>6.3f} "
                  f"{rmse_d[feat]:>6.3f}  {ok}")

    _block("СЧЁТНЫЕ ПРИЗНАКИ (CSDI direct)",
           metrics['KS_count'], metrics['Wasserstein_count'],
           metrics['Coverage_90_count'], metrics['Coverage_50_count'],
           metrics['MAE_count'], metrics['RMSE_count'], COUNT_FEATURES)

    _block("ПРОИЗВОДНЫЕ ЧАСТОТЫ (аналитически из COUNT + 2pN)",
           metrics['KS_rate'], metrics['Wasserstein_rate'],
           metrics['Coverage_90_rate'], metrics['Coverage_50_rate'],
           metrics['MAE_rate'], metrics['RMSE_rate'], RATE_FEATURES)
    print()


def save_metrics_json(metrics: dict,
                      path: str = 'hybrid_v3_metrics.json'):
    saveable = {k: v for k, v in metrics.items() if not k.startswith('_')}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(saveable, f, indent=2, ensure_ascii=False)
    print(f"[METRICS] {path}")


# ═══════════════════════════════════════════════════════════
# [8] ВИЗУАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════

def plot_training_curves(history: dict,
                         path: str = 'hybrid_v3_training.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Hybrid CSDI+LGB v3 — Кривые обучения CSDI (COUNT-only)',
                 fontsize=11, fontweight='bold')
    epochs = range(1, len(history['loss']) + 1)
    for ax, scale in zip(axes, ['linear', 'log']):
        ax.plot(epochs, history['loss'],     '#2196F3', lw=1.5, label='Train')
        ax.plot(epochs, history['val_loss'], '#F44336', lw=1.5,
                ls='--', label='Val')
        if scale == 'log': ax.set_yscale('log')
        ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
        ax.legend(); ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] {path}")


def plot_evaluation(metrics: dict,
                    path: str = 'hybrid_v3_evaluation.png'):
    p_preg     = metrics['_p_preg']
    y_true     = metrics['_y_true']
    count_all  = metrics['_count_all']
    rate_all   = metrics['_rate_all']
    act_cnt    = metrics['_actual_cnt']
    act_rate   = metrics['_actual_rate']

    ALL_FEAT_LABELS = {
        "Число Bl":                  "Бластоцисты (всего)",
        "Число Bl хор.кач-ва":       "Бластоцисты хор. кач-ва",
        "Частота формирования бластоцист":                   "Частота бластоцист",
        "Частота формирования бластоцист хорошего качества": "Частота бласт. хор. кач.",
    }

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Hybrid CSDI + LightGBM v3 — Evaluation Report',
                 fontsize=13, fontweight='bold')
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.38)

    # ── Ряд 1: исход ─────────────────────────────────────────────

    # ROC
    ax = fig.add_subplot(gs[0, 0])
    fpr, tpr, thr = roc_curve(y_true, p_preg)
    ax.plot(fpr, tpr, '#2196F3', lw=2,
            label=f'AUC = {metrics["AUROC"]:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.fill_between(fpr, tpr, alpha=0.08, color='#2196F3')
    best = metrics['Best_threshold']
    idx  = np.searchsorted(thr[::-1], best)
    ax.plot(fpr[-idx], tpr[-idx], 'r*', ms=10,
            label=f'Opt @ {best:.2f}')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curve', fontweight='bold')
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Калибровка
    ax = fig.add_subplot(gs[0, 1])
    bns = np.linspace(0, 1, 11)
    bp, bt = [], []
    for i in range(10):
        m = (p_preg >= bns[i]) & (p_preg < bns[i+1])
        if m.sum() >= 5:
            bp.append(p_preg[m].mean()); bt.append(y_true[m].mean())
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.35, label='Идеальная')
    ax.plot(bp, bt, 'o-', color='#F44336', lw=2, markersize=5,
            label='Модель')
    ax.fill_between(bp, bp, bt, alpha=0.1, color='#F44336')
    ax.set_xlabel('Предсказанная P'); ax.set_ylabel('Фактическая')
    ax.set_title(f'Калибровка  ECE={metrics["ECE"]:.3f}',
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Скоры
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(p_preg[y_true==0], bins=25, alpha=0.6, color='#F44336',
            density=True, label='Нет', edgecolor='white', lw=0.4)
    ax.hist(p_preg[y_true==1], bins=25, alpha=0.6, color='#2196F3',
            density=True, label='Да', edgecolor='white', lw=0.4)
    ax.axvline(0.5, color='k', ls=':', alpha=0.5)
    ax.axvline(best, color='orange', ls='--', alpha=0.7,
               label=f'Opt={best:.2f}')
    ax.set_xlabel('P(беременность)')
    ax.set_title('Скор по исходу', fontweight='bold')
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Метрики bar
    ax = fig.add_subplot(gs[0, 3])
    keys = ['AUROC','AUPRC','Brier','ECE']
    bars = ax.barh(keys, [metrics[k] for k in keys],
                   color=['#2196F3','#4CAF50','#F44336','#FF9800'], alpha=0.8)
    for bar, val in zip(bars, [metrics[k] for k in keys]):
        ax.text(val+0.005, bar.get_y()+bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.set_title('Метрики исхода', fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # ── Ряд 2: COUNT гистограммы (прямая генерация) + RATE ───────
    all_gen_data = [
        (count_all[:,:,0], act_cnt[:,0], COUNT_FEATURES[0]),
        (count_all[:,:,1], act_cnt[:,1], COUNT_FEATURES[1]),
        (rate_all[:,:,0],  act_rate[:,0], RATE_FEATURES[0]),
        (rate_all[:,:,1],  act_rate[:,1], RATE_FEATURES[1]),
    ]
    for i, (gen_s, real_v, feat) in enumerate(all_gen_data):
        ax  = fig.add_subplot(gs[1, i])
        gen_v = gen_s.flatten()
        ax.hist(real_v, bins=20, alpha=0.5, color='#6c757d',
                density=True, label='Real', edgecolor='white', lw=0.4)
        ax.hist(gen_v, bins=20, alpha=0.6, color='#2196F3',
                density=True, label='Gen', edgecolor='white', lw=0.4)
        ks_d = (metrics['KS_count'] if feat in COUNT_FEATURES
                else metrics['KS_rate'])
        ks_s = ks_d[feat]['stat']; ks_p = ks_d[feat]['p']
        clr  = '#2e7d32' if ks_p > 0.05 else '#c62828'
        tag  = '▶ прямая' if feat in COUNT_FEATURES else '⇒ производная'
        ax.set_title(f"{ALL_FEAT_LABELS[feat]}\n{tag}",
                     fontsize=8, fontweight='bold')
        ax.set_xlabel(f'KS={ks_s:.3f}  p={ks_p:.3f}',
                      fontsize=7, color=clr)
        ax.legend(fontsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ── Ряд 3: coverage + MAE + Wass ─────────────────────────────

    # Coverage-калибровочная кривая (COUNT)
    ax  = fig.add_subplot(gs[2, 0])
    qls = [0.50, 0.60, 0.70, 0.80, 0.90]
    for j, feat in enumerate(COUNT_FEATURES):
        col = ['#2196F3', '#4CAF50'][j]
        cov = []
        for ql in qls:
            lo = np.quantile(count_all[:,:,j], (1-ql)/2, axis=1)
            hi = np.quantile(count_all[:,:,j], 1-(1-ql)/2, axis=1)
            cov.append(((act_cnt[:,j] >= lo) & (act_cnt[:,j] <= hi)).mean())
        ax.plot(qls, cov, 'o-', color=col, lw=1.5, markersize=4,
                label=feat[:10])
    ax.plot([0,1],[0,1],'k--', alpha=0.3, label='Идеальная')
    ax.set_xlabel('Номинальное'); ax.set_ylabel('Фактическое')
    ax.set_title('Coverage-калибровка\n(COUNT, CSDI прямая)', fontweight='bold')
    ax.legend(fontsize=6)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Coverage-калибровочная кривая (RATE)
    ax  = fig.add_subplot(gs[2, 1])
    for j, feat in enumerate(RATE_FEATURES):
        col = ['#FF9800', '#9C27B0'][j]
        cov = []
        for ql in qls:
            lo = np.quantile(rate_all[:,:,j], (1-ql)/2, axis=1)
            hi = np.quantile(rate_all[:,:,j], 1-(1-ql)/2, axis=1)
            cov.append(((act_rate[:,j] >= lo) & (act_rate[:,j] <= hi)).mean())
        ax.plot(qls, cov, 'o-', color=col, lw=1.5, markersize=4,
                label=feat[:12])
    ax.plot([0,1],[0,1],'k--', alpha=0.3, label='Идеальная')
    ax.set_xlabel('Номинальное'); ax.set_ylabel('Фактическое')
    ax.set_title('Coverage-калибровка\n(RATE, производные)', fontweight='bold')
    ax.legend(fontsize=6)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # MAE (все 4)
    ax    = fig.add_subplot(gs[2, 2])
    lbls  = ['Bl', 'gBl', 'BRate', 'GRate']
    mae_v = ([metrics['MAE_count'][f] for f in COUNT_FEATURES] +
             [metrics['MAE_rate'][f]  for f in RATE_FEATURES])
    clrs2 = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    bars  = ax.bar(lbls, mae_v, color=clrs2, alpha=0.8)
    for bar, val in zip(bars, mae_v):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+max(mae_v)*0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('MAE')
    ax.set_title('MAE по всем признакам\n(▶прямая / ⇒производная)',
                 fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Wasserstein
    ax    = fig.add_subplot(gs[2, 3])
    wv    = ([metrics['Wasserstein_count'][f] for f in COUNT_FEATURES] +
             [metrics['Wasserstein_rate'][f]  for f in RATE_FEATURES])
    bars  = ax.bar(lbls, wv, color=clrs2, alpha=0.8)
    for bar, val in zip(bars, wv):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+max(wv)*0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Wasserstein')
    ax.set_title('Расстояние Вассерштейна', fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] {path}")


def compare_all_versions(paths_and_labels: list,
                         out_path: str = 'comparison_all.png'):
    """
    Сравнение произвольного числа версий по их JSON-метрикам.
    paths_and_labels: [(json_path, label), ...]
    """
    loaded = []
    for p, lbl in paths_and_labels:
        if os.path.exists(p):
            with open(p) as f:
                loaded.append((json.load(f), lbl))
    if len(loaded) < 2:
        print("[COMPARE] Нужно ≥ 2 JSON"); return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Сравнение версий моделей', fontsize=12, fontweight='bold')
    clrs = ['#78909C', '#2196F3', '#4CAF50', '#FF9800']

    # 1. Метрики исхода
    ax   = axes[0]
    keys = ['AUROC', 'AUPRC', 'Brier', 'ECE']
    x    = np.arange(len(keys))
    w    = 0.8 / len(loaded)
    for i, (m, lbl) in enumerate(loaded):
        vals = []
        for k in keys:
            vals.append(m.get(k, m.get(k.upper(), 0)))
        ax.bar(x + i*w - 0.4 + w/2, vals, w, label=lbl,
               color=clrs[i % len(clrs)], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(keys)
    ax.set_title('Метрики исхода'); ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # 2. KS COUNT
    ax = axes[1]
    cnt_feats = COUNT_FEATURES
    x  = np.arange(len(cnt_feats))
    lbl_s = ['Bl', 'gBl']
    for i, (m, lbl) in enumerate(loaded):
        ks_key = 'KS_count' if 'KS_count' in m else 'KS'
        vals   = [m.get(ks_key, {}).get(f, {}).get('stat', 0)
                  for f in cnt_feats]
        ax.bar(x + i*w - 0.4 + w/2, vals, w, label=lbl,
               color=clrs[i % len(clrs)], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(lbl_s)
    ax.set_title('KS (COUNT, ниже = лучше)'); ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # 3. ECE + Смещение
    ax = axes[2]
    lbls3 = ['ECE', '|Δ P|']
    x     = np.arange(len(lbls3))
    for i, (m, lbl) in enumerate(loaded):
        bias = abs(m.get('Pred_prev', 0) - m.get('Actual_prev', 0))
        vals = [m.get('ECE', 0), bias]
        ax.bar(x + i*w - 0.4 + w/2, vals, w, label=lbl,
               color=clrs[i % len(clrs)], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(lbls3)
    ax.set_title('Калибровка (ниже = лучше)'); ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[COMPARE] {out_path}")


# ═══════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("  EMBRYO Hybrid CSDI + LightGBM v3")
    print("  Stage-1: CSDI — COUNT only (Bl, good_Bl)")
    print("  Stage-2: derive_rates(Bl, good_Bl, 2pN)")
    print("  Stage-3: LightGBM + Platt → P(pregnancy)")
    print("  Stage-4: ConformalizationLayer → PI ≥ 0")
    print("=" * 65)

    # ── 1. Данные ───────────────────────────────────────────────
    df = load_data('all_df_with_KPI.xlsx')
    df_train, df_test = train_test_split(
        df, test_size=0.10, random_state=SEED)
    print(f"[SPLIT] Train: {len(df_train)} | Test: {len(df_test)}")
    print(f"        Train P: {df_train['Исход переноса'].mean():.1%} | "
          f"Test P: {df_test['Исход переноса'].mean():.1%}")

    # ── 2. Обучение ─────────────────────────────────────────────
    model = EmbryoHybridV3(
        T              = 1000,
        hidden         = 128,
        n_heads        = 4,
        n_layers       = 6,
        time_emb_dim   = 64,
        dropout        = 0.1,
        lr             = 3e-4,
        epochs         = 200,
        batch_size     = 128,
        warmup_epochs  = 15,
        ddim_steps     = 50,
        n_quantiles    = 1000,
        lgb_n_estimators   = 600,
        lgb_learning_rate  = 0.04,
        lgb_num_leaves     = 31,
    )
    model.fit(df_train)
    plot_training_curves(model.history)

    # ── 3. Пример пациента ──────────────────────────────────────
    patient_example = {
        "Количество фолликулов":  12,
        "Число ОКК":              9,
        "Число инсеминированных": 8,
        "2 pN":                   6,
        "Частота получения ОКК":  0.75,
        "Частота оплодотворения": 0.75,
        "KPIScore":               18,
    }

    print("\n[MC SAMPLE] Генерация 2000 траекторий ...")
    result = model.mc_sample(patient_example, n_samples=2000)

    print(f"\n{'='*58}")
    print(f"  Вероятность беременности: {result['P_pregnancy']:.1%}")
    print(f"  95% CI:                   "
          f"{result['CI_95'][0]:.1%} – {result['CI_95'][1]:.1%}")
    print(f"  Бластоцист всего (мед):   {result['blast_total_median']:.0f}")
    print(f"  Бластоцист хор.кач (мед): {result['good_blast_median']:.0f}")
    print(f"  Частота бластоцист:        {result['blast_rate_median']:.1%}")
    print(f"  TGBDR:                     {result['good_rate_median']:.1%}")
    print(f"\n  90% PI (COUNT, конформальные, lo ≥ 0):")
    for feat, (lo, hi) in result['PI_90_counts'].items():
        print(f"    {feat:<40} [{lo:.0f}, {hi:.0f}]")

    # ── 4. Оценка ───────────────────────────────────────────────
    metrics = evaluate_model(
        model, df_test,
        n_samples_per_patient = 200,
        patient_batch_size    = 100,
    )
    print_metrics(metrics)
    save_metrics_json(metrics)
    plot_evaluation(metrics)

    # Сохраняем оптимальный порог в модель и перезаписываем
    model.best_threshold = metrics['Best_threshold']

    # ── 5. Сохранение всех компонентов ──────────────────────────
    model.save('embryo_v3_model')

    # ── 6. Сравнение всех версий ────────────────────────────────
    compare_all_versions([
        ('csdi_v1_metrics.json',   'v1 CSDI (pure diff)'),
        ('hybrid_v2_metrics.json', 'v2 CSDI+LGB'),
        ('hybrid_v3_metrics.json', 'v3 COUNT-only'),
    ])

    print("\n[DONE] Все файлы сохранены:")
    print("  embryo_v3_model/")
    print("    config.json          — гиперпараметры + имена признаков")
    print("    csdi_weights.pt      — CSDI нейросеть")
    print("    normalizer.pt       — QuantileNormalizer")
    print("    lgb_model.txt        — LightGBM (native)")
    print("    platt_calibrator.pt — Platt scaling")
    print("    conformal.pt        — PI-радиусы")
    print("    training_history.json")
    print("  hybrid_v3_metrics.json")
    print("  hybrid_v3_training.png")
    print("  hybrid_v3_evaluation.png")
    print("  comparison_all.png")

    # ── 7. Пример загрузки (для пайплайна) ──────────────────────
    print("\n" + "─" * 55)
    print("  ПРИМЕР ЗАГРУЗКИ ДЛЯ ПАЙПЛАЙНА")
    print("─" * 55)
    loaded = EmbryoHybridV3.load('embryo_v3_model')
    r2     = loaded.mc_sample(patient_example, n_samples=500)
    print(f"  Загружено. P(pregnancy) = {r2['P_pregnancy']:.1%}  "
          f"(ожидалось ≈ {result['P_pregnancy']:.1%})")
    print(f"  Рекомендуемый порог: {loaded.best_threshold:.3f}")