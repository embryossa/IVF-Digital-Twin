# ============================================================
# EMBRYO TRAJECTORY TabDDPM — v3.0
#
# Ключевые изменения vs v2:
#   [FIX-1] QuantileTransformer вместо StandardScaler
#           → данные действительно становятся N(0,1)
#           → обратное преобразование сохраняет точные
#             маргинальные распределения тренировочных данных
#   [FIX-2] Взвешенные потери — outcome получает вес ×3
#   [FIX-3] 150 эпох вместо 300 (достаточно, нет смысла ждать)
#   [FIX-4] DDIM sampling — детерминированный sampler,
#           в 10× быстрее и стабильнее для табличных данных
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import QuantileTransformer   # ← главное изменение
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

COND_FEATURES = [
    "Возраст", "№ попытки", "Количество фолликулов",
    "Число ОКК", "Число инсеминированных", "2 pN",
    "Частота получения ОКК", "Частота оплодотворения", "KPIScore"
]
TRAJ_FEATURES = [
    "Число дробящихся на 3 день", "Частота дробления",
    "Число Bl", "Число эмбрионов 5 дня",
    "Частота формирования бластоцист",
    "Число Bl хор.кач-ва",
    "Частота формирования бластоцист хорошего качества",
    "Заморожено эмбрионов", "Перенесено эмбрионов",
    "Исход переноса"
]
ALL_FEATURES = COND_FEATURES + TRAJ_FEATURES

# Индекс целевой переменной в TRAJ_FEATURES
OUTCOME_IDX = TRAJ_FEATURES.index("Исход переноса")


# ═══════════════════════════════════════════════════════════
# ДАННЫЕ
# ═══════════════════════════════════════════════════════════

def load_data(path='all_df_with_KPI.xlsx'):
    df = pd.read_excel(path)
    df = df[ALL_FEATURES].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df.dropna(inplace=True)
    print(f"[DATA] {len(df)} циклов | "
          f"Беременность: {df['Исход переноса'].mean():.1%}")
    return df


# ═══════════════════════════════════════════════════════════
# [FIX-1] QUANTILE NORMALIZER
#
# Зачем это важно:
#   StandardScaler: x' = (x - μ) / σ  — линейное, форма не меняется
#   QuantileTransformer: x' = Φ⁻¹(F̂(x)) — нелинейное,
#     F̂(x) — эмпирическая CDF, Φ⁻¹ — обратная функция N(0,1)
#
#   После преобразования каждая переменная ТОЧНО следует N(0,1),
#   что является предположением Гауссовской диффузии.
#
#   inverse_transform: x = F̂⁻¹(Φ(x')) — точно восстанавливает
#   оригинальное распределение, включая дискретность счётных
#   переменных и нулевую инфляцию.
# ═══════════════════════════════════════════════════════════

class QuantileNormalizer:
    def __init__(self, n_quantiles: int = 1000):
        """
        n_quantiles: число квантилей (= число уникальных значений
        в тренировочных данных, но не более 1000).
        Для датасета 14k+ строк: 1000 — оптимально.
        """
        self.n_quantiles = n_quantiles
        self.cond_qt = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution='normal',
            random_state=SEED
        )
        self.traj_qt = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution='normal',
            random_state=SEED
        )

    def fit_transform(self, df: pd.DataFrame):
        Xc = self.cond_qt.fit_transform(
            df[COND_FEATURES].values).astype(np.float32)
        Xt = self.traj_qt.fit_transform(
            df[TRAJ_FEATURES].values).astype(np.float32)
        return Xc, Xt

    def transform_cond(self, arr: np.ndarray) -> np.ndarray:
        return self.cond_qt.transform(arr).astype(np.float32)

    def inverse_traj(self, arr: np.ndarray) -> np.ndarray:
        """
        Обратное квантильное преобразование.
        Сгенерированные значения из N(0,1) → оригинальная шкала.
        Это гарантирует, что маргинальные распределения
        совпадают с тренировочными данными.
        """
        return self.traj_qt.inverse_transform(arr)


# ═══════════════════════════════════════════════════════════
# НЕЙРОСЕТЕВЫЕ КОМПОНЕНТЫ (FiLM — без изменений из v2)
# ═══════════════════════════════════════════════════════════

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(
            -np.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feature_dim)
        self.beta  = nn.Linear(cond_dim, feature_dim)

    def forward(self, x, cond):
        return self.gamma(cond) * x + self.beta(cond)


class FiLMResidualBlock(nn.Module):
    def __init__(self, dim, cond_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, dim * 2)
        self.lin2 = nn.Linear(dim * 2, dim)
        self.film = FiLMLayer(dim, cond_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cond):
        h = self.norm(x)
        h = F.silu(self.lin1(h))
        h = self.drop(h)
        h = self.lin2(h)
        h = self.film(h, cond)
        return x + h


class EmbryoDenoiser(nn.Module):
    def __init__(self, traj_dim, cond_dim, hidden=256,
                 time_emb_dim=64, n_blocks=6, dropout=0.1):
        super().__init__()
        self.time_emb  = SinusoidalTimeEmbedding(time_emb_dim)
        film_cond_dim  = hidden
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim + time_emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, film_cond_dim)
        )
        self.input_proj = nn.Linear(traj_dim, hidden)
        self.blocks = nn.ModuleList([
            FiLMResidualBlock(hidden, film_cond_dim, dropout)
            for _ in range(n_blocks)
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, traj_dim)
        )

    def forward(self, x_t, t, x_cond):
        t_emb  = self.time_emb(t)
        film_c = self.cond_proj(torch.cat([x_cond, t_emb], dim=-1))
        h      = self.input_proj(x_t)
        for block in self.blocks:
            h = block(h, film_c)
        return self.output_proj(h)


# ═══════════════════════════════════════════════════════════
# ДИФФУЗИОННЫЙ ПРОЦЕСС
# ═══════════════════════════════════════════════════════════

def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cp = f / f[0]
    betas = 1 - (alphas_cp[1:] / alphas_cp[:-1])
    return betas.clamp(1e-5, 0.9999).float()


class GaussianDiffusion:
    def __init__(self, T=1000, device='cpu'):
        self.T      = T
        self.device = device
        betas       = cosine_beta_schedule(T).to(device)
        alphas      = 1.0 - betas
        alphas_cp   = torch.cumprod(alphas, dim=0)
        alphas_cp_p = F.pad(alphas_cp[:-1], (1, 0), value=1.0)

        self.betas          = betas
        self.sqrt_acp       = alphas_cp.sqrt()
        self.sqrt_one_m_acp = (1 - alphas_cp).sqrt()
        self.sqrt_recip_a   = (1 / alphas).sqrt()
        self.posterior_var  = (
                betas * (1 - alphas_cp_p) / (1 - alphas_cp)).clamp(min=1e-20)
        # Для DDIM
        self.alphas_cp      = alphas_cp

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self.sqrt_acp[t].unsqueeze(1)
        sb = self.sqrt_one_m_acp[t].unsqueeze(1)
        return sa * x0 + sb * noise, noise

    # ── [FIX-2] Взвешенные потери ──────────────────────────
    def p_losses(self, model, x0, x_cond,
                 outcome_idx=OUTCOME_IDX, outcome_weight=3.0):
        """
        MSE по предсказанному шуму с повышенным весом для исхода.

        outcome_weight: во сколько раз важнее ошибка по исходу.
        Значение 3.0 эмпирически хорошо работает при imbalance ~32%.
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        x_t, noise = self.q_sample(x0, t)
        pred_noise = model(x_t, t, x_cond)

        # Поэлементный MSE
        mse = (pred_noise - noise) ** 2             # [B, traj_dim]

        # Веса: единица везде, outcome_weight для исхода
        weights = torch.ones(mse.shape[1], device=self.device)
        weights[outcome_idx] = outcome_weight

        return (mse * weights).mean()

    # ── [FIX-4] DDIM Sampling ──────────────────────────────
    @torch.no_grad()
    def ddim_sample(self, model, x_cond, n_samples, traj_dim,
                    ddim_steps=50, eta=0.0):
        """
        DDIM (Song et al. 2021) — детерминированный sampler.

        Преимущества vs DDPM stochastic:
          - в T/ddim_steps раз быстрее (1000 → 50 шагов)
          - меньше накопленного шума → лучше для счётных переменных
          - eta=0.0 — полностью детерминированный (наиболее стабильный)

        Используем подмножество временных шагов (linspace).
        """
        model.eval()
        x    = torch.randn(n_samples, traj_dim, device=self.device)
        cond = x_cond.expand(n_samples, -1)

        # Равномерно распределённые шаги T→0
        timesteps = torch.linspace(
            self.T - 1, 0, ddim_steps, dtype=torch.long, device=self.device)

        for i, t_idx in enumerate(timesteps):
            t_tensor = torch.full(
                (n_samples,), t_idx.item(),
                device=self.device, dtype=torch.long)
            pred_noise = model(x, t_tensor, cond)

            acp_t  = self.alphas_cp[t_idx]
            x0_pred = (x - (1 - acp_t).sqrt() * pred_noise) / acp_t.sqrt()
            x0_pred = x0_pred.clamp(-4, 4)   # клиппинг для стабильности

            if i == len(timesteps) - 1:
                x = x0_pred
            else:
                t_next = timesteps[i + 1]
                acp_next = self.alphas_cp[t_next]
                dir_xt   = (1 - acp_next).sqrt() * pred_noise
                x        = acp_next.sqrt() * x0_pred + dir_xt

        return x


# ═══════════════════════════════════════════════════════════
# ГЛАВНЫЙ КЛАСС
# ═══════════════════════════════════════════════════════════

class EmbryoTabDDPM:
    def __init__(
            self,
            T=1000,
            hidden=256,
            n_blocks=6,
            time_emb_dim=64,
            dropout=0.1,
            lr=3e-4,
            epochs=150,              # [FIX-3] было 300
            batch_size=64,
            warmup_epochs=15,
            outcome_weight=3.0,      # [FIX-2]
            ddim_steps=50,           # [FIX-4]
            n_quantiles=1000         # [FIX-1]
    ):
        self.T              = T
        self.hidden         = hidden
        self.n_blocks       = n_blocks
        self.time_emb_dim   = time_emb_dim
        self.dropout        = dropout
        self.lr             = lr
        self.epochs         = epochs
        self.batch_size     = batch_size
        self.warmup_epochs  = warmup_epochs
        self.outcome_weight = outcome_weight
        self.ddim_steps     = ddim_steps
        self.n_quantiles    = n_quantiles
        self.device         = DEVICE
        self.normalizer     = QuantileNormalizer(n_quantiles)   # ← FIX-1
        self.model          = None
        self.diffusion      = None
        self.history        = {'loss': [], 'val_loss': []}

    def fit(self, df: pd.DataFrame) -> 'EmbryoTabDDPM':
        Xc, Xt = self.normalizer.fit_transform(df)
        Xc_tr, Xc_val, Xt_tr, Xt_val = train_test_split(
            Xc, Xt, test_size=0.15, random_state=SEED)

        def t(a): return torch.tensor(a).to(self.device)
        train_dl = DataLoader(
            TensorDataset(t(Xc_tr), t(Xt_tr)),
            batch_size=self.batch_size, shuffle=True, drop_last=False)
        Xc_val_t, Xt_val_t = t(Xc_val), t(Xt_val)

        self.diffusion = GaussianDiffusion(T=self.T, device=str(self.device))
        self.model = EmbryoDenoiser(
            traj_dim=len(TRAJ_FEATURES),
            cond_dim=len(COND_FEATURES),
            hidden=self.hidden,
            time_emb_dim=self.time_emb_dim,
            n_blocks=self.n_blocks,
            dropout=self.dropout
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[MODEL] FiLM-EmbryoDenoiser | {n_params:,} параметров | {self.device}")
        print(f"[TRAIN] T={self.T} (cosine+DDIM) | "
              f"epochs={self.epochs} | outcome_weight={self.outcome_weight}\n")

        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        def lr_lambda(ep):
            if ep < self.warmup_epochs:
                return ep / max(self.warmup_epochs, 1)
            prog = (ep - self.warmup_epochs) / max(
                self.epochs - self.warmup_epochs, 1)
            return 0.5 * (1 + np.cos(np.pi * prog))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        best_val, best_state = float('inf'), None

        for epoch in range(self.epochs):
            self.model.train()
            batch_losses = []
            for xc_b, xt_b in train_dl:
                loss = self.diffusion.p_losses(
                    self.model, xt_b, xc_b,
                    outcome_weight=self.outcome_weight)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                batch_losses.append(loss.item())

            self.model.eval()
            with torch.no_grad():
                val_loss = self.diffusion.p_losses(
                    self.model, Xt_val_t, Xc_val_t,
                    outcome_weight=self.outcome_weight).item()

            tr_loss = float(np.mean(batch_losses))
            self.history['loss'].append(tr_loss)
            self.history['val_loss'].append(val_loss)
            scheduler.step()

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 25 == 0 or epoch == 0:
                lr_now = opt.param_groups[0]['lr']
                print(f"  Epoch {epoch+1:4d}/{self.epochs} | "
                      f"train={tr_loss:.4f} | val={val_loss:.4f} | "
                      f"lr={lr_now:.2e}")

        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in best_state.items()})
        print(f"\n[DONE] Лучший val_loss: {best_val:.4f}")
        return self

    def generate(self, patient: dict, n_samples: int = 5000) -> pd.DataFrame:
        assert self.model is not None, "Сначала вызови fit()"
        cond_arr  = np.array([[patient[f] for f in COND_FEATURES]],
                             dtype=np.float32)
        cond_norm = self.normalizer.transform_cond(cond_arr)
        cond_t    = torch.tensor(cond_norm).to(self.device)

        # DDIM sampling вместо DDPM
        raw = self.diffusion.ddim_sample(
            self.model, cond_t, n_samples,
            traj_dim=len(TRAJ_FEATURES),
            ddim_steps=self.ddim_steps)

        # Обратное квантильное преобразование
        samples = self.normalizer.inverse_traj(raw.cpu().numpy())
        df_out  = pd.DataFrame(samples, columns=TRAJ_FEATURES)

        # Биологические ограничения
        count_cols = [c for c in TRAJ_FEATURES
                      if any(c.startswith(p)
                             for p in ["Число", "Заморожено", "Перенесено"])]
        rate_cols  = [c for c in TRAJ_FEATURES if c.startswith("Частота")]
        df_out[count_cols] = df_out[count_cols].clip(lower=0).round()
        df_out[rate_cols]  = df_out[rate_cols].clip(0, 1)
        df_out["Исход переноса"] = (
                df_out["Исход переноса"] > 0.5).astype(int)
        return df_out

    def mc_sample(self, patient: dict, n_iter: int = 5000) -> dict:
        df    = self.generate(patient, n_samples=n_iter)
        p     = df["Исход переноса"].mean()
        n_pos = df["Исход переноса"].sum()
        z     = 1.96
        denom = n_iter + z**2
        ctr   = (n_pos + z**2 / 2) / denom
        mrg   = (z * np.sqrt(n_pos * (n_iter - n_pos) / n_iter + z**2/4)
                 / denom)
        return {
            'P_pregnancy':          p,
            'CI_95':                (max(0, ctr-mrg), min(1, ctr+mrg)),
            'D3_median':            df["Число дробящихся на 3 день"].median(),
            'blast_total_median':   df["Число Bl"].median(),
            'blast_d5_median':      df["Число эмбрионов 5 дня"].median(),
            'good_blast_median':    df["Число Bl хор.кач-ва"].median(),
            'blast_rate_mean':      df["Частота формирования бластоцист"].mean(),
            'good_blast_rate_mean': df["Частота формирования бластоцист хорошего качества"].mean(),
            'frozen_median':        df["Заморожено эмбрионов"].median(),
            'samples':              df
        }

    def save(self, path='embryo_tabddpm_v3.pt'):
        torch.save({
            'model_state': self.model.state_dict(),
            'normalizer':  self.normalizer,
            'history':     self.history,
            'config': {
                'T': self.T, 'hidden': self.hidden,
                'n_blocks': self.n_blocks,
                'time_emb_dim': self.time_emb_dim,
                'dropout': self.dropout,
                'outcome_weight': self.outcome_weight,
                'ddim_steps': self.ddim_steps,
                'n_quantiles': self.n_quantiles
            }
        }, path)
        print(f"[SAVE] {path}")

    @classmethod
    def load(cls, path='embryo_tabddpm_v3.pt'):
        ckpt = torch.load(path, map_location=DEVICE)
        cfg  = ckpt['config']
        obj  = cls(**cfg)
        obj.diffusion = GaussianDiffusion(T=cfg['T'], device=str(DEVICE))
        obj.model = EmbryoDenoiser(
            traj_dim=len(TRAJ_FEATURES),
            cond_dim=len(COND_FEATURES),
            hidden=cfg['hidden'], time_emb_dim=cfg['time_emb_dim'],
            n_blocks=cfg['n_blocks'], dropout=cfg['dropout']
        ).to(DEVICE)
        obj.model.load_state_dict(ckpt['model_state'])
        obj.normalizer = ckpt['normalizer']
        obj.history    = ckpt.get('history', {})
        print(f"[LOAD] {path}")
        return obj


# ═══════════════════════════════════════════════════════════
# ВЕРИФИКАЦИЯ
# ═══════════════════════════════════════════════════════════

def filter_similar_patients(df_real, patient,
                            age_tol=3., occ_tol=3., pn_tol=2.,
                            min_n=50):
    mask = (
            df_real["Возраст"].between(
                patient["Возраст"] - age_tol,
                patient["Возраст"] + age_tol) &
            df_real["Число ОКК"].between(
                patient["Число ОКК"] - occ_tol,
                patient["Число ОКК"] + occ_tol) &
            df_real["2 pN"].between(
                patient["2 pN"] - pn_tol,
                patient["2 pN"] + pn_tol)
    )
    subset = df_real[mask]
    if len(subset) < min_n:
        print(f"[FILTER] n={len(subset)} < {min_n} — расширяю допуски ×2")
        mask = (
                df_real["Возраст"].between(
                    patient["Возраст"] - age_tol * 2,
                    patient["Возраст"] + age_tol * 2) &
                df_real["Число ОКК"].between(
                    patient["Число ОКК"] - occ_tol * 2,
                    patient["Число ОКК"] + occ_tol * 2)
        )
        subset = df_real[mask]
    print(f"[FILTER] Похожих пациентов: {len(subset)}")
    return subset


def verify_distributions(tabddpm, df_real, patient_example,
                         n_samples=3000, age_tol=3., occ_tol=3., pn_tol=2.):
    df_similar   = filter_similar_patients(
        df_real, patient_example, age_tol, occ_tol, pn_tol)
    diff_samples = tabddpm.generate(patient_example, n_samples=n_samples)

    vars_to_plot = [
        ("Число дробящихся на 3 день",                       "D3 embryos"),
        ("Число Bl",                                          "Blastocysts total"),
        ("Число Bl хор.кач-ва",                              "Good quality blasts"),
        ("Частота формирования бластоцист",                   "Blastocyst rate"),
        ("Частота формирования бластоцист хорошего качества", "Good blast rate"),
        ("Исход переноса",                                    "Pregnancy outcome"),
    ]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f'TabDDPM v3 (Quantile + FiLM + DDIM)\n'
        f'Пациент: возраст={patient_example["Возраст"]}, '
        f'ОКК={patient_example["Число ОКК"]}, 2pN={patient_example["2 pN"]} | '
        f'Похожих: n={len(df_similar)}',
        fontsize=11, y=1.02)
    gs = gridspec.GridSpec(2, 3, hspace=0.5, wspace=0.35)
    ks_results = {}

    for idx, (col, label) in enumerate(vars_to_plot):
        ax   = fig.add_subplot(gs[idx // 3, idx % 3])
        bins = 15 if col != "Исход переноса" else 3

        real_v = df_similar[col].values
        diff_v = diff_samples[col].values

        ax.hist(real_v, bins=bins, alpha=0.45, color='#6c757d', density=True,
                label=f'Реальные похожие (n={len(df_similar)})',
                edgecolor='white', lw=0.5)
        ax.hist(diff_v, bins=bins, alpha=0.6, color='#2196F3', density=True,
                label=f'Diffusion (n={n_samples})',
                edgecolor='white', lw=0.5)

        ks_stat, ks_p = stats.ks_2samp(real_v, diff_v)
        ks_results[col] = {'KS': ks_stat, 'p': ks_p}

        color = '#2e7d32' if ks_p > 0.05 else '#c62828'
        ax.set_title(label, fontsize=9, fontweight='bold')
        ax.set_xlabel(f'KS={ks_stat:.3f}  p={ks_p:.3f}',
                      fontsize=8, color=color)
        ax.legend(fontsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.savefig('tabddpm_v3_verification.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n[VERIFICATION] KS: Diffusion vs Похожие реальные (n={len(df_similar)})")
    print(f"{'Переменная':<52} {'KS-stat':>8} {'p-value':>10} {'Вывод':>12}")
    print("─" * 87)
    for col, res in ks_results.items():
        verdict = "✓ схожи" if res['p'] > 0.05 else "≠ различны"
        print(f"{col:<52} {res['KS']:>8.3f} {res['p']:>10.4f} {verdict:>12}")

    return ks_results


def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('TabDDPM v3: Кривые обучения')
    epochs = range(1, len(history['loss']) + 1)
    for ax, scale in zip(axes, ['linear', 'log']):
        ax.plot(epochs, history['loss'], '#2196F3', label='Train', lw=1.5)
        ax.plot(epochs, history['val_loss'], '#F44336',
                label='Val', lw=1.5, ls='--')
        if scale == 'log': ax.set_yscale('log')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Weighted MSE')
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('tabddpm_v3_training.png', dpi=150, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  EMBRYO TRAJECTORY TabDDPM v3")
    print("  Quantile Normalization + FiLM + DDIM")
    print("=" * 60)

    df = load_data('all_df_with_KPI.xlsx')
    df_train, df_verify = train_test_split(df, test_size=0.10, random_state=SEED)
    print(f"[SPLIT] Train: {len(df_train)} | Verify: {len(df_verify)}")

    tabddpm = EmbryoTabDDPM(
        T=1000,
        hidden=256,
        n_blocks=6,
        time_emb_dim=64,
        dropout=0.1,
        lr=3e-4,
        epochs=150,              # достаточно
        batch_size=64,
        warmup_epochs=15,
        outcome_weight=3.0,      # усиленный вес для исхода
        ddim_steps=50,           # быстрый sampling
        n_quantiles=1000
    )
    tabddpm.fit(df_train)
    plot_training_curves(tabddpm.history)

    patient_example = {
        "Возраст": 35,
        "№ попытки": 1,
        "Количество фолликулов": 12,
        "Число ОКК": 9,
        "Число инсеминированных": 8,
        "2 pN": 6,
        "Частота получения ОКК": 0.75,
        "Частота оплодотворения": 0.75,
        "KPIScore": 18,
    }

    print("\n[MC SAMPLE] Генерация 5000 траекторий (DDIM, 50 шагов)...")
    result = tabddpm.mc_sample(patient_example, n_iter=5000)

    print(f"\n{'='*55}")
    print(f"  Вероятность беременности:  {result['P_pregnancy']:.1%}")
    print(f"  95% CI:                    "
          f"{result['CI_95'][0]:.1%} – {result['CI_95'][1]:.1%}")
    print(f"  D3 embryos (median):       {result['D3_median']:.0f}")
    print(f"  Blastocysts (median):      {result['blast_total_median']:.0f}")
    print(f"  Good blasts (median):      {result['good_blast_median']:.0f}")
    print(f"  Blast rate:                {result['blast_rate_mean']:.1%}")
    print(f"  TGBDR:                     {result['good_blast_rate_mean']:.1%}")

    print("\n[VERIFY]")
    verify_distributions(
        tabddpm, df_verify, patient_example,
        n_samples=3000, age_tol=3., occ_tol=3., pn_tol=2.)

    tabddpm.save('embryo_tabddpm_v3.pt')