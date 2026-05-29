"""
gnn_predictor.py — GNN inference module for IVF Digital Twin v6.2
Место: src/gnn_predictor.py

Загружает обученный Graph Transformer из models/gnn_ivf_model.pt
и делает предсказание для ОДНОГО нового пациента.

Стратегия инференса (transductive extension):
  Новый пациент добавляется как последний узел в граф тренировочных данных.
  k-NN строится по клиническим признакам (без PRAI/p_kat_raw).
  Предсказание берётся для последнего узла [-1].

Использование из app.py:
    from src.gnn_predictor import load_gnn_model, predict_gnn, build_patient_features
    gnn = load_gnn_model(base_dir)
    feats = build_patient_features(age, afc, attempt, res, known, p_kat_raw)
    result = predict_gnn(gnn, feats, prai_score=p_kat_raw)
    # result['gnn_prob']      — вероятность от GNN
    # result['ensemble_prob'] — w*GNN + (1-w)*PRAI
    # result['available']     — False если модель не загружена
"""

import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ── Признаки ──────────────────────────────────────────────────────────────────
# Должны совпадать с теми, что были при обучении (NODE_FEATURES из gnn_ivf_562.py)
_NODE_FEATURES = [
    'Age', 'attempt', 'afc', 'OCC', 'insem', 'two_pn',
    'cleavage_d3', 'Bl', 'Good_Bl', 'emb_d5',
    'frozen', 'transferred',
    'fert_rate', 'cleav_rate', 'blast_rate', 'good_blast_rate', 'occ_rate',
    'KPIScore', 'PRAI', 'p_kat_raw',
]

# Признаки для ТОПОЛОГИИ графа (без KAT-скоров, как при обучении)
_GRAPH_FEATURES = [
    'Age', 'attempt', 'afc', 'OCC', 'insem', 'two_pn',
    'cleavage_d3', 'Bl', 'Good_Bl', 'emb_d5',
    'frozen', 'transferred',
    'fert_rate', 'cleav_rate', 'blast_rate', 'good_blast_rate', 'occ_rate',
    'KPIScore',
]

# Индексы GRAPH_FEATURES в NODE_FEATURES (вычисляются один раз)
_GRAPH_IDX = [_NODE_FEATURES.index(f) for f in _GRAPH_FEATURES if f in _NODE_FEATURES]


# ══════════════════════════════════════════════════════════════════════════════
# АРХИТЕКТУРА МОДЕЛИ (копия из gnn_ivf_562.py — должна совпадать с сохранённой)
# ══════════════════════════════════════════════════════════════════════════════

def _try_import_torch():
    """Ленивый импорт torch/pyg — не блокирует приложение если нет библиотек."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.nn import TransformerConv
        from torch_geometric.utils import to_undirected
        return torch, nn, F, Data, TransformerConv, to_undirected
    except ImportError as e:
        return None, None, None, None, None, None


def _build_model_classes():
    """Строит классы модели после успешного импорта torch."""
    torch, nn, F, Data, TransformerConv, to_undirected = _try_import_torch()
    if torch is None:
        return None, None

    class _GraphTransformerBlock(nn.Module):
        def __init__(self, hidden, heads, ffn_mult, dropout, attn_dropout):
            super().__init__()
            head_dim = hidden // heads
            self.norm1 = nn.LayerNorm(hidden)
            self.attn  = TransformerConv(
                hidden, head_dim, heads=heads, edge_dim=1,
                dropout=attn_dropout, beta=True, concat=True)
            self.drop1 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(hidden)
            self.ffn   = nn.Sequential(
                nn.Linear(hidden, hidden * ffn_mult), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden * ffn_mult, hidden))
            self.drop2 = nn.Dropout(dropout)

        def forward(self, x, edge_index, edge_attr):
            h = self.attn(self.norm1(x), edge_index, edge_attr)
            x = x + self.drop1(h)
            h = self.ffn(self.norm2(x))
            x = x + self.drop2(h)
            return x

    class _IVFGraphTransformer(nn.Module):
        def __init__(self, in_dim, hidden=48, heads=4, n_layers=3,
                     ffn_mult=2, dropout=0.0, attn_dropout=0.0):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
            self.blocks = nn.ModuleList([
                _GraphTransformerBlock(hidden, heads, ffn_mult, dropout, attn_dropout)
                for _ in range(n_layers)])
            self.final_norm = nn.LayerNorm(hidden)
            self.shared = nn.Sequential(
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, 16), nn.GELU())
            self.preg_head = nn.Linear(16, 1)
            self.prai_head = nn.Linear(16, 1)

        def forward(self, x, edge_index, edge_attr=None):
            h = self.input_proj(x)
            for block in self.blocks:
                h = block(h, edge_index, edge_attr)
            h = self.final_norm(h)
            z = self.shared(h)
            p_preg = torch.sigmoid(self.preg_head(z)).squeeze(-1)
            p_prai = torch.sigmoid(self.prai_head(z)).squeeze(-1)
            return p_preg, p_prai

    return _IVFGraphTransformer, (torch, nn, F, Data, TransformerConv, to_undirected)


# ══════════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА МОДЕЛИ
# ══════════════════════════════════════════════════════════════════════════════

def load_gnn_model(base_dir: str = None) -> dict:
    """
    Загружает GNN из models/gnn_ivf_model.pt.

    Returns:
        dict с ключами: available, model, scaler, features, cfg,
                        train_X_scaled, train_medians, w_gnn, graph_idx
        При ошибке: {'available': False, 'error': str}
    """
    torch_libs = _try_import_torch()
    if torch_libs[0] is None:
        return {'available': False,
                'error': 'torch / torch-geometric не установлены'}

    torch = torch_libs[0]
    ModelClass, _ = _build_model_classes()
    if ModelClass is None:
        return {'available': False, 'error': 'Не удалось построить классы модели'}

    # Поиск файла модели
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    candidates = [
        os.path.join(base_dir, "models", "gnn_ivf_model.pt"),
        os.path.join(base_dir, "gnn_ivf_model.pt"),
        "gnn_ivf_model.pt",
    ]
    model_path = next((p for p in candidates if os.path.exists(p)), None)

    if model_path is None:
        return {'available': False,
                'error': f'Файл gnn_ivf_model.pt не найден. '
                         f'Ожидается в: {candidates[0]}'}

    try:
        ckpt = torch.load(model_path, map_location='cpu')
        cfg  = ckpt['cfg']

        model = ModelClass(
            in_dim       = ckpt['in_dim'],
            hidden       = cfg['hidden_dim'],
            heads        = cfg['n_heads'],
            n_layers     = cfg['n_layers'],
            ffn_mult     = cfg['ffn_multiplier'],
            dropout      = 0.0,      # eval mode — отключаем dropout
            attn_dropout = 0.0,
        )
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        # Индексы graph-фич в saved feature list
        saved_features = ckpt.get('features', _NODE_FEATURES)
        graph_idx = [saved_features.index(f)
                     for f in _GRAPH_FEATURES
                     if f in saved_features]

        return {
            'available':      True,
            'model':          model,
            'scaler':         ckpt['scaler'],
            'features':       saved_features,
            'cfg':            cfg,
            'train_X_scaled': ckpt.get('train_X_scaled'),   # np.array или None
            'train_medians':  ckpt.get('train_medians', {}), # dict feature→median
            'w_gnn':          ckpt.get('w_gnn', 0.35),
            'model_path':     model_path,
        }

    except Exception as e:
        return {'available': False, 'error': f'Ошибка загрузки модели: {e}'}


# ══════════════════════════════════════════════════════════════════════════════
# ПОСТРОЕНИЕ k-NN ГРАФА
# ══════════════════════════════════════════════════════════════════════════════

def _build_knn_graph_numpy(X_norm: np.ndarray, k: int,
                            threshold: float, min_edges: int = 2):
    """CPU k-NN граф без torch_geometric (используется только numpy/sklearn)."""
    from sklearn.metrics.pairwise import cosine_similarity

    n = X_norm.shape[0]
    k_eff = min(k, n - 1)

    sim_mat = cosine_similarity(X_norm)
    indices = np.argsort(-sim_mat, axis=1)[:, 1:k_eff + 1]
    sims    = np.take_along_axis(sim_mat, indices, axis=1)

    src, dst, wgt = [], [], []
    for i in range(n):
        added = 0
        for pos in range(k_eff):
            j   = int(indices[i, pos])
            sim = float(sims[i, pos])
            if sim >= threshold or added < min_edges:
                src.append(i); dst.append(j)
                wgt.append(max(sim, 0.0))
                added += 1

    return np.array(src), np.array(dst), np.array(wgt)


def _build_torch_graph(X_norm: np.ndarray, k: int,
                        threshold: float, min_edges: int = 2):
    """Строит PyG Data-граф из нормированных признаков."""
    torch, nn, F, Data, TransformerConv, to_undirected = _try_import_torch()
    if torch is None:
        return None

    src_np, dst_np, wgt_np = _build_knn_graph_numpy(X_norm, k, threshold, min_edges)

    edge_index = torch.tensor([src_np, dst_np], dtype=torch.long)
    edge_weight = torch.tensor(wgt_np, dtype=torch.float)
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, X_norm.shape[0])

    return edge_index, edge_weight.unsqueeze(1)


# ══════════════════════════════════════════════════════════════════════════════
# ПРЕДСКАЗАНИЕ
# ══════════════════════════════════════════════════════════════════════════════

def predict_gnn(bundle: dict,
                patient_features: dict,
                prai_score: float = None) -> dict:
    """
    Предсказывает вероятность беременности для нового пациента.

    Алгоритм:
      1. Строим вектор признаков пациента (raw), заполняем пропуски медианами.
      2. Масштабируем через сохранённый scaler.
      3. Добавляем как последний узел к тренировочным данным.
      4. Строим k-NN граф по клиническим признакам.
      5. Forward pass → берём предсказание для последнего узла.
      6. Ансамбль: w_gnn × GNN + (1-w_gnn) × PRAI.

    Args:
        bundle:          результат load_gnn_model()
        patient_features: словарь {feature_name: raw_value}
                          Ключи = из _NODE_FEATURES (английские имена)
        prai_score:      вероятность от KAT (float 0-1), нужна для ансамбля

    Returns:
        dict: {
          'available':      bool,
          'gnn_prob':       float или None,
          'ensemble_prob':  float или None,  # w*GNN + (1-w)*PRAI
          'w_gnn':          float,
          'error':          str (только при ошибке)
        }
    """
    if not bundle or not bundle.get('available'):
        return {
            'available':     False,
            'gnn_prob':      None,
            'ensemble_prob': None,
            'w_gnn':         0.35,
            'error':         bundle.get('error', 'Модель не загружена'),
        }

    torch_libs = _try_import_torch()
    torch = torch_libs[0]
    if torch is None:
        return {'available': False, 'gnn_prob': None, 'ensemble_prob': None,
                'w_gnn': 0.35, 'error': 'torch не установлен'}

    try:
        model          = bundle['model']
        scaler         = bundle['scaler']
        features       = bundle['features']
        cfg            = bundle['cfg']
        train_X_scaled = bundle.get('train_X_scaled')
        train_medians  = bundle.get('train_medians', {})
        w_gnn          = bundle.get('w_gnn', 0.35)

        # ── Вектор признаков пациента (raw) ──────────────────────────────────
        x_raw = np.array([
            float(patient_features.get(f, train_medians.get(f, np.nan)))
            for f in features
        ], dtype=float).reshape(1, -1)

        # Заполняем NaN медианами из scaler (mean_)
        nan_mask = np.isnan(x_raw[0])
        if nan_mask.any():
            x_raw[0, nan_mask] = scaler.mean_[nan_mask]

        # ── Масштабирование ───────────────────────────────────────────────────
        x_scaled = scaler.transform(x_raw)  # shape [1, n_features]

        # ── Составной граф: training + новый пациент ─────────────────────────
        if train_X_scaled is not None and len(train_X_scaled) > 0:
            X_aug = np.vstack([train_X_scaled, x_scaled])   # [N+1, d]
        else:
            # Fallback: граф только из одного пациента (нет message passing)
            X_aug = x_scaled

        n_total      = X_aug.shape[0]
        patient_idx  = n_total - 1

        # Нормировка для косинусного сходства
        # Используем только graph-признаки для топологии
        graph_idx = [features.index(f) for f in _GRAPH_FEATURES if f in features]
        X_graph   = X_aug[:, graph_idx]
        norms     = np.linalg.norm(X_graph, axis=1, keepdims=True) + 1e-9
        X_norm    = (X_graph / norms).astype('float32')

        k         = min(cfg.get('k_neighbors', 10), n_total - 1)
        threshold = cfg.get('sim_threshold', 0.60)
        min_edges = cfg.get('min_edges_per_node', 2)

        edge_index, edge_attr = _build_torch_graph(X_norm, k, threshold, min_edges)
        if edge_index is None:
            raise RuntimeError('Не удалось построить граф')

        x_tensor = torch.tensor(X_aug, dtype=torch.float)

        # ── Forward pass ──────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            p_preg_all, _ = model(x_tensor, edge_index, edge_attr)

        gnn_prob = float(p_preg_all[patient_idx].item())
        gnn_prob = float(np.clip(gnn_prob, 0.0, 1.0))

        # ── Соседи пациентки для визуализации ───────────────────────────────
        # Берём top-10 ближайших из тренировочной выборки по косинусному сходству
        from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
        _k_vis    = min(10, n_total - 1)
        _pat_vec  = X_norm[patient_idx:patient_idx+1]          # [1, d_graph]
        _train_vecs = X_norm[:patient_idx]                     # [N, d_graph]
        _sims_pat = _cos_sim(_pat_vec, _train_vecs)[0]         # [N]
        _top_idx  = np.argsort(-_sims_pat)[:_k_vis]           # индексы топ-k
        _top_sims = _sims_pat[_top_idx]                        # cosine sims

        # GNN вероятности соседей (из того же forward pass)
        _neigh_probs = p_preg_all[_top_idx].numpy()            # [k]

        # Имена клинических признаков для hover-подписей (первые 5)
        _feat_labels = [f for f in _GRAPH_FEATURES if f in features][:5]
        _feat_idx_vis = [features.index(f) for f in _feat_labels]

        # Raw значения соседей (inverse_transform через scaler)
        try:
            _neigh_raw = scaler.inverse_transform(
                train_X_scaled[_top_idx])                      # [k, d]
            _pat_raw   = scaler.inverse_transform(x_scaled)    # [1, d]
        except Exception:
            _neigh_raw = None
            _pat_raw   = None

        # Фоновое облако — PCA-проекция всей тренировочной выборки
        try:
            from sklearn.decomposition import PCA as _PCA
            _pca2       = _PCA(n_components=2, random_state=42)
            _all_coords = _pca2.fit_transform(_train_vecs)   # [N, 2]
            _pat_coord  = _pca2.transform(_pat_vec)          # [1, 2]
            _all_probs  = p_preg_all[:patient_idx].numpy()   # [N]
            _all_sims   = _sims_pat                          # [N]
        except Exception:
            _all_coords = None
            _pat_coord  = None
            _all_probs  = None
            _all_sims   = None

        neighbors_data = {
            'indices':    _top_idx,
            'sims':       _top_sims,
            'probs':      _neigh_probs,
            'feat_labels':_feat_labels,
            'feat_idx':   _feat_idx_vis,
            'neigh_raw':  _neigh_raw,
            'pat_raw':    _pat_raw,
            # Облако всех точек
            'all_coords': _all_coords,
            'pat_coord':  _pat_coord,
            'all_probs':  _all_probs,
            'all_sims':   _all_sims,
        }

        # ── Ансамбль ─────────────────────────────────────────────────────────
        ensemble_prob = None
        if prai_score is not None:
            prai = float(np.clip(prai_score, 0.0, 1.0))
            ensemble_prob = float(np.clip(
                w_gnn * gnn_prob + (1.0 - w_gnn) * prai, 0.0, 1.0))

        return {
            'available':     True,
            'gnn_prob':      gnn_prob,
            'ensemble_prob': ensemble_prob,
            'w_gnn':         w_gnn,
            'neighbors':     neighbors_data,   # для визуализации
        }

    except Exception as e:
        import traceback
        return {
            'available':     False,
            'gnn_prob':      None,
            'ensemble_prob': None,
            'w_gnn':         0.35,
            'error':         f'Ошибка инференса GNN: {e}',
            'traceback':     traceback.format_exc(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# ХЕЛПЕР: СБОРКА ПРИЗНАКОВ ИЗ ДАННЫХ ПАЙПЛАЙНА
# ══════════════════════════════════════════════════════════════════════════════

def build_patient_features(age: float,
                            afc: int,
                            attempt: int,
                            res: dict,
                            known,
                            p_kat_raw: float = None) -> dict:
    """
    Собирает словарь признаков для GNN из данных app.py.

    Args:
        age:       возраст пациентки
        afc:       АФЧ (sidebar)
        attempt:   номер попытки (sidebar)
        res:       словарь результатов pipeline (okk_med, mii_med, ...)
        known:     объект с известными mid-cycle значениями (known.okk, ...)
        p_kat_raw: непрерывный скор KAT (0-1)

    Returns:
        dict с ключами из _NODE_FEATURES
    """
    def _known(attr):
        """Достаёт значение из known-объекта (NamedTuple или dict)."""
        if known is None:
            return None
        if hasattr(known, attr):
            v = getattr(known, attr)
            return float(v) if v is not None else None
        if isinstance(known, dict):
            v = known.get(attr)
            return float(v) if v is not None else None
        return None

    def _res(key, fallback=None):
        if isinstance(res, dict):
            v = res.get(key, fallback)
            return float(v) if v is not None else fallback
        return fallback

    # ── Абсолютные счётчики ─────────────────────────────────────────────────
    OCC        = _known('okk')    or _res('okk_med',    0.0)
    insem      = _known('mii')    or _res('mii_med',    0.0)
    two_pn     = _known('pn2')    or _res('pn2_med',    0.0)
    Bl         = _known('blasts') or _res('blasts_med', 0.0)
    Good_Bl    = _known('good')   or _res('good_med',   0.0)
    emb_d5     = Bl               # бластоцисты на 5-й день ≈ Bl
    cleavage_d3 = two_pn          # прокси: дробящихся ≈ 2PN (нет прямой колонки)
    transferred = _res('warmed_med', 1.0)
    frozen      = max(0.0, Bl - transferred)

    # ── Производные частоты ─────────────────────────────────────────────────
    fert_rate       = two_pn  / max(insem, 1)
    cleav_rate      = cleavage_d3 / max(two_pn, 1)
    blast_rate      = Bl      / max(two_pn, 1)
    good_blast_rate = Good_Bl / max(Bl, 1)
    occ_rate        = OCC     / max(afc, 1)

    # ── KPIScore — берём из res если есть, иначе NaN ─────────────────────────
    KPIScore = _res('kpi_score', None) or _res('kpi', None)

    # ── KAT признаки ─────────────────────────────────────────────────────────
    PRAI      = float(p_kat_raw) if p_kat_raw is not None else None
    p_kat_bin = 1.0 if (p_kat_raw is not None and p_kat_raw > 0.5) else 0.0

    return {
        'Age':             float(age),
        'attempt':         float(attempt),
        'afc':             float(afc),
        'OCC':             float(OCC),
        'insem':           float(insem),
        'two_pn':          float(two_pn),
        'cleavage_d3':     float(cleavage_d3),
        'Bl':              float(Bl),
        'Good_Bl':         float(Good_Bl),
        'emb_d5':          float(emb_d5),
        'frozen':          float(frozen),
        'transferred':     float(transferred),
        'fert_rate':       float(np.clip(fert_rate, 0, 1)),
        'cleav_rate':      float(np.clip(cleav_rate, 0, 1)),
        'blast_rate':      float(np.clip(blast_rate, 0, 1)),
        'good_blast_rate': float(np.clip(good_blast_rate, 0, 1)),
        'occ_rate':        float(np.clip(occ_rate, 0, 1)),
        'KPIScore':        float(KPIScore) if KPIScore is not None else float('nan'),
        'PRAI':            float(PRAI) if PRAI is not None else float('nan'),
        'p_kat_raw':       float(p_kat_bin),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ВИЗУАЛИЗАЦИЯ: ГРАФ СОСЕДЕЙ ДЛЯ PDF-ОТЧЁТА
# ══════════════════════════════════════════════════════════════════════════════


def build_gnn_neighborhood_figure(gnn_result: dict,
                                   gnn_prob: float = None,
                                   ensemble_prob: float = None):
    """
    Двухпанельный график:
    Левая — облако всех тренировочных точек (PCA) + 10 соседей поверх + звезда пациентки.
    Правая — горизонтальные бары 10 ближайших соседей.
    Цветовая шкала: красный ≤35%, янтарь ~45%, зелёный ≥55%.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    neighbors = gnn_result.get('neighbors') if gnn_result else None
    if neighbors is None or gnn_prob is None:
        return None

    sims  = np.array(neighbors['sims'],  dtype=float)
    probs = np.array(neighbors['probs'], dtype=float)
    k     = len(sims)
    if k == 0:
        return None

    # ── Цветовая шкала ────────────────────────────────────────────────────────
    _C_RED   = (183, 28,  28)
    _C_AMBER = (245, 127, 23)
    _C_GREEN = (46,  125, 50)

    def _lerp(a, b, t):
        return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

    def _prob_to_rgb(p):
        p = float(np.clip(p, 0.0, 1.0))
        if p <= 0.35: return _C_RED
        if p >= 0.55: return _C_GREEN
        t = (p - 0.35) / 0.20
        if t < 0.5:   return _lerp(_C_RED,   _C_AMBER, t * 2)
        return              _lerp(_C_AMBER, _C_GREEN, (t - 0.5) * 2)

    def _rgba(p, a=1.0):
        r, g, b = _prob_to_rgb(p)
        return f'rgba({r},{g},{b},{a:.2f})'

    # ── Данные облака ─────────────────────────────────────────────────────────
    all_coords = neighbors.get('all_coords')  # [N, 2] PCA всей выборки
    pat_coord  = neighbors.get('pat_coord')   # [1, 2]
    all_probs  = neighbors.get('all_probs')   # [N]
    all_sims   = neighbors.get('all_sims')    # [N]
    top_idx    = neighbors.get('indices')     # [k] индексы топ-10 в облаке

    has_cloud  = (all_coords is not None and all_probs is not None
                  and len(all_coords) > 0)

    # ── Позиции 10 соседей — из PCA если есть, иначе spring ──────────────────
    if has_cloud and top_idx is not None:
        # Нормализуем PCA координаты в [-1, 1]
        coords = np.array(all_coords, dtype=float)
        span   = np.abs(coords).max(axis=0) + 1e-9
        coords = coords / span * 0.95
        neigh_x = coords[top_idx, 0]
        neigh_y = coords[top_idx, 1]
        pat_x   = float(pat_coord[0, 0] / span[0] * 0.95) if pat_coord is not None else 0.0
        pat_y   = float(pat_coord[0, 1] / span[1] * 0.95) if pat_coord is not None else 0.0
        cloud_x = coords[:, 0]
        cloud_y = coords[:, 1]
    else:
        # Fallback: spring layout только для 10+1 точек
        has_cloud = False
        neigh_raw = neighbors.get('neigh_raw')
        n_nodes   = k + 1
        sim_matrix = np.zeros((n_nodes, n_nodes))
        for i in range(k):
            sim_matrix[i, k] = sims[i]
            sim_matrix[k, i] = sims[i]
        if neigh_raw is not None:
            from numpy.linalg import norm as _norm
            for i in range(k):
                for j in range(i+1, k):
                    cs = float(np.dot(neigh_raw[i], neigh_raw[j]) /
                               (_norm(neigh_raw[i]) * _norm(neigh_raw[j]) + 1e-9))
                    sim_matrix[i,j] = sim_matrix[j,i] = max(0.0, cs)
        np.random.seed(42)
        angles0 = np.linspace(0, 2*np.pi, k, endpoint=False)
        pos = np.zeros((n_nodes, 2))
        pos[:k, 0] = np.cos(angles0) * 0.8
        pos[:k, 1] = np.sin(angles0) * 0.8
        EDGE_THR = 0.30
        for _ in range(60):
            forces = np.zeros((n_nodes, 2))
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    delta = pos[j] - pos[i]; dist = np.linalg.norm(delta) + 1e-6
                    if sim_matrix[i,j] >= EDGE_THR:
                        f = delta/dist*(dist-0.5)*sim_matrix[i,j]*0.08
                        forces[i] += f; forces[j] -= f
                    rep = delta/(dist**2)*0.04
                    forces[i] += rep; forces[j] -= rep
            forces[k] -= pos[k] * 0.15
            pos += forces * 0.12
            pos = np.clip(pos, -1.8, 1.8)
        span = max(np.abs(pos).max(), 0.1)
        pos  = pos / span * 0.95
        neigh_x = pos[:k, 0]; neigh_y = pos[:k, 1]
        pat_x = pos[k, 0];    pat_y = pos[k, 1]
        cloud_x = cloud_y = None

    sim_n      = (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)

    # ── Hover-тексты ──────────────────────────────────────────────────────────
    feat_labels = neighbors.get('feat_labels', [])
    feat_idx    = neighbors.get('feat_idx', [])
    neigh_raw   = neighbors.get('neigh_raw')
    pat_raw     = neighbors.get('pat_raw')

    hover_texts = []
    for i in range(k):
        lines = [f"<b>Сосед #{i+1}</b>",
                 f"GNN P(бер.): <b>{probs[i]*100:.1f}%</b>",
                 f"Сходство: <b>{sims[i]:.3f}</b>"]
        if neigh_raw is not None and feat_labels:
            lines.append("─────────")
            for lbl, fi in zip(feat_labels, feat_idx):
                try: lines.append(f"{lbl}: {neigh_raw[i, fi]:.1f}")
                except Exception: pass
        hover_texts.append("<br>".join(lines))

    # ── Subplot ───────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.57, 0.43],
        subplot_titles=("Граф клинических соседей",
                        "Распределение GNN-вероятностей"),
        horizontal_spacing=0.10,
    )

    # ── ПАНЕЛЬ 1 ──────────────────────────────────────────────────────────────

    # Слой 1: всё облако тренировочных точек — очень мелкие маркеры
    if has_cloud and cloud_x is not None:
        _ap = np.array(all_probs, dtype=float)
        cloud_colors = [_rgba(p, 0.45) for p in _ap]
        fig.add_trace(go.Scatter(
            x=cloud_x, y=cloud_y,
            mode='markers',
            marker=dict(
                size=4,
                color=[p * 100 for p in _ap],
                colorscale=[
                    [0.00, f'rgb{_C_RED}'],
                    [0.35, f'rgb{_C_RED}'],
                    [0.45, f'rgb{_C_AMBER}'],
                    [0.55, f'rgb{_C_GREEN}'],
                    [1.00, f'rgb{_C_GREEN}'],
                ],
                cmin=0, cmax=100,
                showscale=False,
                line=dict(color='white', width=0.6),
                opacity=0.55,
            ),
            hoverinfo='skip',
            showlegend=False,
        ), row=1, col=1)

        # Тонкие рёбра: пациентка → топ-10 соседей
        for i in range(k):
            p_edge = (probs[i] + gnn_prob) / 2.0
            w_e    = 0.5 + 1.2 * float(sim_n[i])
            fig.add_trace(go.Scatter(
                x=[pat_x, neigh_x[i], None],
                y=[pat_y, neigh_y[i], None],
                mode='lines',
                line=dict(color=_rgba(p_edge, 0.30 + 0.20 * float(sim_n[i])),
                          width=w_e),
                hoverinfo='skip', showlegend=False,
            ), row=1, col=1)

    else:
        # Без облака — рёбра spring-layout между всеми
        EDGE_THR2 = 0.30
        for i in range(k + 1):
            for j in range(i+1, k + 1):
                if i < k and j < k:
                    w_ij = sim_n[i] * sim_n[j] if neigh_raw is None else 0.0
                else:
                    w_ij = float(sim_n[i]) if i < k else float(sim_n[j])
                if w_ij < 0.05: continue
                p_i = probs[i] if i < k else gnn_prob
                p_j = probs[j] if j < k else gnn_prob
                xi  = neigh_x[i] if i < k else pat_x
                yi  = neigh_y[i] if i < k else pat_y
                xj  = neigh_x[j] if j < k else pat_x
                yj  = neigh_y[j] if j < k else pat_y
                fig.add_trace(go.Scatter(
                    x=[xi, xj, None], y=[yi, yj, None],
                    mode='lines',
                    line=dict(color=_rgba((p_i+p_j)/2, 0.25 + 0.20*w_ij),
                              width=0.5 + 1.5*w_ij),
                    hoverinfo='skip', showlegend=False,
                ), row=1, col=1)

    # Слой 2: топ-10 соседей — чуть крупнее облака, с обводкой
    node_sizes = 8 + 7 * sim_n   # 8–15 px

    _colorscale = [
        [0.00, f'rgb{_C_RED}'],
        [0.35, f'rgb{_C_RED}'],
        [0.45, f'rgb{_C_AMBER}'],
        [0.55, f'rgb{_C_GREEN}'],
        [1.00, f'rgb{_C_GREEN}'],
    ]

    fig.add_trace(go.Scatter(
        x=neigh_x, y=neigh_y,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=[p * 100 for p in probs],
            colorscale=_colorscale,
            cmin=0, cmax=100,
            colorbar=dict(
                title=dict(text='P(бер.)%',
                           font=dict(size=9, family='Inter, Arial, sans-serif')),
                thickness=10, len=0.45,
                tickvals=[0, 35, 55, 100],
                ticktext=['0%', '35%', '55%', '100%'],
                tickfont=dict(size=8, family='Inter, Arial, sans-serif'),
                x=0.55, y=0.5,
                bgcolor='rgba(255,255,255,0.90)',
                bordercolor='#ddd', borderwidth=1,
            ),
            line=dict(color='white', width=1.5),
            opacity=0.95,
        ),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>',
        showlegend=False,
    ), row=1, col=1)

    # Подписи % вне узлов топ-10
    for i in range(k):
        nx_, ny_ = neigh_x[i], neigh_y[i]
        dist = np.sqrt((nx_ - pat_x)**2 + (ny_ - pat_y)**2) + 1e-9
        off  = (node_sizes[i] / 2 + 5) / 200
        lx   = nx_ + (nx_ - pat_x) / dist * off * 2.5
        ly   = ny_ + (ny_ - pat_y) / dist * off * 2.5
        r, g, b = _prob_to_rgb(probs[i])
        fig.add_annotation(
            x=lx, y=ly,
            text=f"{probs[i]*100:.0f}%",
            showarrow=False,
            font=dict(size=7, color=f'rgb({r},{g},{b})',
                      family='Inter, Arial, sans-serif'),
            xref='x', yref='y',
            bgcolor='rgba(255,255,255,0.72)',
            borderpad=1,
        )

    # Слой 3: пациентка — звезда компактная
    pr, pg, pb = _prob_to_rgb(gnn_prob)
    if pat_raw is not None and feat_labels:
        extra = "<br>".join(
            f"{lbl}: {pat_raw[0, fi]:.1f}"
            for lbl, fi in zip(feat_labels, feat_idx)
            if fi < pat_raw.shape[1]
        )
        pat_hover = (f"<b>Пациентка</b><br>"
                     f"GNN P(бер.): <b>{gnn_prob*100:.1f}%</b><br>"
                     f"─────────<br>{extra}")
    else:
        pat_hover = f"<b>Пациентка</b><br>GNN P(бер.): <b>{gnn_prob*100:.1f}%</b>"

    fig.add_trace(go.Scatter(
        x=[pat_x], y=[pat_y],
        mode='markers',
        marker=dict(
            symbol='star',
            size=20,
            color=f'rgb({pr},{pg},{pb})',
            line=dict(color='white', width=2.0),
            opacity=1.0,
        ),
        hovertemplate=pat_hover + '<extra></extra>',
        showlegend=False,
    ), row=1, col=1)

    fig.add_annotation(
        x=pat_x, y=pat_y + 0.12,
        text=f"<b>Пациентка {gnn_prob*100:.1f}%</b>",
        showarrow=False,
        font=dict(size=8, color=f'rgb({pr},{pg},{pb})',
                  family='Inter, Arial, sans-serif'),
        xref='x', yref='y',
        bgcolor='rgba(255,255,255,0.88)',
        bordercolor=f'rgb({pr},{pg},{pb})',
        borderwidth=1, borderpad=2,
        align='center',
    )

    # ── ПАНЕЛЬ 2: Горизонтальные бары 10 соседей ──────────────────────────────
    sort_ord   = np.argsort(probs)
    bar_probs  = probs[sort_ord] * 100
    bar_sims   = sims[sort_ord]
    bar_colors = [_rgba(p / 100, 0.75) for p in bar_probs]
    bar_border = [_rgba(p / 100, 0.95) for p in bar_probs]
    bar_labels = [f"#{sort_ord[i]+1}  {bar_sims[i]:.2f}" for i in range(k)]

    fig.add_trace(go.Bar(
        x=bar_probs, y=bar_labels,
        orientation='h',
        marker=dict(color=bar_colors,
                    line=dict(color=bar_border, width=1.0)),
        text=[f"{v:.1f}%" for v in bar_probs],
        textposition='outside',
        textfont=dict(size=8.5, family='Inter, Arial, sans-serif', color='#333333'),
        hovertemplate='Сосед %{y}<br>P(беременность): %{x:.1f}%<extra></extra>',
        showlegend=False,
    ), row=1, col=2)

    fig.add_vline(
        x=gnn_prob * 100,
        line=dict(color=f'rgba({pr},{pg},{pb},0.90)', width=2.0, dash='dash'),
        row=1, col=2,
        annotation_text=f"Пациентка {gnn_prob*100:.1f}%",
        annotation_position="top",
        annotation_font=dict(size=8, family='Inter, Arial, sans-serif',
                             color=f'rgb({pr},{pg},{pb})'),
        annotation_bgcolor='rgba(255,255,255,0.85)',
    )

    med_prob = float(np.median(probs)) * 100
    fig.add_vline(
        x=med_prob,
        line=dict(color='rgba(100,100,100,0.45)', width=1.2, dash='dot'),
        row=1, col=2,
        annotation_text=f"Медиана {med_prob:.1f}%",
        annotation_position="bottom",
        annotation_font=dict(size=7.5, family='Inter, Arial, sans-serif',
                             color='#666666'),
        annotation_bgcolor='rgba(255,255,255,0.80)',
    )

    # ── Оформление ────────────────────────────────────────────────────────────
    ens_str = (f"  ·  Ансамбль GAT+KAT: <b>{ensemble_prob*100:.1f}%</b>"
               if ensemble_prob is not None else "")
    fig.update_layout(
        title=dict(
            text=(f"GNN (Graph Transformer)  ·  "
                  f"P(беременность): <b>{gnn_prob*100:.1f}%</b>{ens_str}"),
            font=dict(size=11, family='Inter, Arial, sans-serif', color='#1A3A5C'),
            x=0.02, xanchor='left',
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=500,
        margin=dict(l=20, r=20, t=55, b=20),
        font=dict(family='Inter, Arial, sans-serif', size=9, color='#1C2833'),
    )

    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False,
                     range=[-1.30, 1.30], row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False,
                     range=[-1.30, 1.30], row=1, col=1,
                     scaleanchor='x', scaleratio=1)

    _x_max = min(108, max(bar_probs) * 1.28 + 5)
    fig.update_xaxes(
        title=dict(text='GNN P(беременность) %',
                   font=dict(size=8, family='Inter, Arial, sans-serif')),
        range=[0, _x_max], tickfont=dict(size=8, family='Inter, Arial, sans-serif'),
        gridcolor='rgba(200,210,220,0.35)', zeroline=False, row=1, col=2)
    fig.update_yaxes(
        tickfont=dict(size=8, family='Inter, Arial, sans-serif'),
        gridcolor='rgba(200,210,220,0.35)', row=1, col=2)

    for ann in fig.layout.annotations:
        if ann.text in ("Граф клинических соседей",
                        "Распределение GNN-вероятностей"):
            ann.font = dict(size=10, color='#1A3A5C',
                            family='Inter, Arial, sans-serif')

    return fig

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    neighbors = gnn_result.get('neighbors') if gnn_result else None
    if neighbors is None or gnn_prob is None:
        return None

    sims  = np.array(neighbors['sims'],  dtype=float)
    probs = np.array(neighbors['probs'], dtype=float)
    k     = len(sims)
    if k == 0:
        return None

    # ── Цветовая шкала с заданными порогами ──────────────────────────────────
    # Зелёный  ≥ 0.55,  Красный ≤ 0.35,  градиент между ними
    _C_RED    = (183, 28,  28)   # #B71C1C
    _C_AMBER  = (245, 127, 23)   # #F57F17
    _C_GREEN  = (46,  125, 50)   # #2E7D32

    def _lerp(a, b, t):
        return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

    def _prob_to_rgb(p):
        """p ∈ [0,1] → (r,g,b) с порогами 0.35 (красный) и 0.55 (зелёный)."""
        p = float(np.clip(p, 0.0, 1.0))
        if p <= 0.35:
            return _C_RED
        if p >= 0.55:
            return _C_GREEN
        t = (p - 0.35) / 0.20          # 0→1 между порогами
        if t < 0.5:
            return _lerp(_C_RED, _C_AMBER, t * 2)
        return _lerp(_C_AMBER, _C_GREEN, (t - 0.5) * 2)

    def _rgba(p, alpha=1.0):
        r, g, b = _prob_to_rgb(p)
        return f'rgba({r},{g},{b},{alpha:.2f})'

    # ── Позиции узлов ─────────────────────────────────────────────────────────
    angles  = np.linspace(0, 2 * np.pi, k, endpoint=False) - np.pi / 2
    sim_n   = (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)
    radii   = 0.62 + 0.28 * sim_n          # от 0.62 до 0.90 — компактнее
    neigh_x = radii * np.cos(angles)
    neigh_y = radii * np.sin(angles)

    # Размер узла ∝ сходство (14–28 px)
    node_sizes = 14 + 14 * sim_n

    # Подписи вынесены радиально за пределы узла (радиус 1.08–1.15)
    label_r = radii + 0.22
    label_x = label_r * np.cos(angles)
    label_y = label_r * np.sin(angles)

    # ── Hover-тексты ──────────────────────────────────────────────────────────
    feat_labels = neighbors.get('feat_labels', [])
    neigh_raw   = neighbors.get('neigh_raw')
    pat_raw     = neighbors.get('pat_raw')
    feat_idx    = neighbors.get('feat_idx', [])

    hover_texts = []
    for i in range(k):
        lines = [f"<b>Сосед #{i+1}</b>",
                 f"GNN P(бер.): <b>{probs[i]*100:.1f}%</b>",
                 f"Сходство: <b>{sims[i]:.3f}</b>"]
        if neigh_raw is not None and feat_labels:
            lines.append("─────────")
            for lbl, fi in zip(feat_labels, feat_idx):
                try:
                    lines.append(f"{lbl}: {neigh_raw[i, fi]:.1f}")
                except Exception:
                    pass
        hover_texts.append("<br>".join(lines))

    # ── Subplot ───────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=("Граф клинических соседей",
                        "Распределение GNN-вероятностей"),
        horizontal_spacing=0.10,
    )

    # ── ПАНЕЛЬ 1: Граф ────────────────────────────────────────────────────────

    # Рёбра: тонкие, цвет = средняя вероятность пары (сосед + пациентка/2)
    for i in range(k):
        p_edge  = (probs[i] + gnn_prob) / 2.0
        e_color = _rgba(p_edge, alpha=0.40)
        e_width = 0.8 + 2.2 * float(sim_n[i])   # 0.8–3.0 px — воздушные линии
        fig.add_trace(go.Scatter(
            x=[0, neigh_x[i]], y=[0, neigh_y[i]],
            mode='lines',
            line=dict(color=e_color, width=e_width),
            hoverinfo='skip', showlegend=False,
        ), row=1, col=1)

    # Узлы соседей — цвет через colorscale с нашими порогами
    _colorscale = [
        [0.00, f'rgb{_C_RED}'],
        [0.35, f'rgb{_C_RED}'],
        [0.45, f'rgb{_C_AMBER}'],
        [0.55, f'rgb{_C_GREEN}'],
        [1.00, f'rgb{_C_GREEN}'],
    ]

    fig.add_trace(go.Scatter(
        x=neigh_x, y=neigh_y,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=[p * 100 for p in probs],
            colorscale=_colorscale,
            cmin=0, cmax=100,
            colorbar=dict(
                title=dict(text='P(бер.)%', font=dict(size=9,
                           family='Inter, Arial, sans-serif')),
                thickness=10, len=0.50,
                tickvals=[0, 35, 55, 100],
                ticktext=['0%', '35%', '55%', '100%'],
                tickfont=dict(size=8, family='Inter, Arial, sans-serif'),
                x=0.53, y=0.5,
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor='#ddd', borderwidth=1,
            ),
            line=dict(color='white', width=1.5),
            opacity=0.92,
        ),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>',
        showlegend=False,
        name='Соседи',
    ), row=1, col=1)

    # Подписи вероятностей — вынесены радиально, не перекрывают узлы
    for i in range(k):
        r, g, b = _prob_to_rgb(probs[i])
        fig.add_annotation(
            x=label_x[i], y=label_y[i],
            text=f"{probs[i]*100:.0f}%",
            showarrow=False,
            font=dict(size=8, color=f'rgb({r},{g},{b})',
                      family='Inter, Arial, sans-serif'),
            xref='x', yref='y',
            bgcolor='rgba(255,255,255,0.72)',
            borderpad=1,
        )

    # Узел пациентки (звезда в центре)
    pat_r, pat_g, pat_b = _prob_to_rgb(gnn_prob)
    pat_color = f'rgb({pat_r},{pat_g},{pat_b})'

    if pat_raw is not None and feat_labels:
        feat_idx_f = neighbors.get('feat_idx', [])
        extra = "<br>".join(
            f"{lbl}: {pat_raw[0, fi]:.1f}"
            for lbl, fi in zip(feat_labels, feat_idx_f)
            if fi < pat_raw.shape[1]
        )
        pat_hover = (f"<b>Пациентка</b><br>"
                     f"GNN P(бер.): <b>{gnn_prob*100:.1f}%</b><br>"
                     f"─────────<br>{extra}")
    else:
        pat_hover = f"<b>Пациентка</b><br>GNN P(бер.): <b>{gnn_prob*100:.1f}%</b>"

    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(
            symbol='star',
            size=32,
            color=pat_color,
            line=dict(color='white', width=2.5),
            opacity=1.0,
        ),
        hovertemplate=pat_hover + '<extra></extra>',
        showlegend=False,
        name='Пациентка',
    ), row=1, col=1)

    # Подпись пациентки — чуть выше звезды, не перекрывается
    fig.add_annotation(
        x=0, y=0.20,
        text=f"<b>Пациентка<br>{gnn_prob*100:.1f}%</b>",
        showarrow=False,
        font=dict(size=9, color=f'rgb({pat_r},{pat_g},{pat_b})',
                  family='Inter, Arial, sans-serif'),
        xref='x', yref='y',
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor=f'rgb({pat_r},{pat_g},{pat_b})',
        borderwidth=1, borderpad=3,
        align='center',
    )

    # ── ПАНЕЛЬ 2: Горизонтальные бары ────────────────────────────────────────
    sort_ord   = np.argsort(probs)          # снизу вверх = по возрастанию
    bar_probs  = probs[sort_ord] * 100
    bar_sims   = sims[sort_ord]
    bar_colors = [_rgba(p / 100, 0.78) for p in bar_probs]
    bar_border = [_rgba(p / 100, 0.95) for p in bar_probs]
    bar_labels = [f"#{sort_ord[i]+1}  {bar_sims[i]:.2f}"
                  for i in range(k)]

    fig.add_trace(go.Bar(
        x=bar_probs,
        y=bar_labels,
        orientation='h',
        marker=dict(
            color=bar_colors,
            line=dict(color=bar_border, width=1.0),
        ),
        text=[f"{v:.1f}%" for v in bar_probs],
        textposition='outside',
        textfont=dict(size=8.5, family='Inter, Arial, sans-serif',
                      color='#333333'),
        hovertemplate='Сосед %{y}<br>P(беременность): %{x:.1f}%<extra></extra>',
        showlegend=False,
    ), row=1, col=2)

    # Линия — вероятность пациентки
    pr, pg, pb = _prob_to_rgb(gnn_prob)
    fig.add_vline(
        x=gnn_prob * 100,
        line=dict(color=f'rgba({pr},{pg},{pb},0.90)', width=2.0, dash='dash'),
        row=1, col=2,
        annotation_text=f"Пациентка {gnn_prob*100:.1f}%",
        annotation_position="top",
        annotation_font=dict(size=8, family='Inter, Arial, sans-serif',
                             color=f'rgb({pr},{pg},{pb})'),
        annotation_bgcolor='rgba(255,255,255,0.85)',
    )

    # Линия медианы соседей
    med_prob = float(np.median(probs)) * 100
    fig.add_vline(
        x=med_prob,
        line=dict(color='rgba(100,100,100,0.45)', width=1.2, dash='dot'),
        row=1, col=2,
        annotation_text=f"Медиана {med_prob:.1f}%",
        annotation_position="bottom",
        annotation_font=dict(size=7.5, family='Inter, Arial, sans-serif',
                             color='#666666'),
        annotation_bgcolor='rgba(255,255,255,0.80)',
    )

    # ── Оформление ────────────────────────────────────────────────────────────
    ens_str = (f"  ·  Ансамбль GAT+KAT: <b>{ensemble_prob*100:.1f}%</b>"
               if ensemble_prob is not None else "")
    title_text = (f"GNN (Graph Transformer)  ·  "
                  f"P(беременность): <b>{gnn_prob*100:.1f}%</b>{ens_str}")

    fig.update_layout(
        title=dict(text=title_text,
                   font=dict(size=11, family='Inter, Arial, sans-serif',
                             color='#1A3A5C'),
                   x=0.02, xanchor='left'),
        paper_bgcolor='white',
        plot_bgcolor='rgba(248,250,252,1)',
        height=480,
        margin=dict(l=20, r=20, t=60, b=30),
        font=dict(family='Inter, Arial, sans-serif', size=9, color='#1C2833'),
    )

    # Панель 1 — без осей, равный масштаб
    fig.update_xaxes(
        showgrid=False, zeroline=False, showticklabels=False,
        range=[-1.35, 1.35], row=1, col=1)
    fig.update_yaxes(
        showgrid=False, zeroline=False, showticklabels=False,
        range=[-1.35, 1.35], row=1, col=1,
        scaleanchor='x', scaleratio=1)

    # Панель 2 — ось X с запасом для textposition='outside'
    _x_max = min(108, max(bar_probs) * 1.25 + 5)
    fig.update_xaxes(
        title=dict(text='GNN P(беременность) %',
                   font=dict(size=8, family='Inter, Arial, sans-serif')),
        range=[0, _x_max],
        tickfont=dict(size=8, family='Inter, Arial, sans-serif'),
        gridcolor='rgba(200,210,220,0.35)',
        zeroline=False,
        row=1, col=2)
    fig.update_yaxes(
        tickfont=dict(size=8, family='Inter, Arial, sans-serif'),
        gridcolor='rgba(200,210,220,0.35)',
        row=1, col=2)

    # Подписи заголовков subplot
    for ann in fig.layout.annotations:
        if ann.text in ("Граф клинических соседей",
                        "Распределение GNN-вероятностей"):
            ann.font = dict(size=10, color='#1A3A5C',
                            family='Inter, Arial, sans-serif')

    return fig

