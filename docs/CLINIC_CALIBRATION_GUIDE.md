# Руководство по калибровке IVF Digital Twin под клинику

> Версия: 2.0 · Актуально для проекта v7.0  
> Это инструкция для себя — выполняется удалённо на компьютере клиники.

---

## Обзор: что происходит

```
Клиника заполняет Excel          Твой ПК (на месте в клинике)
─────────────────────        ─────────────────────────────────────────
clinic_intake_template.xlsx  →  validate_clinic_data.py
                             →  calibrate_for_clinic.py
                             →  clinic_adaptation_*.json  ─┐
                             →  gbdt_tau_*.joblib          ─┼→  изменения в 2 файлах
                             →  befe_ood_stats.npz         ─┘    проекта + перезапуск
```

Что даёт калибровка:
- **T_kat, T_gat, T_L1** — температурное масштабирование, убирает miscalibration
- **Динамический τ_KAT(X)** — GBDT повышает/снижает доверие к KAT vs GAT по профилю пациентки
- **OOD baseline** — Mahalanobis-детектор настроен на распределение именно этой клиники

---

## Предварительные требования

На компьютере клиники должны быть установлены:

```bash
python --version          # 3.9+
pip install pandas numpy scipy scikit-learn openpyxl xgboost joblib tqdm
```

Проект должен быть полностью установлен и работающий (`streamlit run app.py` запускается).

---

## 1. Структура файлов — куда что класть

```
project_root/               ← рядом с app.py
│
├── app.py
├── befe_app.py
├── clinic_config.json
│
├── validate_clinic_data.py         ← ПОЛОЖИТЬ СЮДА (скрипт валидатора)
├── calibrate_for_clinic.py         ← ПОЛОЖИТЬ СЮДА (скрипт калибровки)
│
├── data/
│   └── clinic_data_raw.xlsx        ← СЮДА кладём Excel от клиники
│
└── models/
    ├── Prediction_KAN.pth
    ├── FTTransformer.joblib
    ├── gnn_ivf_model.pt
    ├── befe_ood_stats.npz          ← ПЕРЕЗАПИШЕТСЯ после калибровки
    ├── clinic_adaptation_*.json    ← СОЗДАЁТСЯ калибровкой (главный файл)
    └── gbdt_tau_*.joblib           ← СОЗДАЁТСЯ если ≥200 исходов
```

**Важно:** `validate_clinic_data.py` и `calibrate_for_clinic.py` кладутся в корень проекта — туда же, где `app.py`. Оба скрипта ищут модели относительно текущей директории.

---

## 2. Пошаговый процесс

### Шаг 0 — Получить данные от клиники

Клиника прислала заполненный `clinic_intake_template.xlsx`. Кладёшь его в `data/`:

```bash
cp ~/Downloads/clinic_data.xlsx data/clinic_data_raw.xlsx
```

---

### Шаг 1 — Валидация данных

```bash
cd /path/to/project_root

python validate_clinic_data.py \
    --input  data/clinic_data_raw.xlsx \
    --output data/clinic_data_validated.csv \
    --report data/validation_report.txt
```

**Проверь отчёт:**

```bash
cat data/validation_report.txt
```

Если есть строки с `[ERROR]` — открой Excel клиники, исправь конкретные строки (номера строк указаны в отчёте), пересохрани, повтори Шаг 1.

**Приемлемый результат:**
```
Всего строк: 280
С ошибками:  8  (исключены)
Валидных:    272
Строк с известным исходом (для калибровки): 241
```

Если `Строк с известным исходом < 50` — калибровка ненадёжна. Запросить у клиники больше данных.

---

### Шаг 2 — Запуск калибровки

```bash
python calibrate_for_clinic.py \
    --data   data/clinic_data_validated.csv \
    --clinic "Название Клиники"
```

> ⚠️ Первый запуск долгий (20–40 мин при 250 строках) — прогоняет каждую строку  
> через все нейросети. Последующие запуски с `--skip-inference` быстрые.

**При повторной калибровке** (если модели уже прогнаны):

```bash
# Сначала сохраняем файл с предсказаниями который создался в прошлый раз
python calibrate_for_clinic.py \
    --data   data/clinic_data_validated_with_predictions.csv \
    --clinic "Название Клиники" \
    --skip-inference
```

**Ожидаемый вывод:**

```
[CALIB] Клиника: Название Клиники
[CALIB] Строк с исходом: 241  (беременность 34.4%)

[CALIB] Температурное масштабирование (per-model)...
  T_kat     T=1.124  ECE 0.0623 → 0.0381  (+38.8%)  n=241
  T_gat     T=0.891  ECE 0.0712 → 0.0498  (+30.1%)  n=198
  T_L1      T=1.056  ECE 0.0445 → 0.0390  (+12.4%)  n=241

[CALIB] OOD baseline (Mahalanobis)...
  Клиническое подпространство : n=272
  Лабораторное подпространство: n=248

[CALIB] GBDT meta-learner (динамический τ)...
[GBDT] Обучено на 198 примерах. Corr(pred, target) = 0.312

[OK] Адаптация сохранена: models/clinic_adaptation_НазваниеКлиники_20241215.json
```

---

### Шаг 3 — Обновить OOD-статистики

Скрипт `fit_befe_ood.py` использует иные названия колонок. Нужно запустить его с маппингом:

**Отредактируй `fit_befe_ood.py`** — найди `COLUMN_MAP` (строка ~34) и временно замени на:

```python
COLUMN_MAP = {
    "age":   "age",
    "amh":   "amh",
    "afc":   "afc",
    "bmi":   "bmi",
    "okk":   "okk",
    "mii":   "mii",
    "pn2":   "pn2",
    "blast": "blasts_total",
    "kpi":   "kpi_score",      # ← вычисляется на следующем шаге
}
```

Затем подготовь файл с kpi_score:

```bash
python -c "
import pandas as pd, sys
sys.path.insert(0,'.')
from calibrate_for_clinic import add_derived_features
df = pd.read_csv('data/clinic_data_validated.csv')
df = add_derived_features(df)
df.to_csv('data/clinic_data_for_ood.csv', index=False)
print('OK:', len(df), 'rows')
"
```

Запусти OOD fitting:

```bash
python fit_befe_ood.py data/clinic_data_for_ood.csv
```

Должно появиться:
```
Saved OOD stats -> models/befe_ood_stats.npz
Restart the app — the BEFE OOD detector is now ON.
```

> После работы **верни `COLUMN_MAP` в `fit_befe_ood.py`** к исходным русским именам  
> (те что были до правки) — иначе сломается работа с оригинальными данными обучения.

---

### Шаг 4 — Изменения в основном коде

Это нужно сделать **один раз**. После — автоматически применяется при каждом перезапуске.

#### 4.1 — `befe_app.py` : добавить параметр `tau_kat_override`

Открой `befe_app.py`. Найди функцию `build_befe_result` (строка ~245):

```python
def build_befe_result(res, *, p_kat_raw=None, ci_kat=(None, None),
                      p_gnn_raw=None, gnn_result=None, w_gnn=0.35,
                      csdi_result=None, age=None, amh=None, afc=None, bmi=None,
                      ood_stats=None):
```

Добавь параметр `tau_kat_override=None` в конец:

```python
def build_befe_result(res, *, p_kat_raw=None, ci_kat=(None, None),
                      p_gnn_raw=None, gnn_result=None, w_gnn=0.35,
                      csdi_result=None, age=None, amh=None, afc=None, bmi=None,
                      ood_stats=None, tau_kat_override=None):   # ← добавлено
```

Найди блок `tau_evidence` (~строка 283):

```python
    tau_evidence = {
        "KAT": 2.4 if p_kat is not None else 1e-6,   # KAT best-calibrated to our data
        "GAT": 1.0 if p_gat is not None else 1e-6,
    }
```

Замени на:

```python
    tau_evidence = {
        # tau_kat_override: динамический τ из GBDT meta-learner (clinic adaptation)
        # Если None — используется базовое 2.4 (обучение на общей когорте)
        "KAT": (tau_kat_override if tau_kat_override is not None else 2.4) if p_kat is not None else 1e-6,
        "GAT": 1.0 if p_gat is not None else 1e-6,
    }
```

---

#### 4.2 — `app.py` : загрузка адаптации + температурная калибровка + τ

Найди строку (~1278):

```python
    if "_befe_ood_stats" not in st.session_state:
```

**Сразу перед ней** (после строки `_befe_res, _befe_map = (None, {})`) добавь блок загрузки адаптации:

```python
# ── Clinic Adaptation — загрузка JSON (рядом с OOD-статистиками) ──────────
if "_clinic_adaptation" not in st.session_state:
    import glob as _glob
    _adapt_files = sorted(
        _glob.glob(os.path.join(_BASE_DIR, "models", "clinic_adaptation_*.json"))
    )
    if _adapt_files:
        try:
            with open(_adapt_files[-1], encoding="utf-8") as _af:
                st.session_state["_clinic_adaptation"] = json.load(_af)
        except Exception:
            st.session_state["_clinic_adaptation"] = None
    else:
        st.session_state["_clinic_adaptation"] = None
```

Найди блок `try:` который вызывает `build_befe_result` (~строка 1291):

```python
    try:
        _befe_res, _befe_map = build_befe_result(
            res,
            p_kat_raw   = _p_kat_raw,
```

**Сразу перед этим `try:`** добавь блок температурной калибровки:

```python
    # ── Применяем clinic adaptation (температура + динамический τ) ────────────
    _adapt     = st.session_state.get("_clinic_adaptation")
    _tau_kat_dyn = None
    if _adapt:
        try:
            from calibrate_for_clinic import (
                apply_clinic_calibration, compute_dynamic_tau_kat
            )
            _T = _adapt.get("temperature", {})
            # Температурное масштабирование raw вероятностей
            if _p_kat_raw is not None and abs(_T.get("T_kat", 1.0) - 1.0) > 0.01:
                _p_kat_raw = apply_clinic_calibration(_p_kat_raw, "T_kat", _adapt)
            if _p_gnn_raw is not None and abs(_T.get("T_gat", 1.0) - 1.0) > 0.01:
                _p_gnn_raw = apply_clinic_calibration(_p_gnn_raw, "T_gat", _adapt)
            # Динамический τ_KAT через GBDT meta-learner
            if _adapt.get("gbdt_tau_available"):
                _gbdt_feats = {
                    "age":          float(age),
                    "amh":          float(amh),
                    "afc":          int(afc),
                    "bmi":          float(bmi),
                    "attempt_number": int(attempt),
                    "okk":          float(res.get("okk_med",    0)),
                    "mii":          float(res.get("mii_med",    0)),
                    "pn2":          float(res.get("pn2_med",    0)),
                    "blasts_total": float(res.get("blasts_med", 0)),
                    "blasts_good":  float(res.get("good_med",   0)),
                }
                _tau_kat_dyn = compute_dynamic_tau_kat(_gbdt_feats, _adapt)
        except Exception as _adapt_exc:
            pass   # адаптация недоступна — работаем без неё
```

Найди вызов `build_befe_result` и добавь в него `tau_kat_override`:

```python
        _befe_res, _befe_map = build_befe_result(
            res,
            p_kat_raw        = _p_kat_raw,
            ci_kat           = _ci_kat,
            p_gnn_raw        = _p_gnn_raw,
            gnn_result       = _gnn_result,
            w_gnn            = _w_gnn,
            csdi_result      = st.session_state.get("csdi_result"),
            age=float(age), amh=float(amh), afc=int(afc), bmi=float(bmi),
            ood_stats        = st.session_state.get("_befe_ood_stats"),
            tau_kat_override = _tau_kat_dyn,   # ← добавлено
        )
```

---

### Шаг 5 — Обновить `clinic_config.json`

```json
{
  "clinic_name": "Название Клиники",
  "use_clinic_data": true,
  "batches": [
    [19, 43],
    [15, 38]
  ]
}
```

`batches` — реальные батчи клиники: `[успехов, всего переносов]`. Берёшь из данных клиники.  
Формат: каждый батч = один квартал или период сбора данных.

---

### Шаг 6 — Перезапуск приложения

```bash
# Останови текущий Streamlit (Ctrl+C или kill)
streamlit run app.py
```

После запуска в логах должно быть видно загрузку моделей. Для проверки — введи тестовую пациентку и убедись что BEFE tab работает.

---

## 3. Проверка что всё работает

```bash
# Проверить что JSON создался
ls -la models/clinic_adaptation_*.json

# Проверить что OOD stats обновились
python -c "
import numpy as np
z = np.load('models/befe_ood_stats.npz')
print('Clinical mean:', z['clinical_mu'])
print('Embryo mean:  ', z['embryo_mu'])
print('OK')
"

# Быстрый тест адаптации
python -c "
import json
with open(sorted(__import__('glob').glob('models/clinic_adaptation_*.json'))[-1]) as f:
    a = json.load(f)
print('Клиника:        ', a['clinic_name'])
print('Дата:           ', a['calibration_date'])
print('Циклов/исходов: ', a['n_cycles_total'], '/', a['n_cycles_outcomes'])
print('T_kat:          ', a['temperature']['T_kat'])
print('T_gat:          ', a['temperature']['T_gat'])
print('ECE KAT до/после:', a['ece']['before'].get('T_kat','—'), '→', a['ece']['after'].get('T_kat','—'))
print('GBDT tau:       ', a['gbdt_tau_available'])
"
```

---

## 4. Квартальная перекалибровка

Через 3 месяца — повторить процесс с новыми данными клиники.  
Клиника присылает выгрузку за истекший квартал.

```bash
# Новые данные добавить к существующим (объединить) или использовать отдельно
python validate_clinic_data.py \
    --input  data/new_quarter_data.xlsx \
    --output data/new_quarter_validated.csv

# Можно объединить с предыдущими данными
python -c "
import pandas as pd
old = pd.read_csv('data/clinic_data_validated.csv')
new = pd.read_csv('data/new_quarter_validated.csv')
combined = pd.concat([old, new], ignore_index=True).drop_duplicates(subset='cycle_id')
combined.to_csv('data/all_data_combined.csv', index=False)
print('Combined:', len(combined), 'rows')
"

python calibrate_for_clinic.py \
    --data   data/all_data_combined.csv \
    --clinic "Название Клиники" \
    --skip-inference   # используем уже прогнанные предсказания из CSV
```

> Если данные новые и предсказаний нет — убрать `--skip-inference`.

Затем повторить **Шаг 3** (OOD stats) и **Шаг 5** (clinic_config.json).  
Шаги 4.1 и 4.2 (изменения в коде) делать **не надо** — они уже сделаны раз и навсегда.

---

## 5. Файлы которые меняются при каждой новой клинике

| Файл | Что меняется | Как |
|---|---|---|
| `models/clinic_adaptation_*.json` | Создаётся новый | `calibrate_for_clinic.py` |
| `models/gbdt_tau_*.joblib` | Создаётся новый (если ≥200 исходов) | `calibrate_for_clinic.py` |
| `models/befe_ood_stats.npz` | Перезаписывается | `fit_befe_ood.py` |
| `clinic_config.json` | Название + batches | Вручную |

**Код** (`app.py`, `befe_app.py`) меняется **один раз** и потом не трогается.  
При переходе к другой клинике — просто заменить `.json`, `.joblib`, `.npz` и `clinic_config.json`.

---

## 6. Как хранить адаптации нескольких клиник

```
models/
├── clinic_adaptation_КлиникаА_20241215.json    ← активна (последняя по дате)
├── clinic_adaptation_КлиникаА_20240915.json    ← архив
├── gbdt_tau_КлиникаА.joblib                    ← активна
├── clinic_adaptation_КлиникаБ_20241110.json    ← неактивна
└── befe_ood_stats.npz                          ← всегда одна (текущей клиники)
```

`app.py` автоматически берёт **последний по имени файл** `clinic_adaptation_*.json` через `sorted()`.  
При смене активной клиники достаточно скопировать её `.json` и `.npz` в `models/` и перезапустить.

---

## 7. Устранение проблем

### `ModuleNotFoundError: calibrate_for_clinic`
```bash
# calibrate_for_clinic.py не в корне проекта
# Убедиться что файл рядом с app.py:
ls app.py calibrate_for_clinic.py
```

### `ECE before/after одинаковые` (T ≈ 1.0)
Модель уже хорошо откалибрована для этой клиники — это нормально. Продолжать.

### `GBDT пропущен: только N строк`
Меньше 200 строк с известными исходами. Запросить больше данных у клиники.  
Пока используется фиксированный τ_KAT = 2.4.

### `OOD baseline: n=0`
Все строки содержат NaN в клинических полях (age/amh/afc/bmi).  
Проверить, правильно ли заполнен шаблон (`validate_clinic_data.py` выдаст детали).

### `ImportError при загрузке adaaptation в app.py`
Ошибка в блоке `try/except` — она подавляется (`pass`). Адаптация просто не применяется.  
Для диагностики: убрать `except ... pass`, перезапустить и смотреть traceback.

### Нужно откатиться к состоянию без калибровки
```bash
# Удалить JSON и NPZ — app.py работает без них
mv models/clinic_adaptation_*.json /tmp/
```

---

## Контрольный чеклист (распечатать и отмечать)

```
□ 1. Excel от клиники получен и скопирован в data/
□ 2. Запущен validate_clinic_data.py, отчёт проверен
□ 3. Ошибки в Excel исправлены, валидация повторена (если нужно)
□ 4. Запущен calibrate_for_clinic.py, выходные файлы появились в models/
□ 5. Обновлён COLUMN_MAP в fit_befe_ood.py (под наши имена колонок)
□ 6. Запущен fit_befe_ood.py, models/befe_ood_stats.npz обновлён
□ 7. COLUMN_MAP в fit_befe_ood.py возвращён к исходному (русские имена)
□ 8. В befe_app.py добавлен параметр tau_kat_override (строка ~245 и ~283)
□ 9. В app.py добавлен блок загрузки адаптации (перед OOD loading)
□ 10. В app.py добавлен блок температурной калибровки (перед build_befe_result)
□ 11. В app.py в вызов build_befe_result добавлен tau_kat_override
□ 12. Обновлён clinic_config.json (название + batches)
□ 13. Приложение перезапущено, тестовая пациентка введена, BEFE tab работает
□ 14. Запущен python-скрипт проверки (Раздел 3) — всё OK
```
