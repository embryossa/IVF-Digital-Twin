"""
ПАТЧ ДЛЯ app.py — IVF Digital Twin v7.0
========================================

ШАГ 1.  Добавить импорт dt_ui в начало файла (после импорта befe):

    try:
        import dt_ui as UI
        _UI_OK = True
    except ImportError:
        _UI_OK = False

ШАГ 2.  Заменить блок CSS (строки 400–428 в app.py):

    # БЫЛО:
    st.markdown(\"\"\"<style>...</style>\"\"\", unsafe_allow_html=True)

    # СТАЛО:
    if _UI_OK:
        UI.inject_css()
    else:
        st.markdown(\"\"\"<style>
            .main { background-color: #f8fafc; }
            ...старый CSS...
        </style>\"\"\", unsafe_allow_html=True)

ШАГ 3.  Добавить заголовок в каждую вкладку.
         Вставлять ПЕРВОЙ строкой внутри блока `with tabs[N]:`.

──────────────────────────────────────────────────────
TAB 0  — 🔬 Pipeline
──────────────────────────────────────────────────────
with tabs[0]:
    if _UI_OK: UI.tab_header_by_key("pipeline")
    col_f, col_v = st.columns([1, 2])
    ...

──────────────────────────────────────────────────────
TAB 1  — 📈 Беременность
──────────────────────────────────────────────────────
with tabs[1]:
    if _UI_OK: UI.tab_header_by_key("pregnancy")
    col_a, col_b = st.columns(2)
    ...

──────────────────────────────────────────────────────
TAB 2  — 🧠 Кластер
──────────────────────────────────────────────────────
with tabs[2]:
    if _UI_OK: UI.tab_header_by_key("cluster")
    col_pca, col_info = st.columns([3, 2])
    ...

──────────────────────────────────────────────────────
TAB 3  — 📉 Байес + попытки
──────────────────────────────────────────────────────
with tabs[3]:
    if _UI_OK: UI.tab_header_by_key("bayes")
    col_bay, col_att = st.columns(2)
    ...

──────────────────────────────────────────────────────
TAB 4  — ⚠️ Риски
──────────────────────────────────────────────────────
with tabs[4]:
    if _UI_OK: UI.tab_header_by_key("risks")
    col_r1, col_r2 = st.columns(2)
    ...

──────────────────────────────────────────────────────
TAB 5  — 🏦 Банкинг
──────────────────────────────────────────────────────
with tabs[5]:
    if _UI_OK: UI.tab_header_by_key("banking")
    eb = _eb
    ...

──────────────────────────────────────────────────────
TAB 6  — 🧬 Diffusion
──────────────────────────────────────────────────────
with tabs[6]:
    if _UI_OK: UI.tab_header_by_key("diffusion")
    ...

──────────────────────────────────────────────────────
TAB 7  — 🕸️ GAT Graph
──────────────────────────────────────────────────────
with tabs[7]:
    if _UI_OK: UI.tab_header_by_key("gat")
    ...

──────────────────────────────────────────────────────
TAB 8  — ⚖️ BEFE (L7)  ← обработан автоматически в render_befe_tab
──────────────────────────────────────────────────────
with tabs[8]:
    # Заголовок вставляет сам render_befe_tab из befe_app.py
    if not _BEFE_OK:
        ...

ШАГ 4.  Опциональная замена подзаголовков внутри вкладок.

    # БЫЛО:
    st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;
                margin:0 0 6px 0">Воронка (медианы)</p>', unsafe_allow_html=True)

    # СТАЛО:
    if _UI_OK:
        UI.section_header("Воронка (медианы)")
    else:
        st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;
                    margin:0 0 6px 0">Воронка (медианы)</p>', unsafe_allow_html=True)

    # ИЛИ короче (везде одинаково):
    (UI.section_header if _UI_OK else
     lambda t: st.markdown(f'<p style="font-size:15px;font-weight:600;color:#1B4F72;'
                            f'margin:0 0 6px 0">{t}</p>', unsafe_allow_html=True)
    )("Воронка (медианы)")

ШАГ 5.  Замена блоков result-box в TAB 2 (беременность):

    # БЫЛО:
    st.markdown(f\"\"\"
    <div class="result-box">
    <b>Трёхуровневая декомпозиция...</b>
    ...
    </div>\"\"\", unsafe_allow_html=True)

    # СТАЛО:
    if _UI_OK:
        UI.result_box(f\"\"\"
        <b>Трёхуровневая декомпозиция вероятности беременности:</b><br><br>
        <b>[1] На один перенос:</b> {res['p_per_transfer']*100:.1f}%
        &nbsp;&nbsp;(если перенос состоится)<br>
        <b>[2] Если цикл viable (≥1 перенос):</b> {res['p_cum_if_viable']*100:.1f}%
        &nbsp;&nbsp;(95% CI: {res['rate_ci'][0]*100:.1f}–{res['rate_ci'][1]*100:.1f}%)<br>
        <b>[3] Успех цикла (от стимуляции):</b> {res['p_overall_cycle']*100:.1f}%
        &nbsp;&nbsp;= P(viable {res['p_viable']*100:.0f}%) × [2]
        \"\"\", kind="info")
    else:
        st.markdown(f'<div class="result-box">...</div>', unsafe_allow_html=True)

ШАГ 6.  Обновление заголовка приложения (title + disclaimer):

    # БЫЛО:
    st.title("IVF Digital Twin")
    st.markdown('<div class="disclaimer">...</div>', unsafe_allow_html=True)

    # СТАЛО:
    st.title("IVF Digital Twin v7.0")
    if _UI_OK:
        st.markdown(
            '<div class="dt-disclaimer">⚠ <b>Только для поддержки клинического решения.</b> '
            'Все прогнозы являются вероятностными оценками. '
            'Окончательное решение принимает врач-репродуктолог.<br>'
            '<i>IVF Digital Twin v7.0 · Sergeev et al., 2025</i></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="disclaimer">...</div>', unsafe_allow_html=True)

===========================================================
МИНИМАЛЬНЫЙ ВАРИАНТ (только CSS + BEFE + заголовки вкладок):
===========================================================

1.  Скопировать dt_ui.py в папку проекта (рядом с app.py)
2.  Скопировать befe_app.py (новую версию) в папку проекта
3.  Добавить в начало app.py:
        try:
            import dt_ui as UI
            _UI_OK = True
        except ImportError:
            _UI_OK = False
4.  Заменить CSS-блок:
        if _UI_OK: UI.inject_css()
5.  Добавить по одной строке в начало каждого with tabs[N]::
        if _UI_OK: UI.tab_header_by_key("<key>")

Это уже даёт красивые вкладки, метрики, шапки секций и полностью
переработанную вкладку BEFE без изменения остальной логики.
"""

# Этот файл — инструкция. Реальные изменения в dt_ui.py и befe_app.py.
