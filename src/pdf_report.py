"""
IVF Digital Twin v6.2 — Клинический PDF-отчёт
Пастельный дизайн, читаемые графики, пригоден для печати.
"""

import io, os, copy
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image as RLImage,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Шрифты ───────────────────────────────────────────────────
_FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts")

def _register_fonts():
    r = os.path.join(_FONTS_DIR, "DejaVuSans.ttf")
    b = os.path.join(_FONTS_DIR, "DejaVuSans-Bold.ttf")
    i = os.path.join(_FONTS_DIR, "DejaVuSans-Oblique.ttf")
    if os.path.exists(r) and os.path.exists(b):
        pdfmetrics.registerFont(TTFont("DV",   r))
        pdfmetrics.registerFont(TTFont("DV-B", b))
        pdfmetrics.registerFont(TTFont("DV-I", i if os.path.exists(i) else r))
        return "DV", "DV-B", "DV-I"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"

F, FB, FI = _register_fonts()

# ── Пастельная палитра ────────────────────────────────────────
# Заголовки разделов
C_HDR_BG   = colors.HexColor("#D6E4F0")   # нежно-голубой фон заголовка
C_HDR_TXT  = colors.HexColor("#1A3A5C")   # тёмно-синий текст
C_HDR_LINE = colors.HexColor("#7BAFD4")   # акцентная линия

# Страничный хедер
C_PAGE_HDR = colors.HexColor("#EBF5FB")
C_PAGE_HDR_LINE = colors.HexColor("#7BAFD4")

# Метрические карточки
C_CARD_BLUE   = colors.HexColor("#EBF5FB")
C_CARD_GREEN  = colors.HexColor("#E9F7EF")
C_CARD_AMBER  = colors.HexColor("#FEF9E7")
C_CARD_BORDER_BLUE  = colors.HexColor("#85C1E9")
C_CARD_BORDER_GREEN = colors.HexColor("#82E0AA")
C_CARD_BORDER_AMBER = colors.HexColor("#F9E79F")

# Текст
C_NAVY   = colors.HexColor("#1A3A5C")
C_TEAL   = colors.HexColor("#1A7A7A")
C_GREEN  = colors.HexColor("#1E7B4B")
C_AMBER  = colors.HexColor("#9A6B00")
C_RED    = colors.HexColor("#922B21")
C_GREY   = colors.HexColor("#6B7280")
C_BLACK  = colors.HexColor("#1C2833")
C_WHITE  = colors.white

# Таблицы
C_TBL_ODD  = colors.HexColor("#FDFEFE")
C_TBL_EVEN = colors.HexColor("#F4F9FD")
C_TBL_HDR  = colors.HexColor("#D6E4F0")
C_BORDER   = colors.HexColor("#BDC3C7")

W, H = A4
MARGIN = 1.8*cm
PAGE_W = W - 2*MARGIN

# ── Стили текста ─────────────────────────────────────────────
def S(name, **kw): return ParagraphStyle(name, **kw)

ST = {
    "cover_title": S("ct", fontName=FB, fontSize=26, textColor=C_NAVY,
                     alignment=TA_CENTER, spaceAfter=5, leading=32),
    "cover_sub":   S("cs", fontName=F,  fontSize=12, textColor=C_TEAL,
                     alignment=TA_CENTER, spaceAfter=4),
    "cover_info":  S("ci", fontName=F,  fontSize=9,  textColor=C_GREY,
                     alignment=TA_CENTER),

    "sec_hdr":     S("sh", fontName=FB, fontSize=11, textColor=C_HDR_TXT,
                     leading=15),
    "subsec":      S("ss", fontName=FB, fontSize=10, textColor=C_NAVY,
                     spaceBefore=8, spaceAfter=4),
    "body":        S("b",  fontName=F,  fontSize=9,  textColor=C_BLACK,
                     spaceAfter=3, leading=14),
    "body_sm":     S("bsm",fontName=F,  fontSize=8,  textColor=C_GREY,
                     leading=12, spaceAfter=2),
    "label":       S("lb", fontName=FB, fontSize=9,  textColor=C_BLACK),
    "label_hdr":   S("lh", fontName=FB, fontSize=9,  textColor=C_NAVY),

    "mv":          S("mv", fontName=FB, fontSize=20, textColor=C_NAVY,
                     alignment=TA_CENTER, leading=24),
    "mv_g":        S("mvg",fontName=FB, fontSize=20, textColor=C_GREEN,
                     alignment=TA_CENTER, leading=24),
    "mv_a":        S("mva",fontName=FB, fontSize=20, textColor=C_AMBER,
                     alignment=TA_CENTER, leading=24),
    "ml":          S("ml", fontName=F,  fontSize=7.5,textColor=C_GREY,
                     alignment=TA_CENTER, leading=10),
    "caption":     S("cp", fontName=FI, fontSize=7.5,textColor=C_GREY,
                     alignment=TA_CENTER, spaceAfter=2),
    "warn":        S("wn", fontName=FB, fontSize=9,  textColor=C_RED, spaceAfter=3),
    "reco":        S("rc", fontName=F,  fontSize=9,  textColor=C_GREEN,
                     leftIndent=10, leading=14, spaceAfter=3),
    "note":        S("nt", fontName=FI, fontSize=8,  textColor=C_GREY,
                     leading=12, spaceAfter=4),
    "footer":      S("ft", fontName=F,  fontSize=7,  textColor=C_GREY,
                     alignment=TA_CENTER),
}


# ── Элементы оформления ───────────────────────────────────────

def sec_header(title):
    """Пастельный заголовок раздела с левой цветной полосой."""
    p = Paragraph(title, ST["sec_hdr"])
    t = Table([[p]], colWidths=[PAGE_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_HDR_BG),
        ("LINEBELOW",     (0,0),(-1,-1), 2, C_HDR_LINE),
        ("TOPPADDING",    (0,0),(-1,-1), 7),
        ("BOTTOMPADDING", (0,0),(-1,-1), 7),
        ("LEFTPADDING",   (0,0),(-1,-1), 12),
    ]))
    return t


def kv_table(rows, col1=5.5*cm):
    col2 = PAGE_W - col1
    data = [[Paragraph(k, ST["label"]), Paragraph(str(v), ST["body"])]
            for k, v in rows]
    t = Table(data, colWidths=[col1, col2])
    t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0,0),(-1,-1), [C_TBL_ODD, C_TBL_EVEN]),
        ("GRID",           (0,0),(-1,-1), 0.4, C_BORDER),
        ("TOPPADDING",     (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
        ("LEFTPADDING",    (0,0),(-1,-1), 8),
        ("VALIGN",         (0,0),(-1,-1), "MIDDLE"),
    ]))
    return t


def metric_card(value, label, mv_style="mv",
                bg=C_CARD_BLUE, border=C_CARD_BORDER_BLUE, width=3.2*cm):
    data = [[Paragraph(value, ST[mv_style])],
            [Paragraph(label, ST["ml"])]]
    t = Table(data, colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), bg),
        ("BOX",           (0,0),(-1,-1), 1.2, border),
        ("TOPPADDING",    (0,0),(-1,-1), 8),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
    ]))
    return t


def _prepare_fig(fig, extra_margin_b=60, extra_margin_l=70,
                 extra_margin_r=30, extra_margin_t=40):
    """
    Создаёт копию figure с белым фоном, Inter шрифтом и увеличенными отступами
    для PDF-рендера (чтобы подписи осей, легенда и аннотации не обрезались).
    """
    try:
        f = copy.deepcopy(fig)
        f.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="rgba(248,250,252,1)",
            margin=dict(
                l=extra_margin_l,
                r=extra_margin_r,
                t=extra_margin_t,
                b=extra_margin_b,
            ),
            font=dict(family="Inter, Arial, sans-serif", size=11, color="#1C2833"),
            # Легенда — горизонтальная, по центру сверху (как в макете)
            legend=dict(
                orientation="h",
                x=0.5, xanchor="center",
                y=1.04, yanchor="bottom",
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="#dddddd",
                borderwidth=1,
                font=dict(size=10),
            ),
            # Сетка — тонкая, читаемая
            xaxis=dict(
                gridcolor="rgba(190,200,215,0.45)",
                tickfont=dict(size=10),
            ),
            yaxis=dict(
                gridcolor="rgba(190,200,215,0.45)",
                tickfont=dict(size=10),
            ),
        )
        # Подзаголовки subplot тоже читаемы
        for ann in f.layout.annotations:
            if ann.font and ann.font.size and ann.font.size < 10:
                ann.font.size = 10
        return f
    except Exception:
        return fig


def fig_to_image(fig, w_cm, h_cm, margin_b=65, margin_l=75, margin_r=30, margin_t=50):
    """Plotly figure → ReportLab Image с нормальными отступами и 2× разрешением."""
    if fig is None:
        return None
    try:
        f = _prepare_fig(fig, extra_margin_b=margin_b, extra_margin_l=margin_l,
                         extra_margin_r=margin_r, extra_margin_t=margin_t)
        # 2× DPI для чёткости при печати
        px_w = int(w_cm * 55)
        px_h = int(h_cm * 55)
        png = f.to_image(format="png", width=px_w, height=px_h, scale=2)
        return RLImage(io.BytesIO(png), width=w_cm * cm, height=h_cm * cm)
    except Exception:
        return None


def chart_block(fig, caption_text, w_cm=None, h_cm=7.5,
                margin_b=65, margin_l=75, margin_r=30, margin_t=50):
    """График на полную ширину + подпись под ним."""
    if w_cm is None:
        w_cm = PAGE_W / cm
    img = fig_to_image(fig, w_cm, h_cm, margin_b=margin_b, margin_l=margin_l,
                       margin_r=margin_r, margin_t=margin_t)
    if img is None:
        return []

    # Рамка вокруг графика
    chart_t = Table([[img]], colWidths=[w_cm * cm])
    chart_t.setStyle(TableStyle([
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
    ]))
    return [
        chart_t,
        Paragraph(caption_text, ST["caption"]),
        Spacer(1, 6),
    ]


# ── Хедер/футер страницы ─────────────────────────────────────

class _PageTemplate:
    def __init__(self, clinic, patient_name, logo_path):
        self.clinic       = clinic
        self.patient_name = patient_name
        self.logo_path    = logo_path

    def __call__(self, canvas, doc):
        canvas.saveState()
        w, h = A4

        # Верхняя полоса — пастельный голубой
        canvas.setFillColor(C_PAGE_HDR)
        canvas.rect(0, h - 1.1*cm, w, 1.1*cm, fill=1, stroke=0)
        canvas.setFillColor(C_PAGE_HDR_LINE)
        canvas.rect(0, h - 1.1*cm, w, 1.5*mm, fill=1, stroke=0)

        # Логотип
        if self.logo_path and os.path.exists(self.logo_path):
            canvas.drawImage(self.logo_path,
                             MARGIN, h - 1.0*cm,
                             width=0.75*cm, height=0.75*cm,
                             preserveAspectRatio=True, mask="auto")

        canvas.setFont(FB, 9.5)
        canvas.setFillColor(C_NAVY)
        canvas.drawString(MARGIN + 1.0*cm, h - 0.70*cm, "IVF Digital Twin v6.2")

        if self.patient_name:
            canvas.setFont(F, 8.5)
            canvas.setFillColor(C_GREY)
            canvas.drawRightString(w - MARGIN, h - 0.70*cm, self.patient_name)

        # Нижняя полоса
        canvas.setFillColor(colors.HexColor("#F8F9FA"))
        canvas.rect(0, 0, w, 0.9*cm, fill=1, stroke=0)
        canvas.setFillColor(C_PAGE_HDR_LINE)
        canvas.rect(0, 0.9*cm, w, 0.5*mm, fill=1, stroke=0)

        canvas.setFont(F, 6.5)
        canvas.setFillColor(C_GREY)
        canvas.drawString(MARGIN, 0.34*cm,
            "IVF Digital Twin v6.2  ·  Sergeev et al., 2025  ·  embryossa@gmail.com  ·  "
            "Research prototype — not for standalone clinical use")
        canvas.drawRightString(w - MARGIN, 0.34*cm, f"Стр. {doc.page}")
        canvas.restoreState()


# ═════════════════════════════════════════════════════════════
#  ГЛАВНАЯ ФУНКЦИЯ
# ═════════════════════════════════════════════════════════════

def generate_patient_report(
    patient_name, patient_id,
    age, amh, afc, bmi, attempt, sperm_source,
    known, res, eb, post,
    fig_funnel=None, fig_violin=None, fig_bar=None,
    fig_pca=None, fig_bayes=None, fig_attempts=None,
    fig_risks=None, fig_csdi=None,
    csdi_result=None,
    clinic_name="",
    cluster_recommendations="",
    warnings_list=None,
    p_kat_raw=None,
    p_nvsa=None,           # оставлено для совместимости (не отображается)
    ci_kat=(None, None),
    ci_nvsa=(None, None),  # оставлено для совместимости (не отображается)
    p_gnn_ens=None,        # GAT Ансамбль: итоговый скор (w*GNN + (1-w)*KAT)
    p_gnn_raw=None,        # GAT: чистый выход Graph Transformer
    w_gnn=0.35,            # вес GNN в ансамбле
    fig_gnn=None,          # Plotly-фигура: граф соседей
) -> bytes:

    buf = io.BytesIO()
    now = datetime.now()
    logo = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "logo22.png")
    tpl = _PageTemplate(clinic_name, patient_name, logo)

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=1.6*cm, bottomMargin=1.4*cm,
        title=f"IVF Digital Twin — {patient_name}",
        author="IVF Digital Twin v6.2",
    )

    def pct(v):
        try: return f"{float(v)*100:.1f}%"
        except: return "—"

    def _kv_get(obj, attr):
        if obj is None: return None
        if hasattr(obj, attr): return getattr(obj, attr)
        if isinstance(obj, dict): return obj.get(attr)
        return None

    story = []

    # ══════════════════════════════════════════════════════════
    #  СТРАНИЦА 1 — ТИТУЛ + ДАННЫЕ + ПОКАЗАТЕЛИ
    # ══════════════════════════════════════════════════════════
    story.append(Spacer(1, 0.4*cm))

    # Заголовок
    if os.path.exists(logo):
        logo_img = RLImage(logo, width=1.8*cm, height=1.8*cm)
        hdr = Table(
            [[logo_img,
              [Paragraph("IVF Digital Twin v6.2", ST["cover_title"]),
               Paragraph("Индивидуальный клинический отчёт", ST["cover_sub"])]]],
            colWidths=[2.2*cm, PAGE_W - 2.2*cm]
        )
        hdr.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                                  ("LEFTPADDING",(0,0),(-1,-1),0)]))
        story.append(hdr)
    else:
        story.append(Paragraph("IVF Digital Twin v6.2", ST["cover_title"]))
        story.append(Paragraph("Индивидуальный клинический отчёт", ST["cover_sub"]))

    story.append(Spacer(1, 4))

    # Инфополоска
    info = Table(
        [[Paragraph(f"Клиника: <b>{clinic_name or '—'}</b>", ST["cover_info"]),
          Paragraph(f"Дата: <b>{now.strftime('%d.%m.%Y')}</b>", ST["cover_info"]),
          Paragraph(f"Время: <b>{now.strftime('%H:%M')}</b>",  ST["cover_info"])]],
        colWidths=[PAGE_W/3]*3
    )
    info.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_TBL_EVEN),
        ("BOX",           (0,0),(-1,-1), 0.5, C_BORDER),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
    ]))
    story.append(info)
    story.append(Spacer(1, 10))

    # Данные пациентки
    story.append(sec_header("Данные пациентки"))
    story.append(Spacer(1, 4))

    obs_parts = []
    for lbl, attr in [("ОКК", "okk"),("MII","mii"),("2PN","pn2"),
                      ("Бласт","blasts"),("Хор.кач","good"),("Эупл","euploid")]:
        v = _kv_get(known, attr)
        if v: obs_parts.append(f"{lbl}: {v}")

    rows_p = [
        ("ФИО пациентки",    patient_name or "—"),
        ("Номер карты / ID", patient_id   or "—"),
        ("Возраст",          f"{age:.0f} лет"),
        ("АМГ",              f"{amh:.2f} нг/мл"),
        ("АФС",              f"{afc} антральных фолликулов"),
        ("ИМТ",              f"{bmi:.1f} кг/м²"),
        ("Попытка ЭКО",      f"№ {attempt}"),
        ("Источник спермы",  sperm_source or "—"),
    ]
    if obs_parts:
        rows_p.append(("Данные mid-cycle", "  ·  ".join(obs_parts)))
    story.append(kv_table(rows_p))
    story.append(Spacer(1, 12))

    # Ключевые показатели
    story.append(sec_header("Ключевые показатели прогноза"))
    story.append(Spacer(1, 8))

    p_transfer = res.get("p_per_transfer", 0)
    p_viable   = res.get("p_cum_if_viable", 0)
    p_cycle    = res.get("p_overall_cycle", 0)
    p_bayes    = post.get("mean", 0)
    kpi        = res.get("kpi_score_median", 0)
    ci_lo      = post.get("ci_low", 0)
    ci_hi      = post.get("ci_high", 0)

    def _mstyle(v):
        if v >= 0.55: return "mv_g", C_CARD_GREEN,  C_CARD_BORDER_GREEN
        if v >= 0.35: return "mv",   C_CARD_BLUE,   C_CARD_BORDER_BLUE
        return "mv_a", C_CARD_AMBER, C_CARD_BORDER_AMBER

    cw = (PAGE_W - 4*4) / 5
    kpi_s = "mv_g" if kpi>=18 else ("mv" if kpi>=12 else "mv_a")
    kpi_bg = C_CARD_GREEN if kpi>=18 else (C_CARD_BLUE if kpi>=12 else C_CARD_AMBER)
    kpi_bc = C_CARD_BORDER_GREEN if kpi>=18 else (C_CARD_BORDER_BLUE if kpi>=12 else C_CARD_BORDER_AMBER)

    # KAT raw и GAT Ансамбль
    _kat_val  = pct(p_kat_raw) if p_kat_raw  is not None else "—"
    _gnn_val  = pct(p_gnn_ens) if p_gnn_ens  is not None else "—"
    _kat_sty  = _mstyle(p_kat_raw) if p_kat_raw is not None else ("mv", C_CARD_BLUE,  C_CARD_BORDER_BLUE)
    _gnn_sty  = _mstyle(p_gnn_ens) if p_gnn_ens is not None else ("mv", C_CARD_BLUE,  C_CARD_BORDER_BLUE)

    cards = Table([[
        metric_card(pct(p_transfer), "P(беременность)\nна перенос",  *_mstyle(p_transfer), cw),
        metric_card(pct(p_viable),   "Если viable\nцикл",            *_mstyle(p_viable),   cw),
        metric_card(pct(p_cycle),    "P(успех\nцикла)",              *_mstyle(p_cycle),    cw),
        metric_card(_kat_val,        "KAT\n(ансамбль NN)",           *_kat_sty,            cw),
        metric_card(_gnn_val,        "GAT\nАнсамбль",                *_gnn_sty,            cw),
    ]], colWidths=[cw]*5, hAlign="CENTER")
    cards.setStyle(TableStyle([
        ("ALIGN",  (0,0),(-1,-1),"CENTER"),
        ("VALIGN", (0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",  (0,0),(-1,-1),4),
        ("RIGHTPADDING", (0,0),(-1,-1),4),
    ]))
    story.append(cards)
    story.append(Spacer(1, 6))
    ci_parts = []
    if p_kat_raw is not None and ci_kat[0] is not None:
        ci_parts.append(f"KAT 95% CI: <b>{pct(ci_kat[0])} – {pct(ci_kat[1])}</b>")
    if p_gnn_ens is not None:
        _w_kat = round(1.0 - w_gnn, 2)
        _gnn_str = (f"GAT Ансамбль: <b>{pct(p_gnn_ens)}</b>  "
                    f"(Graph Transformer {pct(p_gnn_raw) if p_gnn_raw is not None else '—'}  "
                    f"· w={w_gnn:.2f}×GNN + {_w_kat:.2f}×KAT)")
        ci_parts.append(_gnn_str)
    if ci_parts:
        story.append(Paragraph("  ·  ".join(ci_parts), ST["body"]))
    if warnings_list:
        story.append(Spacer(1, 4))
        for w in warnings_list:
            story.append(Paragraph(f"⚠  {w}", ST["warn"]))

    # ══════════════════════════════════════════════════════════
    #  СТРАНИЦА 2 — MONTE CARLO
    # ══════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(sec_header("Monte Carlo — результаты моделирования"))
    story.append(Spacer(1, 8))

    half = (PAGE_W - 1*cm) / 2
    mc_l = [("ОКК получено",       f"{res.get('okk_med',0):.1f}"),
            ("MII ооцитов",        f"{res.get('mii_med',0):.1f}"),
            ("2PN зиготы",         f"{res.get('pn2_med',0):.1f}"),
            ("Бластоцисты всего",  f"{res.get('bl5_med',0):.1f}")]
    mc_r = [("Хор. кач. бласт",   f"{res.get('good_med',0):.1f}"),
            ("Эуплоидные (ПГТ-А)",f"{res.get('euploid_med', res.get('good_med',0)*0.6):.1f}"),
            ("KPI Score",          f"{res.get('kpi_score_median',0):.1f} / 25"),
            ("Риск отмены цикла",  f"{res.get('p_cancel',0)*100:.1f}%" if res.get('p_cancel') else "—")]

    def mini_kv(rows, w):
        data = [[Paragraph(k, ST["label"]), Paragraph(v, ST["body"])] for k,v in rows]
        t = Table(data, colWidths=[w*0.55, w*0.45])
        t.setStyle(TableStyle([
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[C_TBL_ODD, C_TBL_EVEN]),
            ("GRID",(0,0),(-1,-1),0.4,C_BORDER),
            ("TOPPADDING",(0,0),(-1,-1),4),
            ("BOTTOMPADDING",(0,0),(-1,-1),4),
            ("LEFTPADDING",(0,0),(-1,-1),6),
        ]))
        return t

    mc_tbl = Table([[mini_kv(mc_l, half), Spacer(1*cm,1), mini_kv(mc_r, half)]],
                   colWidths=[half, 1*cm, half])
    mc_tbl.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                                 ("LEFTPADDING",(0,0),(-1,-1),0),
                                 ("RIGHTPADDING",(0,0),(-1,-1),0)]))
    story.append(mc_tbl)
    story.append(Spacer(1, 10))

    # Воронка — побольше высота, увеличенный отступ снизу для подписей
    if fig_funnel:
        story.extend(chart_block(fig_funnel,
            "Воронка ЭКО — медианные значения на каждом этапе",
            h_cm=7.5, margin_b=50, margin_l=120, margin_r=30, margin_t=45))

    # Скрипки — нужен большой отступ снизу для подписей осей
    if fig_violin:
        story.extend(chart_block(fig_violin,
            "Распределения Monte Carlo по стадиям (5 000 итераций)",
            h_cm=8.5, margin_b=70, margin_l=65, margin_r=30, margin_t=65))

    # ══════════════════════════════════════════════════════════
    #  СТРАНИЦА 3 — ВЕРОЯТНОСТЬ + КЛАСТЕР
    # ══════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(sec_header("Анализ вероятности беременности"))
    story.append(Spacer(1, 6))

    if fig_bar:
        story.extend(chart_block(fig_bar,
            "Вероятность достижения ≥k клинических беременностей в данном цикле",
            h_cm=8, margin_b=60, margin_l=70, margin_r=30, margin_t=60))

    story.append(Spacer(1, 6))
    story.append(sec_header("Кластерный анализ пациентки"))
    story.append(Spacer(1, 5))

    ca  = res.get("cluster_analysis", {})
    dom = ca.get("dominant_cluster", "—")
    story.append(Paragraph(f"Доминирующий кластер: <b>{dom}</b>", ST["body"]))
    if cluster_recommendations:
        story.append(Spacer(1, 3))
        story.append(Paragraph("Рекомендации:", ST["subsec"]))
        story.append(Paragraph(cluster_recommendations, ST["reco"]))

    if fig_pca:
        story.extend(chart_block(fig_pca,
            "PCA-проекция: положение пациентки относительно кластеров (18D → 2D)",
            h_cm=9, margin_b=65, margin_l=70, margin_r=30, margin_t=65))

    # ══════════════════════════════════════════════════════════
    #  СТРАНИЦА 4 — БАЙЕС + РИСКИ
    # ══════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(sec_header("Байесовский анализ и динамика по попыткам"))
    story.append(Spacer(1, 6))

    story.append(kv_table([
        ("Posterior (среднее)",      pct(post.get("mean",0))),
        ("95% ДИ — нижняя граница",  pct(post.get("ci_low",0))),
        ("95% ДИ — верхняя граница", pct(post.get("ci_high",0))),
    ], col1=6.5*cm))
    story.append(Spacer(1, 8))

    # Байес и попытки — каждый на полную ширину для читаемости
    if fig_bayes:
        story.extend(chart_block(fig_bayes,
            "Байесовский posterior: prior vs posterior",
            h_cm=7, margin_b=60, margin_l=70, margin_r=30, margin_t=60))

    if fig_attempts:
        story.extend(chart_block(fig_attempts,
            "Кумулятивная вероятность успеха по попыткам ЭКО",
            h_cm=7, margin_b=60, margin_l=70, margin_r=30, margin_t=65))

    # ══════════════════════════════════════════════════════════
    #  СТРАНИЦА 5 — БАНКИНГ
    # ══════════════════════════════════════════════════════════
    if eb:
        story.append(PageBreak())
        story.append(sec_header("Стратегия банкинга эмбрионов (модель Esteves)"))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "Модель оценивает вероятность получения эуплоидной бластоцисты "
            "на каждую зрелую ооциту (MII) с учётом возраста пациентки и источника спермы.",
            ST["body"]
        ))
        story.append(Spacer(1, 6))

        eb_rows = [
            ("P(эуплоид) на один MII-ооцит", pct(eb.get("p_per_mii",0))),
            ("Частота оплодотворения",        pct(eb.get("fert_r",0))),
            ("Частота бластуляции (D5)",      pct(eb.get("blast_r",0))),
            ("Частота эуплоидии",             pct(eb.get("eupl_r",0))),
        ]
        if eb.get("patient_mii_median"):
            eb_rows.append(("MII медиана (данная пациентка)",
                            f"{eb['patient_mii_median']:.0f} ооцитов"))
        story.append(kv_table(eb_rows, col1=6.5*cm))
        story.append(Spacer(1, 12))

        mii_table = eb.get("mii_table", {})
        if mii_table:
            story.append(Paragraph(
                "Необходимое число MII-ооцитов для получения k эуплоидных эмбрионов:",
                ST["subsec"]
            ))
            story.append(Spacer(1, 4))

            hdr_style = ParagraphStyle("th", fontName=FB, fontSize=9,
                                       textColor=C_NAVY, alignment=TA_CENTER)
            hdr = [Paragraph(t, hdr_style) for t in
                   ["Цель\n(k эуплоидных)", "Уверенность\n70%",
                    "Уверенность\n80%", "Уверенность\n90%"]]
            tdata = [hdr]
            for k, cd in sorted(mii_table.items()):
                tdata.append([
                    Paragraph(f"{k} эмбрион{'а' if k in(2,3,4) else ''}", ST["body"]),
                    Paragraph(str(cd.get(0.70,"—")), ST["body"]),
                    Paragraph(str(cd.get(0.80,"—")), ST["body"]),
                    Paragraph(str(cd.get(0.90,"—")), ST["body"]),
                ])
            cw4 = PAGE_W / 4
            mt = Table(tdata, colWidths=[cw4]*4)
            mt.setStyle(TableStyle([
                ("BACKGROUND",     (0,0),(-1,0), C_TBL_HDR),
                ("ROWBACKGROUNDS", (0,1),(-1,-1),[C_TBL_ODD, C_TBL_EVEN]),
                ("GRID",           (0,0),(-1,-1), 0.4, C_BORDER),
                ("TOPPADDING",     (0,0),(-1,-1), 6),
                ("BOTTOMPADDING",  (0,0),(-1,-1), 6),
                ("LEFTPADDING",    (0,0),(-1,-1), 8),
                ("ALIGN",          (1,0),(-1,-1), "CENTER"),
                ("VALIGN",         (0,0),(-1,-1), "MIDDLE"),
            ]))
            story.append(mt)
            story.append(Spacer(1, 6))
            story.append(Paragraph(
                "Таблица отражает минимальное число MII-ооцитов для достижения "
                "целевого числа эуплоидных бластоцист с заданной вероятностью.",
                ST["note"]
            ))

    # ══════════════════════════════════════════════════════════
    #  СТРАНИЦА 6 — CSDI
    # ══════════════════════════════════════════════════════════
    if csdi_result:
        story.append(PageBreak())
        story.append(sec_header("CSDI Hybrid v3 — Диффузионная модель (L5)"))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "Гибридная генеративная модель, обученная на лабораторном этапе ЭКО. "
            "Генерирует синтетические пары (число бластоцист / бластоцисты хор. качества) "
            "без параметрических допущений MC-пайплайна. Исход беременности предсказывает "
            "откалиброванный LightGBM + Platt-scaling.",
            ST["body"]
        ))
        story.append(Spacer(1, 8))

        p_csdi = csdi_result.get("P_pregnancy", 0)
        ci_c   = csdi_result.get("CI_95", [0, 0])
        story.append(kv_table([
            ("P(беременность) — CSDI Hybrid v3", f"<b>{pct(p_csdi)}</b>"),
            ("95% доверительный интервал",        f"{pct(ci_c[0])} – {pct(ci_c[1])}"),
        ], col1=7*cm))
        story.append(Spacer(1, 8))

        if fig_csdi:
            story.extend(chart_block(fig_csdi,
                "CSDI: синтетические образцы и распределение P(беременность)",
                h_cm=9, margin_b=60, margin_l=70, margin_r=30, margin_t=60))

        # ── Конформальные предиктивные интервалы ─────────────
        _pi90 = csdi_result.get("PI_90_counts", {})
        _pi50 = csdi_result.get("PI_50_counts", {})
        if _pi90:
            story.append(Spacer(1, 6))
            story.append(sec_header("Конформальные предиктивные интервалы (COUNT)"))
            story.append(Spacer(1, 6))

            _hdr_sty = ParagraphStyle("pi_hdr", fontName=FB, fontSize=9,
                                      textColor=C_NAVY, alignment=TA_CENTER)
            _pi_hdr = [Paragraph(t, _hdr_sty) for t in
                       ["Признак", "50% PI", "90% PI", "Медиана"]]

            _bl_med  = csdi_result.get("blast_total_median", "—")
            _gb_med  = csdi_result.get("good_blast_median",  "—")
            _medians = [
                f"{float(_bl_med):.0f}" if _bl_med != "—" else "—",
                f"{float(_gb_med):.0f}" if _gb_med != "—" else "—",
            ]

            _pi_keys = list(_pi90.keys())
            _pi_data = [_pi_hdr]
            for idx, key in enumerate(_pi_keys):
                lo90, hi90 = _pi90[key]
                pi50_str = "—"
                if _pi50 and key in _pi50:
                    lo50, hi50 = _pi50[key]
                    pi50_str = f"[{lo50:.0f}, {hi50:.0f}]"
                _pi_data.append([
                    Paragraph(key, ST["body"]),
                    Paragraph(pi50_str, ST["body"]),
                    Paragraph(f"[{lo90:.0f}, {hi90:.0f}]", ST["body"]),
                    Paragraph(_medians[idx] if idx < len(_medians) else "—", ST["body"]),
                ])

            _cw_pi = [PAGE_W * 0.40, PAGE_W * 0.20, PAGE_W * 0.20, PAGE_W * 0.20]
            _pi_tbl = Table(_pi_data, colWidths=_cw_pi)
            _pi_tbl.setStyle(TableStyle([
                ("BACKGROUND",     (0, 0), (-1, 0),  C_TBL_HDR),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_TBL_ODD, C_TBL_EVEN]),
                ("GRID",           (0, 0), (-1, -1), 0.4, C_BORDER),
                ("TOPPADDING",     (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
                ("LEFTPADDING",    (0, 0), (-1, -1), 8),
                ("ALIGN",          (1, 0), (-1, -1), "CENTER"),
                ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(_pi_tbl)
            story.append(Spacer(1, 4))
            story.append(Paragraph(
                "50% PI — медианный предиктивный интервал; 90% PI — широкий интервал. "
                "Конформальная калибровка обеспечивает гарантированное покрытие на hold-out выборке.",
                ST["note"]
            ))

    # ══════════════════════════════════════════════════════════
    #  СТРАНИЦА 7 — РИСКИ + ГРАФ КЛИНИЧЕСКИХ СОСЕДЕЙ
    # ══════════════════════════════════════════════════════════
    _has_risks = fig_risks is not None
    _has_gnn   = fig_gnn is not None or p_gnn_ens is not None

    if _has_risks or _has_gnn:
        story.append(PageBreak())

    if _has_risks:
        story.append(sec_header("Анализ рисков"))
        story.append(Spacer(1, 6))
        story.extend(chart_block(fig_risks,
            "Вероятность нежелательных исходов цикла",
            h_cm=7, margin_b=110, margin_l=70, margin_r=35, margin_t=55))
    else:
        if _has_gnn:
            story.append(sec_header("Анализ рисков"))
            story.append(Spacer(1, 4))
            story.append(Paragraph("График рисков недоступен.", ST["body_sm"]))
            story.append(Spacer(1, 8))

    if _has_gnn:
        if not _has_risks:
            pass  # уже на новой странице
        story.append(sec_header(
            "Graph Attention Transformer (GAT) — анализ клинических соседей"))
        story.append(Spacer(1, 6))

        story.append(Paragraph(
            "Graph Transformer обучен на клинических протоколах ЭКО. "
            "Модель строит граф сходства пациентов по клиническим показателям "
            "и передаёт информацию между похожими случаями через механизм "
            "графового внимания (Graph Attention). "
            "Итоговая вероятность формируется как взвешенный ансамбль "
            "Graph Transformer и KAT-нейросети.",
            ST["body"]
        ))
        story.append(Spacer(1, 8))

        _gat_rows = []
        if p_gnn_raw is not None:
            _gat_rows.append(("Graph Transformer (raw)",
                               f"<b>{pct(p_gnn_raw)}</b>"))
        if p_gnn_ens is not None:
            _gat_rows.append(("GAT Ансамбль (GNN + KAT)",
                               f"<b>{pct(p_gnn_ens)}</b>"))
        if _gat_rows:
            story.append(kv_table(_gat_rows, col1=6.5*cm))
            story.append(Spacer(1, 10))

        if fig_gnn is not None:
            story.extend(chart_block(
                fig_gnn,
                "Граф клинических соседей (левая панель) и распределение их "
                "GNN-вероятностей (правая панель). Звезда = текущая пациентка. "
                "Цвет узлов: зелёный — высокая вероятность беременности, "
                "красный — низкая. Размер узла и толщина ребра ∝ косинусное сходство.",
                h_cm=10,
                margin_b=55, margin_l=55, margin_r=25, margin_t=65,
            ))
        else:
            story.append(Paragraph(
                "График недоступен (GNN-инференс не был запущен).",
                ST["body_sm"]))

    # ══════════════════════════════════════════════════════════
    #  ИТОГОВОЕ РЕЗЮМЕ
    # ══════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(sec_header("Итоговое резюме"))
    story.append(Spacer(1, 8))

    summary = [
        ("Пациентка",                 f"{patient_name or '—'}  [{patient_id or '—'}]"),
        ("Возраст / АМГ / АФС",       f"{age:.0f} лет  ·  АМГ {amh:.2f} нг/мл  ·  {afc} фолликулов"),
        ("P(беременность) на перенос",pct(p_transfer)),
        ("Кумулятивная (viable цикл)", pct(p_viable)),
        ("P(успех цикла)",            pct(p_cycle)),
        ("Байесовский posterior",     f"{pct(p_bayes)}  (95% ДИ: {pct(ci_lo)}–{pct(ci_hi)})"),
        ("KAT (ансамбль NN)",          pct(p_kat_raw) if p_kat_raw is not None else "—"),
        ("GAT Ансамбль (GNN + KAT)",   pct(p_gnn_ens) if p_gnn_ens is not None else "—"),
        ("  ↳ Graph Transformer (raw)",pct(p_gnn_raw) if p_gnn_raw is not None else "—"),
        ("Доминирующий кластер",      str(dom)),
    ]
    if csdi_result:
        summary.append(("CSDI P(беременность)", pct(csdi_result.get("P_pregnancy",0))))
    story.append(kv_table(summary, col1=6.5*cm))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        f"Дата формирования: <b>{now.strftime('%d.%m.%Y %H:%M')}</b>  ·  "
        f"Клиника: <b>{clinic_name or '—'}</b>",
        ST["body"]
    ))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER,
                             spaceBefore=4, spaceAfter=6))
    story.append(Paragraph(
        "Данный отчёт сформирован системой IVF Digital Twin v6.2 и предназначен "
        "для использования врачом-репродуктологом в качестве вспомогательного "
        "инструмента. Не является самостоятельным клиническим заключением.",
        ST["note"]
    ))

    doc.build(story, onFirstPage=tpl, onLaterPages=tpl)
    return buf.getvalue()
