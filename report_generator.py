"""Builds downloadable PDF reports from data already computed elsewhere in
NEXUS (a /api/predict ticker report, a /api/portfolio_optimize result, and
the client-side INVESTRA chat transcript) -- this module has no DB or model
dependencies of its own, it only lays out numbers it's handed."""

import io
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")  # headless -- no display available inside the container
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable,
)

# Matches quant_core.MODEL_NAMES -- duplicated here (rather than imported)
# so report generation doesn't pull in torch/xgboost/prophet just for one
# ordering constant.
MODEL_NAMES = ["LSTM", "Prophet", "MNN", "XGBoost"]

# NSE's own timezone -- reports were showing UTC, which reads as "wrong" to
# a user comparing against their own (IST) clock.
IST = ZoneInfo("Asia/Kolkata")

# reportlab's built-in base-14 fonts (Helvetica etc.) don't include the
# Rupee sign (U+20B9) or other non-Latin-1 glyphs -- they'd silently render
# as a black box. DejaVu Sans ships inside the matplotlib package itself
# (no extra system font package needed) and covers it, so register it as
# the report's font instead of Helvetica.
_DEJAVU_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DejaVuSans", os.path.join(_DEJAVU_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", os.path.join(_DEJAVU_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSans-Oblique", os.path.join(_DEJAVU_DIR, "DejaVuSans-Oblique.ttf")))

# Print-safe versions of the dashboard's own neon brand palette (--nexus-cyan
# / --nexus-magenta / --nexus-orange / --nexus-green / --nexus-red in
# style.css) -- the neon values read fine on the dashboard's dark background
# but wash out to near-invisible on white paper, so these are darkened
# equivalents that keep the same hue identity per model/status.
NAVY = colors.HexColor("#1B2A6B")
CYAN = colors.HexColor("#0E7C86")       # LSTM
MAGENTA = colors.HexColor("#9B1C6E")    # Prophet
ORANGE = colors.HexColor("#B45309")     # MNN
GREEN = colors.HexColor("#15803D")      # XGBoost / positive / bullish
RED = colors.HexColor("#B91C1C")        # negative / bearish
AMBER = colors.HexColor("#A16207")      # borderline
DIM = colors.HexColor("#444444")
MUTED = colors.HexColor("#8A8FA3")
LIGHT_GRID = colors.HexColor("#E3E6EF")
CALLOUT_BG = colors.HexColor("#EEF2FC")

MODEL_COLORS = {"LSTM": CYAN, "Prophet": MAGENTA, "MNN": ORANGE, "XGBoost": GREEN}

_styles = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle("NexusTitle", parent=_styles["Title"], fontName="DejaVuSans-Bold",
                             fontSize=20, textColor=NAVY, spaceAfter=2)
SUBTITLE_STYLE = ParagraphStyle("NexusSubtitle", parent=_styles["Normal"], fontName="DejaVuSans",
                                fontSize=9.5, textColor=MUTED, spaceAfter=4)
SECTION_TEXT_STYLE = ParagraphStyle("NexusSection", parent=_styles["Heading2"], fontName="DejaVuSans-Bold",
                                    fontSize=12.5, textColor=NAVY, spaceBefore=0, spaceAfter=0,
                                    leftIndent=8)
BODY_STYLE = ParagraphStyle("NexusBody", parent=_styles["Normal"], fontName="DejaVuSans",
                            fontSize=9.5, leading=14, textColor=DIM, spaceAfter=6)
CALLOUT_LABEL_STYLE = ParagraphStyle("NexusCalloutLabel", parent=_styles["Normal"], fontName="DejaVuSans-Bold",
                                     fontSize=11.5, textColor=NAVY, leading=16)
CHAT_ROLE_USER_STYLE = ParagraphStyle("NexusChatRoleUser", parent=_styles["Normal"], fontName="DejaVuSans-Bold",
                                      fontSize=9.5, textColor=NAVY, spaceBefore=8, spaceAfter=2)
CHAT_ROLE_AI_STYLE = ParagraphStyle("NexusChatRoleAI", parent=_styles["Normal"], fontName="DejaVuSans-Bold",
                                    fontSize=9.5, textColor=MAGENTA, spaceBefore=8, spaceAfter=2)
CHAT_TEXT_STYLE = ParagraphStyle("NexusChatText", parent=_styles["Normal"], fontName="DejaVuSans",
                                 fontSize=9.5, leading=13.5, textColor=DIM, spaceAfter=2)

PAGE_W, PAGE_H = letter


def _now_str():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")


def _fmt_currency(value, symbol):
    if value is None:
        return "--"
    return f"{symbol}{value:,.2f}"


def _fmt_pct(value, digits=2):
    if value is None:
        return "--"
    return f"{value:.{digits}f}%"


def _sentiment_color(avg):
    if avg is None:
        return MUTED
    if avg > 0.55:
        return GREEN
    if avg < 0.45:
        return RED
    return AMBER


def _win_rate_color(rate):
    if rate is None:
        return MUTED
    if rate > 55:
        return GREEN
    if rate >= 50:
        return AMBER
    return RED


def _sharpe_color(sharpe):
    if sharpe is None:
        return MUTED
    return GREEN if sharpe >= 0 else RED


def _mpl_hex(color):
    """reportlab's Color.hexval() returns '0xrrggbb'; matplotlib wants
    '#rrggbb'. Both reportlab Paragraph markup and matplotlib need the same
    color objects, so this just reformats for the latter."""
    return "#" + color.hexval()[2:]


# ---------------------------------------------------------------------------
# Page chrome -- a thin brand-colored top bar + "NEXUS" wordmark, and a
# footer with page number + disclaimer, drawn on every page via reportlab's
# onPage canvas hook rather than left to Platypus flowables.
# ---------------------------------------------------------------------------
def _draw_page_chrome(canvas, doc):
    canvas.saveState()

    # Top accent bar: navy with a short cyan-to-magenta tick, echoing the
    # dashboard's own gradient buttons/section accents.
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 0.16 * inch, PAGE_W, 0.16 * inch, stroke=0, fill=1)
    canvas.setFillColor(CYAN)
    canvas.rect(0, PAGE_H - 0.16 * inch, 1.6 * inch, 0.16 * inch, stroke=0, fill=1)
    canvas.setFillColor(MAGENTA)
    canvas.rect(1.6 * inch, PAGE_H - 0.16 * inch, 1.6 * inch, 0.16 * inch, stroke=0, fill=1)

    canvas.setFont("DejaVuSans-Bold", 8.5)
    canvas.setFillColor(NAVY)
    canvas.drawString(0.75 * inch, PAGE_H - 0.34 * inch, "NEXUS")
    canvas.setFont("DejaVuSans", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(PAGE_W - 0.75 * inch, PAGE_H - 0.34 * inch,
                           "Multi-Modal Algorithmic Trading Terminal")

    # Footer: page number + disclaimer, small and out of the way.
    canvas.setFont("DejaVuSans", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.75 * inch, 0.5 * inch, "For research purposes only, not investment advice.")
    canvas.drawRightString(PAGE_W - 0.75 * inch, 0.5 * inch, f"Page {canvas.getPageNumber()}")

    canvas.restoreState()


def render_price_chart_png(dates, closes, predicted_series=None, winner_name=None,
                            width_in=6.6, height_in=2.6):
    """Renders a price-trend PNG (actual close, plus the winning model's
    predicted path if given) via matplotlib, returned as raw PNG bytes for
    embedding in the PDF."""
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=150)
    if dates and closes:
        ax.plot(dates, closes, color="#1B2A6B", linewidth=1.4, label="Close (actual)")

    if predicted_series:
        pred_dates = [row.get("date") for row in predicted_series]
        pred_prices = [row.get("predicted_price") for row in predicted_series]
        label = f"{winner_name} predicted" if winner_name else "Predicted"
        pred_color = _mpl_hex(MODEL_COLORS.get(winner_name, RED)) if winner_name else "#C0264A"
        ax.plot(pred_dates, pred_prices, color=pred_color, linewidth=1.3,
                linestyle="--", label=label)

    ax.set_title("Price Trend", fontsize=10, color="#1B2A6B", loc="left", fontweight="bold")
    ax.tick_params(axis="both", labelsize=6.5, colors="#444444")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Sparse x-ticks -- a year of daily dates would otherwise overlap into
    # an unreadable smear at this print width.
    n = len(dates) if dates else 0
    if n > 8:
        step = max(1, n // 8)
        ax.set_xticks(range(0, n, step))
        ax.set_xticklabels([dates[i] for i in range(0, n, step)], rotation=30, ha="right")

    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _header_block(title, subtitle):
    return [
        Spacer(1, 6),
        Paragraph(title, TITLE_STYLE),
        Paragraph(subtitle, SUBTITLE_STYLE),
        HRFlowable(width="100%", thickness=1.4, color=NAVY, spaceBefore=2, spaceAfter=1),
        HRFlowable(width="100%", thickness=0.8, color=CYAN, spaceAfter=14),
    ]


def _section_heading(text):
    """A section title with a colored left accent bar, mirroring the
    dashboard's own .nexus-section-title (border-left: 3px solid cyan)."""
    bar_and_text = Table(
        [[" ", Paragraph(text, SECTION_TEXT_STYLE)]],
        colWidths=[0.06 * inch, 6.5 * inch],
    )
    bar_and_text.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), CYAN),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("LEFTPADDING", (1, 0), (1, 0), 8),
    ]))
    return [Spacer(1, 10), bar_and_text, Spacer(1, 6)]


def _winner_callout(winner, winner_metrics, currency_symbol):
    """A tinted, colored-left-border box (mirroring .winner-callout on the
    dashboard) highlighting the winning model, its last close, and its
    next-day prediction -- with the predicted value colored green/red
    depending on whether it's up or down versus last close."""
    if not winner:
        return []

    last_close = winner_metrics.get("past_day_value")
    predicted = winner_metrics.get("predicted_price")
    model_color = MODEL_COLORS.get(winner, NAVY)

    predicted_html = _fmt_currency(predicted, currency_symbol)
    if last_close is not None and predicted is not None:
        up = predicted >= last_close
        arrow = "&#9650;" if up else "&#9660;"
        move_color = GREEN if up else RED
        predicted_html = (
            f'<font color="{move_color.hexval()}">{arrow} '
            f'{_fmt_currency(predicted, currency_symbol)}</font>'
        )

    text = (
        f'Winning model: <font color="{model_color.hexval()}"><b>{winner}</b></font> '
        f'&mdash; last close <font color="{DIM.hexval()}">{_fmt_currency(last_close, currency_symbol)}</font>, '
        f'next-day prediction {predicted_html}.'
    )
    box = Table([[Paragraph(text, CALLOUT_LABEL_STYLE)]], colWidths=[6.9 * inch])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CALLOUT_BG),
        ("LINEBEFORE", (0, 0), (0, 0), 3, model_color),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
    ]))
    return [box, Spacer(1, 12)]


def _model_breakdown_table(model_breakdown, winner, currency_symbol):
    header = ["Model", "Predicted Value", "Error Margin (RMSE)", "MAE", "Win Rate", "Sharpe"]
    rows = [header]
    winner_row_idx = None
    extra_styles = []

    for i, name in enumerate(MODEL_NAMES, start=1):
        m = model_breakdown.get(name)
        if not m:
            continue
        if name == winner:
            winner_row_idx = i
        sharpe = m.get("sharpe_ratio")
        win_rate = m.get("win_rate")
        rows.append([
            name,
            _fmt_currency(m.get("predicted_price"), currency_symbol),
            _fmt_currency(m.get("rmse"), currency_symbol),
            _fmt_currency(m.get("mae"), currency_symbol),
            _fmt_pct(win_rate, 1),
            f"{sharpe:.2f}" if sharpe is not None else "N/A",
        ])
        row_idx = len(rows) - 1
        extra_styles.append(("TEXTCOLOR", (0, row_idx), (0, row_idx), MODEL_COLORS.get(name, DIM)))
        extra_styles.append(("FONTNAME", (0, row_idx), (0, row_idx), "DejaVuSans-Bold"))
        extra_styles.append(("TEXTCOLOR", (4, row_idx), (4, row_idx), _win_rate_color(win_rate)))
        extra_styles.append(("TEXTCOLOR", (5, row_idx), (5, row_idx), _sharpe_color(sharpe)))

    table = Table(rows, colWidths=[0.9 * inch, 1.25 * inch, 1.35 * inch, 0.9 * inch, 0.85 * inch, 0.75 * inch])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "DejaVuSans-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "DejaVuSans"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_GRID),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F8FC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    style.extend(extra_styles)
    if winner_row_idx is not None:
        style.append(("BACKGROUND", (0, winner_row_idx), (-1, winner_row_idx), CALLOUT_BG))
    table.setStyle(TableStyle(style))
    return table


def _sentiment_block(sentiment_summary):
    blocks = _section_heading("Sentiment Overview")
    avg = sentiment_summary.get("average_sentiment") if sentiment_summary else None
    if avg is None:
        blocks.append(Paragraph("No sentiment data has been ingested for this ticker yet.", BODY_STYLE))
        return blocks
    label = "bullish" if avg > 0.55 else ("bearish" if avg < 0.45 else "neutral")
    color = _sentiment_color(avg)
    blocks.append(Paragraph(
        f'Average sentiment score: <font color="{color.hexval()}"><b>{avg:.2f}</b></font> '
        f'(<font color="{color.hexval()}">{label}</font>, 0 = bearish, 1 = bullish).',
        BODY_STYLE,
    ))
    headlines = (sentiment_summary.get("headlines") or [])[:5]
    for h in headlines:
        blocks.append(Paragraph(
            f'&bull; <font color="{MUTED.hexval()}">[{h.get("date", "--")}]</font> {h.get("headline", "")}',
            BODY_STYLE,
        ))
    return blocks


def _ticker_story_blocks(ticker, display_name, currency_symbol, ticker_report,
                         sentiment_summary, ai_summary, price_dates, price_closes):
    """Shared per-ticker layout used by both the single-ticker report and
    each ticker's section inside the full session report."""
    model_breakdown = ticker_report.get("model_breakdown", {})
    winner = ticker_report.get("winner")
    winner_metrics = model_breakdown.get(winner, {})

    blocks = []
    blocks.extend(_header_block(
        f"{display_name} ({ticker}) &mdash; Forecast Report",
        f"Generated {_now_str()} &middot; Currency: {currency_symbol}",
    ))

    blocks.extend(_winner_callout(winner, winner_metrics, currency_symbol))

    blocks.extend(_section_heading("Model Breakdown"))
    blocks.append(_model_breakdown_table(model_breakdown, winner, currency_symbol))
    blocks.append(Spacer(1, 10))

    if price_dates and price_closes:
        blocks.extend(_section_heading("Price Trend"))
        chart_png = render_price_chart_png(
            price_dates, price_closes,
            predicted_series=winner_metrics.get("predicted_series"),
            winner_name=winner,
        )
        blocks.append(Image(io.BytesIO(chart_png), width=6.6 * inch, height=2.6 * inch))
        blocks.append(Spacer(1, 6))

    blocks.extend(_sentiment_block(sentiment_summary))

    blocks.extend(_section_heading("AI Analyst Summary"))
    blocks.append(Paragraph(ai_summary or "An AI summary was not available when this report was generated.", BODY_STYLE))

    return blocks


def _portfolio_story_blocks(portfolio_result):
    blocks = _section_heading("Portfolio Management")
    if not portfolio_result:
        blocks.append(Paragraph("No Portfolio Management result was available for this session.", BODY_STYLE))
        return blocks

    objective = portfolio_result.get("objective_used", "--")
    horizon = portfolio_result.get("horizon", "--")
    blocks.append(Paragraph(
        f'Objective: <font color="{CYAN.hexval()}"><b>{objective}</b></font> &middot; '
        f'Horizon: <font color="{CYAN.hexval()}"><b>{horizon}</b></font>', BODY_STYLE
    ))

    sharpe_val = portfolio_result.get("sharpe_ratio")
    sharpe_color = _sharpe_color(sharpe_val)
    metrics_rows = [
        ["Expected Annual Return", "Annual Volatility", "Sharpe Ratio"],
        [
            _fmt_pct(portfolio_result.get("expected_annual_return_pct")),
            _fmt_pct(portfolio_result.get("annual_volatility_pct")),
            f"{sharpe_val:.2f}" if sharpe_val is not None else "N/A",
        ],
    ]
    metrics_table = Table(metrics_rows, colWidths=[2.2 * inch, 2.2 * inch, 2.2 * inch])
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "DejaVuSans-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "DejaVuSans-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_GRID),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TEXTCOLOR", (2, 1), (2, 1), sharpe_color),
    ]))
    blocks.append(metrics_table)
    blocks.append(Spacer(1, 10))

    breakdown = portfolio_result.get("breakdown") or []
    if breakdown:
        alloc_rows = [["Ticker", "Weight", "Price", "Shares", "Allocated"]]
        for row in breakdown:
            alloc_rows.append([
                row.get("ticker", "--"),
                _fmt_pct(row.get("weight_pct")),
                f"{row.get('price', 0):,.2f}",
                str(row.get("shares", "--")),
                f"{row.get('allocated_amount', 0):,.2f}",
            ])
        alloc_table = Table(alloc_rows, colWidths=[1.5 * inch, 1.1 * inch, 1.2 * inch, 1.1 * inch, 1.5 * inch])
        alloc_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "DejaVuSans-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "DejaVuSans"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_GRID),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F8FC")]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        blocks.append(alloc_table)
        blocks.append(Spacer(1, 8))

    total_allocated = portfolio_result.get("total_allocated")
    leftover = portfolio_result.get("leftover_cash")
    if total_allocated is not None:
        blocks.append(Paragraph(
            f"Total allocated: <b>&#8377;{total_allocated:,.2f}</b> &middot; "
            f"Leftover cash: &#8377;{leftover:,.2f}" if leftover is not None else
            f"Total allocated: <b>&#8377;{total_allocated:,.2f}</b>",
            BODY_STYLE,
        ))
    if portfolio_result.get("under_minimum"):
        blocks.append(Paragraph(
            f'<font color="{AMBER.hexval()}">Note: whole-share rounding could not reach the requested minimum amount.</font>',
            BODY_STYLE,
        ))

    explanation = portfolio_result.get("explanation")
    blocks.extend(_section_heading("INVESTRA's Explanation"))
    blocks.append(Paragraph(explanation or "No explanation was available for this result.", BODY_STYLE))

    return blocks


def _chat_transcript_blocks(chat_transcript):
    blocks = _section_heading("INVESTRA Chat Transcript")
    if not chat_transcript:
        blocks.append(Paragraph("No chat messages were exchanged this session.", BODY_STYLE))
        return blocks
    for msg in chat_transcript:
        is_user = msg.get("role") == "user"
        blocks.append(Paragraph("You" if is_user else "INVESTRA",
                                CHAT_ROLE_USER_STYLE if is_user else CHAT_ROLE_AI_STYLE))
        blocks.append(Paragraph(msg.get("text", ""), CHAT_TEXT_STYLE))
    return blocks


def build_ticker_pdf(ticker, display_name, currency_symbol, ticker_report,
                     sentiment_summary, ai_summary, price_dates, price_closes):
    """A single-ticker forecast report ("half report")."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.55 * inch, bottomMargin=0.75 * inch,
    )
    story = _ticker_story_blocks(
        ticker, display_name, currency_symbol, ticker_report,
        sentiment_summary, ai_summary, price_dates, price_closes,
    )
    # The fuller disclaimer already runs in the per-page footer (see
    # _draw_page_chrome); appending it again here as a flowable used to
    # spill a whole second page for one sentence.
    doc.build(story, onFirstPage=_draw_page_chrome, onLaterPages=_draw_page_chrome)
    buf.seek(0)
    return buf.read()


def build_full_report_pdf(ticker_sections, portfolio_result, chat_transcript):
    """The combined "full report": every predicted ticker's forecast section,
    then Portfolio Management, then the INVESTRA chat transcript.

    `ticker_sections` is a list of dicts, one per ticker, each with the same
    keys build_ticker_pdf takes: ticker, display_name, currency_symbol,
    ticker_report, sentiment_summary, ai_summary, price_dates, price_closes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.55 * inch, bottomMargin=0.75 * inch,
    )

    story = _header_block(
        "NEXUS &mdash; Full Session Report",
        f"Generated {_now_str()} &middot; Covers {len(ticker_sections)} ticker(s) and the current portfolio allocation.",
    )

    for i, section in enumerate(ticker_sections):
        story.extend(_ticker_story_blocks(
            section["ticker"], section.get("display_name", section["ticker"]),
            section.get("currency_symbol", "$"), section["ticker_report"],
            section.get("sentiment_summary"), section.get("ai_summary"),
            section.get("price_dates"), section.get("price_closes"),
        ))
        story.append(PageBreak())

    story.extend(_portfolio_story_blocks(portfolio_result))
    story.append(PageBreak())
    story.extend(_chat_transcript_blocks(chat_transcript))
    # The fuller disclaimer already runs in the per-page footer (see
    # _draw_page_chrome); no separate flowable needed here.

    doc.build(story, onFirstPage=_draw_page_chrome, onLaterPages=_draw_page_chrome)
    buf.seek(0)
    return buf.read()
