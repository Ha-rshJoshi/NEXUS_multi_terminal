"""Builds downloadable PDF reports from data already computed elsewhere in
NEXUS (a /api/predict ticker report, a /api/portfolio_optimize result, and
the client-side INVESTRA chat transcript) -- this module has no DB or model
dependencies of its own, it only lays out numbers it's handed."""

import io
import os
from datetime import datetime

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

# reportlab's built-in base-14 fonts (Helvetica etc.) don't include the
# Rupee sign (U+20B9) or other non-Latin-1 glyphs -- they'd silently render
# as a black box. DejaVu Sans ships inside the matplotlib package itself
# (no extra system font package needed) and covers it, so register it as
# the report's font instead of Helvetica.
_DEJAVU_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DejaVuSans", os.path.join(_DEJAVU_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", os.path.join(_DEJAVU_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSans-Oblique", os.path.join(_DEJAVU_DIR, "DejaVuSans-Oblique.ttf")))

NAVY = colors.HexColor("#1B2A6B")
CYAN = colors.HexColor("#0E7C86")
DIM = colors.HexColor("#444444")
LIGHT_GRID = colors.HexColor("#E3E6EF")
WINNER_BG = colors.HexColor("#EAF7F1")

_styles = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle("NexusTitle", parent=_styles["Title"], fontName="DejaVuSans-Bold",
                             fontSize=20, textColor=NAVY, spaceAfter=4)
SUBTITLE_STYLE = ParagraphStyle("NexusSubtitle", parent=_styles["Normal"], fontName="DejaVuSans",
                                fontSize=9.5, textColor=DIM, spaceAfter=14)
SECTION_STYLE = ParagraphStyle("NexusSection", parent=_styles["Heading2"], fontName="DejaVuSans-Bold",
                               fontSize=13.5, textColor=NAVY, spaceBefore=14, spaceAfter=8)
BODY_STYLE = ParagraphStyle("NexusBody", parent=_styles["Normal"], fontName="DejaVuSans",
                            fontSize=9.5, leading=14, textColor=DIM, spaceAfter=6)
CALLOUT_STYLE = ParagraphStyle("NexusCallout", parent=_styles["Normal"], fontName="DejaVuSans-Bold",
                               fontSize=11, textColor=NAVY, spaceAfter=10)
DISCLAIMER_STYLE = ParagraphStyle("NexusDisclaimer", parent=_styles["Normal"], fontName="DejaVuSans-Oblique",
                                  fontSize=8, textColor=colors.HexColor("#888888"), spaceBefore=16)
CHAT_ROLE_STYLE = ParagraphStyle("NexusChatRole", parent=_styles["Normal"], fontName="DejaVuSans-Bold",
                                 fontSize=9.5, textColor=NAVY, spaceBefore=6, spaceAfter=2)
CHAT_TEXT_STYLE = ParagraphStyle("NexusChatText", parent=_styles["Normal"], fontName="DejaVuSans",
                                 fontSize=9.5, leading=13.5, textColor=DIM, spaceAfter=6)

DISCLAIMER_TEXT = ("NEXUS is a research/learning tool. Nothing in this report is investment "
                   "advice; forecasts are historical backtests, not guarantees of future performance.")


def _now_str():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


def _fmt_currency(value, symbol):
    if value is None:
        return "--"
    return f"{symbol}{value:,.2f}"


def _fmt_pct(value, digits=2):
    if value is None:
        return "--"
    return f"{value:.{digits}f}%"


def render_price_chart_png(dates, closes, predicted_series=None, winner_name=None,
                            width_in=6.6, height_in=2.6):
    """Renders a price-trend PNG (actual close, plus the winning model's
    predicted path if given) via matplotlib, returned as raw PNG bytes for
    embedding in the PDF. Deliberately plain/dark-on-light -- print-friendly,
    not trying to visually match the neon dashboard theme."""
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=150)
    if dates and closes:
        ax.plot(dates, closes, color="#1B2A6B", linewidth=1.4, label="Close (actual)")

    if predicted_series:
        pred_dates = [row.get("date") for row in predicted_series]
        pred_prices = [row.get("predicted_price") for row in predicted_series]
        label = f"{winner_name} predicted" if winner_name else "Predicted"
        ax.plot(pred_dates, pred_prices, color="#C0264A", linewidth=1.2,
                linestyle="--", label=label)

    ax.set_title("Price Trend", fontsize=10, color="#1B2A6B", loc="left")
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
        Paragraph(title, TITLE_STYLE),
        Paragraph(subtitle, SUBTITLE_STYLE),
        HRFlowable(width="100%", thickness=0.7, color=LIGHT_GRID, spaceAfter=10),
    ]


def _model_breakdown_table(model_breakdown, winner, currency_symbol):
    header = ["Model", "Predicted Value", "Error Margin (RMSE)", "MAE", "Win Rate", "Sharpe"]
    rows = [header]
    winner_row_idx = None
    for i, name in enumerate(MODEL_NAMES, start=1):
        m = model_breakdown.get(name)
        if not m:
            continue
        if name == winner:
            winner_row_idx = i
        sharpe = m.get("sharpe_ratio")
        rows.append([
            name,
            _fmt_currency(m.get("predicted_price"), currency_symbol),
            _fmt_currency(m.get("rmse"), currency_symbol),
            _fmt_currency(m.get("mae"), currency_symbol),
            _fmt_pct(m.get("win_rate"), 1),
            f"{sharpe:.2f}" if sharpe is not None else "N/A",
        ])

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
    if winner_row_idx is not None:
        style.append(("BACKGROUND", (0, winner_row_idx), (-1, winner_row_idx), WINNER_BG))
        style.append(("FONTNAME", (0, winner_row_idx), (-1, winner_row_idx), "DejaVuSans-Bold"))
    table.setStyle(TableStyle(style))
    return table


def _sentiment_block(sentiment_summary):
    blocks = [Paragraph("Sentiment Overview", SECTION_STYLE)]
    avg = sentiment_summary.get("average_sentiment") if sentiment_summary else None
    if avg is None:
        blocks.append(Paragraph("No sentiment data has been ingested for this ticker yet.", BODY_STYLE))
        return blocks
    label = "bullish" if avg > 0.55 else ("bearish" if avg < 0.45 else "neutral")
    blocks.append(Paragraph(f"Average sentiment score: <b>{avg:.2f}</b> ({label}, 0 = bearish, 1 = bullish).", BODY_STYLE))
    headlines = (sentiment_summary.get("headlines") or [])[:5]
    for h in headlines:
        blocks.append(Paragraph(f"&bull; [{h.get('date', '--')}] {h.get('headline', '')}", BODY_STYLE))
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

    if winner:
        predicted = _fmt_currency(winner_metrics.get("predicted_price"), currency_symbol)
        last_close = _fmt_currency(winner_metrics.get("past_day_value"), currency_symbol)
        blocks.append(Paragraph(
            f"Winning model: <b>{winner}</b> &mdash; last close {last_close}, next-day prediction {predicted}.",
            CALLOUT_STYLE,
        ))

    blocks.append(Paragraph("Model Breakdown", SECTION_STYLE))
    blocks.append(_model_breakdown_table(model_breakdown, winner, currency_symbol))
    blocks.append(Spacer(1, 10))

    if price_dates and price_closes:
        blocks.append(Paragraph("Price Trend", SECTION_STYLE))
        chart_png = render_price_chart_png(
            price_dates, price_closes,
            predicted_series=winner_metrics.get("predicted_series"),
            winner_name=winner,
        )
        blocks.append(Image(io.BytesIO(chart_png), width=6.6 * inch, height=2.6 * inch))
        blocks.append(Spacer(1, 6))

    blocks.extend(_sentiment_block(sentiment_summary))

    blocks.append(Paragraph("AI Analyst Summary", SECTION_STYLE))
    blocks.append(Paragraph(ai_summary or "An AI summary was not available when this report was generated.", BODY_STYLE))

    return blocks


def _portfolio_story_blocks(portfolio_result):
    blocks = [Paragraph("Portfolio Management", SECTION_STYLE)]
    if not portfolio_result:
        blocks.append(Paragraph("No Portfolio Management result was available for this session.", BODY_STYLE))
        return blocks

    objective = portfolio_result.get("objective_used", "--")
    horizon = portfolio_result.get("horizon", "--")
    blocks.append(Paragraph(
        f"Objective: <b>{objective}</b> &middot; Horizon: <b>{horizon}</b>", BODY_STYLE
    ))

    metrics_rows = [
        ["Expected Annual Return", "Annual Volatility", "Sharpe Ratio"],
        [
            _fmt_pct(portfolio_result.get("expected_annual_return_pct")),
            _fmt_pct(portfolio_result.get("annual_volatility_pct")),
            f"{portfolio_result['sharpe_ratio']:.2f}" if portfolio_result.get("sharpe_ratio") is not None else "N/A",
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
            "Note: whole-share rounding could not reach the requested minimum amount.", BODY_STYLE,
        ))

    explanation = portfolio_result.get("explanation")
    blocks.append(Paragraph("INVESTRA's Explanation", SECTION_STYLE))
    blocks.append(Paragraph(explanation or "No explanation was available for this result.", BODY_STYLE))

    return blocks


def _chat_transcript_blocks(chat_transcript):
    blocks = [Paragraph("INVESTRA Chat Transcript", SECTION_STYLE)]
    if not chat_transcript:
        blocks.append(Paragraph("No chat messages were exchanged this session.", BODY_STYLE))
        return blocks
    for msg in chat_transcript:
        role = "You" if msg.get("role") == "user" else "INVESTRA"
        blocks.append(Paragraph(role, CHAT_ROLE_STYLE))
        blocks.append(Paragraph(msg.get("text", ""), CHAT_TEXT_STYLE))
    return blocks


def build_ticker_pdf(ticker, display_name, currency_symbol, ticker_report,
                     sentiment_summary, ai_summary, price_dates, price_closes):
    """A single-ticker forecast report ("half report")."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )
    story = _ticker_story_blocks(
        ticker, display_name, currency_symbol, ticker_report,
        sentiment_summary, ai_summary, price_dates, price_closes,
    )
    story.append(Paragraph(DISCLAIMER_TEXT, DISCLAIMER_STYLE))
    doc.build(story)
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
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
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
    story.append(Paragraph(DISCLAIMER_TEXT, DISCLAIMER_STYLE))

    doc.build(story)
    buf.seek(0)
    return buf.read()
