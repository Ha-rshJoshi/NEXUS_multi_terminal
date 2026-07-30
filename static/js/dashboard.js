/* ==========================================================================
   NEXUS dashboard client
   - Polls /api/status until all tickers are ready, then reveals
     "Start Prediction" (Section 3: never auto-predicted).
   - Listens on Socket.IO 'progress' events to drive the two independent
     bottom-right toasts (live data / sentiment+news).
   - Renders the Sentiment Accordion, Model Breakdown Accordion and the
     3-chart Chart.js graphs section per stock.
   - Drives the Portfolio Management form against /api/portfolio_optimize.
   - Drives the INVESTRA floating chat widget against /api/chat -- aware of
     the most recently computed portfolio, so follow-up questions can be
     grounded in those numbers too.
   ========================================================================== */

(function () {
  "use strict";

  const TICKERS = window.NEXUS_TICKERS || {};
  const tickerIds = Object.keys(TICKERS);
  const charts = {}; // { ticker: { price: Chart, sentiment: Chart, models: Chart, live: Chart } }
  const MODEL_NAMES = ["LSTM", "Prophet", "MNN", "XGBoost"];
  const MODEL_COLORS = { LSTM: "#22d3ee", Prophet: "#c026f7", MNN: "#fb923c", XGBoost: "#34d399" };

  // Single source of truth for INVESTRA's branding lives server-side in
  // app.py's INVESTRA_CONFIG -- this just falls back to sane defaults if
  // the template variable is somehow missing.
  const INVESTRA_CONFIG = Object.assign({
    name: "INVESTRA",
    tagline: "Your AI Trading & Portfolio Advisor",
    opening_message: "Hi, I'm INVESTRA. Ask me about any tracked ticker.",
    logo_url: null,
  }, window.NEXUS_INVESTRA_CONFIG || {});

  // Client-side only -- the most recently computed Portfolio Management
  // result, so INVESTRA can answer "why this split" grounded in the actual
  // numbers. Never persisted; resets on page reload.
  let lastPortfolioResult = null;

  // { ticker: { modelName: { predicted_series: [{date, predicted_price, actual_price}], color } } }
  const predictionCache = {};

  // { ticker: full tickerReport object from /api/predict } -- kept as-is
  // (not just the pieces rendered to the DOM) so a PDF report can be built
  // from it later without re-running the backtest.
  const predictionReports = {};

  // Every INVESTRA chat message this session, in order -- used only for the
  // "Download Full Report" PDF; never persisted beyond the page's lifetime.
  const chatTranscript = [];

  function safeId(ticker) {
    return ticker.replace(/[^a-zA-Z0-9]/g, "_");
  }

  function formatCurrency(amount, symbol) {
    if (amount === null || amount === undefined || Number.isNaN(amount)) return "--";
    return `${symbol}${amount.toFixed(2)}`;
  }

  function tickerCurrencySymbol(ticker) {
    return ticker.endsWith(".NS") || ticker.endsWith(".BO") ? "₹" : "$";
  }

  function formatClockTime(isoString) {
    if (!isoString) return "";
    const d = new Date(isoString.endsWith("Z") ? isoString : `${isoString}Z`);
    if (Number.isNaN(d.getTime())) return isoString;
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function winRateBarClass(rate) {
    if (rate > 55) return "bg-success";
    if (rate >= 50) return "bg-warning";
    return "bg-danger";
  }

  // ------------------------------------------------------------------
  // Page 1 -> Page 2 transition. Page 2 (per-ticker slider + Execution
  // Desk) starts hidden so the landing page only ever shows the hero +
  // Data Ingestion Status without scrolling; it's revealed the first time
  // a prediction run succeeds.
  // ------------------------------------------------------------------
  const pageDetailsEl = document.getElementById("page-details");
  const tickerCarouselEl = document.getElementById("ticker-carousel");
  let tickerCarouselInstance = null;
  let pageDetailsRevealed = false;

  function resizeAllCharts() {
    Object.values(charts).forEach((c) => {
      if (c.price) c.price.resize();
      if (c.sentiment) c.sentiment.resize();
      if (c.models) c.models.resize();
      if (c.live) c.live.resize();
    });
  }

  function revealPageDetails() {
    if (!pageDetailsEl) return;
    const wasHidden = pageDetailsEl.style.display === "none" || !pageDetailsRevealed;
    pageDetailsEl.style.display = "block";

    if (tickerCarouselEl && !tickerCarouselInstance) {
      tickerCarouselInstance = new bootstrap.Carousel(tickerCarouselEl, { interval: false, wrap: true });
      // Chart.js canvases were built while this section had display:none,
      // so they measured 0x0 -- resize once the section is actually
      // visible, and again whenever the slider moves to a new ticker.
      tickerCarouselEl.addEventListener("slid.bs.carousel", resizeAllCharts);
    }

    if (wasHidden) {
      pageDetailsRevealed = true;
      // Let the browser finish the display:block layout pass before
      // resizing/scrolling, otherwise the canvases still measure 0x0.
      requestAnimationFrame(() => {
        resizeAllCharts();
        pageDetailsEl.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    } else {
      resizeAllCharts();
    }

    startLiveRefreshLoop();
  }

  // ------------------------------------------------------------------
  // Live Price auto-refresh -- polls /api/live_ticks for every tracked
  // ticker on an interval while page 2 (the ticker carousel/detail view)
  // is visible, so the "as of HH:MM:SS" readout and intraday chart stay
  // current without the user needing to manually re-click the Live Price
  // tab. Started once, from revealPageDetails; guarded against
  // double-starting since revealPageDetails can be called more than once
  // (preview link, then a real prediction run).
  // ------------------------------------------------------------------
  let liveRefreshTimer = null;
  const LIVE_REFRESH_INTERVAL_MS = 18000;

  function startLiveRefreshLoop() {
    if (liveRefreshTimer) return;
    tickerIds.forEach((t) => loadLiveTicks(t));
    liveRefreshTimer = setInterval(() => {
      if (!pageDetailsRevealed) return;
      tickerIds.forEach((t) => loadLiveTicks(t));
    }, LIVE_REFRESH_INTERVAL_MS);
  }

  // ------------------------------------------------------------------
  // Toasts
  // ------------------------------------------------------------------
  const toastEls = {
    live_data: document.getElementById("toast-live-data"),
    sentiment_news: document.getElementById("toast-sentiment-news"),
  };
  const toastInstances = {};

  function showToast(channel) {
    const el = toastEls[channel];
    if (!el) return;
    if (!toastInstances[channel]) {
      toastInstances[channel] = new bootstrap.Toast(el, { autohide: false });
    }
    toastInstances[channel].show();
  }

  function updateToastProgress(channel, percent) {
    const el = toastEls[channel];
    if (!el) return;
    showToast(channel);
    const bar = el.querySelector(".progress-bar");
    const label = el.querySelector(".toast-percent");
    if (bar) bar.style.width = `${percent}%`;
    if (label) label.textContent = `${percent}%`;
    if (percent >= 100) {
      setTimeout(() => toastInstances[channel] && toastInstances[channel].hide(), 1500);
    }
  }

  // ------------------------------------------------------------------
  // Socket.IO -- guarded so a failed/blocked CDN load for this script
  // (e.g. a network that blocks cdn.socket.io) degrades to "no live
  // progress toasts" instead of throwing and killing the whole file,
  // which would otherwise also take down status polling/prediction/
  // portfolio/chat below.
  // ------------------------------------------------------------------
  const socket = typeof io === "function" ? io() : null;
  if (socket) {
    socket.on("connect", () => console.log("[NEXUS] socket connected"));
    socket.on("progress", (data) => {
      updateToastProgress(data.channel, data.percent);
    });
    socket.on("ingestion_complete", () => {
      refreshStatus();
    });
  } else {
    console.warn("[NEXUS] Socket.IO client unavailable -- ingestion progress toasts disabled, but status polling continues.");
  }

  // ------------------------------------------------------------------
  // /api/status polling -> gates the Start Prediction button
  // ------------------------------------------------------------------
  const statusPanel = document.getElementById("nexus-status-panel");
  const startBtn = document.getElementById("start-prediction-btn");
  const fetchingLabel = document.getElementById("nexus-fetching-label");
  let statusPollTimer = null;

  function liveStatusLabel(info) {
    // Section 5: live readiness is now "has a tick landed in the last
    // live_target_seconds", not a 15-day calendar count -- so the label
    // shows either "waiting for first tick" or how many seconds ago the
    // last one arrived, instead of a day-count fraction.
    if (info.live_tick_age_seconds === null || info.live_tick_age_seconds === undefined) {
      return "Live: waiting for first tick";
    }
    if (info.live_ready) {
      return `Live: active (${info.live_tick_age_seconds}s ago)`;
    }
    return `Live: stale (${info.live_tick_age_seconds}s ago)`;
  }

  function renderStatus(payload) {
    statusPanel.innerHTML = "";
    Object.entries(payload.tickers).forEach(([ticker, info]) => {
      const row = document.createElement("div");
      row.className = "status-row";
      row.innerHTML = `
        <span>${info.name} <span class="text-muted">(${ticker})</span></span>
        <span>
          <span class="status-pill ${info.historical_ready ? "ready" : "pending"}">Hist ${info.historical_days}/${info.historical_target}d</span>
          <span class="status-pill ${info.sentiment_ready ? "ready" : "pending"}">Sentiment ${info.sentiment_ready ? "OK" : "..."}</span>
          <span class="status-pill ${info.live_ready ? "ready" : "pending"}">${liveStatusLabel(info)}</span>
        </span>`;
      statusPanel.appendChild(row);
    });

    if (payload.all_ready) {
      fetchingLabel.textContent = "All data verified -- ready to run the multi-model forecast.";
      startBtn.disabled = false;
      if (statusPollTimer) {
        clearInterval(statusPollTimer);
        statusPollTimer = null;
      }
    } else {
      fetchingLabel.textContent = payload.ingestion_running
        ? "Fetching data..."
        : "Waiting on data sufficiency (365d history, sentiment, 15d live)...";
      startBtn.disabled = true;
    }
  }

  async function refreshStatus() {
    try {
      const res = await fetch("/api/status");
      const payload = await res.json();
      renderStatus(payload);
    } catch (err) {
      console.error("[NEXUS] status poll failed", err);
    }
  }

  function beginIngestionAndPolling() {
    fetch("/api/ingest", { method: "POST" }).catch((e) => console.error(e));
    refreshStatus();
    statusPollTimer = setInterval(refreshStatus, 4000);
  }

  // ------------------------------------------------------------------
  // "Preview page 2 layout" -- lets you check the carousel / Model
  // Breakdown / Execution Desk styling without waiting on a full
  // prediction run (which is gated behind 15 days of live data). This
  // still loads real price/sentiment charts from whatever has already
  // been ingested -- only the Model Breakdown cards stay at their
  // "Run Start Prediction to populate" placeholder state, since those
  // numbers only exist after an actual backtest.
  // ------------------------------------------------------------------
  const previewLink = document.getElementById("preview-page2-link");
  if (previewLink) {
    previewLink.addEventListener("click", async (e) => {
      e.preventDefault();
      revealPageDetails();
      await Promise.all(tickerIds.map((t) => loadChartData(t)));
      resizeAllCharts();
    });
  }

  // ------------------------------------------------------------------
  // Start Prediction -> /api/predict -> render breakdown + charts
  // ------------------------------------------------------------------
  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;
    startBtn.textContent = "Running multi-model forecast...";
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tickers: tickerIds }),
      });
      const payload = await res.json();
      Object.entries(payload.report || {}).forEach(([ticker, tickerReport]) => {
        predictionReports[ticker] = tickerReport;
        renderModelBreakdown(ticker, tickerReport);

        const downloadBtn = document.getElementById(`download-report-btn-${safeId(ticker)}`);
        if (downloadBtn) {
          downloadBtn.classList.remove("d-none");
          downloadBtn.onclick = () => downloadTickerReport(ticker);
        }
      });
      revealPageDetails();
      await Promise.all(tickerIds.map((t) => loadChartData(t)));
      resizeAllCharts();
    } catch (err) {
      console.error("[NEXUS] prediction failed", err);
    } finally {
      startBtn.textContent = "Re-run Prediction";
      startBtn.disabled = false;
    }
  });

  function renderModelBreakdown(ticker, tickerReport) {
    const sid = safeId(ticker);
    const breakdown = tickerReport.model_breakdown || {};
    const currencySymbol = tickerReport.currency_symbol || "$";
    predictionCache[ticker] = predictionCache[ticker] || {};

    MODEL_NAMES.forEach((modelName) => {
      const m = breakdown[modelName];
      if (!m) return;

      // Cache this model's full backtest-window predicted price path so it
      // can be drawn on the price chart -- each model's data here is
      // independent, nothing is averaged across MNN/Prophet/LSTM/XGBoost.
      predictionCache[ticker][modelName] = {
        predicted_series: m.predicted_series || [],
        color: MODEL_COLORS[modelName],
      };

      const summaryEl = document.getElementById(`model-summary-${modelName}-${sid}`);
      if (summaryEl) {
        summaryEl.textContent = `Past Day: ${formatCurrency(m.past_day_value, currencySymbol)}`;
      }

      // Card 1: Predicted Value -- next-step forward price forecast.
      const predictedEl = document.getElementById(`predicted-value-${modelName}-${sid}`);
      if (predictedEl) predictedEl.textContent = formatCurrency(m.predicted_price, currencySymbol);

      // Card 2: Baseline Accuracy -- Error Margin (MAE), currency-formatted.
      const maeEl = document.getElementById(`mae-value-${modelName}-${sid}`);
      if (maeEl) maeEl.textContent = formatCurrency(m.mae, currencySymbol);

      // Card 3: Trading Viability -- Directional Win Rate, color-coded progress bar.
      const winRateEl = document.getElementById(`winrate-value-${modelName}-${sid}`);
      if (winRateEl) winRateEl.textContent = `${m.win_rate.toFixed(1)}%`;
      const winBar = document.getElementById(`winrate-bar-${modelName}-${sid}`);
      if (winBar) {
        const clamped = Math.min(100, Math.max(0, m.win_rate));
        winBar.style.width = `${clamped}%`;
        winBar.setAttribute("aria-valuenow", String(m.win_rate));
        winBar.classList.remove("bg-success", "bg-warning", "bg-danger");
        winBar.classList.add(winRateBarClass(m.win_rate));
      }

      // Card 4: Tail Risk -- Volatility Risk (RMSE), currency-formatted.
      const rmseEl = document.getElementById(`rmse-value-${modelName}-${sid}`);
      if (rmseEl) rmseEl.textContent = formatCurrency(m.rmse, currencySymbol);

      // Card 5: Risk-Adjusted Performance -- Sharpe Ratio of the simple
      // long/short strategy derived from this model's predicted direction
      // (see evaluation.py's strategy_sharpe_ratio). Can be null when the
      // strategy's return series has zero variance (e.g. a model that
      // never flips its predicted sign across the whole test window) --
      // shown as "N/A" rather than a misleading 0.00.
      const sharpeEl = document.getElementById(`sharpe-value-${modelName}-${sid}`);
      if (sharpeEl) {
        if (m.sharpe_ratio === null || m.sharpe_ratio === undefined) {
          sharpeEl.textContent = "N/A";
          sharpeEl.classList.remove("text-success", "text-danger");
        } else {
          sharpeEl.textContent = m.sharpe_ratio.toFixed(2);
          sharpeEl.classList.remove("text-success", "text-danger");
          sharpeEl.classList.add(m.sharpe_ratio >= 0 ? "text-success" : "text-danger");
        }
      }

      // All four models' predicted price paths are drawn on the chart
      // automatically -- there's no per-model toggle anymore, since all
      // four always run together on one Start Prediction click.
      applyModelChartOverlay(ticker, modelName, true);

      // Highlight whichever model outperformed the others for this ticker
      // (server-computed winner, see evaluation.py's performance_backtest).
      const isWinner = modelName === tickerReport.winner;
      const itemEl = document.getElementById(`model-item-${modelName}-${sid}`);
      if (itemEl) itemEl.classList.toggle("is-winner", isWinner);
      const badgeEl = document.getElementById(`best-badge-${modelName}-${sid}`);
      if (badgeEl) badgeEl.classList.toggle("d-none", !isWinner);
    });

    const winnerEl = document.getElementById(`winner-badge-${sid}`);
    if (winnerEl) winnerEl.textContent = `Winner: ${tickerReport.winner}`;

    // Prominent winner callout: surfaces the outperforming model's name,
    // Directional Win Rate and Predicted Value right at the top of the
    // ticker card, instead of requiring the user to open the accordion.
    const winnerModel = breakdown[tickerReport.winner];
    const calloutEl = document.getElementById(`winner-callout-${sid}`);
    if (calloutEl && winnerModel) {
      calloutEl.classList.remove("d-none");
      const nameEl = document.getElementById(`winner-callout-name-${sid}`);
      if (nameEl) nameEl.textContent = tickerReport.winner;
      const winrateEl = document.getElementById(`winner-callout-winrate-${sid}`);
      if (winrateEl) winrateEl.textContent = `${winnerModel.win_rate.toFixed(1)}%`;
      const predictedEl = document.getElementById(`winner-callout-predicted-${sid}`);
      if (predictedEl) predictedEl.textContent = formatCurrency(winnerModel.predicted_price, currencySymbol);
    }
  }

  // ------------------------------------------------------------------
  // Multi-Model Chart.js Sync -- all four models' predicted-price paths
  // are drawn on the primary price chart automatically once a prediction
  // completes, so they can be visually contrasted against each other and
  // against the actual price line.
  // ------------------------------------------------------------------
  function applyModelChartOverlay(ticker, modelName, show) {
    const c = charts[ticker];
    if (!c || !c.price) return;
    const datasetLabel = `${modelName} Predicted`;
    const existingIdx = c.price.data.datasets.findIndex((d) => d.label === datasetLabel);

    if (!show) {
      if (existingIdx !== -1) {
        c.price.data.datasets.splice(existingIdx, 1);
        c.price.update();
      }
      return;
    }

    const cached = predictionCache[ticker] && predictionCache[ticker][modelName];
    if (!cached || !cached.predicted_series || !cached.predicted_series.length) {
      console.warn(`[NEXUS] No predicted_series cached yet for ${ticker}/${modelName} -- run Start Prediction first.`);
      return;
    }

    const labels = c.price.data.labels;
    if (!labels || !labels.length) {
      console.warn(`[NEXUS] Price chart has no labels loaded yet for ${ticker} -- try again once the chart finishes loading.`);
      return;
    }

    const seriesByDate = {};
    cached.predicted_series.forEach((row) => {
      seriesByDate[row.date] = row.predicted_price;
    });
    // Aligned 1:1 against the price chart's existing date labels, with
    // `null` gaps outside the model's backtest window (Chart.js renders a
    // break in the line there instead of a misleading flat/zero segment).
    const aligned = labels.map((label) => (Object.prototype.hasOwnProperty.call(seriesByDate, label) ? seriesByDate[label] : null));

    const dataset = {
      label: datasetLabel,
      data: aligned,
      borderColor: cached.color,
      backgroundColor: "transparent",
      borderDash: [6, 3],
      tension: 0.2,
      pointRadius: 0,
      spanGaps: false,
    };

    if (existingIdx !== -1) {
      c.price.data.datasets[existingIdx] = dataset;
    } else {
      c.price.data.datasets.push(dataset);
    }
    c.price.update();
  }

  // ------------------------------------------------------------------
  // Sentiment accordion (lazy-loaded on first expand)
  // ------------------------------------------------------------------
  function sentimentBadgeClass(score) {
    if (score >= 0.6) return "bull";
    if (score <= 0.4) return "bear";
    return "neutral";
  }

  async function loadSentiment(ticker) {
    const listEl = document.getElementById(`sentiment-list-${safeId(ticker)}`);
    const scoreEl = document.getElementById(`sentiment-score-${safeId(ticker)}`);
    if (!listEl || listEl.dataset.loaded === "1") return;
    listEl.innerHTML = `<div class="text-muted small">Loading headlines...</div>`;
    try {
      const res = await fetch(`/api/news/${encodeURIComponent(ticker)}`);
      const payload = await res.json();
      scoreEl.textContent = payload.average_sentiment.toFixed(3);
      scoreEl.className = `sentiment-badge ${sentimentBadgeClass(payload.average_sentiment)}`;
      listEl.innerHTML = "";
      if (!payload.top_headlines.length) {
        listEl.innerHTML = `<div class="text-muted small">No headlines ingested yet.</div>`;
      }
      payload.top_headlines.forEach((item) => {
        const div = document.createElement("div");
        div.className = "headline-item";
        div.innerHTML = `<span class="headline-date">${item.date}</span>${item.headline}
          <span class="sentiment-badge ${sentimentBadgeClass(item.sentiment_score)}">${item.sentiment_score.toFixed(2)}</span>`;
        listEl.appendChild(div);
      });
      listEl.dataset.loaded = "1";
    } catch (err) {
      listEl.innerHTML = `<div class="text-danger small">Failed to load headlines.</div>`;
    }
  }

  // ------------------------------------------------------------------
  // Chart.js -- 3 toggleable charts per stock
  // ------------------------------------------------------------------
  Chart.defaults.color = "#8b93c4";
  Chart.defaults.borderColor = "#1c2444";

  function buildChartsForTicker(ticker) {
    const id = safeId(ticker);
    const priceCtx = document.getElementById(`chart-price-${id}`);
    const sentimentCtx = document.getElementById(`chart-sentiment-${id}`);
    const modelCtx = document.getElementById(`chart-models-${id}`);
    const liveCtx = document.getElementById(`chart-live-${id}`);

    charts[ticker] = {
      price: new Chart(priceCtx, {
        type: "line",
        data: { labels: [], datasets: [
          { label: "Close", data: [], borderColor: "#22d3ee", backgroundColor: "rgba(34,211,238,0.08)", tension: 0.25, pointRadius: 0, fill: true },
        ] },
        options: chartBaseOptions("Price (Historical Only)"),
      }),
      sentiment: new Chart(sentimentCtx, {
        type: "line",
        data: { labels: [], datasets: [
          { label: "Sentiment Score", data: [], borderColor: "#c026f7", backgroundColor: "rgba(192,38,247,0.08)", tension: 0.25, pointRadius: 0, fill: true },
        ] },
        options: chartBaseOptions("Sentiment (0=Bear, 1=Bull)"),
      }),
      models: new Chart(modelCtx, {
        type: "bar",
        data: { labels: [], datasets: [
          { label: "Error Margin", data: [], backgroundColor: "#f87171" },
        ] },
        options: chartBaseOptions("Model Comparison"),
      }),
      // Section 5: intraday tick history -- genuinely separate from the
      // "Price (Historical Only)" chart above (which now only ever pulls
      // source='historical' daily candles). This one plots every polled
      // live tick for today, so the user can see the price move at any
      // instant during the trading session.
      live: new Chart(liveCtx, {
        type: "line",
        data: { labels: [], datasets: [
          { label: "Live Price", data: [], borderColor: "#34d399", backgroundColor: "rgba(52,211,153,0.08)", tension: 0.15, pointRadius: 0, fill: true },
        ] },
        options: chartBaseOptions("Live Intraday Price"),
      }),
    };
  }

  function chartBaseOptions(title) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#e9ecff" } },
        title: { display: true, text: title, color: "#e9ecff" },
      },
      scales: {
        x: { grid: { color: "#1c2444" }, ticks: { color: "#8b93c4", maxTicksLimit: 10 } },
        y: { grid: { color: "#1c2444" }, ticks: { color: "#8b93c4" } },
      },
    };
  }

  async function loadChartData(ticker) {
    try {
      const res = await fetch(`/api/chart_data/${encodeURIComponent(ticker)}`);
      const payload = await res.json();
      const c = charts[ticker];
      if (!c) return;

      const priceLabels = payload.price_series.map((r) => r.date);
      const priceValues = payload.price_series.map((r) => r.close);
      c.price.data.labels = priceLabels;
      c.price.data.datasets[0].data = priceValues;
      c.price.update();

      // Re-align every model's prediction overlay against the freshly
      // (re)loaded label set -- `labels` was just replaced wholesale above,
      // so a previously-aligned overlay dataset would otherwise silently
      // drift out of alignment. All four models are always drawn together.
      MODEL_NAMES.forEach((modelName) => {
        if (predictionCache[ticker] && predictionCache[ticker][modelName]) {
          applyModelChartOverlay(ticker, modelName, true);
        }
      });

      const sentLabels = payload.sentiment_series.map((r) => r.date);
      const sentValues = payload.sentiment_series.map((r) => r.sentiment_score);
      c.sentiment.data.labels = sentLabels;
      c.sentiment.data.datasets[0].data = sentValues;
      c.sentiment.update();

      const models = payload.model_comparison || [];
      c.models.data.labels = models.map((m) => m.model_name);
      c.models.data.datasets[0].data = models.map((m) => m.error_margin);
      c.models.update();

      await loadLiveTicks(ticker);
    } catch (err) {
      console.error(`[NEXUS] chart data load failed for ${ticker}`, err);
    }
  }

  // ------------------------------------------------------------------
  // Live Price section -- separate from loadChartData's historical-only
  // price series. Pulls the append-only intraday tick log plus the latest
  // price/timestamp, and drives both the "Live Price" chart tab and the
  // current-price readout (value, change vs last close, "as of HH:MM:SS",
  // pulsing live indicator dot).
  // ------------------------------------------------------------------
  async function loadLiveTicks(ticker) {
    const id = safeId(ticker);
    const c = charts[ticker];
    try {
      const res = await fetch(`/api/live_ticks/${encodeURIComponent(ticker)}`);
      const payload = await res.json();
      const ticks = payload.ticks || [];

      if (c && c.live) {
        c.live.data.labels = ticks.map((t) => formatClockTime(t.timestamp));
        c.live.data.datasets[0].data = ticks.map((t) => t.price);
        c.live.update();
      }

      const symbol = tickerCurrencySymbol(ticker);
      const valueEl = document.getElementById(`live-price-value-${id}`);
      const changeEl = document.getElementById(`live-price-change-${id}`);
      const asofEl = document.getElementById(`live-price-asof-${id}`);
      const dotEl = document.getElementById(`live-dot-${id}`);

      if (valueEl) valueEl.textContent = formatCurrency(payload.latest_price, symbol);

      if (changeEl) {
        if (payload.latest_price != null && payload.last_close) {
          const diff = payload.latest_price - payload.last_close;
          const pct = (diff / payload.last_close) * 100;
          const sign = diff >= 0 ? "+" : "";
          changeEl.textContent = `${sign}${diff.toFixed(2)} (${sign}${pct.toFixed(2)}%)`;
          changeEl.classList.remove("text-success", "text-danger");
          changeEl.classList.add(diff >= 0 ? "text-success" : "text-danger");
        } else {
          changeEl.textContent = "";
        }
      }

      if (asofEl) {
        asofEl.textContent = payload.latest_timestamp
          ? `as of ${formatClockTime(payload.latest_timestamp)}`
          : "Waiting for live tick...";
      }

      if (dotEl) dotEl.classList.toggle("live-dot-active", !!payload.latest_price);
    } catch (err) {
      console.error(`[NEXUS] live ticks load failed for ${ticker}`, err);
    }
  }

  function wireChartToggles(ticker) {
    const id = safeId(ticker);
    const group = document.getElementById(`chart-toggle-${id}`);
    if (!group) return;
    group.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", () => {
        group.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const target = btn.dataset.chart;
        ["price", "sentiment", "models", "live"].forEach((key) => {
          const wrap = document.getElementById(`chart-wrap-${key}-${id}`);
          if (wrap) wrap.style.display = key === target ? "block" : "none";
        });

        if (target === "live") {
          loadLiveTicks(ticker);
        }

        // Replay the line-draw-in animation every time a chart tab is
        // opened, instead of showing a line that's already fully plotted
        // and static the moment it becomes visible -- a chart built while
        // hidden (display:none) never got to animate its entrance, and
        // Chart.js doesn't automatically replay it just because the canvas
        // becomes visible again, so it's triggered manually here.
        const targetChart = charts[ticker] && charts[ticker][target];
        if (targetChart) {
          requestAnimationFrame(() => {
            targetChart.reset();
            targetChart.update();
          });
        }
      });
    });
  }

  // ------------------------------------------------------------------
  // INVESTRA floating chat widget (RAG-grounded on ingested news, plus the
  // last computed Portfolio Management result if there is one).
  // ------------------------------------------------------------------
  const chatForm = document.getElementById("nexus-chat-form");
  const chatInput = document.getElementById("nexus-chat-input");
  const chatMessages = document.getElementById("nexus-chat-messages");
  const chatTickerSelect = document.getElementById("nexus-chat-ticker");
  const investraLauncher = document.getElementById("investra-launcher");
  const investraPanel = document.getElementById("investra-panel");
  const investraCloseBtn = document.getElementById("investra-close-btn");

  function applyInvestraBranding() {
    const nameEl = document.getElementById("investra-name");
    const taglineEl = document.getElementById("investra-tagline");
    const avatarEls = [
      document.getElementById("investra-avatar"),
      document.getElementById("investra-launcher-avatar"),
    ];
    if (nameEl) nameEl.textContent = INVESTRA_CONFIG.name;
    if (taglineEl) taglineEl.textContent = INVESTRA_CONFIG.tagline;
    avatarEls.forEach((el) => {
      if (!el) return;
      if (INVESTRA_CONFIG.logo_url) {
        el.innerHTML = `<img src="${INVESTRA_CONFIG.logo_url}" alt="${INVESTRA_CONFIG.name}">`;
      } else {
        el.textContent = (INVESTRA_CONFIG.name || "I").charAt(0).toUpperCase();
      }
    });
  }

  function openInvestraPanel() {
    if (!investraPanel) return;
    investraPanel.classList.remove("d-none");
    if (investraLauncher) investraLauncher.classList.add("d-none");
    if (chatInput) chatInput.focus();
  }

  function closeInvestraPanel() {
    if (!investraPanel) return;
    investraPanel.classList.add("d-none");
    if (investraLauncher) investraLauncher.classList.remove("d-none");
  }

  if (investraLauncher) {
    investraLauncher.addEventListener("click", openInvestraPanel);
    investraLauncher.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") openInvestraPanel();
    });
  }
  if (investraCloseBtn) investraCloseBtn.addEventListener("click", closeInvestraPanel);

  function appendBubble(role, text) {
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${role}`;
    bubble.textContent = text;
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    chatTranscript.push({ role, text });
    return bubble;
  }

  function appendTyping() {
    const bubble = document.createElement("div");
    bubble.className = "chat-bubble ai typing-indicator";
    bubble.innerHTML = "<span></span><span></span><span></span>";
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return bubble;
  }

  if (chatForm) {
    chatForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const query = chatInput.value.trim();
      if (!query) return;
      appendBubble("user", query);
      chatInput.value = "";
      const typingEl = appendTyping();

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query,
            ticker: chatTickerSelect.value || null,
            // Grounds follow-up questions ("why this split?") in the actual
            // computed numbers if Portfolio Management has been run this
            // session -- null otherwise, so the prompt-building on the
            // server side just skips that block entirely.
            portfolio_context: lastPortfolioResult,
          }),
        });
        const payload = await res.json();
        typingEl.remove();
        if (payload.error) {
          appendBubble("ai", `Error: ${payload.error}`);
        } else {
          appendBubble("ai", payload.response);
        }
      } catch (err) {
        typingEl.remove();
        appendBubble("ai", "Connection error -- please try again.");
      }
    });
  }

  // ------------------------------------------------------------------
  // Portfolio Management (Markowitz optimization form)
  // ------------------------------------------------------------------
  const portfolioForm = document.getElementById("portfolio-form");
  const portfolioSubmitBtn = document.getElementById("portfolio-submit-btn");
  const portfolioFormError = document.getElementById("portfolio-form-error");
  const portfolioResultsEl = document.getElementById("portfolio-results");

  const OBJECTIVE_LABELS = {
    min_volatility: "Capital Preservation (Min Volatility)",
    max_sharpe: "Growth-Oriented (Max Sharpe)",
    target_return: "Balanced (Target Return)",
    single_ticker_no_diversification: "Single Stock (No Diversification Possible)",
  };

  function formatRupees(amount) {
    if (amount === null || amount === undefined || Number.isNaN(amount)) return "--";
    return `₹${amount.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;
  }

  function renderPortfolioResult(result) {
    document.getElementById("portfolio-objective-badge").textContent =
      OBJECTIVE_LABELS[result.objective_used] || result.objective_used;

    const underMinNote = document.getElementById("portfolio-under-min-note");
    underMinNote.textContent = result.under_minimum
      ? `Note: whole-share rounding couldn't reach your minimum of ${formatRupees(result.amount_min)} -- see allocation below.`
      : "";

    document.getElementById("portfolio-expected-return").textContent =
      `${result.expected_annual_return_pct.toFixed(2)}%`;
    document.getElementById("portfolio-volatility").textContent =
      `${result.annual_volatility_pct.toFixed(2)}%`;
    document.getElementById("portfolio-sharpe").textContent =
      result.sharpe_ratio === null || result.sharpe_ratio === undefined
        ? "N/A" : result.sharpe_ratio.toFixed(2);

    const body = document.getElementById("portfolio-breakdown-body");
    body.innerHTML = "";
    result.breakdown.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${row.ticker}</td>
        <td>${row.weight_pct.toFixed(2)}%</td>
        <td>${formatRupees(row.price)}</td>
        <td>${row.shares}</td>
        <td>${formatRupees(row.allocated_amount)}</td>`;
      body.appendChild(tr);
    });

    document.getElementById("portfolio-cash-note").textContent =
      `Total allocated: ${formatRupees(result.total_allocated)} -- leftover cash: ${formatRupees(result.leftover_cash)} ` +
      `(against your amount ceiling of ${formatRupees(result.amount_max)}).`;

    const explanationEl = document.getElementById("portfolio-explanation");
    explanationEl.textContent = result.explanation
      || "INVESTRA's explanation is unavailable right now (no AI provider configured or the call failed) -- the numbers above are still accurate.";

    portfolioResultsEl.classList.remove("d-none");
  }

  if (portfolioForm) {
    portfolioForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      portfolioFormError.textContent = "";

      const checkedTickers = Array.from(
        document.querySelectorAll("#portfolio-ticker-checks input[type=checkbox]:checked")
      ).map((el) => el.value);
      if (checkedTickers.length === 0) {
        portfolioFormError.textContent = "Select at least one ticker.";
        return;
      }

      const amountMinRaw = document.getElementById("portfolio-amount-min").value;
      const amountMaxRaw = document.getElementById("portfolio-amount-max").value;
      const amountMin = parseFloat(amountMinRaw);
      if (!amountMinRaw || Number.isNaN(amountMin) || amountMin <= 0) {
        portfolioFormError.textContent = "Enter a valid amount.";
        return;
      }
      const amountMax = amountMaxRaw ? parseFloat(amountMaxRaw) : amountMin;
      if (Number.isNaN(amountMax) || amountMax < amountMin) {
        portfolioFormError.textContent = "Max amount must be >= Min amount.";
        return;
      }

      const horizon = document.getElementById("portfolio-horizon").value;

      portfolioSubmitBtn.disabled = true;
      portfolioSubmitBtn.textContent = "Optimizing...";

      try {
        const res = await fetch("/api/portfolio_optimize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            tickers: checkedTickers,
            amount_min: amountMin,
            amount_max: amountMax,
            horizon,
          }),
        });
        const payload = await res.json();
        if (payload.error) {
          portfolioFormError.textContent = payload.error;
          return;
        }
        lastPortfolioResult = payload;
        renderPortfolioResult(payload);
      } catch (err) {
        portfolioFormError.textContent = "Connection error -- please try again.";
      } finally {
        portfolioSubmitBtn.disabled = false;
        portfolioSubmitBtn.textContent = "Optimize Portfolio";
      }
    });
  }

  // ------------------------------------------------------------------
  // PDF report downloads -- both routes return a raw PDF; fetched as a
  // blob and downloaded via a throwaway <a>, since these aren't simple
  // GET links (they need a JSON body of data already sitting in memory).
  // ------------------------------------------------------------------
  async function downloadPdfBlob(url, body, filename) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      throw new Error(payload.error || `Report generation failed (${res.status}).`);
    }
    const blob = await res.blob();
    const blobUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = blobUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(blobUrl);
  }

  async function downloadTickerReport(ticker) {
    const btn = document.getElementById(`download-report-btn-${safeId(ticker)}`);
    const report = predictionReports[ticker];
    if (!report) return;
    if (btn) { btn.disabled = true; btn.textContent = "Generating..."; }
    try {
      await downloadPdfBlob(
        "/api/report/ticker_pdf",
        { ticker, report },
        `NEXUS_${ticker.replace(".", "_")}_Report.pdf`,
      );
    } catch (err) {
      console.error("[NEXUS] ticker report download failed", err);
      alert(err.message || "Report generation failed.");
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = "Download Report"; }
    }
  }

  const downloadFullReportBtn = document.getElementById("download-full-report-btn");
  if (downloadFullReportBtn) {
    downloadFullReportBtn.addEventListener("click", async () => {
      downloadFullReportBtn.disabled = true;
      downloadFullReportBtn.textContent = "Generating...";
      try {
        await downloadPdfBlob(
          "/api/report/full_pdf",
          {
            tickers_reports: predictionReports,
            portfolio_result: lastPortfolioResult,
            chat_transcript: chatTranscript,
          },
          "NEXUS_Full_Report.pdf",
        );
      } catch (err) {
        console.error("[NEXUS] full report download failed", err);
        alert(err.message || "Report generation failed.");
      } finally {
        downloadFullReportBtn.disabled = false;
        downloadFullReportBtn.textContent = "Download Full Report (PDF)";
      }
    });
  }

  // ------------------------------------------------------------------
  // Init
  // ------------------------------------------------------------------
  document.addEventListener("DOMContentLoaded", () => {
    tickerIds.forEach((ticker) => {
      buildChartsForTicker(ticker);
      wireChartToggles(ticker);
      const accordionEl = document.getElementById(`sentiment-collapse-${safeId(ticker)}`);
      if (accordionEl) {
        accordionEl.addEventListener("show.bs.collapse", () => loadSentiment(ticker));
      }
    });

    // Bootstrap tooltips for the RMSE "Volatility Risk" / amount-range info icons.
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach((el) => new bootstrap.Tooltip(el));

    applyInvestraBranding();
    if (chatMessages) appendBubble("ai", INVESTRA_CONFIG.opening_message);

    beginIngestionAndPolling();
  });
})();
