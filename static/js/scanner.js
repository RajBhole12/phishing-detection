// scanner.js — drives the main URL scanning page

const analyzeBtn = document.getElementById("analyze-btn");
const urlInput = document.getElementById("url-input");
const loadingEl = document.getElementById("loading");
const resultWrap = document.getElementById("result-wrap");
const errorBox = document.getElementById("error-box");

if (analyzeBtn) {
  analyzeBtn.addEventListener("click", runScan);
  urlInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") runScan();
  });
}

async function runScan() {
  const url = urlInput.value.trim();
  errorBox.innerHTML = "";
  resultWrap.style.display = "none";

  if (!url) {
    errorBox.innerHTML = `<div class="alert alert-danger">Please enter a URL to analyze.</div>`;
    return;
  }

  loadingEl.style.display = "flex";
  analyzeBtn.disabled = true;

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();

    loadingEl.style.display = "none";
    analyzeBtn.disabled = false;

    if (!res.ok) {
      errorBox.innerHTML = `<div class="alert alert-danger">${escapeHtml(data.error || "Something went wrong.")}</div>`;
      return;
    }

    renderResult(data);
  } catch (err) {
    loadingEl.style.display = "none";
    analyzeBtn.disabled = false;
    errorBox.innerHTML = `<div class="alert alert-danger">Network error: ${escapeHtml(err.message)}</div>`;
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function riskLevelClass(pred) {
  if (pred === "Phishing") return "phishing";
  if (pred === "Suspicious") return "suspicious";
  return "legitimate";
}

function renderResult(data) {
  const cls = riskLevelClass(data.prediction);

  const featureChips = Object.entries(data.features).map(([k, v]) => `
    <div class="feature-chip">
      <div class="k">${k.replace(/_/g, " ")}</div>
      <div class="v">${typeof v === "number" ? v : v}</div>
    </div>
  `).join("");

  const riskFactors = data.risk_factors.map(r => `
    <div class="risk-factor">
      <span class="risk-dot ${r.level}"></span>
      <span>${escapeHtml(r.text)}</span>
    </div>
  `).join("");

  resultWrap.innerHTML = `
    <div class="result-card ${cls}">
      <div class="result-label ${cls}">${data.prediction.toUpperCase()}</div>
      <div class="result-url">${escapeHtml(data.url)}</div>
      <div class="result-metrics">
        <div><div class="result-metric-value">${data.risk_score}/100</div><div class="result-metric-label">Risk Score</div></div>
        <div><div class="result-metric-value">${data.confidence}%</div><div class="result-metric-label">Confidence</div></div>
        <div><div class="result-metric-value">${data.model_probability}%</div><div class="result-metric-label">Model Probability</div></div>
        <div><div class="result-metric-value" style="font-size:20px;">${data.model_used}</div><div class="result-metric-label">Model Used</div></div>
      </div>
    </div>

    <div class="alert ${cls === 'phishing' ? 'alert-danger' : cls === 'suspicious' ? 'alert-warn' : 'alert-info'}" style="margin-bottom:18px;">
      ${escapeHtml(data.explanation)}
    </div>

    <div class="card" style="margin-bottom:18px;">
      <div class="card-header"><span class="card-title">Risk Factors &amp; AI Explanation</span></div>
      ${riskFactors}
    </div>

    <div class="card" style="margin-bottom:18px;">
      <div class="card-header"><span class="card-title">Prediction Probability</span></div>
      <canvas id="prob-chart" height="80"></canvas>
    </div>

    <div class="card">
      <div class="card-header"><span class="card-title">Extracted Features (${Object.keys(data.features).length})</span></div>
      <div class="feature-grid">${featureChips}</div>
    </div>
  `;
  resultWrap.style.display = "block";

  const ctx = document.getElementById("prob-chart");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Legitimate", "Phishing"],
      datasets: [{
        data: [data.probability_legitimate, data.probability_phishing],
        backgroundColor: [CHART_COLORS.safe, CHART_COLORS.danger],
        borderRadius: 6,
      }],
    },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: {
        x: { max: 100, grid: { color: CHART_COLORS.grid }, ticks: { callback: v => v + "%" } },
        y: { grid: { display: false } },
      },
    },
  });

  resultWrap.scrollIntoView({ behavior: "smooth", block: "start" });
}
