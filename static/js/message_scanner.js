// message_scanner.js — drives the new Message Scanner page

const analyzeMessageBtn = document.getElementById("analyze-message-btn");
const messageInput = document.getElementById("message-text");
const msgLoadingEl = document.getElementById("loading");
const msgResultWrap = document.getElementById("result-wrap");
const msgErrorBox = document.getElementById("error-box");

if (analyzeMessageBtn) {
  analyzeMessageBtn.addEventListener("click", runMessageScan);
}

async function runMessageScan() {
  const message = messageInput.value.trim();

  msgErrorBox.innerHTML = "";
  msgResultWrap.style.display = "none";

  if (!message) {
    msgErrorBox.innerHTML = `<div class="alert alert-danger">Please paste a message to analyze.</div>`;
    return;
  }

  msgLoadingEl.style.display = "flex";
  analyzeMessageBtn.disabled = true;

  try {
    const res = await fetch("/api/predict-message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();

    msgLoadingEl.style.display = "none";
    analyzeMessageBtn.disabled = false;

    if (!res.ok) {
      msgErrorBox.innerHTML = `<div class="alert alert-danger">${escapeHtmlMsg(data.error || "Something went wrong.")}</div>`;
      return;
    }

    renderMessageResult(data);
  } catch (err) {
    msgLoadingEl.style.display = "none";
    analyzeMessageBtn.disabled = false;
    msgErrorBox.innerHTML = `<div class="alert alert-danger">Network error: ${escapeHtmlMsg(err.message)}</div>`;
  }
}

function escapeHtmlMsg(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function riskLevelClassMsg(pred) {
  if (pred === "Phishing") return "phishing";
  if (pred === "Suspicious") return "suspicious";
  return "legitimate";
}

function renderMessageResult(data) {
  const cls = riskLevelClassMsg(data.prediction);

  const featureChips = Object.entries(data.features).map(([k, v]) => `
    <div class="feature-chip">
      <div class="k">${k.replace(/_/g, " ")}</div>
      <div class="v">${typeof v === "number" ? v : v}</div>
    </div>
  `).join("");

  const riskFactors = data.risk_factors.map(r => `
    <div class="risk-factor">
      <span class="risk-dot ${r.level}"></span>
      <span>${escapeHtmlMsg(r.text)}</span>
    </div>
  `).join("");

  msgResultWrap.innerHTML = `
    <div class="result-card ${cls}">
      <div class="result-label ${cls}">${data.prediction.toUpperCase()}</div>
      <div class="result-metrics">
        <div><div class="result-metric-value">${data.risk_score}/100</div><div class="result-metric-label">Risk Score</div></div>
        <div><div class="result-metric-value">${data.confidence}%</div><div class="result-metric-label">Confidence</div></div>
        <div><div class="result-metric-value">${data.model_probability}%</div><div class="result-metric-label">Model Probability</div></div>
        <div><div class="result-metric-value" style="font-size:20px;">${data.model_used}</div><div class="result-metric-label">Model Used</div></div>
      </div>
    </div>

    <div class="alert ${cls === 'phishing' ? 'alert-danger' : cls === 'suspicious' ? 'alert-warn' : 'alert-info'}" style="margin-bottom:18px;">
      ${escapeHtmlMsg(data.explanation)}
    </div>

    <div class="card" style="margin-bottom:18px;">
      <div class="card-header"><span class="card-title">Risk Factors &amp; AI Explanation</span></div>
      ${riskFactors}
    </div>

    <div class="card">
      <div class="card-header"><span class="card-title">Extracted Message Features (${Object.keys(data.features).length})</span></div>
      <div class="feature-grid">${featureChips}</div>
    </div>
  `;
  msgResultWrap.style.display = "block";
  msgResultWrap.scrollIntoView({ behavior: "smooth", block: "start" });
}
