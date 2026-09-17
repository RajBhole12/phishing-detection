// email_scanner.js — drives the new Email Scanner page

const analyzeEmailBtn = document.getElementById("analyze-email-btn");
const senderInput = document.getElementById("email-sender");
const subjectInput = document.getElementById("email-subject");
const bodyInput = document.getElementById("email-body");
const loadingEl = document.getElementById("loading");
const resultWrap = document.getElementById("result-wrap");
const errorBox = document.getElementById("error-box");

if (analyzeEmailBtn) {
  analyzeEmailBtn.addEventListener("click", runEmailScan);
}

async function runEmailScan() {
  const sender = senderInput.value.trim();
  const subject = subjectInput.value.trim();
  const body = bodyInput.value.trim();

  errorBox.innerHTML = "";
  resultWrap.style.display = "none";

  if (!body) {
    errorBox.innerHTML = `<div class="alert alert-danger">Please paste the email body to analyze.</div>`;
    return;
  }

  loadingEl.style.display = "flex";
  analyzeEmailBtn.disabled = true;

  try {
    const res = await fetch("/api/predict-email", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sender, subject, body }),
    });
    const data = await res.json();

    loadingEl.style.display = "none";
    analyzeEmailBtn.disabled = false;

    if (!res.ok) {
      errorBox.innerHTML = `<div class="alert alert-danger">${escapeHtmlLocal(data.error || "Something went wrong.")}</div>`;
      return;
    }

    renderEmailResult(data);
  } catch (err) {
    loadingEl.style.display = "none";
    analyzeEmailBtn.disabled = false;
    errorBox.innerHTML = `<div class="alert alert-danger">Network error: ${escapeHtmlLocal(err.message)}</div>`;
  }
}

function escapeHtmlLocal(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function riskLevelClassLocal(pred) {
  if (pred === "Phishing") return "phishing";
  if (pred === "Suspicious") return "suspicious";
  return "legitimate";
}

function renderEmailResult(data) {
  const cls = riskLevelClassLocal(data.prediction);

  const featureChips = Object.entries(data.features).map(([k, v]) => `
    <div class="feature-chip">
      <div class="k">${k.replace(/_/g, " ")}</div>
      <div class="v">${typeof v === "number" ? v : v}</div>
    </div>
  `).join("");

  const riskFactors = data.risk_factors.map(r => `
    <div class="risk-factor">
      <span class="risk-dot ${r.level}"></span>
      <span>${escapeHtmlLocal(r.text)}</span>
    </div>
  `).join("");

  resultWrap.innerHTML = `
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
      ${escapeHtmlLocal(data.explanation)}
    </div>

    <div class="card" style="margin-bottom:18px;">
      <div class="card-header"><span class="card-title">Risk Factors &amp; AI Explanation</span></div>
      ${riskFactors}
    </div>

    <div class="card">
      <div class="card-header"><span class="card-title">Extracted Email Features (${Object.keys(data.features).length})</span></div>
      <div class="feature-grid">${featureChips}</div>
    </div>
  `;
  resultWrap.style.display = "block";
  resultWrap.scrollIntoView({ behavior: "smooth", block: "start" });
}
