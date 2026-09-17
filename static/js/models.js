// models.js — populates the Model Analytics page

async function loadModelPerformance() {
  const banner = document.getElementById("best-model-banner");
  if (!banner) return; // models not trained, page shows the warning block instead

  try {
    const res = await fetch("/api/model-performance");
    const data = await res.json();
    if (!res.ok) return;

    const info = data.best_model_info;
    banner.innerHTML = `Primary prediction model: <strong>${info.best_model}</strong> — selected automatically by highest ROC-AUC on the held-out test set. Trained on ${info.dataset_size} samples (${info.n_train} train / ${info.n_test} test).`;

    renderComparisonChart(data.metrics);
    renderMetricsTable(data.metrics, info.best_model);
  } catch (err) {
    console.error(err);
  }
}

function renderComparisonChart(metrics) {
  const ctx = document.getElementById("chart-model-comparison");
  const models = Object.keys(metrics);
  const metricKeys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"];
  const colors = [CHART_COLORS.accent, CHART_COLORS.safe, CHART_COLORS.warn, CHART_COLORS.danger, "#a78bfa"];

  new Chart(ctx, {
    type: "bar",
    data: {
      labels: models,
      datasets: metricKeys.map((m, i) => ({
        label: m.replace("_", " ").toUpperCase(),
        data: models.map(mod => metrics[mod][m]),
        backgroundColor: colors[i],
        borderRadius: 4,
      })),
    },
    options: {
      scales: {
        y: { min: 0, max: 1, grid: { color: CHART_COLORS.grid } },
        x: { grid: { display: false } },
      },
    },
  });
}

function renderMetricsTable(metrics, bestName) {
  const body = document.getElementById("metrics-body");
  body.innerHTML = Object.entries(metrics).map(([name, m]) => `
    <tr>
      <td>${name} ${name === bestName ? '<span class="badge badge-safe">BEST</span>' : ""}</td>
      <td>${(m.accuracy * 100).toFixed(2)}%</td>
      <td>${(m.precision * 100).toFixed(2)}%</td>
      <td>${(m.recall * 100).toFixed(2)}%</td>
      <td>${(m.f1_score * 100).toFixed(2)}%</td>
      <td>${m.roc_auc.toFixed(4)}</td>
    </tr>
  `).join("");
}

loadModelPerformance();
