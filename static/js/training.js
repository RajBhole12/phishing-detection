// training.js — dataset stats + Train Models button + polling

const trainBtn = document.getElementById("train-btn");
const statusEl = document.getElementById("train-status");
let pollTimer = null;

async function loadDatasetStats() {
  try {
    const res = await fetch("/api/model-performance");
    if (!res.ok) return; // not trained yet, stats unavailable
    const data = await res.json();
    const info = data.best_model_info;
    document.getElementById("s-size").textContent = info.dataset_size;
    document.getElementById("s-features").textContent = info.n_features;
    document.getElementById("s-train").textContent = info.n_train;
    document.getElementById("s-test").textContent = info.n_test;
    document.getElementById("class-dist").innerHTML =
      `Legitimate: <strong>${info.class_distribution.legitimate}</strong> &nbsp;•&nbsp; Phishing: <strong>${info.class_distribution.phishing}</strong>`;
  } catch (err) {
    console.error(err);
  }
}

if (trainBtn) {
  trainBtn.addEventListener("click", async () => {
    trainBtn.disabled = true;
    statusEl.innerHTML = `<div class="alert alert-info">Starting training…</div>`;

    const res = await fetch("/api/train", { method: "POST" });
    const data = await res.json();

    if (!res.ok) {
      statusEl.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
      trainBtn.disabled = false;
      return;
    }

    pollTimer = setInterval(pollStatus, 1500);
  });
}

async function pollStatus() {
  const res = await fetch("/api/train-status");
  const data = await res.json();

  if (data.in_progress) {
    statusEl.innerHTML = `<div class="alert alert-info">${data.message}</div>`;
    return;
  }

  clearInterval(pollTimer);
  trainBtn.disabled = false;

  if (data.error) {
    statusEl.innerHTML = `<div class="alert alert-danger">Training failed: ${data.error}</div>`;
  } else {
    statusEl.innerHTML = `<div class="alert alert-info">${data.message} Reload the page to see updated stats, or visit Model Analytics.</div>`;
    loadDatasetStats();
  }
}

loadDatasetStats();
