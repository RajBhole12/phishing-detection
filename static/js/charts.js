// charts.js — shared Chart.js theming helpers used across pages

const CHART_COLORS = {
  accent: "#4f8cff",
  safe: "#38d996",
  warn: "#ffb454",
  danger: "#ff5c7a",
  grid: "#1e2538",
  text: "#8b93b0",
};

Chart.defaults.color = CHART_COLORS.text;
Chart.defaults.font.family = "Inter, sans-serif";
Chart.defaults.borderColor = CHART_COLORS.grid;

function fmtDate(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  } catch (e) {
    return iso;
  }
}

function predictionBadge(pred) {
  const map = { Phishing: "badge-danger", Suspicious: "badge-warn", Legitimate: "badge-safe" };
  const cls = map[pred] || "badge-warn";
  return `<span class="badge ${cls}">${pred}</span>`;
}
