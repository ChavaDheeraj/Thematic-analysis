// Validated reference palette (fixed order — never cycled). Light and dark are
// separately stepped for their surfaces.
const LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
const DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];

export const OTHER_LIGHT = "#b4b2a9";
export const OTHER_DARK = "#5f5e58";

export function isDark(): boolean {
  const t = document.documentElement.dataset.theme;
  if (t === "dark") return true;
  if (t === "light") return false;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
}

export function series(i: number): string {
  const p = isDark() ? DARK : LIGHT;
  if (i < 0 || i >= p.length) return isDark() ? OTHER_DARK : OTHER_LIGHT;
  return p[i];
}

export function chrome() {
  const d = isDark();
  return {
    surface: d ? "#1a1a19" : "#fcfcfb",
    ink: d ? "#f0efec" : "#0b0b0b",
    ink2: d ? "#c3c2b7" : "#52514e",
    muted: "#898781",
    grid: d ? "#2c2c2a" : "#e1e0d9",
    axis: d ? "#383835" : "#c3c2b7",
    neutral: d ? "#383835" : "#d6d5ce",
  };
}

/** `color` laid over `base` at `alpha`, as an opaque colour, so lines passing behind a shape stay hidden. */
export function tint(color: string, base: string, alpha: number): string {
  const ch = (hex: string, i: number) => parseInt(hex.slice(1 + 2 * i, 3 + 2 * i), 16);
  return `rgb(${[0, 1, 2].map((i) => Math.round(ch(color, i) * alpha + ch(base, i) * (1 - alpha))).join(",")})`;
}

// Sentiment: polarity (blue ↔ red with a gray midpoint) plus two non-polar states.
export function sentimentColor(label: string): string {
  const d = isDark();
  switch (label) {
    case "positive": return d ? "#3987e5" : "#2a78d6";
    case "negative": return d ? "#e66767" : "#e34948";
    case "mixed": return d ? "#9085e9" : "#6250d6";
    case "neutral": return d ? "#5f5e58" : "#b4b2a9";
    default: return d ? "#383835" : "#e1e0d9";
  }
}

// Sequential single-hue blue ramp for magnitude (heatmaps).
export const SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
