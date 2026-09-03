export const number = (value: number, digits = 2) =>
  Number.isFinite(value)
    ? value.toLocaleString("en-US", {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
      })
    : "—";

export const price = (value: number) =>
  Number.isFinite(value) && value > 0
    ? value.toLocaleString("en-US", {
        maximumFractionDigits: 2,
        minimumFractionDigits: 2,
      })
    : "—";

export const signed = (value: number, digits = 2) => {
  if (!Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${number(value, digits)}`;
};

export const percent = (value: number, digits = 0) =>
  `${number(value * 100, digits)}%`;

export const age = (value: number) =>
  value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(1)} s`;

export function formatTimestamp(timestamp: string) {
  if (!timestamp) return "No snapshot yet";
  const date = new Date(timestamp);
  return Number.isNaN(date.valueOf())
    ? timestamp
    : date.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
}
