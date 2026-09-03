export const API_DEFAULT = "http://localhost:8000";
export const MARKET_WS_PATH = "/ws/market";

export function websocketUrl(apiUrl: string) {
  return `${apiUrl.replace(/\/$/, "").replace(/^http/, "ws")}${MARKET_WS_PATH}`;
}
