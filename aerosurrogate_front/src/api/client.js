/**
 * API client — always uses relative /api/ path.
 * nginx proxies /api/ → http://api:8000/  so this works in Docker and locally.
 */
const API = "/api";

export async function apiHealth() {
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(4000) });
    return r.ok ? await r.json() : null;
  } catch { return null; }
}

export async function apiPredict(params) {
  const r = await fetch(`${API}/predict`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}

export async function apiBatch(inputs) {
  const r = await fetch(`${API}/predict/batch`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ inputs }),
  });
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}

export async function apiField(params) {
  try {
    const r = await fetch(`${API}/predict/field`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    return r.ok ? r.json() : null;
  } catch { return null; }
}

export async function apiCompare(configA, configB) {
  const r = await fetch(`${API}/compare`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ configA, configB }),
  });
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}

export async function apiExportSweep(points) {
  const r = await fetch(`${API}/export/sweep`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points }),
  });
  if (!r.ok) throw new Error(`${r.status}`);
  return r.blob();
}
