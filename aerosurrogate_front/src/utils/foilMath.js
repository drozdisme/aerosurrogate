export function generateFoil(thickness, camber, camberPos, n = 100) {
  const upper = [];
  const lower = [];
  const t = Math.max(0.01, Math.min(thickness, 0.4));
  const m = camber;
  const p = Math.max(0.1, Math.min(camberPos, 0.9));

  for (let i = 0; i <= n; i++) {
    const x = (1 - Math.cos(Math.PI * i / n)) / 2;
    const yt = 5 * t * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x * x * x - 0.1015 * x * x * x * x);

    let yc, dy;
    if (x < p) {
      yc = (m / (p * p)) * (2 * p * x - x * x);
      dy = (2 * m / (p * p)) * (p - x);
    } else {
      yc = (m / ((1 - p) * (1 - p))) * (1 - 2 * p + 2 * p * x - x * x);
      dy = (2 * m / ((1 - p) * (1 - p))) * (p - x);
    }

    const theta = Math.atan(dy);
    upper.push({ x: x - yt * Math.sin(theta), y: yc + yt * Math.cos(theta) });
    lower.push({ x: x + yt * Math.sin(theta), y: yc - yt * Math.cos(theta) });
  }
  return { upper, lower };
}

export function estimateCp(alpha, mach, thickness, camber, camberPos) {
  const a = (alpha * Math.PI) / 180;
  const beta = Math.sqrt(Math.max(1 - mach * mach, 0.01));
  const p = Math.max(0.1, Math.min(camberPos, 0.9));
  const points = [];

  for (let i = 0; i < 200; i++) {
    const x = i / 199;
    let dy;
    if (x < p) dy = (2 * camber) / (p * p) * (p - x);
    else dy = (2 * camber) / ((1 - p) * (1 - p)) * (p - x);

    const te = thickness * (1 / Math.sqrt(Math.max(x, 0.001)) - 1) * 0.15;
    const ce = dy * 2;
    const ae = (a / Math.sqrt(Math.max(x, 0.001))) * 0.4;

    const vu = 1 + ae + ce + Math.abs(te);
    const vl = 1 - ae - ce + Math.abs(te) * 0.3;
    const stag = Math.exp(-x * 40) * (1 - 0.5 * Math.abs(a));

    points.push({
      x: +x.toFixed(4),
      cpU: +Math.max(-6, Math.min(1, (1 - vu * vu) / beta + stag * 0.8)).toFixed(4),
      cpL: +Math.max(-6, Math.min(1, (1 - vl * vl) / beta + stag * 0.8)).toFixed(4),
    });
  }
  return points;
}