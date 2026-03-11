import { colors } from "../../constants/theme";

const pulse = `
@keyframes aero-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
`;

if (typeof document !== "undefined") {
  const style = document.createElement("style");
  style.textContent = pulse;
  document.head.appendChild(style);
}

export function Skeleton({ width = "100%", height = 16, radius = 4, style: extra = {} }) {
  return (
    <div
      style={{
        width,
        height,
        borderRadius: radius,
        background: colors.border,
        animation: "aero-pulse 1.5s ease-in-out infinite",
        ...extra,
      }}
    />
  );
}

export function SkeletonChart({ height = 240 }) {
  return (
    <div style={{ width: "100%", height, position: "relative", overflow: "hidden", borderRadius: 4 }}>
      <Skeleton width="100%" height={height} radius={4} />
      <div style={{ position: "absolute", bottom: 12, left: 12, right: 12, display: "flex", flexDirection: "column", gap: 6 }}>
        {[60, 80, 50, 90, 40].map((w, i) => (
          <Skeleton key={i} width={`${w}%`} height={3} radius={2} />
        ))}
      </div>
    </div>
  );
}

export function SkeletonTable({ rows = 4 }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} style={{ display: "flex", gap: 12 }}>
          <Skeleton width="40%" height={14} />
          <Skeleton width="25%" height={14} />
          <Skeleton width="25%" height={14} />
        </div>
      ))}
    </div>
  );
}
