import { Layers, BarChart3, FileText, Sliders, CheckCircle, XCircle, ArrowLeftRight } from "lucide-react";
import { colors, fonts } from "../../constants/theme";

const NAV_ITEMS = [
  { id: "analysis", label: "Analysis",          icon: Layers },
  { id: "sweep",    label: "Parameter Sweep",   icon: Sliders },
  { id: "compare",  label: "Compare Airfoils",  icon: ArrowLeftRight },
  { id: "batch",    label: "Batch Analysis",    icon: FileText },
  { id: "explorer", label: "Design Explorer",   icon: BarChart3 },
];

export function Sidebar({ currentPage, onPageChange, health }) {
  const isConnected = health?.status === "ok";
  const demoMode     = health?.demo_mode;
  const deeponet     = health?.deeponet_available;
  const metrics      = health?.metrics;

  return (
    <nav style={{ width: 200, background: colors.card, borderRight: `1px solid ${colors.border}`,
        display: "flex", flexDirection: "column", flexShrink: 0 }}>
      <div style={{ padding: "14px 14px 10px", borderBottom: `1px solid ${colors.border}` }}>
        <div style={{ fontFamily: fonts.serif, fontSize: 16, fontWeight: 700 }}>AeroSurrogate</div>
        <div style={{ fontSize: 10, color: colors.muted, marginTop: 1 }}>Engineering Dashboard v2.0</div>
      </div>

      <div style={{ padding: "6px 0", flex: 1 }}>
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
          <button key={id} onClick={() => onPageChange(id)}
            style={{
              display: "flex", alignItems: "center", gap: 9,
              width: "100%", padding: "8px 14px",
              background: currentPage === id ? "#f0f0ee" : "transparent",
              border: "none",
              borderLeft: currentPage === id ? `3px solid ${colors.accent}` : "3px solid transparent",
              color: currentPage === id ? colors.accent : colors.textSecondary,
              fontSize: 12, fontWeight: currentPage === id ? 600 : 400,
              fontFamily: fonts.sans, cursor: "pointer", textAlign: "left", transition: "all .1s",
            }}>
            <Icon size={14} />
            {label}
          </button>
        ))}
      </div>

      <div style={{ padding: 10, borderTop: `1px solid ${colors.border}`, fontSize: 10, color: colors.muted, lineHeight: 1.8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          {isConnected ? (
            <><CheckCircle size={11} color={colors.success} /><span style={{ color: colors.success }}>API Connected</span></>
          ) : (
            <><XCircle size={11} color={colors.error} /><span style={{ color: colors.error }}>API Offline</span></>
          )}
        </div>
        {demoMode && (
          <div style={{ color: colors.warning, fontSize: 9, fontWeight: 600 }}>⚡ Demo mode</div>
        )}
        {metrics && (
          <div style={{ marginTop: 4, lineHeight: 1.6 }}>
            <div>Cl R² {metrics.Cl_R2}</div>
            <div>Cd R² {metrics.Cd_R2}</div>
            <div>Cm R² {metrics.Cm_R2}</div>
          </div>
        )}
        <div>DeepONet: {deeponet ? "available" : "not loaded"}</div>
        <div style={{ marginTop: 4, fontFamily: fonts.mono, fontSize: 9, color: "#bbb" }}>
          {typeof window !== "undefined" && window.location.hostname === "localhost"
            ? "http://localhost:8000" : "http://api:8000"}
        </div>
      </div>
    </nav>
  );
}
