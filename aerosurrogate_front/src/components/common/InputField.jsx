import { Info } from "lucide-react";
import { Tooltip } from "./Tooltip";
import { colors, fonts } from "../../constants/theme";

export function InputField({ label, value, onChange, step = 0.01, unit, tooltip }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <label
        style={{
          fontSize: 10,
          color: colors.muted,
          fontWeight: 600,
          fontFamily: fonts.sans,
          letterSpacing: 0.3,
          display: "flex",
          alignItems: "center",
          gap: 3,
        }}
      >
        {label}
        {unit && <span style={{ fontWeight: 400 }}> ({unit})</span>}
        {tooltip && (
          <Tooltip text={tooltip}>
            <Info size={10} strokeWidth={1.8} style={{ cursor: "help", color: colors.muted }} />
          </Tooltip>
        )}
      </label>
      <input
        type="number"
        value={value}
        step={step}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        style={{
          fontFamily: fonts.mono,
          fontSize: 12,
          padding: "4px 7px",
          border: `1px solid ${colors.border}`,
          borderRadius: 3,
          outline: "none",
          background: colors.card,
          color: colors.text,
          width: "100%",
          boxSizing: "border-box",
        }}
        onFocus={(e) => (e.target.style.borderColor = colors.accent)}
        onBlur={(e) => (e.target.style.borderColor = colors.border)}
      />
    </div>
  );
}
