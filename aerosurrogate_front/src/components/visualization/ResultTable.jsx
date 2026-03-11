import { colors, fonts } from "../../constants/theme";

const thStyle = {
  fontSize: 10,
  fontWeight: 700,
  color: colors.muted,
  textTransform: "uppercase",
  letterSpacing: 0.5,
  padding: "7px 10px",
  textAlign: "left",
  borderBottom: `1px solid ${colors.border}`,
  background: "#f5f5f5",
  fontFamily: fonts.sans,
};

const tdStyle = {
  padding: "7px 10px",
  fontSize: 12,
};

export function ResultTable({ predictions }) {
  const labels = {
    Cl: "Lift coefficient",
    Cd: "Drag coefficient",
    Cm: "Pitching moment",
    K: "Aerodynamic efficiency",
  };

  return (
    <table style={{ width: "100%", borderCollapse: "collapse", border: `1px solid ${colors.border}`, borderRadius: 4, overflow: "hidden", background: colors.card, fontFamily: fonts.sans }}>
      <thead>
        <tr>
          <th style={thStyle}>Output</th>
          <th style={thStyle}>Value</th>
          <th style={{ ...thStyle, textAlign: "right" }}>Uncertainty</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(predictions).map(([key, val], idx, arr) => (
          <tr key={key}>
            <td style={{ ...tdStyle, color: colors.textSecondary, fontWeight: 500, borderBottom: idx < arr.length - 1 ? `1px solid ${colors.borderLight}` : "none" }}>
              {labels[key] || key} <span style={{ color: colors.muted }}>({key})</span>
            </td>
            <td style={{ ...tdStyle, fontFamily: fonts.mono, borderBottom: idx < arr.length - 1 ? `1px solid ${colors.borderLight}` : "none" }}>
              {val.value.toFixed(6)}
            </td>
            <td style={{ ...tdStyle, fontFamily: fonts.mono, color: colors.muted, textAlign: "right", borderBottom: idx < arr.length - 1 ? `1px solid ${colors.borderLight}` : "none" }}>
              ±{val.std.toFixed(6)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}