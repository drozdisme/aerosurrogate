import { colors, fonts } from "../../constants/theme";

export function CpLegend() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6, fontSize: 10, color: colors.muted, fontFamily: fonts.sans }}>
      <span>Cp:</span>
      <div style={{ display: "flex", height: 10, borderRadius: 2, overflow: "hidden", flex: 1, maxWidth: 200 }}>
        {["#0000ff", "#4040ff", "#8080ff", "#c8c8c8", "#ffc880", "#ff8000", "#ff0000"].map((c, i) => (
          <div key={i} style={{ flex: 1, background: c }} />
        ))}
      </div>
      <span style={{ fontFamily: fonts.mono }}>-3</span>
      <span style={{ fontFamily: fonts.mono }}>+1</span>
      <span style={{ marginLeft: 12 }}>
        <span style={{ color: colors.error, fontWeight: 600 }}>&#9679;</span> Stagnation
      </span>
    </div>
  );
}