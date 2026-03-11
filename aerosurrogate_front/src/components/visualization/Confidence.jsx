import { colors, fonts } from "../../constants/theme";

const STYLES = {
  HIGH:   { bg: colors.successBg, border: colors.successBorder, text: colors.success },
  MEDIUM: { bg: colors.warningBg, border: colors.warningBorder, text: colors.warning },
  LOW:    { bg: colors.errorBg,   border: colors.errorBorder,   text: colors.error   },
};

export function Confidence({ confidence }) {
  if (!confidence) return null;
  // normalise regardless of what case the backend sends
  const level = String(confidence.level).toUpperCase();
  const s = STYLES[level] || STYLES.MEDIUM;

  const bar = Math.round(confidence.score * 100);

  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between",
        background: s.bg, border: `1px solid ${s.border}`, borderRadius: "4px 4px 0 0",
        padding: "6px 12px" }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: s.text, fontFamily: fonts.sans }}>
          Confidence: {level}
        </span>
        <span style={{ fontFamily: fonts.mono, fontSize: 12, fontWeight: 700, color: s.text }}>
          {confidence.score.toFixed(3)}
        </span>
      </div>
      <div style={{ height: 4, background: colors.borderLight, borderRadius: "0 0 4px 4px", overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${bar}%`,
          background: s.text, transition: "width 0.4s ease" }} />
      </div>
    </div>
  );
}
