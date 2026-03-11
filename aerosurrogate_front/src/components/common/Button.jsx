import { colors, fonts } from "../../constants/theme";

export function Button({ children, onClick, disabled, primary, style = {} }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "8px 20px",
        background: primary ? colors.accent : colors.card,
        color: primary ? "#fff" : colors.accent,
        border: `1px solid ${primary ? colors.accent : colors.border}`,
        borderRadius: 4,
        fontSize: 13,
        fontWeight: 600,
        fontFamily: fonts.sans,
        cursor: disabled ? "default" : "pointer",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 7,
        opacity: disabled ? 0.55 : 1,
        transition: "all .12s",
        ...style,
      }}
    >
      {children}
    </button>
  );
}