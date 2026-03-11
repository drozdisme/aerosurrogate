import { colors, fonts } from "../../constants/theme";

export function SectionTitle({ children }) {
  return (
    <div
      style={{
        fontSize: 10,
        fontWeight: 700,
        color: colors.muted,
        letterSpacing: 1.2,
        textTransform: "uppercase",
        borderBottom: `1px solid ${colors.borderLight}`,
        paddingBottom: 4,
        marginBottom: 10,
        fontFamily: fonts.sans,
      }}
    >
      {children}
    </div>
  );
}