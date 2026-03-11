import { colors, fonts } from "../../constants/theme";
import { AlertTriangle } from "lucide-react";

export function ErrorMessage({ message }) {
  if (!message) return null;
  return (
    <div
      style={{
        background: colors.errorBg,
        border: `1px solid ${colors.errorBorder}`,
        borderRadius: 4,
        padding: "7px 12px",
        fontSize: 11,
        color: colors.error,
        fontFamily: fonts.sans,
        display: "flex",
        alignItems: "center",
        gap: 8,
      }}
    >
      <AlertTriangle size={13} />
      {message}
    </div>
  );
}