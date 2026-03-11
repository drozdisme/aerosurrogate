import { colors, fonts } from "../../constants/theme";
import { AlertTriangle } from "lucide-react";

export function StatusBar({ health }) {
  const apiUrl = window.location.hostname === "localhost" ? "http://localhost:8000" : "http://api:8000";
  return (
    <div
      style={{
        background: colors.warningBg,
        border: `1px solid ${colors.warningBorder}`,
        borderRadius: 4,
        padding: "8px 14px",
        marginBottom: 14,
        fontSize: 12,
        color: colors.warning,
        fontFamily: fonts.sans,
        display: "flex",
        alignItems: "center",
        gap: 8,
      }}
    >
      <AlertTriangle size={14} />
      API not reachable at {apiUrl}. Start: <code style={{ fontFamily: fonts.mono }}>docker-compose up</code>
    </div>
  );
}