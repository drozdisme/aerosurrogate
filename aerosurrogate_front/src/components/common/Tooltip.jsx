import { useState } from "react";
import { colors, fonts } from "../../constants/theme";

export function Tooltip({ text, children }) {
  const [visible, setVisible] = useState(false);

  return (
    <span
      style={{ position: "relative", display: "inline-flex", alignItems: "center" }}
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      {children}
      {visible && text && (
        <span
          style={{
            position: "absolute",
            bottom: "calc(100% + 6px)",
            left: "50%",
            transform: "translateX(-50%)",
            background: "#1a1a1a",
            color: "#fff",
            padding: "4px 8px",
            borderRadius: 4,
            fontSize: 11,
            fontFamily: fonts.sans,
            whiteSpace: "nowrap",
            zIndex: 9999,
            pointerEvents: "none",
            boxShadow: "0 2px 8px rgba(0,0,0,0.25)",
          }}
        >
          {text}
          <span
            style={{
              position: "absolute",
              top: "100%",
              left: "50%",
              transform: "translateX(-50%)",
              borderWidth: 4,
              borderStyle: "solid",
              borderColor: "#1a1a1a transparent transparent transparent",
            }}
          />
        </span>
      )}
    </span>
  );
}
