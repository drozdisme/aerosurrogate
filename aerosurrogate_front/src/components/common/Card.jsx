import { cardStyle } from "../../constants/theme";

export function Card({ children, style = {} }) {
  return <div style={{ ...cardStyle, ...style }}>{children}</div>;
}