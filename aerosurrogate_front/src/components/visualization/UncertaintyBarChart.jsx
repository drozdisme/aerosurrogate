import { ResponsiveContainer, BarChart, CartesianGrid, XAxis, YAxis, Tooltip, Bar } from "recharts";
import { colors, fonts } from "../../constants/theme";

export function UncertaintyBarChart({ predictions }) {
  const data = Object.entries(predictions).map(([key, val]) => ({
    name: key,
    std: val.std,
  }));

  return (
    <ResponsiveContainer width="100%" height={130}>
      <BarChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
        <XAxis dataKey="name" tick={{ fontSize: 10 }} />
        <YAxis tick={{ fontSize: 9 }} />
        <Tooltip contentStyle={{ fontFamily: fonts.mono, fontSize: 10 }} />
        <Bar dataKey="std" fill={colors.accent} fillOpacity={0.6} radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}