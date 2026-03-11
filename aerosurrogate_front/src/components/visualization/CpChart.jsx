import { ResponsiveContainer, LineChart, CartesianGrid, XAxis, YAxis,
         Tooltip, Line, Legend, ReferenceLine } from "recharts";
import { colors, fonts } from "../../constants/theme";

/**
 * CpChart handles two data formats:
 *  1. ML field:   {x:[...], Cp:[...]}        → isDeepONet=true
 *  2. Analytical: [{x, cpU, cpL}, ...]       → isDeepONet=false
 */
function normalizeField(data) {
  if (!data) return [];
  // Format 1: object with x and Cp arrays
  if (Array.isArray(data.x) && Array.isArray(data.Cp)) {
    return data.x.map((xv, i) => ({
      x:  parseFloat(xv.toFixed(4)),
      Cp: parseFloat((data.Cp[i] ?? 0).toFixed(4)),
    }));
  }
  // Format 2: already array of objects
  if (Array.isArray(data)) return data;
  return [];
}

export function CpChart({ data, isDeepONet }) {
  const chartData = isDeepONet ? normalizeField(data) : (Array.isArray(data) ? data : []);

  if (!chartData.length) return (
    <div style={{ height: 220, display: "flex", alignItems: "center",
      justifyContent: "center", color: colors.muted, fontSize: 12 }}>
      No Cp data
    </div>
  );

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={chartData} margin={{ top: 5, right: 15, bottom: 18, left: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
        <XAxis dataKey="x" tick={{ fontSize: 9 }}
          label={{ value: "x/c", position: "insideBottom", offset: -8, style: { fontSize: 10 } }} />
        <YAxis reversed tick={{ fontSize: 9 }}
          label={{ value: "−Cp", angle: -90, position: "insideLeft", style: { fontSize: 9 } }} />
        <ReferenceLine y={0} stroke="#aaa" strokeDasharray="3 3" />
        <Tooltip
          contentStyle={{ fontFamily: fonts.mono, fontSize: 10 }}
          formatter={(v) => [v.toFixed(4), isDeepONet ? "Cp (ML)" : undefined]}
        />
        {isDeepONet ? (
          <Line type="monotone" dataKey="Cp" stroke={colors.accent}
            strokeWidth={2} dot={false} name="ML surrogate" />
        ) : (
          <>
            <Line type="monotone" dataKey="cpU" stroke={colors.blue}
              strokeWidth={1.5} dot={false} name="Upper" />
            <Line type="monotone" dataKey="cpL" stroke={colors.red}
              strokeWidth={1.5} dot={false} name="Lower" />
          </>
        )}
        <Legend verticalAlign="top" wrapperStyle={{ fontSize: 10 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}
