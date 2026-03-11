import { useState } from "react";
import { Play } from "lucide-react";
import { Cd2, Sec, InputField, Button, ErrorMessage } from "../common";
import { ResponsiveContainer, ScatterChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend, Scatter, BarChart, Bar, Cell } from "recharts";
import { apiBatch } from "../../api/client";
import { colors, fonts } from "../../constants/theme";

export function ExplorerPage({ params }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [count, setCount] = useState(100);

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      const inputs = [];
      for (let i = 0; i < count; i++) {
        inputs.push({
          ...params,
          alpha: -3 + Math.random() * 18,
          mach: 0.1 + Math.random() * 0.8,
          thickness_ratio: 0.06 + Math.random() * 0.18,
          camber: Math.random() * 0.08,
          aspect_ratio: 4 + Math.random() * 8,
          sweep_angle: Math.random() * 40,
        });
      }
      const res = await apiBatch(inputs);
      setData(
        res.results.map((r) => ({
          Cd: r.predictions.Cd.value,
          Cl: r.predictions.Cl.value,
          K: r.predictions.K.value,
          conf: r.confidence.score,
          level: r.confidence.level,
        }))
      );
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const levelColors = {
    HIGH: colors.success,
    MEDIUM: colors.warning,
    LOW: colors.error,
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <Cd2>
        <Sec>Design Space</Sec>
        <div style={{ display: "flex", gap: 10, alignItems: "end" }}>
          <InputField label="Samples" value={count} onChange={(v) => setCount(Math.max(10, Math.min(500, Math.round(v))))} step={10} />
          <Button onClick={handleRun} disabled={loading} primary>
            <Play size={13} /> {loading ? "..." : "Explore"}
          </Button>
        </div>
        <ErrorMessage message={error} />
      </Cd2>

      {data && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          <Cd2>
            <Sec>L/D vs Drag</Sec>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart margin={{ top: 10, right: 15, bottom: 5, left: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey="Cd" type="number" tick={{ fontSize: 9 }} />
                <YAxis dataKey="K" tick={{ fontSize: 9 }} />
                <Tooltip
                  contentStyle={{ fontFamily: fonts.mono, fontSize: 10 }}
                  formatter={(v) => (typeof v === "number" ? v.toFixed(4) : v)}
                />
                {["HIGH", "MEDIUM", "LOW"].map((l) => (
                  <Scatter
                    key={l}
                    name={l}
                    data={data.filter((d) => d.level === l)}
                    fill={levelColors[l]}
                    fillOpacity={0.55}
                    r={4}
                  />
                ))}
                <Legend verticalAlign="top" wrapperStyle={{ fontSize: 10 }} />
              </ScatterChart>
            </ResponsiveContainer>
          </Cd2>

          <Cd2>
            <Sec>Cl vs Cd — Pareto</Sec>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart margin={{ top: 10, right: 15, bottom: 5, left: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey="Cd" type="number" tick={{ fontSize: 9 }} />
                <YAxis dataKey="Cl" tick={{ fontSize: 9 }} />
                <Tooltip
                  contentStyle={{ fontFamily: fonts.mono, fontSize: 10 }}
                  formatter={(v) => (typeof v === "number" ? v.toFixed(4) : v)}
                />
                {["HIGH", "MEDIUM", "LOW"].map((l) => (
                  <Scatter
                    key={l}
                    name={l}
                    data={data.filter((d) => d.level === l)}
                    fill={levelColors[l]}
                    fillOpacity={0.55}
                    r={4}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </Cd2>

          <Cd2 style={{ gridColumn: "1/3" }}>
            <Sec>Confidence Distribution</Sec>
            <ResponsiveContainer width="100%" height={140}>
              <BarChart
                data={[
                  { level: "HIGH", count: data.filter((d) => d.level === "HIGH").length },
                  { level: "MEDIUM", count: data.filter((d) => d.level === "MEDIUM").length },
                  { level: "LOW", count: data.filter((d) => d.level === "LOW").length },
                ]}
                margin={{ top: 5, right: 15, bottom: 5, left: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey="level" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 9 }} />
                <Tooltip />
                <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                  {[colors.success, colors.warning, colors.error].map((c, i) => (
                    <Cell key={i} fill={c} fillOpacity={0.75} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Cd2>
        </div>
      )}
    </div>
  );
}