import { useState, useEffect } from "react";
import { Cd2, Sec, InputField, Button, ErrorMessage, SkeletonChart } from "../common";
import {
  ResponsiveContainer, ComposedChart, CartesianGrid, XAxis, YAxis, Tooltip,
  Area, Line, ScatterChart, Scatter, LineChart, ReferenceLine, AreaChart,
} from "recharts";
import { apiBatch, apiExportSweep } from "../../api/client";
import { sweepOptions } from "../../constants/config";
import { colors, fonts } from "../../constants/theme";
import { Download } from "lucide-react";

export function SweepPage({ params }) {
  const [sweepKey, setSweepKey] = useState("alpha");
  const [from, setFrom] = useState(-5);
  const [to, setTo] = useState(15);
  const [steps, setSteps] = useState(40);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    const opt = sweepOptions.find((o) => o.key === sweepKey);
    if (opt) { setFrom(opt.from); setTo(opt.to); }
  }, [sweepKey]);

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      const inputs = [];
      for (let i = 0; i <= steps; i++) {
        const val = from + ((to - from) * i) / steps;
        inputs.push({ ...params, [sweepKey]: val });
      }
      const res = await apiBatch(inputs);
      setData(
        res.results.map((r, i) => ({
          [sweepKey]: +(from + ((to - from) * i) / steps).toFixed(4),
          Cl: r.predictions.Cl.value,
          Cd: r.predictions.Cd.value,
          Cm: r.predictions.Cm.value,
          K:  r.predictions.K.value,
          conf: r.confidence.score,
          Cl_lo: r.predictions.Cl.value - r.predictions.Cl.std,
          Cl_hi: r.predictions.Cl.value + r.predictions.Cl.std,
        }))
      );
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    if (!data) return;
    setExporting(true);
    try {
      const points = data.map((d) => ({
        alpha: d[sweepKey],
        Cl: d.Cl,
        Cd: d.Cd,
        Cm: d.Cm,
        K:  d.K,
        confidence: d.conf,
      }));
      const blob = await apiExportSweep(points);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `sweep_${sweepKey}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(`Export failed: ${e.message}`);
    } finally {
      setExporting(false);
    }
  };

  const xLabel = sweepOptions.find((o) => o.key === sweepKey)?.label || sweepKey;
  const margin = { top: 5, right: 15, bottom: 5, left: 5 };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <Cd2>
        <Sec>Sweep Configuration</Sec>
        <div style={{ display: "flex", gap: 12, alignItems: "end", flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 160 }}>
            <label style={{ fontSize: 10, color: colors.muted, fontWeight: 600, display: "block", marginBottom: 3, fontFamily: fonts.sans }}>
              Parameter
            </label>
            <select
              value={sweepKey}
              onChange={(e) => setSweepKey(e.target.value)}
              style={{ width: "100%", padding: "5px 8px", border: `1px solid ${colors.border}`, borderRadius: 3, fontFamily: fonts.sans, fontSize: 12, background: colors.card }}
            >
              {sweepOptions.map((opt) => (
                <option key={opt.key} value={opt.key}>{opt.label}</option>
              ))}
            </select>
          </div>
          <InputField label="From" value={from} onChange={setFrom} step={sweepKey === "reynolds" ? 1e5 : 0.5} />
          <InputField label="To"   value={to}   onChange={setTo}   step={sweepKey === "reynolds" ? 1e5 : 0.5} />
          <InputField label="Steps" value={steps} onChange={(v) => setSteps(Math.max(5, Math.min(100, Math.round(v))))} step={1} />
          <Button onClick={handleRun} disabled={loading} primary>{loading ? "..." : "Run Sweep"}</Button>
          {data && (
            <Button onClick={handleExport} disabled={exporting} style={{ display: "flex", gap: 5 }}>
              <Download size={14} />
              {exporting ? "Saving..." : "Download CSV"}
            </Button>
          )}
        </div>
        <ErrorMessage message={error} />
      </Cd2>

      {loading && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          {["Lift Curve", "Drag Polar", "Efficiency (L/D)", "Pitching Moment"].map((t) => (
            <Cd2 key={t}><Sec>{t}</Sec><SkeletonChart height={240} /></Cd2>
          ))}
        </div>
      )}

      {data && !loading && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          <Cd2>
            <Sec>Lift Curve + Uncertainty</Sec>
            <ResponsiveContainer width="100%" height={240}>
              <ComposedChart data={data} margin={margin}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey={sweepKey} tick={{ fontSize: 9 }} label={{ value: xLabel, position: "insideBottom", offset: -2, style: { fontSize: 9 } }} />
                <YAxis tick={{ fontSize: 9 }} label={{ value: "Cl", angle: -90, position: "insideLeft", style: { fontSize: 9 } }} />
                <Tooltip contentStyle={{ fontFamily: fonts.mono, fontSize: 10 }} />
                <Area dataKey="Cl_lo" stackId="1" stroke="none" fill={colors.blue} fillOpacity={0} />
                <Area dataKey="Cl_hi" stackId="1" stroke="none" fill={colors.blue} fillOpacity={0.08} />
                <Line type="monotone" dataKey="Cl" stroke={colors.blue} strokeWidth={2} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </Cd2>

          <Cd2>
            <Sec>Drag Polar</Sec>
            <ResponsiveContainer width="100%" height={240}>
              <ScatterChart margin={margin}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey="Cd" type="number" tick={{ fontSize: 9 }} label={{ value: "Cd", position: "insideBottom", offset: -2, style: { fontSize: 9 } }} />
                <YAxis dataKey="Cl" tick={{ fontSize: 9 }} label={{ value: "Cl", angle: -90, position: "insideLeft", style: { fontSize: 9 } }} />
                <Tooltip contentStyle={{ fontFamily: fonts.mono, fontSize: 10 }} />
                <Scatter data={data} fill={colors.accent} r={3} />
              </ScatterChart>
            </ResponsiveContainer>
          </Cd2>

          <Cd2>
            <Sec>Efficiency (L/D)</Sec>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={data} margin={margin}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey={sweepKey} tick={{ fontSize: 9 }} />
                <YAxis tick={{ fontSize: 9 }} label={{ value: "Cl/Cd", angle: -90, position: "insideLeft", style: { fontSize: 9 } }} />
                <Tooltip contentStyle={{ fontFamily: fonts.mono, fontSize: 10 }} />
                <Line type="monotone" dataKey="K" stroke={colors.green} strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Cd2>

          <Cd2>
            <Sec>Pitching Moment</Sec>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={data} margin={margin}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey={sweepKey} tick={{ fontSize: 9 }} />
                <YAxis tick={{ fontSize: 9 }} label={{ value: "Cm", angle: -90, position: "insideLeft", style: { fontSize: 9 } }} />
                <Tooltip contentStyle={{ fontFamily: fonts.mono, fontSize: 10 }} />
                <ReferenceLine y={0} stroke="#aaa" strokeDasharray="3 3" />
                <Line type="monotone" dataKey="Cm" stroke={colors.red} strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Cd2>

          <Cd2 style={{ gridColumn: "1/3" }}>
            <Sec>Confidence</Sec>
            <ResponsiveContainer width="100%" height={130}>
              <AreaChart data={data} margin={margin}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey={sweepKey} tick={{ fontSize: 9 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 9 }} />
                <Tooltip contentStyle={{ fontFamily: fonts.mono, fontSize: 10 }} />
                <ReferenceLine y={0.7} stroke={colors.success}  strokeDasharray="4 4" />
                <ReferenceLine y={0.4} stroke={colors.error}    strokeDasharray="4 4" />
                <Area type="monotone" dataKey="conf" stroke={colors.accent} fill={colors.accent} fillOpacity={0.1} strokeWidth={1.5} />
              </AreaChart>
            </ResponsiveContainer>
          </Cd2>
        </div>
      )}
    </div>
  );
}
