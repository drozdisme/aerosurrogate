import { useState } from "react";
import { Cd2, Sec, InputField, Button, ErrorMessage, SkeletonChart, SkeletonTable } from "../common";
import { apiCompare } from "../../api/client";
import { colors, fonts } from "../../constants/theme";
import {
  ResponsiveContainer, LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend,
} from "recharts";

const defaultConfig = {
  thickness_ratio: 0.12, camber: 0.04, camber_position: 0.4,
  leading_edge_radius: 0.02, trailing_edge_angle: 15,
  aspect_ratio: 8, taper_ratio: 0.5, sweep_angle: 20,
  twist_angle: 0, dihedral_angle: 3,
  mach: 0.3, reynolds: 1e6, alpha: 5, beta: 0, altitude: 0,
};

function ConfigPanel({ label, config, onChange }) {
  const ip = (lbl, key, opts = {}) => (
    <InputField label={lbl} value={config[key]}
      onChange={(v) => onChange({ ...config, [key]: v })} {...opts} />
  );
  return (
    <Cd2>
      <Sec>{label}</Sec>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 6 }}>
        {ip("Thickness", "thickness_ratio", { step: 0.01 })}
        {ip("Camber", "camber", { step: 0.005 })}
        {ip("Camber pos.", "camber_position", { step: 0.05 })}
        {ip("Aspect ratio", "aspect_ratio", { step: 0.5 })}
        {ip("Sweep", "sweep_angle", { unit: "deg" })}
        {ip("Taper", "taper_ratio", { step: 0.05 })}
        {ip("Mach", "mach", { step: 0.01 })}
        {ip("Reynolds", "reynolds", { step: 1e5 })}
        {ip("Alpha", "alpha", { unit: "deg", step: 0.5 })}
      </div>
    </Cd2>
  );
}

function DeltaBadge({ val }) {
  const pos = val > 0;
  const zero = Math.abs(val) < 1e-5;
  const bg = zero ? colors.border : pos ? colors.successBg : colors.errorBg;
  const col = zero ? colors.muted : pos ? colors.success : colors.error;
  return (
    <span style={{ fontSize: 11, fontFamily: fonts.mono, fontWeight: 700,
      color: col, background: bg, padding: "1px 6px", borderRadius: 3 }}>
      {zero ? "—" : `${pos ? "+" : ""}${val.toFixed(4)}`}
    </span>
  );
}

function CompareTable({ result }) {
  const { A, B, delta } = result;
  const rows = [
    { label: "Cl  (lift)", a: A.Cl, b: B.Cl, d: delta.Cl },
    { label: "Cd  (drag)", a: A.Cd, b: B.Cd, d: delta.Cd },
    { label: "Cm  (moment)", a: A.Cm, b: B.Cm, d: delta.Cm },
    { label: "K = L/D",   a: A.K,  b: B.K,  d: delta.K  },
  ];
  const th = { fontSize: 10, color: colors.muted, fontWeight: 700, padding: "4px 8px",
    borderBottom: `1px solid ${colors.border}`, textAlign: "right", fontFamily: fonts.sans };
  const td = { fontSize: 12, fontFamily: fonts.mono, padding: "5px 8px",
    borderBottom: `1px solid ${colors.borderLight}`, textAlign: "right" };
  return (
    <table style={{ width: "100%", borderCollapse: "collapse" }}>
      <thead>
        <tr>
          <th style={{ ...th, textAlign: "left" }}>Parameter</th>
          <th style={{ ...th, color: colors.blue }}>Config A</th>
          <th style={{ ...th, color: "#b45309" }}>Config B</th>
          <th style={th}>Δ (B − A)</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(({ label, a, b, d }) => (
          <tr key={label}>
            <td style={{ ...td, textAlign: "left", color: colors.textSecondary, fontFamily: fonts.sans, fontSize: 11 }}>{label}</td>
            <td style={{ ...td, color: colors.blue }}>{a.toFixed(4)}</td>
            <td style={{ ...td, color: "#b45309" }}>{b.toFixed(4)}</td>
            <td style={{ ...td }}><DeltaBadge val={d} /></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function CpOverlay({ cpA, cpB }) {
  if (!cpA || !cpB) return null;
  const data = cpA.x.map((x, i) => ({
    x: parseFloat(x.toFixed(3)),
    A: parseFloat((cpA.Cp[i] ?? 0).toFixed(4)),
    B: parseFloat((cpB.Cp[i] ?? 0).toFixed(4)),
  }));
  return (
    <Cd2>
      <Sec>Cp(x) Overlay</Sec>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 5, right: 15, bottom: 5, left: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
          <XAxis dataKey="x" tick={{ fontSize: 9 }} label={{ value: "x/c", position: "insideBottom", offset: -2, style: { fontSize: 9 } }} />
          <YAxis reversed tick={{ fontSize: 9 }} label={{ value: "Cp", angle: -90, position: "insideLeft", style: { fontSize: 9 } }} />
          <Tooltip contentStyle={{ fontFamily: fonts.mono, fontSize: 10 }} />
          <Legend iconSize={10} wrapperStyle={{ fontSize: 10 }} />
          <Line type="monotone" dataKey="A" stroke={colors.blue}    strokeWidth={2} dot={false} name="Config A" />
          <Line type="monotone" dataKey="B" stroke="#b45309"         strokeWidth={2} dot={false} name="Config B" />
        </LineChart>
      </ResponsiveContainer>
    </Cd2>
  );
}

export function ComparePage({ params }) {
  const [configA, setConfigA] = useState({ ...defaultConfig, ...params });
  const [configB, setConfigB] = useState({ ...defaultConfig, sweep_angle: 30, camber: 0.06 });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleCompare = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiCompare(configA, configB);
      setResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <ConfigPanel label="Config A" config={configA} onChange={setConfigA} />
        <ConfigPanel label="Config B" config={configB} onChange={setConfigB} />
      </div>

      <div style={{ display: "flex", gap: 10 }}>
        <Button onClick={handleCompare} disabled={loading} primary>
          {loading ? "Comparing..." : "Compare Airfoils"}
        </Button>
      </div>
      <ErrorMessage message={error} />

      {loading && (
        <Cd2><Sec>Results</Sec><SkeletonTable rows={4} /></Cd2>
      )}

      {result && !loading && (
        <>
          <Cd2>
            <Sec>Comparison Results</Sec>
            {result.demo_mode && (
              <div style={{ fontSize: 10, color: colors.warning, marginBottom: 8, padding: "3px 7px",
                background: colors.warningBg, borderRadius: 3, border: `1px solid ${colors.warningBorder}` }}>
                Demo mode — analytical model
              </div>
            )}
            <CompareTable result={result} />
          </Cd2>
          <CpOverlay cpA={result.cpA} cpB={result.cpB} />
        </>
      )}
    </div>
  );
}
