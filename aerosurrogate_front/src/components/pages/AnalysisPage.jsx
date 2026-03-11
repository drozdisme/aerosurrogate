import { useState, useMemo } from "react";
import { Cd2, Sec, InputField, Button, ErrorMessage, SkeletonChart, SkeletonTable } from "../common";
import { Viewer3D, CpLegend, CpChart, UncertaintyBarChart, ResultTable, Confidence } from "../visualization";
import { apiPredict, apiField } from "../../api/client";
import { estimateCp } from "../../utils/foilMath";
import { colors } from "../../constants/theme";
import { Info } from "lucide-react";

const TOOLTIPS = {
  thickness_ratio: "Максимальная относительная толщина профиля t/c",
  camber: "Максимальная кривизна средней линии профиля",
  camber_position: "Положение максимальной кривизны (доля хорды)",
  leading_edge_radius: "Радиус передней кромки (доля хорды)",
  trailing_edge_angle: "Угол заостряемой задней кромки (градусы)",
  aspect_ratio: "Удлинение крыла λ = b²/S",
  taper_ratio: "Сужение крыла η = c_tip / c_root",
  sweep_angle: "Угол стреловидности по линии 25% хорд",
  dihedral_angle: "Поперечный угол V-образности крыла",
};

export function AnalysisPage({ params, setParams }) {
  const [result, setResult] = useState(null);
  const [cpField, setCpField] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const cpAnalytical = useMemo(
    () => estimateCp(params.alpha, params.mach, params.thickness_ratio, params.camber, params.camber_position),
    [params.alpha, params.mach, params.thickness_ratio, params.camber, params.camber_position]
  );

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiPredict(params);
      setResult(res);
      try {
        const field = await apiField(params);
        setCpField(field);
      } catch {
        // field optional
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const ip = (label, key, opts = {}) => (
    <InputField
      label={label}
      value={params[key]}
      onChange={(v) => setParams({ ...params, [key]: v })}
      tooltip={TOOLTIPS[key]}
      {...opts}
    />
  );

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18, alignItems: "start" }}>
      {/* Left — geometry + flow */}
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <Cd2>
          <Sec>Geometry + Pressure Field</Sec>
          <Viewer3D {...params} />
          <CpLegend />
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 6, marginTop: 10 }}>
            {ip("Thickness", "thickness_ratio", { step: 0.01 })}
            {ip("Camber", "camber", { step: 0.005 })}
            {ip("Camber pos.", "camber_position", { step: 0.05 })}
            {ip("LE radius", "leading_edge_radius", { step: 0.005 })}
            {ip("TE angle", "trailing_edge_angle", { unit: "deg" })}
            {ip("Aspect ratio", "aspect_ratio", { step: 0.5 })}
            {ip("Taper", "taper_ratio", { step: 0.05 })}
            {ip("Sweep", "sweep_angle", { unit: "deg" })}
            {ip("Dihedral", "dihedral_angle", { unit: "deg" })}
          </div>
        </Cd2>
        <Cd2>
          <Sec>Flow Conditions</Sec>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 6 }}>
            {ip("Mach", "mach", { step: 0.01 })}
            {ip("Reynolds", "reynolds", { step: 1e5 })}
            {ip("Alpha", "alpha", { unit: "deg", step: 0.5 })}
            {ip("Beta", "beta", { unit: "deg", step: 0.5 })}
            {ip("Altitude", "altitude", { unit: "m", step: 500 })}
          </div>
        </Cd2>
        <Button onClick={handleRun} disabled={loading} primary style={{ width: "100%" }}>
          {loading ? "Computing..." : "Run Analysis"}
        </Button>
        <ErrorMessage message={error} />
      </div>

      {/* Right — results */}
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {loading ? (
          <>
            <Cd2>
              <Sec>Integral Coefficients</Sec>
              <SkeletonTable rows={4} />
            </Cd2>
            <Cd2>
              <Sec>Surface Pressure Distribution</Sec>
              <SkeletonChart height={200} />
            </Cd2>
            <Cd2>
              <Sec>Uncertainty per Output</Sec>
              <SkeletonChart height={120} />
            </Cd2>
          </>
        ) : result ? (
          <>
            <Cd2>
              <Sec>Integral Coefficients</Sec>
              {result.demo_mode && (
                <div style={{ fontSize: 10, color: colors.warning, marginBottom: 6, padding: "3px 7px", background: colors.warningBg, borderRadius: 3, border: `1px solid ${colors.warningBorder}` }}>
                  Demo mode — analytical approximation
                </div>
              )}
              <Confidence confidence={result.confidence} />
              <ResultTable predictions={result.predictions} />
            </Cd2>
            <Cd2>
              <Sec>Surface Pressure Distribution</Sec>
              <CpChart data={cpField || cpAnalytical} isDeepONet={!!cpField} />
            </Cd2>
            <Cd2>
              <Sec>Uncertainty per Output</Sec>
              <UncertaintyBarChart predictions={result.predictions} />
            </Cd2>
          </>
        ) : (
          <Cd2 style={{ minHeight: 500, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", color: colors.muted }}>
            <Info size={28} strokeWidth={1.5} />
            <p style={{ marginTop: 10, fontSize: 13 }}>Configure parameters and run analysis</p>
            <p style={{ fontSize: 11, maxWidth: 260, textAlign: "center", lineHeight: 1.5, marginTop: 4 }}>
              3D model updates in real-time. Hover parameter labels for descriptions.
            </p>
          </Cd2>
        )}
      </div>
    </div>
  );
}
