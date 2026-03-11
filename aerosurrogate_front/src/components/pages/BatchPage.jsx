import { useState } from "react";
import { Cd2, Sec, Button, ErrorMessage, SkeletonTable } from "../common";
import { Upload, Play, Download, FileText } from "lucide-react";
import { apiBatch } from "../../api/client";
import { colors, fonts } from "../../constants/theme";
import { defaultParams } from "../../constants/config";

// ── Table styles ──────────────────────────────────────────────────────────────
const thStyle = {
  fontSize: 9, fontWeight: 700, color: colors.muted, textTransform: "uppercase",
  letterSpacing: 0.5, padding: "5px 8px", textAlign: "left",
  borderBottom: `1px solid ${colors.border}`, background: "#f5f5f5",
  fontFamily: fonts.sans, whiteSpace: "nowrap",
};

// ── Sample CSV content ────────────────────────────────────────────────────────
const SAMPLE_CSV = `mach,reynolds,alpha,thickness_ratio,camber,sweep_angle
0.30,1000000,0,0.12,0.04,20
0.30,1000000,4,0.12,0.04,20
0.30,1000000,8,0.12,0.04,20
0.50,2000000,0,0.15,0.02,25
0.50,2000000,5,0.15,0.02,25
0.70,3000000,0,0.10,0.00,30
0.70,3000000,3,0.10,0.00,30
0.20,500000,10,0.12,0.06,15`;

function downloadBlob(content, filename, type = "text/csv") {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

function LevelBadge({ level }) {
  const lvl = String(level).toUpperCase();
  const col = lvl === "HIGH" ? colors.success : lvl === "LOW" ? colors.error : colors.warning;
  const bg  = lvl === "HIGH" ? colors.successBg : lvl === "LOW" ? colors.errorBg : colors.warningBg;
  return (
    <span style={{ fontSize: 9, fontWeight: 700, color: col, background: bg,
      padding: "1px 5px", borderRadius: 3, fontFamily: fonts.sans }}>
      {lvl}
    </span>
  );
}

export function BatchPage() {
  const [csv, setCsv]       = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState(null);
  const [progress, setProgress] = useState(0);

  const handleFile = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const lines = ev.target.result.trim().split("\n");
      const headers = lines[0].split(",").map((h) => h.trim().replace(/\r/g,""));
      const rows = lines.slice(1)
        .filter(l => l.trim())
        .map((line) => {
          const values = line.split(",");
          const obj = {};
          headers.forEach((h, i) => { obj[h] = parseFloat(values[i]) || 0; });
          return obj;
        });
      setCsv({ headers, rows, name: file.name });
      setResults(null);
      setError(null);
    };
    reader.readAsText(file);
  };

  const handleRun = async () => {
    if (!csv) return;
    setLoading(true);
    setError(null);
    setProgress(0);
    try {
      // Process in chunks of 50 to show progress
      const CHUNK = 50;
      const all = [];
      const total = csv.rows.length;
      for (let i = 0; i < total; i += CHUNK) {
        const chunk = csv.rows.slice(i, i + CHUNK);
        const inputs = chunk.map((row) => ({ ...defaultParams, ...row }));
        const res = await apiBatch(inputs);
        all.push(...res.results);
        setProgress(Math.round(((i + chunk.length) / total) * 100));
      }
      setResults(
        all.map((r, i) => ({
          "#": i + 1,
          ...Object.fromEntries(
            Object.entries(r.predictions).map(([k, v]) => [k, v.value.toFixed(5)])
          ),
          "±Cl": r.predictions.Cl.std.toFixed(5),
          conf: r.confidence.score.toFixed(3),
          level: r.confidence.level,
        }))
      );
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setProgress(0);
    }
  };

  const handleDownload = () => {
    if (!results) return;
    const keys = Object.keys(results[0]).filter(k => k !== "level");
    const content = [keys.join(","),
      ...results.map((row) => keys.map((k) => row[k]).join(","))].join("\n");
    downloadBlob(content, "aerosurrogate_batch_results.csv");
  };

  const cols = results ? Object.keys(results[0]) : [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

      {/* Upload area */}
      <Cd2>
        <Sec>Batch Prediction</Sec>
        <p style={{ fontSize: 11, color: colors.textSecondary, margin: "0 0 10px",
          fontFamily: fonts.sans, lineHeight: 1.6 }}>
          Upload a CSV with aerodynamic parameters. Required columns:{" "}
          {["mach","reynolds","alpha"].map(c => (
            <code key={c} style={{ fontFamily: fonts.mono, fontSize: 10,
              background: "#f0f0ee", padding: "1px 4px", borderRadius: 3, marginRight: 4 }}>
              {c}
            </code>
          ))}
          — all other columns use defaults.
        </p>

        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
          <label style={{ flex: 1, minWidth: 200, display: "flex", alignItems: "center",
            justifyContent: "center", gap: 8, padding: "14px 0",
            border: `1.5px dashed ${csv ? colors.accent : colors.border}`,
            borderRadius: 5, cursor: "pointer",
            color: csv ? colors.accent : colors.muted, fontSize: 12, fontFamily: fonts.sans,
            background: csv ? "#f0f4ff" : "transparent", transition: "all .15s" }}>
            <Upload size={14} />
            {csv ? `${csv.name}  (${csv.rows.length} rows)` : "Select CSV file"}
            <input type="file" accept=".csv" onChange={handleFile} style={{ display: "none" }} />
          </label>

          <Button onClick={() => downloadBlob(SAMPLE_CSV, "aero_template.csv")}
            style={{ display: "flex", gap: 6 }}>
            <FileText size={13} /> Download template
          </Button>
        </div>
      </Cd2>

      {/* Preview + controls */}
      {csv && (
        <>
          <Cd2 style={{ maxHeight: 180, overflowY: "auto" }}>
            <Sec>Preview — first {Math.min(csv.rows.length, 5)} of {csv.rows.length} rows</Sec>
            <table style={{ width: "100%", borderCollapse: "collapse",
              fontSize: 10, fontFamily: fonts.mono }}>
              <thead>
                <tr>{csv.headers.map(h => <th key={h} style={thStyle}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {csv.rows.slice(0, 5).map((row, i) => (
                  <tr key={i}>{csv.headers.map(h => (
                    <td key={h} style={{ padding: "3px 8px",
                      borderBottom: `1px solid ${colors.borderLight}`, fontSize: 10 }}>
                      {typeof row[h] === "number" ? row[h].toFixed(4) : row[h]}
                    </td>
                  ))}</tr>
                ))}
              </tbody>
            </table>
          </Cd2>

          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <Button onClick={handleRun} disabled={loading} primary
              style={{ flex: 1, display: "flex", gap: 6 }}>
              <Play size={13} />
              {loading ? `Computing… ${progress}%` : `Run ${csv.rows.length} predictions`}
            </Button>
            {results && (
              <Button onClick={handleDownload} style={{ display: "flex", gap: 6 }}>
                <Download size={13} /> Download CSV
              </Button>
            )}
          </div>

          {/* Progress bar */}
          {loading && (
            <div style={{ height: 4, background: colors.borderLight, borderRadius: 2, overflow: "hidden" }}>
              <div style={{ height: "100%", width: `${progress}%`,
                background: colors.accent, transition: "width 0.2s ease" }} />
            </div>
          )}

          <ErrorMessage message={error} />
        </>
      )}

      {/* Skeleton while loading */}
      {loading && <Cd2><Sec>Results</Sec><SkeletonTable rows={6} /></Cd2>}

      {/* Results table */}
      {results && !loading && (
        <Cd2 style={{ maxHeight: 420, overflowY: "auto" }}>
          <Sec>Results — {results.length} predictions</Sec>
          <table style={{ width: "100%", borderCollapse: "collapse",
            fontSize: 10, fontFamily: fonts.mono }}>
            <thead>
              <tr>{cols.map(h => <th key={h} style={thStyle}>{h}</th>)}</tr>
            </thead>
            <tbody>
              {results.map((row, i) => {
                const lvl = String(row.level).toUpperCase();
                const bg = lvl==="HIGH" ? colors.successBg
                         : lvl==="LOW"  ? colors.errorBg : "transparent";
                return (
                  <tr key={i} style={{ background: bg }}>
                    {cols.map(k => (
                      <td key={k} style={{ padding: "3px 8px",
                        borderBottom: `1px solid ${colors.borderLight}` }}>
                        {k === "level" ? <LevelBadge level={row[k]} /> : row[k]}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Cd2>
      )}
    </div>
  );
}
