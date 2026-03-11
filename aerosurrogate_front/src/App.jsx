import { useState, useEffect } from "react";
import { Sidebar, StatusBar } from "./components/layout";
import { AnalysisPage, SweepPage, BatchPage, ExplorerPage, ComparePage } from "./components/pages";
import { apiHealth } from "./api/client";
import { defaultParams } from "./constants/config";
import { colors, fonts } from "./constants/theme";

export default function App() {
  const [page, setPage] = useState("analysis");
  const [health, setHealth] = useState(null);
  const [params, setParams] = useState(defaultParams);

  useEffect(() => {
    const check = () => apiHealth().then(setHealth);
    check();
    const id = setInterval(check, 60000);
    return () => clearInterval(id);
  }, []);

  return (
    <div style={{ display: "flex", minHeight: "100vh", background: colors.bg, fontFamily: fonts.sans, color: colors.text }}>
      <Sidebar currentPage={page} onPageChange={setPage} health={health} />
      <main style={{ flex: 1, padding: 20, overflow: "auto" }}>
        {health && !health.model_loaded && !health.demo_mode && <StatusBar health={health} />}
        {page === "analysis" && <AnalysisPage params={params} setParams={setParams} />}
        {page === "sweep"    && <SweepPage    params={params} />}
        {page === "compare"  && <ComparePage  params={params} />}
        {page === "batch"    && <BatchPage />}
        {page === "explorer" && <ExplorerPage params={params} />}
      </main>
    </div>
  );
}
