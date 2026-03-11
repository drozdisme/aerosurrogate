"""AeroSurrogate v2.0 — Engineering interface."""

import io
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(page_title="AeroSurrogate v2.0", page_icon=None, layout="wide", initial_sidebar_state="collapsed")

# ── CSS (same academic style as v1 UI) ─────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;500&display=swap');
.stApp { background-color: #fafafa !important; }
header[data-testid="stHeader"] { background-color: #fafafa !important; }
.block-container { padding-top: 1.5rem !important; max-width: 960px !important; }
h1,h2,h3,h4,h5,h6 { font-family: 'Source Serif 4', Georgia, serif !important; color: #1a1a1a !important; }
.stTabs [data-baseweb="tab-list"] { gap:0 !important; border-bottom: 2px solid #e5e5e5 !important; background:transparent !important; }
.stTabs [data-baseweb="tab"] { background:transparent !important; color:#a3a3a3 !important; border:none !important; border-bottom:2px solid transparent !important; margin-bottom:-2px !important; padding:8px 20px !important; font-family:'Source Sans 3',sans-serif !important; font-size:14px !important; font-weight:600 !important; }
.stTabs [aria-selected="true"] { color:#1a1a1a !important; border-bottom:2px solid #1a1a1a !important; }
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"] { display:none !important; }
.stButton>button { background:#fff !important; color:#1a1a1a !important; border:1px solid #d4d4d4 !important; border-radius:4px !important; font-family:'Source Sans 3',sans-serif !important; font-weight:600 !important; font-size:14px !important; }
.stButton>button:hover { background:#f5f5f5 !important; border-color:#a3a3a3 !important; }
.stNumberInput input { font-family:'IBM Plex Mono',monospace !important; font-size:13px !important; }
.stDownloadButton>button { background:#fff !important; color:#1a1a1a !important; border:1px solid #d4d4d4 !important; border-radius:4px !important; font-family:'Source Sans 3',sans-serif !important; font-size:13px !important; }
.section-heading { font-family:'Source Sans 3',sans-serif; font-size:12px; font-weight:600; color:#737373; text-transform:uppercase; letter-spacing:1px; margin:0 0 12px 0; padding:0 0 6px 0; border-bottom:1px solid #e5e5e5; }
.conf-block { border-radius:4px; padding:10px 14px; margin:16px 0; display:flex; align-items:center; justify-content:space-between; }
.results-table { width:100%; border-collapse:collapse; font-family:'Source Sans 3',sans-serif; border:1px solid #d4d4d4; border-radius:4px; overflow:hidden; background:#fff; }
.results-table thead th { background:#f5f5f5; color:#737373; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; padding:8px 14px; text-align:left; border-bottom:1px solid #d4d4d4; }
.results-table tbody td { padding:10px 14px; font-size:14px; color:#1a1a1a; border-bottom:1px solid #e5e5e5; }
.results-table tbody tr:last-child td { border-bottom:none; }
.mono { font-family:'IBM Plex Mono',monospace; font-size:13px; }
.muted { color:#737373; }
.disclaimer { font-family:'Source Sans 3',sans-serif; font-size:12px; color:#a3a3a3; margin-top:16px; line-height:1.6; font-style:italic; }
.subtle-divider { border:none; border-top:1px solid #e5e5e5; margin:16px 0; }
</style>""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    model_ok = health.get("model_loaded", False)
    has_field = health.get("deeponet_available", False)
except Exception:
    model_ok = False
    has_field = False
    health = None

status_cls = "aero-status-ok" if model_ok else ("aero-status-warn" if health else "aero-status-err")
status_text = "Connected" if model_ok else ("Model not loaded" if health else "API unavailable")
field_note = " + Field model" if has_field else ""

col_h, col_s = st.columns([4, 1])
with col_h:
    st.markdown(f"""<div style="border-bottom:1px solid #e5e5e5; padding-bottom:10px; margin-bottom:20px;">
        <span style="font-family:'Source Serif 4',Georgia,serif; font-size:22px; font-weight:600;">AeroSurrogate</span>
        <span style="font-size:12px; color:#a3a3a3;">&ensp;v2.0{field_note}</span></div>""", unsafe_allow_html=True)
with col_s:
    bg = {"aero-status-ok":"#f0fdf4","aero-status-warn":"#fefce8","aero-status-err":"#fef2f2"}
    fg = {"aero-status-ok":"#166534","aero-status-warn":"#854d0e","aero-status-err":"#991b1b"}
    bd = {"aero-status-ok":"#bbf7d0","aero-status-warn":"#fef08a","aero-status-err":"#fecaca"}
    st.markdown(f'<span style="font-family:IBM Plex Mono,monospace; font-size:11px; padding:3px 8px; border-radius:3px; background:{bg[status_cls]}; color:{fg[status_cls]}; border:1px solid {bd[status_cls]};">{status_text}</span>', unsafe_allow_html=True)

# ── Helpers ─────────────────────────────────────────────────────────
TARGET_META = {"Cl":"Lift coefficient","Cd":"Drag coefficient","Cm":"Pitching moment coefficient","K":"Aerodynamic efficiency (L/D)"}

def render_confidence(conf):
    s = {"HIGH":("#f0fdf4","#86efac","#166534"),"MEDIUM":("#fefce8","#fde047","#854d0e"),"LOW":("#fef2f2","#fca5a5","#991b1b")}
    bg,bd,fg = s.get(conf["level"], s["MEDIUM"])
    st.markdown(f'<div class="conf-block" style="background:{bg};border:1px solid {bd};"><span style="font-size:13px;font-weight:600;color:{fg};">Confidence: {conf["level"]}</span><span class="mono" style="color:{fg};">{conf["score"]:.3f}</span></div>', unsafe_allow_html=True)

def render_results(preds):
    rows=""
    for k,v in preds.items():
        rows+=f'<tr><td>{TARGET_META.get(k,k)} <span class="muted">({k})</span></td><td class="mono">{v["value"]:.6f}</td><td class="mono muted">&plusmn;&thinsp;{v["std"]:.6f}</td></tr>'
    st.markdown(f'<table class="results-table"><thead><tr><th style="width:45%">Output</th><th style="width:30%">Value</th><th style="width:25%">Uncertainty (1&sigma;)</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)

def build_payload():
    return {k: st.session_state.get(k, v) for k, v in {
        "thickness_ratio":0.12,"camber":0.04,"camber_position":0.4,"leading_edge_radius":0.02,"trailing_edge_angle":15.0,
        "aspect_ratio":8.0,"taper_ratio":0.5,"sweep_angle":20.0,"twist_angle":0.0,"dihedral_angle":3.0,
        "mach":0.5,"reynolds":1e6,"alpha":5.0,"beta":0.0,"altitude":0.0}.items()}

# ── Tabs ────────────────────────────────────────────────────────────
tabs = ["Single prediction", "Batch analysis"]
if has_field:
    tabs.append("Field prediction (Cp)")
tab_list = st.tabs(tabs)

# ==================== SINGLE PREDICTION ====================
with tab_list[0]:
    st.markdown('<div class="section-heading">Geometry</div>', unsafe_allow_html=True)
    gc = st.columns(5)
    gc[0].number_input("Thickness ratio",value=0.12,step=0.01,format="%.2f",key="thickness_ratio")
    gc[1].number_input("Camber",value=0.04,step=0.005,format="%.3f",key="camber")
    gc[2].number_input("Camber position",value=0.40,step=0.05,format="%.2f",key="camber_position")
    gc[3].number_input("LE radius",value=0.02,step=0.005,format="%.3f",key="leading_edge_radius")
    gc[4].number_input("TE angle, deg",value=15.0,step=1.0,format="%.1f",key="trailing_edge_angle")
    gc2 = st.columns(5)
    gc2[0].number_input("Aspect ratio",value=8.0,step=0.5,format="%.1f",key="aspect_ratio")
    gc2[1].number_input("Taper ratio",value=0.50,step=0.05,format="%.2f",key="taper_ratio")
    gc2[2].number_input("Sweep, deg",value=20.0,step=1.0,format="%.1f",key="sweep_angle")
    gc2[3].number_input("Twist, deg",value=0.0,step=0.5,format="%.1f",key="twist_angle")
    gc2[4].number_input("Dihedral, deg",value=3.0,step=0.5,format="%.1f",key="dihedral_angle")

    st.markdown('<div class="section-heading" style="margin-top:16px;">Flow conditions</div>', unsafe_allow_html=True)
    fc = st.columns(5)
    fc[0].number_input("Mach",value=0.50,step=0.01,format="%.2f",key="mach")
    fc[1].number_input("Reynolds",value=1.0e6,step=1e5,format="%.0f",key="reynolds")
    fc[2].number_input("Alpha, deg",value=5.0,step=0.5,format="%.1f",key="alpha")
    fc[3].number_input("Beta, deg",value=0.0,step=0.5,format="%.1f",key="beta")
    fc[4].number_input("Altitude, m",value=0.0,step=500.0,format="%.0f",key="altitude")

    _,bc,_ = st.columns([3,1,3])
    if bc.button("Compute", type="primary", use_container_width=True):
        try:
            resp = requests.post(f"{API_URL}/predict", json=build_payload(), timeout=30)
            resp.raise_for_status()
            data = resp.json()
            st.markdown('<div class="section-heading" style="margin-top:20px;">Results</div>', unsafe_allow_html=True)
            render_confidence(data["confidence"])
            render_results(data["predictions"])
            st.markdown('<p class="disclaimer">Surrogate model predictions. Validate final designs with CFD or experimental data.</p>', unsafe_allow_html=True)
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach API server.")
        except Exception as e:
            st.error(f"Error: {e}")

# ==================== BATCH ====================
with tab_list[1]:
    st.markdown('<div class="section-heading">CSV Upload</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:13px;color:#525252;margin-bottom:14px;">Required: <code>mach</code>, <code>reynolds</code>, <code>alpha</code>. Geometry columns optional.</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.caption(f"{len(df)} rows, {len(df.columns)} columns")
            st.dataframe(df.head(10), use_container_width=True)
            _,bc2,_ = st.columns([3,1,3])
            if bc2.button("Compute", type="primary", use_container_width=True, key="batch_btn"):
                required = {"mach","reynolds","alpha"}
                missing = required - set(df.columns)
                if missing:
                    st.error(f"Missing: {', '.join(missing)}")
                else:
                    defaults={"thickness_ratio":0.12,"camber":0.04,"camber_position":0.4,"leading_edge_radius":0.02,"trailing_edge_angle":15.0,"aspect_ratio":8.0,"taper_ratio":0.5,"sweep_angle":20.0,"twist_angle":0.0,"dihedral_angle":3.0,"beta":0.0,"altitude":0.0}
                    for c,v in defaults.items():
                        if c not in df.columns: df[c]=v
                    with st.spinner(""):
                        resp=requests.post(f"{API_URL}/predict/batch",json={"inputs":df.to_dict(orient="records")},timeout=300)
                        resp.raise_for_status()
                        bd2=resp.json()
                    rows=[]
                    for i,r in enumerate(bd2["results"]):
                        row={"#":i+1}
                        for t,v in r["predictions"].items(): row[t]=round(v["value"],6); row[f"{t}_std"]=round(v["std"],6)
                        row["Confidence"]=round(r["confidence"]["score"],3); row["Level"]=r["confidence"]["level"]
                        rows.append(row)
                    rdf=pd.DataFrame(rows)
                    st.markdown(f'<div class="section-heading" style="margin-top:16px;">Results — {len(rdf)} points</div>', unsafe_allow_html=True)
                    def sl(val):
                        m={"HIGH":"background-color:#f0fdf4;color:#166534","MEDIUM":"background-color:#fefce8;color:#854d0e","LOW":"background-color:#fef2f2;color:#991b1b"}
                        return m.get(val,"")
                    st.dataframe(rdf.style.map(sl,subset=["Level"]),use_container_width=True,height=400)
                    buf=io.StringIO(); rdf.to_csv(buf,index=False)
                    st.download_button("Download CSV",buf.getvalue(),"aerosurrogate_results.csv","text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

# ==================== FIELD PREDICTION ====================
if has_field and len(tab_list) > 2:
    with tab_list[2]:
        st.markdown('<div class="section-heading">Surface Pressure Distribution</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:13px;color:#525252;margin-bottom:14px;">DeepONet predicts the Cp(x) distribution over the surface. Set parameters in the Single Prediction tab, then compute here.</p>', unsafe_allow_html=True)
        _,fc2,_ = st.columns([3,1,3])
        if fc2.button("Compute Cp field", type="primary", use_container_width=True, key="field_btn"):
            try:
                resp = requests.post(f"{API_URL}/predict/field", json=build_payload(), timeout=60)
                resp.raise_for_status()
                fdata = resp.json()
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fdata["x"], y=fdata["Cp"], mode="lines", line=dict(color="#2563eb", width=2)))
                fig.update_layout(xaxis_title="x/c", yaxis_title="Cp", yaxis_autorange="reversed",
                                  template="simple_white", height=400, margin=dict(l=60,r=20,t=30,b=50),
                                  font=dict(family="Source Sans 3, sans-serif"))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Field prediction error: {e}")
