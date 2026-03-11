/**
 * Viewer3D.jsx — Standalone aerodynamic wing visualizer
 * Light theme · static camera (drag to rotate) · streamlines on both surfaces
 *
 * Prop aliases (all accepted):
 *   thickness | t            camber | m          camber_position | camberPos | p
 *   alpha | a                mach | mc            sweep_angle | sweep | sw
 *   aspect_ratio | aspect | ar   taper_ratio | taper | tp   dihedral_angle | dihedral | di
 */

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

// ─── NACA 4-digit foil ────────────────────────────────────────────────────────
function genFoil(t, m, p, n = 100) {
  const upper = [], lower = [];
  const th = Math.max(0.01, Math.min(t, 0.4));
  const pp = Math.max(0.1, Math.min(p, 0.9));
  for (let i = 0; i <= n; i++) {
    const x = (1 - Math.cos(Math.PI * i / n)) / 2;
    const yt = 5 * th * (0.2969*Math.sqrt(x) - 0.126*x - 0.3516*x*x + 0.2843*x*x*x - 0.1015*x*x*x*x);
    let yc, dy;
    if (x < pp) { yc = m/(pp*pp)*(2*pp*x-x*x);                dy = 2*m/(pp*pp)*(pp-x); }
    else         { yc = m/((1-pp)*(1-pp))*(1-2*pp+2*pp*x-x*x); dy = 2*m/((1-pp)*(1-pp))*(pp-x); }
    const ang = Math.atan(dy);
    upper.push({ x: x - yt*Math.sin(ang), y: yc + yt*Math.cos(ang) });
    lower.push({ x: x + yt*Math.sin(ang), y: yc - yt*Math.cos(ang) });
  }
  return { upper, lower };
}

// ─── Cp estimator ─────────────────────────────────────────────────────────────
function estimateCp(alpha, mach, t, cam, cpPos) {
  const aRad = alpha * Math.PI / 180;
  const beta = Math.sqrt(Math.max(1 - mach*mach, 0.01));
  const pp   = Math.max(0.1, Math.min(cpPos, 0.9));
  return Array.from({ length: 200 }, (_, i) => {
    const x  = i / 199;
    const dy = x < pp ? 2*cam/(pp*pp)*(pp-x) : 2*cam/((1-pp)*(1-pp))*(pp-x);
    const te = t * (1/Math.sqrt(Math.max(x, 0.001)) - 1) * 0.15;
    const ce = dy * 2;
    const ae = aRad / Math.sqrt(Math.max(x, 0.001)) * 0.4;
    const vu = 1 + ae + ce + Math.abs(te);
    const vl = 1 - ae - ce + Math.abs(te)*0.3;
    const sg = Math.exp(-x*40) * (1 - 0.5*Math.abs(aRad));
    return {
      x:   +x.toFixed(4),
      cpU: +Math.max(-6, Math.min(1, (1-vu*vu)/beta + sg*0.8)).toFixed(4),
      cpL: +Math.max(-6, Math.min(1, (1-vl*vl)/beta + sg*0.8)).toFixed(4),
    };
  });
}

// ─── Cp value → THREE.Color ───────────────────────────────────────────────────
function cpColor(val) {
  const n = Math.max(0, Math.min(1, (val + 3) / 4));
  return new THREE.Color(n < 0.5 ? n*2 : 1, n < 0.5 ? n*2 : 2-n*2, n < 0.5 ? 1 : 2-n*2);
}

// ─── Wing half mesh ───────────────────────────────────────────────────────────
function buildWingHalf(foil, cpData, halfSpan, taperV, sweepRad, dihRad, nSec) {
  const pos = [], col = [];
  const nPts = foil.upper.length;
  for (let s = 0; s < nSec; s++) {
    const t1 = s/nSec, t2 = (s+1)/nSec;
    const z1 = t1*halfSpan, z2 = t2*halfSpan;
    const c1 = 1*(1-t1)+taperV*t1,      c2 = 1*(1-t2)+taperV*t2;
    const xo1 = t1*halfSpan*Math.tan(sweepRad), xo2 = t2*halfSpan*Math.tan(sweepRad);
    const yo1 = t1*halfSpan*Math.tan(dihRad),   yo2 = t2*halfSpan*Math.tan(dihRad);
    for (let i = 0; i < nPts-1; i++) {
      const ci = Math.min(Math.floor(i/(nPts-1)*199), 198);
      // upper (normal winding)
      const [uA,uB,uC,uD] = mkQuad(foil.upper[i], foil.upper[i+1], xo1,yo1,c1, xo2,yo2,c2, z1,z2);
      pos.push(...uA,...uB,...uD,...uB,...uC,...uD);
      const cu = cpColor(cpData[ci].cpU);
      for (let k=0;k<6;k++) col.push(cu.r,cu.g,cu.b);
      // lower (flipped winding for correct normals)
      const [lA,lB,lC,lD] = mkQuad(foil.lower[i], foil.lower[i+1], xo1,yo1,c1, xo2,yo2,c2, z1,z2);
      pos.push(...lA,...lD,...lB,...lB,...lD,...lC);
      const cl = cpColor(cpData[ci].cpL);
      for (let k=0;k<6;k++) col.push(cl.r,cl.g,cl.b);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("color",    new THREE.Float32BufferAttribute(col, 3));
  geo.computeVertexNormals();
  return geo;
}
function mkQuad(p1, p2, xo1,yo1,c1, xo2,yo2,c2, z1,z2) {
  return [
    [p1.x*c1+xo1, p1.y*c1+yo1, z1],
    [p1.x*c2+xo2, p1.y*c2+yo2, z2],
    [p2.x*c2+xo2, p2.y*c2+yo2, z2],
    [p2.x*c1+xo1, p2.y*c1+yo1, z1],
  ];
}

// ─── Streamline path (wraps both upper and lower surfaces) ────────────────────
// h       : far-field Y height of this streamline
// stagH   : Y of the stagnation streamline
// isStall : whether we're in stall (adds wake turbulence)
function computeStreamline(h, zPos, foil, halfSpan, taperV, sweepRad, dihRad, alphaRad, stagH, isStall) {
  const isUpper = h >= stagH;
  const hDist   = Math.abs(h - stagH) + 0.004;
  const pts     = [];
  const N       = 260;
  for (let xi = 0; xi < N; xi++) {
    const x   = -2.3 + xi * (5.6 / (N-1));
    const lz  = Math.abs(zPos);
    const sf  = Math.max(0, Math.min(lz / halfSpan, 1));
    const chord = Math.max(0.01, 1*(1-sf) + taperV*sf);
    const xOff  = sf * halfSpan * Math.tan(sweepRad);
    const yOff  = sf * halfSpan * Math.tan(dihRad);
    const spanW = Math.max(0, 1 - lz / (halfSpan * 1.05));
    let y = h;
    if (spanW > 0.01) {
      const xLoc = (x - xOff) / chord;
      if (xLoc >= -0.08 && xLoc <= 1.1) {
        const fi = Math.min(Math.max(Math.round(xLoc * (foil.upper.length-1)), 0), foil.upper.length-1);
        const surfU = foil.upper[fi].y * chord + yOff;
        const surfL = foil.lower[fi].y * chord + yOff;
        const blIn  = Math.max(0, Math.min(1, (xLoc + 0.08) / 0.12));
        const blOut = Math.max(0, Math.min(1, (1.10 - xLoc) / 0.12));
        const blend = blIn * blOut * spanW;
        if (isUpper) {
          y = h + (surfU + hDist * 1.05 - h) * blend;
        } else {
          y = h + (surfL - hDist * 1.05 - h) * blend;
        }
      } else if (xLoc > 1.1 && xLoc < 3.0) {
        const fi  = foil.upper.length - 1;
        const teY = (isUpper ? foil.upper[fi].y : foil.lower[fi].y) * chord + yOff;
        const decay  = Math.exp(-(xLoc - 1.1) * 2.2);
        const target = h + alphaRad * 0.08 * spanW * (isUpper ? 0.5 : -0.5);
        y = target + (teY + hDist*(isUpper?1:-1) - target) * decay;
        if (isStall && isUpper) {
          y += Math.sin(xLoc*30 + zPos*8) * 0.015 * spanW;
        }
      }
    }
    pts.push(new THREE.Vector3(x, y, zPos));
  }
  return pts;
}

// ─── Legend row ───────────────────────────────────────────────────────────────
function LegendRow({ dot, line, color = "#1a1a1a", children }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      {dot  && <div style={{ width: 9, height: 9, borderRadius: "50%", background: dot, flexShrink: 0 }} />}
      {line && <div style={{ width: 16, height: 2, background: line, flexShrink: 0 }} />}
      <span style={{ color }}>{children}</span>
    </div>
  );
}

// ─── Viewer3D ─────────────────────────────────────────────────────────────────
export function Viewer3D(rawProps) {
  // Prop alias resolution
  const thickness_ = rawProps.thickness        ?? rawProps.t   ?? 0.12;
  const camber_    = rawProps.camber           ?? rawProps.m   ?? 0.04;
  const camberPos_ = rawProps.camber_position  ?? rawProps.camberPos ?? rawProps.p ?? 0.4;
  const alpha_     = rawProps.alpha            ?? rawProps.a   ?? 5;
  const mach_      = rawProps.mach             ?? rawProps.mc  ?? 0.5;
  const sweep_     = rawProps.sweep_angle      ?? rawProps.sweep ?? rawProps.sw ?? 20;
  const aspect_    = rawProps.aspect_ratio     ?? rawProps.aspect ?? rawProps.ar ?? 8;
  const taper_     = rawProps.taper_ratio      ?? rawProps.taper  ?? rawProps.tp ?? 0.5;
  const dihedral_  = rawProps.dihedral_angle   ?? rawProps.dihedral ?? rawProps.di ?? 3;

  const canvasRef = useRef(null);
  // Camera state — persists across renders, never auto-rotates
  const camState  = useRef({ animId: null, rotating: false, rotX: 0.18, rotY: -0.5, lastX: 0, lastY: 0 });

  const [showLegend, setShowLegend] = useState(true);
  const [wireframe,  setWireframe]  = useState(false);

  // Overlay estimates
  const cpData     = estimateCp(alpha_, mach_, thickness_, camber_, camberPos_);
  const Cl_est     = (cpData.reduce((s, d) => s + (d.cpL - d.cpU), 0) / cpData.length * 1.75 + alpha_ * 0.097).toFixed(3);
  const Cd_est     = (Math.abs(alpha_) * 0.00085 + thickness_ * 0.028 + 0.005).toFixed(4);
  const LDratio    = (parseFloat(Cl_est) / parseFloat(Cd_est)).toFixed(1);
  const isStall    = Math.abs(alpha_) > 14;
  const isCritMach = mach_ > 0.7;
  const hasSep     = Math.abs(alpha_) > 10;
  const hasShock   = mach_ > 0.6;

  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const W = el.clientWidth  || 640;
    const H = el.clientHeight || 360;

    // ── Scene ────────────────────────────────────────────────────────────────
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf7f7f5);
    scene.fog = new THREE.Fog(0xf7f7f5, 9, 22);

    const camera = new THREE.PerspectiveCamera(28, W/H, 0.01, 100);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    el.innerHTML = "";
    el.appendChild(renderer.domElement);

    // ── Lights ───────────────────────────────────────────────────────────────
    scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9);
    sun.position.set(4, 8, 5);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    scene.add(sun);
    const fill = new THREE.DirectionalLight(0xb0c8e8, 0.45);
    fill.position.set(-3, 1, -3);
    scene.add(fill);
    const under = new THREE.DirectionalLight(0xd0e4f8, 0.2);
    under.position.set(0, -5, 2);
    scene.add(under);

    // ── Grid ─────────────────────────────────────────────────────────────────
    const grid = new THREE.GridHelper(10, 40, 0xcccccc, 0xe2e2e2);
    grid.position.y = -0.6;
    scene.add(grid);

    // ── Wing params ──────────────────────────────────────────────────────────
    const group    = new THREE.Group();
    const foil     = genFoil(thickness_, camber_, camberPos_, 80);
    const halfSpan = (aspect_ * 0.19) / 2;
    const sweepRad = sweep_    * Math.PI / 180;
    const dihRad   = dihedral_ * Math.PI / 180;
    const taperV   = Math.max(0.1, Math.min(taper_, 1.0));
    const alphaRad = alpha_    * Math.PI / 180;
    const stagH    = -alphaRad * 0.055;

    // ── Wing mesh ─────────────────────────────────────────────────────────────
    const wingGeo = buildWingHalf(foil, cpData, halfSpan, taperV, sweepRad, dihRad, 32);
    const wingMat = new THREE.MeshPhongMaterial({
      vertexColors: !wireframe,
      side:         THREE.DoubleSide,
      shininess:    45,
      specular:     new THREE.Color(0x334455),
      ...(wireframe ? { color: 0x1a5276, wireframe: true } : {}),
    });
    const wingR = new THREE.Mesh(wingGeo, wingMat);
    wingR.castShadow = true;
    group.add(wingR);
    const wingL = wingR.clone();
    wingL.scale.z = -1;
    group.add(wingL);

    // ── Spanwise annotation lines ─────────────────────────────────────────────
    const addSpanLine = (xFrac, getSurfY, hexColor, opacity, yBias) => {
      const mat = new THREE.LineBasicMaterial({ color: hexColor, transparent: true, opacity });
      const pR = [], pL = [];
      for (let s = 0; s <= 30; s++) {
        const f = s/30, z = f*halfSpan;
        const c = 1*(1-f)+taperV*f;
        const xo = f*halfSpan*Math.tan(sweepRad), yo = f*halfSpan*Math.tan(dihRad);
        const fi = Math.min(Math.floor(xFrac * (foil.upper.length-1)), foil.upper.length-2);
        const sy = getSurfY(fi) * c + yo;
        pR.push(new THREE.Vector3(foil.upper[fi].x*c+xo, sy+yBias,  z));
        pL.push(new THREE.Vector3(foil.upper[fi].x*c+xo, sy+yBias, -z));
      }
      group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pR), mat));
      group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pL), mat));
    };

    addSpanLine(0,    i => foil.upper[i].y, 0x2255aa, 0.40, 0.008); // leading edge

    if (hasSep) {
      const sf = Math.max(0.15, 1 - (Math.abs(alpha_)-10)/12*0.75);
      addSpanLine(sf, i => foil.upper[i].y, 0xff5500, 0.85, 0.012);  // separation
    }
    if (hasShock) {
      const shf = Math.max(0.25, 0.7 - (mach_-0.6)*0.5);
      addSpanLine(shf, i => foil.upper[i].y, 0xcc9900, 0.90, 0.012); // shock
    }

    // ── Streamlines — upper AND lower surface ─────────────────────────────────
    const streamMat = new THREE.LineBasicMaterial({ color: 0x2255aa, transparent: true, opacity: 0.28 });

    // Z stations (symmetric about root)
    const zSlots = [0, 0.18, 0.38, 0.57, 0.73, 0.85, 0.94].map(f => f * halfSpan);
    // Height offsets from stagnation — 4 upper + 4 lower
    const dists  = [0.020, 0.060, 0.115, 0.190];
    const allH   = [...dists.map(d => stagH + d), ...dists.map(d => stagH - d)];

    for (const h of allH) {
      for (const zAbs of zSlots) {
        for (const sign of [1, -1]) {
          if (zAbs === 0 && sign === -1) continue; // skip z=0 duplicate
          const pts = computeStreamline(
            h, zAbs * sign,
            foil, halfSpan, taperV, sweepRad, dihRad,
            alphaRad, stagH, isStall
          );
          group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), streamMat));
        }
      }
    }

    // ── Wake turbulence at stall ──────────────────────────────────────────────
    if (isStall) {
      const wakeMat = new THREE.LineBasicMaterial({ color: 0xcc3300, transparent: true, opacity: 0.4 });
      for (let k = 0; k < 10; k++) {
        const zz = (k/9 - 0.5) * halfSpan * 1.6;
        const wp  = [];
        for (let dx = 0; dx <= 1.1; dx += 0.02) {
          wp.push(new THREE.Vector3(1+dx, Math.sin(dx*22+k*1.7)*0.03*(1+dx*2.5)-0.015, zz));
        }
        group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(wp), wakeMat));
      }
    }

    // ── Stagnation sphere ─────────────────────────────────────────────────────
    const stagMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.014, 14, 14),
      new THREE.MeshBasicMaterial({ color: 0xdd1111 })
    );
    stagMesh.position.set(0, stagH, 0);
    group.add(stagMesh);

    // ── Center & tilt group ───────────────────────────────────────────────────
    group.rotation.x = -alphaRad * 0.3;
    const box = new THREE.Box3().setFromObject(group);
    group.position.sub(box.getCenter(new THREE.Vector3()));
    scene.add(group);

    // ── Camera sync ───────────────────────────────────────────────────────────
    const st = camState.current;
    const syncCamera = () => {
      const d = 3.6;
      camera.position.set(
        d * Math.sin(st.rotY) * Math.cos(st.rotX),
        d * Math.sin(st.rotX) + 0.3,
        d * Math.cos(st.rotY) * Math.cos(st.rotX),
      );
      camera.lookAt(0, 0, 0);
    };
    syncCamera();

    // ── Input handlers ────────────────────────────────────────────────────────
    const cvs    = renderer.domElement;
    const onDown = e => { st.rotating = true;  st.lastX = e.clientX; st.lastY = e.clientY; cvs.style.cursor = "grabbing"; };
    const onUp   = ()  => { st.rotating = false; cvs.style.cursor = "grab"; };
    const onMove = e  => {
      if (!st.rotating) return;
      st.rotY += (e.clientX - st.lastX) * 0.006;
      st.rotX += (e.clientY - st.lastY) * 0.005;
      st.rotX  = Math.max(-1.2, Math.min(1.2, st.rotX));
      st.lastX = e.clientX; st.lastY = e.clientY;
    };
    const onTD = e => { st.rotating = true;  st.lastX = e.touches[0].clientX; st.lastY = e.touches[0].clientY; };
    const onTU = ()  => { st.rotating = false; };
    const onTM = e  => {
      if (!st.rotating) return;
      st.rotY += (e.touches[0].clientX - st.lastX) * 0.006;
      st.rotX += (e.touches[0].clientY - st.lastY) * 0.005;
      st.lastX = e.touches[0].clientX; st.lastY = e.touches[0].clientY;
    };
    cvs.addEventListener("mousedown",  onDown);
    cvs.addEventListener("mouseup",    onUp);
    cvs.addEventListener("mouseleave", onUp);
    cvs.addEventListener("mousemove",  onMove);
    cvs.addEventListener("touchstart", onTD, { passive: true });
    cvs.addEventListener("touchend",   onTU);
    cvs.addEventListener("touchmove",  onTM, { passive: true });

    // ── Render loop ───────────────────────────────────────────────────────────
    const animate = () => {
      syncCamera();
      renderer.render(scene, camera);
      st.animId = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelAnimationFrame(st.animId);
      cvs.removeEventListener("mousedown",  onDown);
      cvs.removeEventListener("mouseup",    onUp);
      cvs.removeEventListener("mouseleave", onUp);
      cvs.removeEventListener("mousemove",  onMove);
      cvs.removeEventListener("touchstart", onTD);
      cvs.removeEventListener("touchend",   onTU);
      cvs.removeEventListener("touchmove",  onTM);
      renderer.dispose();
      wingGeo.dispose();
      wingMat.dispose();
    };
  }, [thickness_, camber_, camberPos_, alpha_, mach_, sweep_, aspect_, taper_, dihedral_, wireframe]);

  // ── Cp gradient (legend) ─────────────────────────────────────────────────
  const cpGradient = Array.from({ length: 12 }, (_, i) => {
    const n = i / 11;
    const r = Math.round(n < 0.5 ? n*2*255 : 255);
    const g = Math.round(n < 0.5 ? n*2*255 : (2-n*2)*255);
    const b = Math.round(n < 0.5 ? 255 : (2-n*2)*255);
    return `rgb(${r},${g},${b}) ${(n*100).toFixed(0)}%`;
  }).join(", ");

  // ── Shared style tokens (mirrors dashboard) ───────────────────────────────
  const F = { s: "'Instrument Sans',system-ui,sans-serif", m: "'IBM Plex Mono','Courier New',monospace" };
  const C = { tx:"#1a1a1a", mu:"#999", ac:"#1a5276", bd:"#d4d4d4", bg:"rgba(255,255,255,0.88)", warn:"#b7950b", err:"#c0392b", ok:"#1e8449" };

  const panel = { position:"absolute", background:C.bg, border:`1px solid ${C.bd}`, borderRadius:5, padding:"7px 10px", backdropFilter:"blur(4px)", fontFamily:F.m, fontSize:10, color:C.tx, lineHeight:1.85 };
  const secLbl = { display:"block", fontSize:9, fontWeight:700, color:C.mu, letterSpacing:1.1, textTransform:"uppercase", marginBottom:3, fontFamily:F.s };
  const av     = { color:C.ac, fontWeight:600 };
  const btn    = { padding:"4px 12px", background:"#fff", border:`1px solid ${C.bd}`, borderRadius:4, color:C.ac, fontSize:10, fontFamily:F.s, fontWeight:600, cursor:"pointer", letterSpacing:0.3 };

  return (
    <div style={{ position:"relative", width:"100%", height:370, borderRadius:5, overflow:"hidden", border:`1px solid ${C.bd}`, background:"#f7f7f5" }}>

      <div ref={canvasRef} style={{ width:"100%", height:"100%", cursor:"grab" }} />

      {/* Top-left: geometry */}
      <div style={{ ...panel, top:10, left:10 }}>
        <span style={secLbl}>Wing Geometry</span>
        <div>α <span style={av}>{alpha_}°</span>{"  "}M <span style={{ ...av, color: isCritMach ? C.warn : C.ac }}>{mach_}</span></div>
        <div>t/c <span style={av}>{(thickness_*100).toFixed(1)}%</span>{"  "}AR <span style={av}>{aspect_}</span></div>
        <div>Λ <span style={av}>{sweep_}°</span>{"  "}λ <span style={av}>{taper_}</span></div>
        <div>Γ <span style={av}>{dihedral_}°</span>{"  "}cam <span style={av}>{(camber_*100).toFixed(1)}%</span></div>
      </div>

      {/* Top-right: coefficients */}
      <div style={{ ...panel, top:10, right:10, textAlign:"right" }}>
        <span style={{ ...secLbl, textAlign:"right" }}>Estimated Aero</span>
        <div>C<sub>L</sub> <span style={{ color:C.ok, fontWeight:600 }}>{Cl_est}</span></div>
        <div>C<sub>D</sub> <span style={{ color:C.ok, fontWeight:600 }}>{Cd_est}</span></div>
        <div>L/D <span style={av}>{LDratio}</span></div>
        {isStall    && <div style={{ color:C.err,  fontSize:9, fontWeight:700, marginTop:2 }}>⚠ STALL</div>}
        {isCritMach && <div style={{ color:C.warn, fontSize:9, fontWeight:700 }}>⚡ CRIT. MACH</div>}
      </div>

      {/* Bottom-left: Cp legend */}
      {showLegend && (
        <div style={{ ...panel, bottom:38, left:10 }}>
          <span style={secLbl}>Pressure Coefficient C<sub>p</sub></span>
          <div style={{ display:"flex", alignItems:"center", gap:6 }}>
            <span style={{ fontSize:9, color:C.mu }}>−3</span>
            <div style={{ width:100, height:8, borderRadius:3, background:`linear-gradient(to right, ${cpGradient})`, border:`1px solid ${C.bd}` }} />
            <span style={{ fontSize:9, color:C.mu }}>+1</span>
          </div>
          <div style={{ display:"flex", justifyContent:"space-between", width:128, marginTop:1 }}>
            <span style={{ fontSize:8, color:C.mu }}>← suction</span>
            <span style={{ fontSize:8, color:C.mu }}>pressure →</span>
          </div>
        </div>
      )}

      {/* Bottom-right: critical points */}
      {showLegend && (
        <div style={{ ...panel, bottom:38, right:10 }}>
          <span style={secLbl}>Critical Points</span>
          <LegendRow dot="#dd1111">Stagnation</LegendRow>
          <LegendRow line="#2255aa">Leading edge</LegendRow>
          {hasSep   && <LegendRow line="#ff5500" color={C.err}>Separation</LegendRow>}
          {hasShock && <LegendRow line="#cc9900" color={C.warn}>Shock wave</LegendRow>}
          {isStall  && <LegendRow line="#cc3300" color={C.err}>Wake / turbulence</LegendRow>}
        </div>
      )}

      {/* Buttons */}
      <div style={{ position:"absolute", bottom:10, left:"50%", transform:"translateX(-50%)", display:"flex", gap:6 }}>
        <button onClick={() => setWireframe(v => !v)} style={btn}>{wireframe ? "◈ Pressure" : "⬡ Wireframe"}</button>
        <button onClick={() => setShowLegend(v => !v)} style={btn}>{showLegend ? "◻ Hide legend" : "◻ Legend"}</button>
      </div>

      <div style={{ position:"absolute", bottom:13, right:10, fontFamily:F.s, fontSize:9, color:"#bbb", pointerEvents:"none" }}>
        drag to rotate
      </div>
    </div>
  );
}

export default Viewer3D;