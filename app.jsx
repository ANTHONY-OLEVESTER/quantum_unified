import React, { useMemo, useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ScatterChart,
  Scatter,
  ResponsiveContainer,
} from "recharts";

// ---------------------------------------------
// Utility helpers
// ---------------------------------------------
const fmt = (n) => (typeof n === "number" && isFinite(n) ? n.toFixed(4) : "–");
const log10 = (x) => Math.log(x) / Math.log(10);

function linregSlope(x, y) {
  // Simple least squares slope of y on x
  const n = Math.min(x.length, y.length);
  if (n < 2) return NaN;
  const xs = x.slice(0, n);
  const ys = y.slice(0, n);
  const xbar = xs.reduce((a, b) => a + b, 0) / n;
  const ybar = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - xbar) * (ys[i] - ybar);
    den += (xs[i] - xbar) ** 2;
  }
  return num / den;
}

function parseCSVSimple(text) {
  // Expect header line; tolerate commas or semicolons
  // Returns array of objects with inferred numeric values when possible
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) return [];
  const header = lines[0].split(/[;,]/).map((s) => s.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(/[;,]/).map((s) => s.trim());
    const obj = {};
    header.forEach((h, j) => {
      const v = cols[j];
      const num = Number(v);
      obj[h || `col${j}`] = isNaN(num) ? v : num;
    });
    rows.push(obj);
  }
  return rows;
}

function downloadBlob(filename, text) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------
// Demo simulators
// ---------------------------------------------
function simulateVarianceScaling({ Dmin, Dmax, points, c, noise, seed }) {
  // Var(Y) ≈ c / D + noise
  const rng = mulberry32(seed || 42);
  const data = [];
  const Ds = logspace(Dmin, Dmax, points);
  for (const D of Ds) {
    const mean = c / D;
    const v = Math.max(1e-12, mean + noise * (rng() - 0.5) * 2);
    data.push({ D, VarY: v, logD: Math.log(D), logVarY: Math.log(v) });
  }
  const slope = linregSlope(
    data.map((d) => d.logD),
    data.map((d) => d.logVarY)
  );
  return { data, slope };
}

function simulateAlphaFlatness({ Dmin, Dmax, points, k, seed }) {
  // |alpha| ~ N(0, (k^2)/D), signed alpha ~ N(0, k/sqrt(D))
  const rng = mulberry32(seed || 7);
  const data = [];
  const Ds = logspace(Dmin, Dmax, points);
  for (const D of Ds) {
    const sigma = k / Math.sqrt(D);
    const signed = gaussian(rng) * sigma; // mean 0
    data.push({ D, invD: 1 / D, signedAlpha: signed, absAlpha: Math.abs(signed) });
  }
  // Regress signedAlpha vs 1/D to estimate intercept ~ 0
  const slope = linregSlope(
    data.map((d) => d.invD),
    data.map((d) => d.signedAlpha)
  );
  // Intercept
  const x = data.map((d) => d.invD);
  const y = data.map((d) => d.signedAlpha);
  const n = x.length;
  const xbar = x.reduce((a, b) => a + b, 0) / n;
  const ybar = y.reduce((a, b) => a + b, 0) / n;
  const slopeDen = x.reduce((acc, xi) => acc + (xi - xbar) ** 2, 0);
  const b = ybar - (slope * xbar);
  return { data, slope, intercept: b };
}

// PRNG utils
function mulberry32(a) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng) {
  // Box–Muller
  let u = 0,
    v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function logspace(a, b, n) {
  // a,b are powers of 2 for convenience, produce D in [2^a, 2^b]
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const pow = a * (1 - t) + b * t;
    out.push(Math.pow(2, pow));
  }
  return out;
}

// ---------------------------------------------
// UI Components
// ---------------------------------------------
const Pill = ({ children }) => (
  <span className="px-3 py-1 rounded-full bg-slate-800/70 border border-slate-700 text-slate-200 text-xs">
    {children}
  </span>
);

const Section = ({ id, title, subtitle, children }) => (
  <section id={id} className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
    <div className="mb-6">
      <h2 className="text-2xl sm:text-3xl font-semibold text-slate-100">{title}</h2>
      {subtitle && (
        <p className="text-slate-300 mt-2 max-w-3xl leading-relaxed">{subtitle}</p>
      )}
    </div>
    <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-4 sm:p-6 shadow-xl">
      {children}
    </div>
  </section>
);

const Card = ({ title, children, right }) => (
  <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5">
    <div className="flex items-center justify-between gap-4 mb-3">
      <h3 className="text-lg font-medium text-slate-100">{title}</h3>
      {right}
    </div>
    {children}
  </div>
);

// ---------------------------------------------
// Main App
// ---------------------------------------------
export default function App() {
  // Simulator states
  const [varCfg, setVarCfg] = useState({ Dmin: 6, Dmax: 20, points: 10, c: 1.0, noise: 0.05, seed: 1234 });
  const [alpCfg, setAlpCfg] = useState({ Dmin: 6, Dmax: 20, points: 10, k: 1.0, seed: 999 });

  const varSim = useMemo(() => simulateVarianceScaling(varCfg), [varCfg]);
  const alpSim = useMemo(() => simulateAlphaFlatness(alpCfg), [alpCfg]);

  // CSV upload state
  const [csvRows, setCsvRows] = useState([]);
  const [csvFields, setCsvFields] = useState([]);

  function handleCSV(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const rows = parseCSVSimple(String(reader.result || ""));
      setCsvRows(rows);
      const fields = rows.length ? Object.keys(rows[0]) : [];
      setCsvFields(fields);
    };
    reader.readAsText(file);
  }

  function exportSimData() {
    const header = ["D", "VarY", "logD", "logVarY"]; 
    const lines = [header.join(",")].concat(
      varSim.data.map((d) => [d.D, d.VarY, d.logD, d.logVarY].join(","))
    );
    downloadBlob("variance_sim_data.csv", lines.join("\n"));
  }

  useEffect(() => {
    document.title = "Curvature–Information Showcase";
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      {/* Nav */}
      <header className="sticky top-0 z-50 bg-slate-950/80 backdrop-blur border-b border-slate-900">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />
            <span className="font-semibold tracking-wide">Curvature–Information</span>
            <Pill>Live Demos</Pill>
          </div>
          <nav className="hidden sm:flex items-center gap-6 text-sm">
            <a href="#concept" className="hover:text-white">Concept</a>
            <a href="#demos" className="hover:text-white">Simulations</a>
            <a href="#data" className="hover:text-white">Your Data</a>
            <a href="#applications" className="hover:text-white">Applications</a>
            <a href="#experiment" className="hover:text-white">Experiment</a>
          </nav>
        </div>
      </header>

      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(79,70,229,0.15),transparent_50%)]" />
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-14">
          <div className="grid md:grid-cols-2 gap-10 items-center">
            <div>
              <motion.h1
                initial={{ y: 10, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ duration: 0.6 }}
                className="text-3xl sm:text-5xl font-extrabold text-white leading-tight"
              >
                Flatness & D<sup className="align-super text-xs">−1</sup> Concentration
              </motion.h1>
              <p className="mt-4 text-slate-300 text-base sm:text-lg leading-relaxed">
                A lightweight, interactive site to <span className="text-white font-semibold">show—not tell</span>:
                the curvature–information invariant <em>Y</em>, its flat signed-α, and the universal variance law
                <span className="whitespace-nowrap"> Var(Y) ∝ D⁻¹</span> under 2‑designs.
              </p>
              <div className="mt-6 flex flex-wrap gap-3">
                <Pill>Unitary 2‑design intuition</Pill>
                <Pill>Bures/Uhlmann geometry</Pill>
                <Pill>Mutual information</Pill>
                <Pill>Monte‑Carlo demos</Pill>
              </div>
            </div>
            <motion.div
              initial={{ scale: 0.96, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.15 }}
              className="rounded-3xl border border-slate-800 bg-slate-900/60 p-5"
            >
              <div className="text-sm text-slate-300 leading-relaxed">
                <p>
                  <span className="font-semibold text-white">Key predictions:</span> E[Y] = Y₀ + O(D⁻¹), Var(Y) = Θ(D⁻¹),
                  E[α] → 0 with |α| = O<sub>P</sub>(D⁻¹ᐟ²). Twirling restores flatness for structured dynamics.
                </p>
                <ul className="list-disc ml-5 mt-3 space-y-1">
                  <li>Signed‑α confidence intervals include 0 at large D.</li>
                  <li>log Var(Y) vs log D has slope ≈ −1.</li>
                  <li>α vs 1/D extrapolates to intercept 0 within CI.</li>
                </ul>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Concept */}
      <Section
        id="concept"
        title="Concept: the curvature–information invariant"
        subtitle={
          "Y couples Bures geometry (A²) with information (I) and effective dimension (d_eff). These demos visualize the empirical laws and how twirling/2‑designs drive universality."
        }
      >
        <div className="grid md:grid-cols-3 gap-4">
          <Card title="Definition (schematic)">
            <div className="text-slate-300 text-sm leading-relaxed">
              <div>
                <span className="text-white">Y</span> = sqrt(d_eff − 1) · A² / I
              </div>
              <ul className="list-disc ml-5 mt-2 space-y-1">
                <li>A² ≈ ¼ g<sub>Bures</sub>(δρ, δρ) for small perturbations</li>
                <li>I = I(S:E) after one‑step Stinespring dilation</li>
                <li>d_eff = 1 / Tr(ρ′²)</li>
              </ul>
            </div>
          </Card>
          <Card title="Flatness (α → 0)">
            <p className="text-slate-300 text-sm">
              On log–log axes of Y vs (d_eff−1), the slope α averages to 0 and concentrates like D⁻¹ᐟ² under 2‑designs.
            </p>
          </Card>
          <Card title="Concentration (Var(Y) ∝ D⁻¹)">
            <p className="text-slate-300 text-sm">
              The finite‑size variance falls as 1/D. This shows up as a ≈ −1 slope in log Var(Y) vs log D.
            </p>
          </Card>
        </div>
      </Section>

      {/* Demos & Simulations */}
      <Section id="demos" title="Interactive Simulations" subtitle="Illustrate the laws with transparent, browser‑side Monte‑Carlo.">
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Variance scaling */}
          <Card
            title="Variance scaling: fit log Var(Y) vs log D"
            right={<Pill>slope ≈ {fmt(varSim.slope)}</Pill>}
          >
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-3 text-sm">
                <Tuner label="D range (powers of 2)">
                  <RangeRow
                    label="min pow"
                    value={varCfg.Dmin}
                    min={4}
                    max={16}
                    step={1}
                    onChange={(v) => setVarCfg({ ...varCfg, Dmin: v })}
                  />
                  <RangeRow
                    label="max pow"
                    value={varCfg.Dmax}
                    min={varCfg.Dmin + 1}
                    max={22}
                    step={1}
                    onChange={(v) => setVarCfg({ ...varCfg, Dmax: v })}
                  />
                </Tuner>
                <Tuner label="Sampling">
                  <RangeRow
                    label="#points"
                    value={varCfg.points}
                    min={5}
                    max={24}
                    step={1}
                    onChange={(v) => setVarCfg({ ...varCfg, points: v })}
                  />
                  <RangeRow
                    label="c (scale)"
                    value={varCfg.c}
                    min={0.05}
                    max={5}
                    step={0.05}
                    onChange={(v) => setVarCfg({ ...varCfg, c: v })}
                  />
                  <RangeRow
                    label="noise"
                    value={varCfg.noise}
                    min={0}
                    max={0.5}
                    step={0.01}
                    onChange={(v) => setVarCfg({ ...varCfg, noise: v })}
                  />
                </Tuner>
                <div className="flex gap-3">
                  <button
                    className="px-3 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-sm"
                    onClick={() => setVarCfg({ ...varCfg, seed: Math.floor(Math.random() * 1e9) })}
                  >
                    Reroll seed
                  </button>
                  <button
                    className="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-100 text-sm"
                    onClick={exportSimData}
                  >
                    Export CSV
                  </button>
                </div>
                <p className="text-slate-400 text-xs">
                  Expect slope ≈ −1 when Var(Y) ∝ 1/D.
                </p>
              </div>
              <div className="h-64 sm:h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={varSim.data.map((d) => ({ logD: log10(d.D), logVarY: log10(d.VarY) }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#0f172a" />
                    <XAxis dataKey="logD" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                    <YAxis tick={{ fill: "#94a3b8", fontSize: 12 }} />
                    <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937" }} />
                    <Legend />
                    <Line type="monotone" dataKey="logVarY" stroke="#818cf8" dot={false} name="log Var(Y)" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </Card>

          {/* Alpha flatness */}
          <Card
            title="α flatness: regress signed α vs 1/D"
            right={
              <div className="flex gap-2">
                <Pill>slope {fmt(alpSim.slope)}</Pill>
                <Pill>intercept {fmt(alpSim.intercept)}</Pill>
              </div>
            }
          >
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-3 text-sm">
                <Tuner label="D range (powers of 2)">
                  <RangeRow
                    label="min pow"
                    value={alpCfg.Dmin}
                    min={4}
                    max={16}
                    step={1}
                    onChange={(v) => setAlpCfg({ ...alpCfg, Dmin: v })}
                  />
                  <RangeRow
                    label="max pow"
                    value={alpCfg.Dmax}
                    min={alpCfg.Dmin + 1}
                    max={22}
                    step={1}
                    onChange={(v) => setAlpCfg({ ...alpCfg, Dmax: v })}
                  />
                </Tuner>
                <Tuner label="Concentration">
                  <RangeRow
                    label="k (scale)"
                    value={alpCfg.k}
                    min={0.2}
                    max={5}
                    step={0.1}
                    onChange={(v) => setAlpCfg({ ...alpCfg, k: v })}
                  />
                </Tuner>
                <button
                  className="px-3 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-sm"
                  onClick={() => setAlpCfg({ ...alpCfg, seed: Math.floor(Math.random() * 1e9) })}
                >
                  Reroll seed
                </button>
                <p className="text-slate-400 text-xs">
                  Intercept → 0 indicates flatness (signed α averages to 0).
                </p>
              </div>
              <div className="h-64 sm:h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#0f172a" />
                    <XAxis dataKey="invD" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                    <YAxis dataKey="signedAlpha" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                    <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937" }} />
                    <Legend />
                    <Scatter name="signed α" data={alpSim.data} fill="#34d399" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          </Card>
        </div>

        {/* Twirl vs Structured Demo (conceptual) */}
        <div className="mt-6">
          <Card title="Twirling vs Structured Dynamics (conceptual toggle)">
            <TwirlingDemo />
          </Card>
        </div>
      </Section>

      {/* Data ingestion */}
      <Section
        id="data"
        title="Plug in your Phase CSVs"
        subtitle="Drop any of your exported CSVs (e.g., ‘phase9 summary haar sq wls lodo.csv’) and chart key fields."
      >
        <div className="grid md:grid-cols-3 gap-6">
          <div className="space-y-3">
            <label className="block text-sm text-slate-300">Upload CSV</label>
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={handleCSV}
              className="block w-full text-sm file:mr-4 file:py-2 file:px-3 file:rounded-xl file:border-0 file:text-sm file:font-medium file:bg-slate-800 file:text-slate-100 hover:file:bg-slate-700"
            />
            {csvFields.length > 0 && (
              <div className="text-xs text-slate-400">Detected columns: {csvFields.join(", ")}</div>
            )}
            <p className="text-xs text-slate-400">
              Tip: Common pairs — (D, VarY), (invD, alpha), (logD, logVarY)
            </p>
          </div>
          <div className="md:col-span-2">
            <CSVPlot rows={csvRows} />
          </div>
        </div>
      </Section>

      {/* Applications */}
      <Section
        id="applications"
        title="Applications & Live Widgets"
        subtitle="Embed bite‑size widgets to connect the principle to real workflows."
      >
        <div className="grid md:grid-cols-3 gap-6">
          <Card title="Model audit (RB‑style)">
            <p className="text-slate-300 text-sm mb-3">
              Use twirling intuition to visualize ‘isotropization’: structured channels deviate; twirl pulls back to flat.
            </p>
            <MiniDial label="Flatness score" value={87} />
          </Card>
          <Card title="Scaling forecaster">
            <p className="text-slate-300 text-sm mb-3">
              Extrapolate Var(Y) to larger D assuming slope −1 and current c.
            </p>
            <ScalingForecaster c={varCfg.c} />
          </Card>
          <Card title="CI Checker (toy)">
            <p className="text-slate-300 text-sm mb-3">
              Check whether the CI for signed‑α includes 0 at the largest D.
            </p>
            <CIWidget data={alpSim.data} />
          </Card>
        </div>
      </Section>

      {/* Experiment */}
      <Section
        id="experiment"
        title="Minimal experimental protocol"
        subtitle="For a 2–3 qubit platform: apply random local Cliffords (2‑design), single‑qubit tomography, estimate Y, plot α & Var(Y)."
      >
        <ol className="list-decimal ml-6 space-y-2 text-slate-300 text-sm">
          <li>Implement a noisy channel on the target qubit; conjugate by random local Cliffords (pre/post twirl).</li>
          <li>Estimate ρ<sub>S</sub>, ρ′<sub>S</sub>; compute A² via fidelity, I = 2S(ρ′<sub>S</sub>), and d_eff.</li>
          <li>Increase effective D via ancillas or mixing depth; collect (D, Y) points.</li>
          <li>Plot log Var(Y) vs log D → slope ≈ −1; regress α vs 1/D → intercept ≈ 0.</li>
        </ol>
      </Section>

      {/* Footer */}
      <footer className="border-t border-slate-900 mt-10">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6 text-xs text-slate-500">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <span>Curvature–Information Showcase • MIT‑licensed demo scaffold</span>
            <div className="flex items-center gap-3">
              <a className="hover:text-slate-300" href="#concept">Concept</a>
              <a className="hover:text-slate-300" href="#demos">Simulations</a>
              <a className="hover:text-slate-300" href="#data">Your Data</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

// ---------------------------------------------
// Subcomponents
// ---------------------------------------------
function Tuner({ label, children }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-slate-400 mb-1">{label}</div>
      <div className="space-y-2">{children}</div>
    </div>
  );
}

function RangeRow({ label, value, min, max, step, onChange }) {
  return (
    <div className="grid grid-cols-3 gap-2 items-center">
      <div className="text-slate-300 text-xs">{label}</div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="col-span-2 w-full"
      />
      <div className="col-span-3 text-right text-slate-400 text-xs">{value}</div>
    </div>
  );
}

function TwirlingDemo() {
  const [mode, setMode] = useState("structured");
  // synthetic signed alpha distribution
  const data = useMemo(() => {
    const rng = mulberry32(321);
    const out = [];
    const Dvals = [64, 128, 256, 512, 1024, 2048, 4096];
    for (const D of Dvals) {
      const k = mode === "twirled" ? 0.6 : 2.0;
      const shift = mode === "twirled" ? 0.0 : 0.15; // structured bias
      const a = gaussian(rng) * (k / Math.sqrt(D)) + shift;
      out.push({ D, invD: 1 / D, signedAlpha: a });
    }
    return out;
  }, [mode]);

  const slope = useMemo(() => linregSlope(
    data.map((d) => d.invD),
    data.map((d) => d.signedAlpha)
  ), [data]);

  const intercept = useMemo(() => {
    const x = data.map((d) => d.invD);
    const y = data.map((d) => d.signedAlpha);
    const n = x.length;
    const xbar = x.reduce((a, b) => a + b, 0) / n;
    const ybar = y.reduce((a, b) => a + b, 0) / n;
    const b = ybar - linregSlope(x, y) * xbar;
    return b;
  }, [data]);

  return (
    <div className="grid md:grid-cols-3 gap-4 items-start">
      <div className="space-y-3 text-sm">
        <div className="flex gap-2">
          <button
            onClick={() => setMode("structured")}
            className={`px-3 py-2 rounded-xl text-sm ${mode === "structured" ? "bg-rose-600 text-white" : "bg-slate-800 text-slate-100"}`}
          >
            Structured
          </button>
          <button
            onClick={() => setMode("twirled")}
            className={`px-3 py-2 rounded-xl text-sm ${mode === "twirled" ? "bg-emerald-600 text-white" : "bg-slate-800 text-slate-100"}`}
          >
            Twirled (2‑design)
          </button>
        </div>
        <p className="text-slate-300">
          Toggle to see how twirling removes bias in α and tightens concentration.
        </p>
        <div className="text-xs text-slate-400">
          <div>mode: <span className="text-slate-200">{mode}</span></div>
          <div>slope: <span className="text-slate-200">{fmt(slope)}</span></div>
          <div>intercept: <span className="text-slate-200">{fmt(intercept)}</span></div>
        </div>
      </div>
      <div className="md:col-span-2 h-64 sm:h-72">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="#0f172a" />
            <XAxis dataKey="invD" tick={{ fill: "#94a3b8", fontSize: 12 }} />
            <YAxis dataKey="signedAlpha" tick={{ fill: "#94a3b8", fontSize: 12 }} />
            <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937" }} />
            <Legend />
            <Scatter name="signed α" data={data} fill={mode === "twirled" ? "#34d399" : "#fb7185"} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function CSVPlot({ rows }) {
  const [xKey, setXKey] = useState("");
  const [yKey, setYKey] = useState("");

  const keys = useMemo(() => (rows.length ? Object.keys(rows[0]) : []), [rows]);

  useEffect(() => {
    if (keys.length && (!xKey || !yKey)) {
      setXKey(keys[0]);
      setYKey(keys[1] || keys[0]);
    }
  }, [keys]);

  const data = rows.map((r) => ({ ...r }));

  return (
    <div>
      <div className="flex flex-wrap gap-3 items-end mb-3">
        <Select label="X" value={xKey} onChange={setXKey} options={keys} />
        <Select label="Y" value={yKey} onChange={setYKey} options={keys} />
      </div>
      <div className="h-64 sm:h-72">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="#0f172a" />
            <XAxis dataKey={xKey} tick={{ fill: "#94a3b8", fontSize: 12 }} />
            <YAxis dataKey={yKey} tick={{ fill: "#94a3b8", fontSize: 12 }} />
            <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937" }} />
            <Legend />
            <Scatter name="data" data={data} fill="#60a5fa" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Select({ label, value, onChange, options }) {
  return (
    <label className="text-sm text-slate-300">
      <span className="mr-2">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="bg-slate-900 border border-slate-800 rounded-xl px-2 py-1 text-slate-100"
      >
        {options.map((k) => (
          <option key={k} value={k}>
            {k}
          </option>
        ))}
      </select>
    </label>
  );
}

function MiniDial({ label, value }) {
  const pct = Math.max(0, Math.min(100, value));
  return (
    <div className="flex items-center gap-4">
      <div className="relative w-28 h-28">
        <svg viewBox="0 0 100 100" className="w-full h-full">
          <circle cx="50" cy="50" r="40" fill="none" stroke="#111827" strokeWidth="12" />
          <circle
            cx="50" cy="50" r="40" fill="none" stroke="#4f46e5" strokeWidth="12"
            strokeDasharray={`${(pct / 100) * 2 * Math.PI * 40} ${2 * Math.PI * 40}`}
            transform="rotate(-90 50 50)"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <div className="text-xl font-semibold text-white">{pct}%</div>
            <div className="text-[10px] text-slate-400 uppercase tracking-wider">{label}</div>
          </div>
        </div>
      </div>
      <div className="text-slate-300 text-sm max-w-[14rem]">
        Higher is ‘flatter’. This is a toy score; tie to your own α‑CI checker if needed.
      </div>
    </div>
  );
}

function ScalingForecaster({ c }) {
  const [Dtarget, setDtarget] = useState(1 << 18);
  const varY = c / Dtarget;
  return (
    <div className="space-y-2 text-sm">
      <div className="flex items-center gap-2">
        <input
          type="number"
          className="bg-slate-900 border border-slate-800 rounded-xl px-2 py-1 w-40"
          value={Dtarget}
          onChange={(e) => setDtarget(Math.max(2, parseInt(e.target.value || "2", 10)))}
        />
        <span className="text-slate-400">D (effective dimension)</span>
      </div>
      <div className="text-slate-300">Predicted Var(Y) ≈ <span className="text-white">{fmt(varY)}</span></div>
    </div>
  );
}

function CIWidget({ data }) {
  if (!data?.length) return <div className="text-slate-400 text-sm">No data.</div>;
  const last = data[data.length - 1];
  // Toy 95% CI assuming known sigma ~ k/sqrt(D) with k estimated from sample std * sqrt(D)
  const sd = stdev(data.map((d) => d.signedAlpha));
  const kHat = sd * Math.sqrt(data[data.length - 1].D);
  const sigma = kHat / Math.sqrt(last.D);
  const ci = 1.96 * sigma;
  const includesZero = Math.abs(last.signedAlpha) <= ci;
  return (
    <div className="text-sm text-slate-300">
      <div>Largest D = <span className="text-white">{Math.round(last.D)}</span></div>
      <div>signed α = <span className="text-white">{fmt(last.signedAlpha)}</span></div>
      <div>±95% ≈ <span className="text-white">{fmt(ci)}</span></div>
      <div className={`mt-1 font-medium ${includesZero ? "text-emerald-400" : "text-rose-400"}`}>
        {includesZero ? "0 ∈ CI (flat)" : "0 ∉ CI (non‑flat)"}
      </div>
    </div>
  );
}

function stdev(arr) {
  if (!arr.length) return NaN;
  const m = arr.reduce((a, b) => a + b, 0) / arr.length;
  const v = arr.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, arr.length - 1);
  return Math.sqrt(v);
}
