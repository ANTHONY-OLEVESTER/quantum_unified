import React, { useEffect, useMemo, useState } from "react";
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
  const n = Math.min(x.length, y.length);
  if (n < 2) return NaN;
  const xs = x.slice(0, n);
  const ys = y.slice(0, n);
  const xbar = xs.reduce((a, b) => a + b, 0) / n;
  const ybar = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i += 1) {
    num += (xs[i] - xbar) * (ys[i] - ybar);
    den += (xs[i] - xbar) ** 2;
  }
  return den === 0 ? NaN : num / den;
}

function parseCSVSimple(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) return [];
  const header = lines[0].split(/[;,]/).map((s) => s.trim());
  return lines.slice(1).map((line) => {
    const cols = line.split(/[;,]/).map((s) => s.trim());
    const row = {};
    header.forEach((h, idx) => {
      const value = cols[idx];
      const num = Number(value);
      row[h || `col${idx}`] = Number.isFinite(num) ? num : value;
    });
    return row;
  });
}

function downloadBlob(filename, text) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function simulateVarianceScaling({ Dmin, Dmax, points, c, noise, seed }) {
  const rng = mulberry32(seed || 42);
  const Ds = logspace(Dmin, Dmax, points);
  const data = Ds.map((D) => {
    const mean = c / D;
    const jitter = (rng() - 0.5) * 2 * noise * mean;
    const value = Math.max(1e-12, mean + jitter);
    return { D, VarY: value, logD: Math.log(D), logVarY: Math.log(value) };
  });
  const slope = linregSlope(
    data.map((d) => d.logD),
    data.map((d) => d.logVarY)
  );
  return { data, slope };
}

function simulateAlphaFlatness({ Dmin, Dmax, points, k, seed }) {
  const rng = mulberry32(seed || 7);
  const Ds = logspace(Dmin, Dmax, points);
  const data = Ds.map((D) => {
    const sigma = k / Math.sqrt(D);
    const signed = gaussian(rng) * sigma;
    return { D, invD: 1 / D, signedAlpha: signed, absAlpha: Math.abs(signed) };
  });
  const slope = linregSlope(
    data.map((d) => d.invD),
    data.map((d) => d.signedAlpha)
  );
  const xs = data.map((d) => d.invD);
  const ys = data.map((d) => d.signedAlpha);
  const n = xs.length;
  const xbar = xs.reduce((a, b) => a + b, 0) / n;
  const ybar = ys.reduce((a, b) => a + b, 0) / n;
  const intercept = ybar - slope * xbar;
  return { data, slope, intercept };
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return function rng() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function logspace(a, b, n) {
  if (n === 1) return [2 ** a];
  const out = [];
  for (let i = 0; i < n; i += 1) {
    const t = i / (n - 1);
    const pow = a * (1 - t) + b * t;
    out.push(2 ** pow);
  }
  return out;
}

// ---------------------------------------------
// Simulation mode entry point
// ---------------------------------------------
export default function SimulationMode({ onBackToStory }) {
  const [varCfg, setVarCfg] = useState({ Dmin: 6, Dmax: 20, points: 10, c: 1.0, noise: 0.05, seed: 1234 });
  const [alpCfg, setAlpCfg] = useState({ Dmin: 6, Dmax: 20, points: 10, k: 1.0, seed: 999 });
  const [csvRows, setCsvRows] = useState([]);
  const [csvFields, setCsvFields] = useState([]);

  const varSim = useMemo(() => simulateVarianceScaling(varCfg), [varCfg]);
  const alpSim = useMemo(() => simulateAlphaFlatness(alpCfg), [alpCfg]);

  useEffect(() => {
    document.title = "Curvature–Information • Simulation Mode";
  }, []);

  function handleCSV(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const rows = parseCSVSimple(String(reader.result || ""));
      setCsvRows(rows);
      setCsvFields(rows.length ? Object.keys(rows[0]) : []);
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

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      <header className="sticky top-0 z-50 bg-slate-950/80 backdrop-blur border-b border-slate-900">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />
            <span className="font-semibold tracking-wide">Curvature–Information</span>
            <Pill>Simulation Mode</Pill>
          </div>
          <nav className="hidden sm:flex items-center gap-6 text-sm">
            <a href="#abstract" className="hover:text-white">Abstract</a>
            <a href="#results" className="hover:text-white">Results</a>
            <a href="#reproduce" className="hover:text-white">Reproduce</a>
            <a href="#applications" className="hover:text-white">Applications</a>
            <a href="#experiment" className="hover:text-white">Experiment</a>
            <a href="#references" className="hover:text-white">References</a>
          </nav>
          <button
            className="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-100 text-sm"
            onClick={onBackToStory}
          >
            Back to Story
          </button>
        </div>
      </header>

      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(79,70,229,0.15),transparent_55%)]" />
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-14">
          <div className="grid md:grid-cols-2 gap-10 items-center">
            <div>
              <motion.h1
                initial={{ y: 10, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ duration: 0.6 }}
                className="text-3xl sm:text-5xl font-extrabold text-white leading-tight"
              >
                Numerical Universality Lab
              </motion.h1>
              <p className="mt-4 text-slate-300 text-base sm:text-lg leading-relaxed">
                Probe the curvature–information invariant with synthetic sweeps. Fit slopes, examine signed α flatness,
                and plug in your Phase CSVs. This mode mirrors the analysis pipeline used for the paper.
              </p>
              <div className="mt-6 flex flex-wrap gap-3">
                <Pill>Variance collapse</Pill>
                <Pill>Signed-α flatness</Pill>
                <Pill>CSV ingest</Pill>
              </div>
            </div>
            <motion.div
              initial={{ scale: 0.96, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.15 }}
              className="rounded-3xl border border-slate-800 bg-slate-900/60 p-5"
            >
              <div className="text-sm text-slate-300 leading-relaxed space-y-2">
                <p>
                  <span className="font-semibold text-white">Quick claims:</span> E[Y] ≈ constant, Var(Y) ∝ D⁻¹, and signed α → 0 after isotropisation.
                </p>
                <p className="text-slate-400 text-xs">
                  Use the controls to sweep different D ranges, sample counts, and noise levels. Reroll seeds to sanity-check robustness.
                </p>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      <Section
        id="abstract"
        title="Abstract recap"
        subtitle="Geometry (A²) meets information (I) and effective dimension (d_eff)."
      >
        <div className="grid md:grid-cols-3 gap-4">
          <Card title="Invariant">
            <p className="text-slate-300 text-sm">
              Y = sqrt(d_eff − 1) · A² / I. Perturbations read out Bures/Fisher geometry while I captures the correlations the channel creates.
            </p>
          </Card>
          <Card title="Flatness">
            <p className="text-slate-300 text-sm">
              On log–log axes of Y vs (d_eff − 1), the slope α concentrates near 0 with width ≈ D⁻¹ᐟ² under 2-design dynamics.
            </p>
          </Card>
          <Card title="Variance law">
            <p className="text-slate-300 text-sm">
              Finite-size variance falls ∝ 1/D. In log space this is a slope of −1. Twirling structured dynamics restores the law.
            </p>
          </Card>
        </div>
      </Section>

      <Section id="results" title="Interactive diagnostics" subtitle="Tweak parameters and watch the collapse.">
        <div className="grid lg:grid-cols-2 gap-6">
          <Card title="Variance scaling" right={<Pill>slope ≈ {fmt(varSim.slope)}</Pill>}>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-3 text-sm">
                <Tuner label="D range (powers of 2)">
                  <RangeRow label="min pow" value={varCfg.Dmin} min={4} max={16} step={1} onChange={(v) => setVarCfg({ ...varCfg, Dmin: v })} />
                  <RangeRow label="max pow" value={varCfg.Dmax} min={varCfg.Dmin + 1} max={22} step={1} onChange={(v) => setVarCfg({ ...varCfg, Dmax: v })} />
                </Tuner>
                <Tuner label="Sampling">
                  <RangeRow label="#points" value={varCfg.points} min={5} max={24} step={1} onChange={(v) => setVarCfg({ ...varCfg, points: v })} />
                  <RangeRow label="scale c" value={varCfg.c} min={0.05} max={5} step={0.05} onChange={(v) => setVarCfg({ ...varCfg, c: v })} />
                  <RangeRow label="noise" value={varCfg.noise} min={0} max={0.5} step={0.01} onChange={(v) => setVarCfg({ ...varCfg, noise: v })} />
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
                <p className="text-slate-400 text-xs">Slope −1 ⇒ Var(Y) ∝ 1/D.</p>
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

          <Card
            title="α flatness"
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
                  <RangeRow label="min pow" value={alpCfg.Dmin} min={4} max={16} step={1} onChange={(v) => setAlpCfg({ ...alpCfg, Dmin: v })} />
                  <RangeRow label="max pow" value={alpCfg.Dmax} min={alpCfg.Dmin + 1} max={22} step={1} onChange={(v) => setAlpCfg({ ...alpCfg, Dmax: v })} />
                </Tuner>
                <Tuner label="Concentration">
                  <RangeRow label="k scale" value={alpCfg.k} min={0.2} max={5} step={0.1} onChange={(v) => setAlpCfg({ ...alpCfg, k: v })} />
                </Tuner>
                <button
                  className="px-3 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-sm"
                  onClick={() => setAlpCfg({ ...alpCfg, seed: Math.floor(Math.random() * 1e9) })}
                >
                  Reroll seed
                </button>
                <p className="text-slate-400 text-xs">Intercept ≈ 0 ⇒ signed α averages to 0.</p>
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

        <div className="mt-6">
          <Card title="Structured vs twirled dynamics">
            <TwirlingDemo />
          </Card>
        </div>
      </Section>

      <Section id="reproduce" title="Bring your own data" subtitle="Drop any Phase CSV and explore.">
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
            <p className="text-xs text-slate-400">Popular pairs: (D, VarY), (invD, alpha), (logD, logVarY).</p>
          </div>
          <div className="md:col-span-2">
            <CSVPlot rows={csvRows} />
          </div>
        </div>
      </Section>

      <Section
        id="applications"
        title="Applications"
        subtitle="Quick widgets that tie the invariant to experimental intuition."
      >
        <div className="grid md:grid-cols-3 gap-6">
          <Card title="Model audit">
            <p className="text-slate-300 text-sm mb-3">Structured channels drift; twirling pulls them back. Treat the score as a toy RB-style indicator.</p>
            <MiniDial label="Flatness" value={87} />
          </Card>
          <Card title="Scaling forecaster">
            <p className="text-slate-300 text-sm mb-3">Predict Var(Y) at larger D assuming slope −1 and current c.</p>
            <ScalingForecaster c={varCfg.c} />
          </Card>
          <Card title="CI checker">
            <p className="text-slate-300 text-sm mb-3">Check if zero still sits inside the signed α CI at your largest D.</p>
            <CIWidget data={alpSim.data} />
          </Card>
        </div>
      </Section>

      <Section
        id="experiment"
        title="Minimal experimental protocol"
        subtitle="Random local Cliffords, tomography, and a little patience."
      >
        <ol className="list-decimal ml-6 space-y-2 text-slate-300 text-sm">
          <li>Implement a noisy channel on the target qubit; conjugate by random low-depth Cliffords.</li>
          <li>Tomograph ρₛ before/after; compute A² via fidelity, I = 2 S(ρ′ₛ), and d_eff.</li>
          <li>Increase effective dimension via ancillas or mixing depth; collect (D, Y) points.</li>
          <li>Plot log Var(Y) vs log D → slope ≈ −1; regress α vs 1/D → intercept ≈ 0.</li>
        </ol>
      </Section>

      <Section id="references" title="References" subtitle="Pointers to the underlying maths.">
        <ul className="list-disc ml-6 space-y-2 text-slate-300 text-sm">
          <li>Page, D. N. (1993). Average entropy of a subsystem. Phys. Rev. Lett. 71, 1291.</li>
          <li>Collins, B., Śniady, P. (2006). Integration over U(d). Comm. Math. Phys. 264, 773.</li>
          <li>Bures, D. (1969). An extension of Kakutani’s theorem on product measures. Trans. AMS 135, 199.</li>
          <li>Uhlmann, A. (1976). The transition probability in state space of a *-algebra. Rep. Math. Phys. 9, 273.</li>
        </ul>
      </Section>

      <footer className="border-t border-slate-900 mt-10">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6 text-xs text-slate-500">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
            <span>Curvature–Information • Simulation toolkit</span>
            <div className="flex items-center gap-3">
              <a className="hover:text-slate-300" href="#abstract">Abstract</a>
              <a className="hover:text-slate-300" href="#results">Results</a>
              <a className="hover:text-slate-300" href="#reproduce">Reproduce</a>
            </div>
          </div>
          <div className="text-xs text-slate-600">
            Curvature–Information Principle © {new Date().getFullYear()} Anthony Olevester
            <span className="mx-2">•</span>
            DOI: <a href="https://doi.org/10.5281/zenodo.17497059" target="_blank" rel="noopener noreferrer" className="text-indigo-400 hover:text-indigo-300 underline">10.5281/zenodo.17497059</a>
          </div>
        </div>
      </footer>
    </div>
  );
}

// ---------------------------------------------
// Shared UI bits
// ---------------------------------------------
function Pill({ children }) {
  return (
    <span className="px-3 py-1 rounded-full bg-slate-800/70 border border-slate-700 text-slate-200 text-xs">
      {children}
    </span>
  );
}

function Section({ id, title, subtitle, children }) {
  return (
    <section id={id} className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="mb-6">
        <h2 className="text-2xl sm:text-3xl font-semibold text-slate-100">{title}</h2>
        {subtitle && <p className="text-slate-300 mt-2 max-w-3xl leading-relaxed">{subtitle}</p>}
      </div>
      <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-4 sm:p-6 shadow-xl">
        {children}
      </div>
    </section>
  );
}

function Card({ title, right, children }) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5">
      <div className="flex items-center justify-between gap-4 mb-3">
        <h3 className="text-lg font-medium text-slate-100">{title}</h3>
        {right}
      </div>
      {children}
    </div>
  );
}

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
  const data = useMemo(() => {
    const rng = mulberry32(321);
    const Ds = [64, 128, 256, 512, 1024, 2048, 4096];
    const bias = mode === "structured" ? 0.15 : 0;
    const scale = mode === "structured" ? 2.0 : mode === "twirled" ? 0.6 : 1.0;
    return Ds.map((D) => ({
      D,
      invD: 1 / D,
      signedAlpha: gaussian(rng) * (scale / Math.sqrt(D)) + bias,
    }));
  }, [mode]);

  const slope = useMemo(() => linregSlope(
    data.map((d) => d.invD),
    data.map((d) => d.signedAlpha)
  ), [data]);

  const intercept = useMemo(() => {
    const xs = data.map((d) => d.invD);
    const ys = data.map((d) => d.signedAlpha);
    const n = xs.length;
    const xbar = xs.reduce((a, b) => a + b, 0) / n;
    const ybar = ys.reduce((a, b) => a + b, 0) / n;
    return ybar - slope * xbar;
  }, [data, slope]);

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
            onClick={() => setMode("chaotic")}
            className={`px-3 py-2 rounded-xl text-sm ${mode === "chaotic" ? "bg-indigo-600 text-white" : "bg-slate-800 text-slate-100"}`}
          >
            Chaotic
          </button>
          <button
            onClick={() => setMode("twirled")}
            className={`px-3 py-2 rounded-xl text-sm ${mode === "twirled" ? "bg-emerald-600 text-white" : "bg-slate-800 text-slate-100"}`}
          >
            Twirled
          </button>
        </div>
        <p className="text-slate-300">Observe how twirling wipes the bias and tightens concentration.</p>
        <div className="text-xs text-slate-400 space-y-1">
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
            <Scatter name="signed α" data={data} fill={mode === "twirled" ? "#34d399" : mode === "chaotic" ? "#60a5fa" : "#fb7185"} />
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
  }, [keys, xKey, yKey]);

  const data = rows.map((row) => ({ ...row }));

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
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
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
            cx="50"
            cy="50"
            r="40"
            fill="none"
            stroke="#4f46e5"
            strokeWidth="12"
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
        Higher is flatter. Treat as a toy diagnostic.
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
      <div className="text-slate-300">
        Predicted Var(Y) ≈ <span className="text-white">{fmt(varY)}</span>
      </div>
    </div>
  );
}

function CIWidget({ data }) {
  if (!data.length) return <div className="text-slate-400 text-sm">No data loaded.</div>;
  const last = data[data.length - 1];
  const sd = stdev(data.map((d) => d.signedAlpha));
  const kHat = sd * Math.sqrt(last.D);
  const sigma = kHat / Math.sqrt(last.D);
  const ci = 1.96 * sigma;
  const flat = Math.abs(last.signedAlpha) <= ci;
  return (
    <div className="text-sm text-slate-300 space-y-1">
      <div>Largest D = <span className="text-white">{Math.round(last.D)}</span></div>
      <div>signed α = <span className="text-white">{fmt(last.signedAlpha)}</span></div>
      <div>±95% ≈ <span className="text-white">{fmt(ci)}</span></div>
      <div className={`mt-1 font-medium ${flat ? "text-emerald-400" : "text-rose-400"}`}>
        {flat ? "0 ∈ CI (flat)" : "0 ∉ CI (non-flat)"}
      </div>
    </div>
  );
}

function stdev(values) {
  if (!values.length) return NaN;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / Math.max(1, values.length - 1);
  return Math.sqrt(variance);
}

