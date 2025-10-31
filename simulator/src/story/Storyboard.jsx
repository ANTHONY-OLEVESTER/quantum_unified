import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
    Label,
} from "recharts";

const ACT_LABELS = [
    "I. Genesis",
    "II. Flatness",
    "III. D^-1",
    "IV. Theorem",
    "V. Qubits",
    "VI. Finale",
    "VII. Legacy",
];

const powersOfTwo = (minPow = 6, maxPow = 16) =>
    Array.from({ length: maxPow - minPow + 1 }, (_, idx) => 2 ** (idx + minPow));

function linReg(x, y) {
    const n = x.length;
    const sx = x.reduce((acc, value) => acc + value, 0);
    const sy = y.reduce((acc, value) => acc + value, 0);
    const sxx = x.reduce((acc, value) => acc + value * value, 0);
    const sxy = x.reduce((acc, value, idx) => acc + value * y[idx], 0);
    const denom = n * sxx - sx * sx;
    if (denom === 0) {
        return { m: 0, b: 0 };
    }
    const m = (n * sxy - sx * sy) / denom;
    const b = (sy - m * sx) / n;
    return { m, b };
}

function synthVarYSeries(dimensions, c = 0.8) {
    return dimensions.map((d) => {
        const mean = c / d;
        const noise = (Math.random() * 0.4 - 0.2) * mean;
        const value = Math.max(1e-8, mean + noise);
        return {
            D: d,
            VarY: value,
            logD: Math.log10(d),
            logVarY: Math.log10(value),
        };
    });
}

function slopeCI(series) {
    if (series.length < 3) {
        return { mean: 0, lo: 0, hi: 0 };
    }
    const x = series.map((point) => point.logD);
    const y = series.map((point) => point.logVarY);
    const { m } = linReg(x, y);
    const residuals = y.map((value, idx) => value - m * x[idx]);
    const variance =
        residuals.reduce((acc, value) => acc + value * value, 0) /
        Math.max(1, residuals.length - 2);
    const spread = Math.sqrt(Math.max(variance, 0));
    const halfWidth = Math.min(0.02 + spread * 0.1, 0.05);
    return {
        mean: m,
        lo: m - halfWidth,
        hi: m + halfWidth,
    };
}
export default function Storyboard() {
    const [scene, setScene] = useState(0);
    const [autoPlay, setAutoPlay] = useState(false);
    const [pow, setPow] = useState(7);

    useEffect(() => {
        if (!autoPlay) {
            return;
        }
        const id = setInterval(() => {
            setScene((prev) => (prev + 1) % ACT_LABELS.length);
        }, 6000);
        return () => clearInterval(id);
    }, [autoPlay]);

    const dimensions = useMemo(() => powersOfTwo(6, 16), []);
    const series = useMemo(() => synthVarYSeries(dimensions, 1.0), [dimensions]);
    const slope = useMemo(() => slopeCI(series), [series]);
    const currentD = 2 ** pow;

    return (
        <div className="min-h-screen w-full bg-black text-white font-sans">
            <Nav scene={scene} setScene={setScene} autoPlay={autoPlay} setAutoPlay={setAutoPlay} />
            <main className="max-w-6xl mx-auto px-4 pb-20">
                <Section visible={scene === 0}>
                    <ActI />
                </Section>
                <Section visible={scene === 1}>
                    <ActII />
                </Section>
                <Section visible={scene === 2}>
                    <ActIII pow={pow} setPow={setPow} series={series} slope={slope} currentD={currentD} />
                </Section>
                <Section visible={scene === 3}>
                  <ActIV />
                </Section>
                <Section visible={scene === 4}>
                    <ActV />
                </Section>
                <Section visible={scene === 5}>
                    <ActVI />
                </Section>
                <Section visible={scene === 6}>
                    <ActVII />
                </Section>
            </main>
            <Footer />
        </div>
    );
}

function Nav({ scene, setScene, autoPlay, setAutoPlay }) {
    return (
        <div className="sticky top-0 z-30 backdrop-blur bg-black/70 border-b border-white/10">
            <div className="max-w-6xl mx-auto px-4 py-3 flex flex-wrap items-center gap-3">
                <span className="text-lg font-semibold tracking-wide">Curvature Information Principle</span>
                <div className="hidden md:flex items-center gap-2 ml-6">
                    {ACT_LABELS.map((label, idx) => (
                        <button
                            key={label}
                            onClick={() => setScene(idx)}
                            className={`text-sm px-3 py-1 rounded-full transition ${
                                scene === idx ? "bg-white text-black" : "hover:bg-white/10"
                            }`}
                        >
                            {label}
                        </button>
                    ))}
                </div>
                <button
                    onClick={() => setAutoPlay((prev) => !prev)}
                    className={`ml-auto text-xs px-3 py-1.5 rounded-full border border-white/20 hover:bg-white/10 ${
                        autoPlay ? "bg-white/10" : ""
                    }`}
                >
                    {autoPlay ? "Auto: ON" : "Auto: OFF"}
                </button>
            </div>
        </div>
    );
}

function Section({ visible, children }) {
    return (
        <AnimatePresence mode="wait">
            {visible && (
                <motion.section
                    initial={{ opacity: 0, y: 24 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -24 }}
                    transition={{ duration: 0.6, ease: "easeOut" }}
                    className="min-h-[70vh] py-14"
                >
                    {children}
                </motion.section>
            )}
        </AnimatePresence>
    );
}

function Pill({ children }) {
    return <span className="px-2.5 py-1 rounded-full bg-white/10 text-xs mr-2">{children}</span>;
}

function DOIBadge() {
    return (
        <div className="mt-6 pt-4 border-t border-white/5 text-center text-xs text-white/40">
            DOI: <a href="https://doi.org/10.5281/zenodo.17497059" target="_blank" rel="noopener noreferrer" className="text-sky-400/60 hover:text-sky-300 underline">10.5281/zenodo.17497059</a>
        </div>
    );
}

function ActI() {
    const [angle, setAngle] = useState(0);
    const vo = useVoiceOver(VO_SCRIPTS[0], { autoStart:true, rate: 1.0 });
    useEffect(() => {
        const id = setInterval(() => {
            setAngle((prev) => (prev + 6) % 360);
        }, 60);
        return () => clearInterval(id);
    }, []);

    const orbitCount = 14;
    const sparkCount = 48;

    return (
        <div className="grid md:grid-cols-2 gap-8 items-center">
            <div className="order-2 md:order-1">
                <h1 className="text-4xl md:text-5xl font-black leading-tight">
                    When <span className="text-sky-300">geometry</span> bends, <span className="text-pink-300">information</span> flows.
                </h1>
                <p className="text-white/80 mt-5 text-lg max-w-prose">
                    At their intersection lives an invariant:
                    <span className="font-semibold"> Y = sqrt(d_eff - 1) * A^2 / I</span>.
                    This first act sets the tone: soft light, rotating curvature bands, and the law hovering at the center.
                </p>
                <div className="mt-6 flex flex-wrap items-center gap-2">
                    <Pill>Curvature</Pill>
                    <Pill>Mutual Information</Pill>
                    <Pill>Bures Geometry</Pill>
                    <Pill>2-Designs</Pill>
                </div>
            </div>

            <div className="order-1 md:order-2 relative aspect-square w-full">
                <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-slate-950 to-black" />
                {Array.from({ length: orbitCount }).map((_, idx) => (
                    <motion.div
                        key={`orbit-${idx}`}
                        className="absolute inset-8 border border-cyan-300/20 rounded-full"
                        animate={{ rotate: idx * 18 + angle }}
                        transition={{ type: "tween", duration: 0.6, ease: "linear" }}
                    />
                ))}
                {Array.from({ length: sparkCount }).map((_, idx) => {
                    const theta = (idx / sparkCount) * 2 * Math.PI;
                    const radius = 0.35 + (idx % 5) * 0.08;
                    const x = 50 + Math.cos(theta + (angle * Math.PI) / 180) * radius * 50;
                    const y = 50 + Math.sin(theta + (angle * Math.PI) / 180) * radius * 50;
                    return (
                        <motion.div
                            key={`spark-${idx}`}
                            className="absolute w-1.5 h-1.5 rounded-full bg-pink-300/80"
                            style={{ top: `${y}%`, left: `${x}%` }}
                            animate={{ opacity: [0.4, 0.9, 0.4] }}
                            transition={{ duration: 3.2, repeat: Infinity, delay: idx * 0.05 }}
                        />
                    );
                })}
                <motion.div
                    className="pointer-events-none absolute inset-0 flex items-center justify-center"
                    animate={{ scale: [0.96, 1, 0.96], opacity: [0.8, 1, 0.8] }}
                    transition={{ duration: 5.4, repeat: Infinity }}
                >
                    <motion.div
                        className="rounded-full bg-white text-black px-6 py-3 text-sm sm:text-base font-semibold shadow-2xl border border-white/70"
                        animate={{ y: [0, -6, 0] }}
                        transition={{ duration: 3.2, repeat: Infinity }}
                    >
                        Y = sqrt(d_eff - 1) * A^2 / I
                    </motion.div>
                </motion.div>
            </div>
            <div className="md:col-span-2">
        <VoiceoverControls vo={vo} />
      </div>
        </div>
        <DOIBadge />
    );
}
function ActII() {
    const [phase, setPhase] = useState(0);
    const [holding, setHolding] = useState(false);
    const [caption, setCaption] = useState(0);
    const vo = useVoiceOver(VO_SCRIPTS[1], { autoStart:true, rate: 1.0 });

    useEffect(() => {
        let direction = 1;
        const id = setInterval(() => {
            setPhase((value) => {
                let next = value + direction * 0.01;
                if (next >= 1) {
                    direction = -1;
                    next = 1;
                }
                if (next <= 0) {
                    direction = 1;
                    next = 0;
                }
                return next;
            });
        }, 30);
        return () => clearInterval(id);
    }, []);

    useEffect(() => {
        const id = setInterval(() => setCaption((value) => (value + 1) % 4), 3200);
        return () => clearInterval(id);
    }, []);

    const alphaCloud = useMemo(() => {
        const bias = 0.8 * (1 - phase);
        const spread = 0.6 * (1 - phase) + 0.35 * phase;
        const twirlSpread = holding ? 0.22 : spread;
        const count = 700;
        return Array.from({ length: count }, () => bias + (Math.random() * 2 - 1) * twirlSpread);
    }, [phase, holding]);

    const tone = useMemo(() => {
        const warm = { from: "from-amber-900/30", dot: "bg-amber-300" };
        const cool = { from: "from-cyan-900/30", dot: "bg-cyan-300" };
        return phase < 0.5 ? warm : cool;
    }, [phase]);

    const captions = [
        "Structure remembers where it came from.",
        "Chaos forgets - and so it forgives.",
        "Symmetry is what remains when bias lets go.",
        "Hold your breath - and twirl the world.",
    ];

    return (
        <div>
            <h2 className="text-3xl font-bold">Birth of Flatness</h2>
            <p className="text-white/75 mt-2 max-w-prose">
                The system breathes on its own. Inhale toward structure, exhale into chaos, and press the cloud flat.
            </p>

            <div className="mt-4 flex items-center gap-4">
                <label className="text-sm text-white/70">Breath / Phase</label>
                <input
                    type="range"
                    min={0}
                    max={100}
                    value={Math.round(phase * 100)}
                    onChange={(event) => setPhase(parseInt(event.target.value, 10) / 100)}
                    className="w-64"
                />
                <span className="text-sm">{phase.toFixed(2)}</span>
                <div className="text-xs text-white/60">Press and hold the plot to twirl</div>
            </div>

            <div className="mt-6 grid md:grid-cols-2 gap-8 items-center">
                <div
                    className={`relative h-80 rounded-2xl border border-white/10 overflow-hidden group bg-gradient-to-br ${tone.from} to-black`}
                    onPointerDown={() => setHolding(true)}
                    onPointerUp={() => setHolding(false)}
                    onPointerLeave={() => setHolding(false)}
                >
                    <AlphaScatter
                        alphaCloud={alphaCloud}
                        title={holding ? "Twirl engaged - alpha -> 0" : "Alpha distribution breathing"}
                    />
                    <AnimatePresence mode="wait">
                        <motion.div
                            key={caption}
                            initial={{ opacity: 0, y: 8 }}
                            animate={{ opacity: 0.9, y: 0 }}
                            exit={{ opacity: 0, y: -8 }}
                            transition={{ duration: 0.5 }}
                            className="absolute bottom-3 left-3 right-3 text-center text-sm text-white/80"
                        >
                            {captions[caption]}
                        </motion.div>
                    </AnimatePresence>
                    <motion.div
                        className="pointer-events-none absolute inset-0 flex items-center justify-center"
                        animate={{ scale: [1, 1.05, 1], opacity: [0.15, 0.25, 0.15] }}
                        transition={{ duration: 5.2, repeat: Infinity }}
                    >
                        <div className="w-40 h-40 rounded-full border border-white/30" />
                    </motion.div>
                </div>
                <div>
                    <ul className="list-disc list-inside text-white/80 space-y-2">
                        <li>
                            <span className="font-semibold">Breathing timeline:</span> a continuous morph from biased structure to centered chaos.
                        </li>
                        <li>
                            <span className="font-semibold">Press and hold:</span> applies a twirl so alpha collapses toward zero in real time.
                        </li>
                        <li>
                            <span className="font-semibold">Color language:</span> warm tones for memory, cool tones for isotropy.
                        </li>
                    </ul>
                    <p className="text-white/70 mt-3 text-sm">Release the hold and bias slowly returns. The world remembers.</p>
                </div>
            </div>
            <VoiceoverControls vo={vo} />
            <DOIBadge />
        </div>
    );
}

function AlphaScatter({ alphaCloud, title }) {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) {
            return;
        }
        const dpr = window.devicePixelRatio || 1;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
            return;
        }
        ctx.scale(dpr, dpr);
        ctx.fillStyle = "#0b0b12";
        ctx.fillRect(0, 0, width, height);

        ctx.strokeStyle = "rgba(255,255,255,0.12)";
        ctx.beginPath();
        ctx.moveTo(40, height / 2);
        ctx.lineTo(width - 10, height / 2);
        ctx.moveTo(40, 10);
        ctx.lineTo(40, height - 10);
        ctx.stroke();

        const min = -2;
        const max = 2;
        ctx.fillStyle = "rgba(220, 230, 255, 0.85)";
        for (let idx = 0; idx < alphaCloud.length; idx += 1) {
            const value = Math.max(min, Math.min(max, alphaCloud[idx]));
            const x = 40 + ((value - min) / (max - min)) * (width - 60);
            const y = height / 2 + (Math.random() * 2 - 1) * height * 0.2;
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, Math.PI * 2);
            ctx.fill();
        }

        ctx.strokeStyle = "rgba(255,255,255,0.6)";
        ctx.setLineDash([6, 6]);
        const zeroX = 40 + ((0 - min) / (max - min)) * (width - 60);
        ctx.beginPath();
        ctx.moveTo(zeroX, 10);
        ctx.lineTo(zeroX, height - 10);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "rgba(255,255,255,0.8)";
        ctx.font = "12px ui-sans-serif";
        ctx.fillText("alpha = 0", zeroX + 6, 20);
    }, [alphaCloud]);

    return (
        <div className="w-full h-full">
            <div className="px-4 pt-3 text-sm text-white/70">{title}</div>
            <canvas ref={canvasRef} className="w-full h-[calc(100%-2rem)]" />
        </div>
    );
}
function ActIII({ pow, setPow, series, slope, currentD }) {
    const [auto, setAuto] = useState(true);
    const [time, setTime] = useState(0);
    const [mic, setMic] = useState(false);
    const [rms, setRms] = useState(0);
    const vo = useVoiceOver(VO_SCRIPTS[2], { autoStart:true, rate: 1.0 });

    useEffect(() => {
        const id = setInterval(() => setTime((value) => value + 0.016), 16);
        return () => clearInterval(id);
    }, []);

    useEffect(() => {
        if (!mic) {
            return undefined;
        }
        let audioCtx;
        let source;
        let analyser;
        let data;
        let raf;
        navigator.mediaDevices
            .getUserMedia({ audio: true })
            .then((stream) => {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                source = audioCtx.createMediaStreamSource(stream);
                analyser = audioCtx.createAnalyser();
                analyser.fftSize = 512;
                data = new Uint8Array(analyser.frequencyBinCount);
                source.connect(analyser);
                const tick = () => {
                    analyser.getByteTimeDomainData(data);
                    let sum = 0;
                    for (let idx = 0; idx < data.length; idx += 1) {
                        const value = (data[idx] - 128) / 128;
                        sum += value * value;
                    }
                    const level = Math.min(1, Math.sqrt(sum / data.length) * 4);
                    setRms(level);
                    raf = requestAnimationFrame(tick);
                };
                raf = requestAnimationFrame(tick);
            })
            .catch(() => setMic(false));
        return () => {
            if (raf) {
                cancelAnimationFrame(raf);
            }
            if (analyser && source) {
                source.disconnect(analyser);
            }
            if (audioCtx) {
                audioCtx.close();
            }
        };
    }, [mic]);

    const effectivePow = useMemo(() => {
        const base = 6 + (Math.sin(time * 0.25) * 0.5 + 0.5) * 10;
        const boost = mic ? rms * 2 : 0;
        return Math.max(6, Math.min(16, auto ? base + boost : pow));
    }, [time, rms, mic, auto, pow]);

    const effectiveD = Math.round(2 ** effectivePow);
    const displaySeries = useMemo(
        () => series.map((point) => ({ ...point, label: `D=${point.D}` })),
        [series],
    );
    const slopeLabel = slope ? `${slope.mean.toFixed(3)} [${slope.lo.toFixed(3)}, ${slope.hi.toFixed(3)}]` : "n/a";

    return (
        <div>
            <div className="flex items-center gap-3">
                <h2 className="text-3xl font-bold">The D^-1 Law</h2>
                <span className="text-xs px-2 py-1 rounded-full bg-white/10">Hands-free mode</span>
            </div>
            <p className="text-white/75 mt-2 max-w-prose">
                The variance collapse runs on its own. Let the dimension grow, or take the controls and feel the slope settle near minus one.
            </p>

            <div className="mt-3 flex items-center gap-3 text-sm">
                <button onClick={() => setAuto((value) => !value)} className="px-3 py-1.5 rounded-full border border-white/20 hover:bg-white/10">
                    {auto ? "Auto: ON" : "Auto: OFF"}
                </button>
                <button
                    onClick={() => setMic((value) => !value)}
                    className={`px-3 py-1.5 rounded-full border border-white/20 hover:bg-white/10 ${mic ? "bg-white/10" : ""}`}
                >
                    {mic ? "Mic: ON" : "Mic: OFF"}
                </button>
                {!auto && (
                    <>
                        <label className="opacity-70">log2 D</label>
                        <input
                            type="range"
                            min={6}
                            max={16}
                            value={pow}
                            onChange={(event) => setPow(parseInt(event.target.value, 10))}
                            className="w-56"
                        />
                        <span>D = {currentD.toLocaleString()}</span>
                    </>
                )}
            </div>

            <div className="mt-6 relative h-80 w-full rounded-2xl overflow-hidden">
                <GravityGrid strength={(effectivePow - 6) / 10} />
                <div className="absolute inset-0 p-3">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={displaySeries} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.15)" />
                            <XAxis dataKey="logD" tick={{ fill: "#bbbbbb", fontSize: 12 }}>
                                <Label value="log10 D" offset={-10} position="insideBottom" fill="#bbbbbb" />
                            </XAxis>
                            <YAxis tick={{ fill: "#bbbbbb", fontSize: 12 }}>
                                <Label value="log10 Var(Y)" angle={-90} position="insideLeft" fill="#bbbbbb" />
                            </YAxis>
                            <Tooltip contentStyle={{ background: "#0b0b12", border: "1px solid rgba(255,255,255,0.1)", color: "#ffffff" }} />
                            <Line type="monotone" dataKey="logVarY" dot={false} strokeWidth={2} stroke="#8fd7ff" />
                            <ReferenceLine y={0} stroke="rgba(255,255,255,0.2)" />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="mt-3 text-sm text-white/80 flex items-center gap-4">
                <span>
                    Estimated slope beta approximately <span className="font-semibold">{slopeLabel}</span>
                </span>
                <span className="opacity-70">Effective D approximately {effectiveD.toLocaleString()}</span>
                {mic && <span className="opacity-60">rms {rms.toFixed(2)}</span>}
            </div>
             <VoiceoverControls vo={vo} />
             <DOIBadge />
        </div>
    );
}

function ActIV() {
  // Cinematic theorem assembly: orbiting glyphs -> bezier converge -> snap to equation
  const canvasRef = useRef(null);
  const [assembled, setAssembled] = useState(false);
  const vo = useVoiceOver(VO_SCRIPTS[3], { autoStart:true, rate: 1.0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    let raf = 0;

    // Responsive sizing
    const fit = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    fit();
    window.addEventListener("resize", fit);

    // Scene params
    let t = 0;
    const tokens = [
      { text: "Y",        w: 34 },
      { text: "=",        w: 24 },
      { text: "√",        w: 28 },
      { text: "(",        w: 20 },
      { text: "d_eff",    w: 56 },
      { text: "–",        w: 24 },
      { text: "1",        w: 20 },
      { text: ")",        w: 20 },
      { text: "·",        w: 18 },
      { text: "A²",       w: 34 },
      { text: "/",        w: 18 },
      { text: "I",        w: 18 },
    ];

    // Layout for final equation (centered)
    function computeTargets() {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      const cx = w / 2, cy = h / 2;
      const spacing = 6;
      const totalW =
        tokens.reduce((acc, tok) => acc + tok.w, 0) + spacing * (tokens.length - 1);
      let x = cx - totalW / 2;
      const y = cy + 4; // slight optical lift
      return tokens.map((tok) => {
        const pos = { x: x + tok.w / 2, y };
        x += tok.w + spacing;
        return pos;
      });
    }

    // Particle state
    const state = {
      parts: [],
      targets: computeTargets(),
      haloR: Math.min(canvas.clientWidth, canvas.clientHeight) * 0.33,
      cx: canvas.clientWidth / 2,
      cy: canvas.clientHeight / 2,
      assembling: false,
      phase: 0, // 0..1 assembly progress
    };

    // Initialize particles on a ring
    function resetParticles() {
      state.parts = tokens.map((tok, i) => {
        const ang = (i / tokens.length) * Math.PI * 2;
        const r = state.haloR;
        const sx = state.cx + Math.cos(ang) * r;
        const sy = state.cy + Math.sin(ang) * r;
        const tx = state.targets[i].x;
        const ty = state.targets[i].y;

        // Random bezier control points for nice arcs
        const jitter = () => (Math.random() * 2 - 1) * r * 0.35;
        const cp1 = { x: (sx + tx) / 2 + jitter(), y: (sy + ty) / 2 + jitter() };
        const cp2 = { x: (sx + tx) / 2 + jitter(), y: (sy + ty) / 2 + jitter() };

        return {
          text: tok.text,
          sx, sy,
          tx, ty,
          cp1, cp2,
          theta: ang,
          orbitSpeed: 0.22 + Math.random() * 0.15,
          snapFlash: 0, // 0..1
        };
      });
    }

    // Bezier interpolation + easing
    const ease = (x) => 1 - Math.pow(1 - x, 3); // cubic ease-out
    const bezier = (p0, p1, p2, p3, u) => {
      const v = 1 - u;
      return {
        x: v*v*v*p0.x + 3*v*v*u*p1.x + 3*v*u*u*p2.x + u*u*u*p3.x,
        y: v*v*v*p0.y + 3*v*v*u*p1.y + 3*v*u*u*p2.y + u*u*u*p3.y,
      };
    };

    resetParticles();

    // Starfield for depth
    const stars = Array.from({ length: 160 }).map(() => ({
      x: Math.random() * canvas.clientWidth,
      y: Math.random() * canvas.clientHeight,
      a: 0.3 + Math.random() * 0.6,
      r: Math.random() * 1.2 + 0.4,
      s: Math.random() * 0.3 + 0.05,
    }));

    function drawBG() {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      // gradient space
      const g = ctx.createLinearGradient(0, 0, w, h);
      g.addColorStop(0, "#0b0b14");
      g.addColorStop(1, "#120b14");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, h);

      // subtle vignetting
      ctx.fillStyle = "rgba(0,0,0,0.25)";
      ctx.beginPath();
      ctx.roundRect(0, 0, w, h, 24);
      ctx.fill();

      // starfield
      stars.forEach((s) => {
        const tw = 0.5 + 0.5 * Math.sin(t * s.s + s.x * 0.02);
        ctx.globalAlpha = s.a * tw;
        ctx.fillStyle = "#9ad0ff";
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.globalAlpha = 1;
    }

    function drawHalo() {
      ctx.strokeStyle = "rgba(255,255,255,0.10)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(state.cx, state.cy, state.haloR, 0, Math.PI * 2);
      ctx.stroke();
    }

    function drawTokens() {
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      state.parts.forEach((p, i) => {
        let x, y;
        if (state.assembling) {
          const u = ease(state.phase);
          const pos = bezier(
            { x: p.sx, y: p.sy },
            p.cp1,
            p.cp2,
            { x: p.tx, y: p.ty },
            u
          );
          x = pos.x; y = pos.y;

          // When nearly snapped, add flash
          if (state.phase > 0.96 && p.snapFlash < 1) {
            p.snapFlash = Math.min(1, p.snapFlash + 0.1);
          } else if (state.phase < 0.9 && p.snapFlash > 0) {
            p.snapFlash = Math.max(0, p.snapFlash - 0.08);
          }
        } else {
          // Free orbit
          p.theta += p.orbitSpeed * 0.016;
          const r = state.haloR * (0.95 + 0.05 * Math.sin(t + i));
          x = state.cx + Math.cos(p.theta) * r;
          y = state.cy + Math.sin(p.theta) * r;
          // relax flash
          p.snapFlash = Math.max(0, p.snapFlash - 0.04);
        }

        // trail (motion blur)
        ctx.strokeStyle = "rgba(150,230,255,0.08)";
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.moveTo(x - 6, y);
        ctx.lineTo(x + 6, y);
        ctx.stroke();

        // glow
        ctx.shadowColor = "rgba(120,255,220,0.55)";
        ctx.shadowBlur = 12;

        // token
        ctx.fillStyle = "rgba(200,255,240,0.95)";
        ctx.font = "bold 28px ui-sans-serif";
        if (p.text === "d_eff") ctx.font = "bold 24px ui-sans-serif";
        if (p.text === "A²") ctx.font = "bold 26px ui-sans-serif";
        ctx.fillText(p.text, x, y);

        // snap flash ring
        if (p.snapFlash > 0) {
          ctx.shadowBlur = 0;
          ctx.strokeStyle = `rgba(180,255,230,${0.35 * p.snapFlash})`;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, 14 + 10 * p.snapFlash, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.shadowBlur = 0;
      });
    }

    function drawEquationBaseline() {
      if (!state.assembling || state.phase < 0.98) return;
      ctx.fillStyle = "rgba(255,255,255,0.92)";
      ctx.font = "700 22px ui-sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Y = √(d_eff − 1) · A² / I", state.cx, state.cy + 64);
    }

    function step() {
      t += 1;
      const w = canvas.clientWidth, h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);

      drawBG();
      drawHalo();

      // progress change
      const target = assembled ? 1 : 0;
      state.phase += (target - state.phase) * 0.06;
      state.assembling = state.phase > 0.01;

      drawTokens();
      drawEquationBaseline();

      raf = requestAnimationFrame(step);
    }
    raf = requestAnimationFrame(step);

    // Toggle handler from outside state changes
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", fit);
    };
  }, [assembled]);

  return (
    <div className="text-center">
      <h2 className="text-3xl font-bold">The Theorem</h2>
      <p className="text-white/75 mt-2">
        Geometry, entropy, and dimension don’t just add up—&nbsp;
        <span className="text-teal-200">they converge</span>.
      </p>

      <div className="mt-6 relative mx-auto max-w-4xl aspect-video rounded-3xl overflow-hidden border border-white/10">
        <canvas ref={canvasRef} className="w-full h-full" />
      </div>

      <button
        onClick={() => setAssembled(v => !v)}
        className="mt-5 px-5 py-2 rounded-full border border-white/20 hover:bg-white/10 text-sm"
      >
        {assembled ? "Disperse" : "Prove"}
      </button>
      <DOIBadge />
    </div>
  );
}


function GravityGrid({ strength = 0.5 }) {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) {
            return;
        }
        const dpr = window.devicePixelRatio || 1;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
            return;
        }
        ctx.scale(dpr, dpr);
        ctx.fillStyle = "rgba(10,10,18,1)";
        ctx.fillRect(0, 0, width, height);
        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.lineWidth = 1;

        const rows = 14;
        const cols = 22;
        for (let i = 0; i <= rows; i += 1) {
            ctx.beginPath();
            for (let j = 0; j <= cols; j += 1) {
                const x = (j / cols) * width;
                const bend = Math.sin((x / width) * Math.PI * 2) * (1 - strength) * 14;
                const y = (i / rows) * height + bend;
                if (j === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
        }
        for (let j = 0; j <= cols; j += 1) {
            ctx.beginPath();
            for (let i = 0; i <= rows; i += 1) {
                const y = (i / rows) * height;
                const bend = Math.sin((y / height) * Math.PI * 2) * (1 - strength) * 14;
                const x = (j / cols) * width + bend;
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
        }
    }, [strength]);

    return <canvas ref={canvasRef} className="absolute inset-0" />;
}
function ActV() {
    const [depth, setDepth] = useState(1);
    const [alpha, setAlpha] = useState(0);
    const [varY, setVarY] = useState(0.4);
    const [mic, setMic] = useState(false);
    const [rms, setRms] = useState(0);
    const [time, setTime] = useState(0);
    const vo = useVoiceOver(VO_SCRIPTS[4], { autoStart:true, rate: 1.0 });

    useEffect(() => {
        const id = setInterval(() => setTime((value) => value + 0.016), 16);
        return () => clearInterval(id);
    }, []);

    useEffect(() => {
        if (!mic) {
            return undefined;
        }
        let audioCtx;
        let source;
        let analyser;
        let data;
        let raf;
        navigator.mediaDevices
            .getUserMedia({ audio: true })
            .then((stream) => {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                source = audioCtx.createMediaStreamSource(stream);
                analyser = audioCtx.createAnalyser();
                analyser.fftSize = 512;
                data = new Uint8Array(analyser.frequencyBinCount);
                source.connect(analyser);
                const tick = () => {
                    analyser.getByteTimeDomainData(data);
                    let sum = 0;
                    for (let idx = 0; idx < data.length; idx += 1) {
                        const value = (data[idx] - 128) / 128;
                        sum += value * value;
                    }
                    const level = Math.min(1, Math.sqrt(sum / data.length) * 4);
                    setRms(level);
                    raf = requestAnimationFrame(tick);
                };
                raf = requestAnimationFrame(tick);
            })
            .catch(() => setMic(false));
        return () => {
            if (raf) {
                cancelAnimationFrame(raf);
            }
            if (analyser && source) {
                source.disconnect(analyser);
            }
            if (audioCtx) {
                audioCtx.close();
            }
        };
    }, [mic]);

    useEffect(() => {
        const autoDepth = 1 + Math.round((Math.sin(time * 0.4) * 0.5 + 0.5) * 7);
        const micDepth = 1 + Math.round(rms * 7);
        const nextDepth = mic ? micDepth : autoDepth;
        setDepth(nextDepth);
    }, [time, rms, mic]);

    useEffect(() => {
        const nextAlpha = (Math.random() * 0.2 - 0.1) / Math.sqrt(depth);
        const nextVarY = Math.max(0.01, 0.6 / (depth * 2));
        setAlpha(nextAlpha);
        setVarY(nextVarY);
    }, [depth]);

    return (
        <div>
            <div className="flex items-center gap-3">
                <h2 className="text-3xl font-bold">From Paper to Qubits</h2>
                <span className="text-xs px-2 py-1 rounded-full bg-white/10">Breath-driven twirl</span>
                <button
                    onClick={() => setMic((value) => !value)}
                    className={`text-xs px-3 py-1.5 rounded-full border border-white/20 hover:bg-white/10 ${mic ? "bg-white/10" : ""}`}
                >
                    {mic ? "Mic: ON" : "Mic: OFF"}
                </button>
                {mic && <span className="text-xs opacity-70">rms {rms.toFixed(2)}</span>}
            </div>
            <p className="text-white/75 mt-2 max-w-prose">
                Blow softly to randomize the circuit. Louder means deeper twirls, flatter alpha, and smaller Var(Y). Without the mic it still breathes on its own.
            </p>

            <div className="mt-6 grid md:grid-cols-3 gap-6">
                <div className="rounded-2xl border border-white/10 p-4">
                    <h3 className="font-semibold">Chip</h3>
                    <OrbitingChip depth={depth} />
                </div>
                <div className="rounded-2xl border border-white/10 p-4">
                    <h3 className="font-semibold">Depth / Twirl</h3>
                    <div className="mt-2 text-sm">Effective depth: <span className="font-semibold">{depth}</span></div>
                    <div className="mt-2 text-sm">alpha approximately <span className="font-semibold">{alpha.toFixed(3)}</span></div>
                    <div className="text-sm">Var(Y) approximately <span className="font-semibold">{varY.toFixed(3)}</span></div>
                </div>
                <div className="rounded-2xl border border-white/10 p-4">
                    <h3 className="font-semibold">Readouts</h3>
                    <ul className="mt-2 text-white/80 text-sm space-y-1">
                        <li className={depth > 2 ? "text-green-200" : ""}>A^2 = arccos^2(sqrt(F(rho_S, rho'_S)))</li>
                        <li className={depth > 3 ? "text-green-200" : ""}>I = 2 * S(rho'_S)</li>
                        <li className={depth > 4 ? "text-green-200" : ""}>d_eff = 1 / Tr[(rho'_S)^2]</li>
                        <li className={depth > 5 ? "text-green-200" : ""}>Y = sqrt(d_eff - 1) * A^2 / I</li>
                    </ul>
                </div>
            </div>
             <VoiceoverControls vo={vo} />
             <DOIBadge />
        </div>
    );
}

function OrbitingChip({ depth }) {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) {
            return;
        }
        const ctx = canvas.getContext("2d");
        if (!ctx) {
            return;
        }
        const dpr = window.devicePixelRatio || 1;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        ctx.scale(dpr, dpr);
        let t = 0;
        let raf;
        const draw = () => {
            t += 0.016;
            ctx.clearRect(0, 0, width, height);
            ctx.fillStyle = "#0b1220";
            ctx.fillRect(0, 0, width, height);
            const cx = width / 2;
            const cy = height / 2;
            const baseR = Math.min(width, height) / 4;
            ctx.strokeStyle = "rgba(255,255,255,0.5)";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(cx, cy, baseR, 0, Math.PI * 2);
            ctx.stroke();
            const nodes = 18;
            const damping = Math.min(0.95, 0.8 + depth * 0.03);
            for (let i = 0; i < nodes; i += 1) {
                const angle = (i / nodes) * Math.PI * 2 + t * (0.4 + i * 0.003);
                const amplitude = (1 - damping) * baseR * 0.8;
                const radius = baseR + Math.sin(angle * 3 + i) * amplitude;
                const x = cx + Math.cos(angle) * radius;
                const y = cy + Math.sin(angle) * radius;
                ctx.fillStyle = `rgba(255,255,255,${0.7 - (i / nodes) * 0.4})`;
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, Math.PI * 2);
                ctx.fill();
            }
            raf = requestAnimationFrame(draw);
        };
        raf = requestAnimationFrame(draw);
        return () => cancelAnimationFrame(raf);
    }, [depth]);

    return (
        <div className="mt-2 aspect-video rounded-xl overflow-hidden">
            <canvas ref={canvasRef} className="w-full h-full" />
        </div>
    );
}

function ActVI() {
    const [rewind, setRewind] = useState(false);
    const canvasRef = useRef(null);
    const [time, setTime] = useState(0);
    const vo = useVoiceOver(VO_SCRIPTS[5], { autoStart:true, rate: 1.0 });

    useEffect(() => {
        const id = setInterval(() => setTime((value) => value + 0.016), 16);
        return () => clearInterval(id);
    }, []);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) {
            return;
        }
        const ctx = canvas.getContext("2d");
        if (!ctx) {
            return;
        }
        const dpr = window.devicePixelRatio || 1;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        ctx.scale(dpr, dpr);
        const particles = Array.from({ length: 420 }, (_, idx) => ({ seed: idx, theta: (idx / 420) * Math.PI * 2 }));
        let localTime = 0;
        let raf;
        const draw = () => {
            localTime += rewind ? -0.016 : 0.016;
            ctx.fillStyle = "rgba(16,12,24,1)";
            ctx.fillRect(0, 0, width, height);
            const centerX = width / 2;
            const centerY = height / 2;
            const baseR = Math.min(width, height) / 3.2;
            const beat = (Math.sin(localTime * 2) + 1) / 2;
            const radius = baseR * (0.95 + beat * 0.05);
            particles.forEach((p) => {
                const wobble = Math.sin(localTime * 0.7 + p.seed) * (0.18 - 0.16 * Math.min(1, Math.abs(localTime) / 6));
                const r = radius * (1 + wobble);
                const x = centerX + Math.cos(p.theta + localTime * 0.1 + p.seed * 0.01) * r;
                const y = centerY + Math.sin(p.theta + localTime * 0.1 + p.seed * 0.01) * r;
                ctx.fillStyle = `rgba(255,255,255,${0.35 + 0.25 * Math.sin(p.seed)})`;
                ctx.beginPath();
                ctx.arc(x, y, 1.6, 0, Math.PI * 2);
                ctx.fill();
            });
            ctx.strokeStyle = "rgba(255,255,255,0.7)";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
            ctx.stroke();
            raf = requestAnimationFrame(draw);
        };
        raf = requestAnimationFrame(draw);
        return () => cancelAnimationFrame(raf);
    }, [rewind]);

    return (
        <div className="text-center">
            <h2 className="text-3xl font-bold">Universality Unveiled</h2>
            <p className="text-white/75 mt-2">From chaos to flatness, the universe preserves its curvature through information.</p>
            <div className="mt-8 relative mx-auto max-w-3xl aspect-video rounded-3xl overflow-hidden border border-white/10">
                <canvas ref={canvasRef} className="w-full h-full" />
                <div className="absolute bottom-4 w-full text-center text-white/80">Curvature Information Principle</div>
            </div>
            <button onClick={() => setRewind((value) => !value)} className="mt-6 text-sm px-4 py-2 rounded-full border border-white/20 hover:bg-white/10">
                {rewind ? "Play Forward" : "Rewind"}
            </button>
              <VoiceoverControls vo={vo} />
              <DOIBadge />
        </div>
    );
}

function Footer() {
    return (
        <footer className="mt-8 border-t border-white/10">
            <div className="max-w-6xl mx-auto px-4 py-6 text-xs text-white/60">
                <div className="flex flex-wrap items-center gap-3 mb-2">
                    <span>© {new Date().getFullYear()} Curvature Information Demo</span>
                    <span className="hidden sm:inline">*</span>
                    <span>Display-only synthesis consistent with D^-1 concentration and alpha flatness narrative.</span>
                </div>
                <div className="text-xs text-white/50">
                    Curvature–Information Principle © {new Date().getFullYear()} Anthony Olevester
                    <span className="mx-2">•</span>
                    DOI: <a href="https://doi.org/10.5281/zenodo.17497059" target="_blank" rel="noopener noreferrer" className="text-sky-400 hover:text-sky-300 underline">10.5281/zenodo.17497059</a>
                </div>
            </div>
        </footer>
    );
}

function ActVII() {
  const canvasRef = useRef(null);
  const [mic,setMic]=useState(false);
  const [rms,setRms]=useState(0);
  const [time,setTime]=useState(0);
  const vo=useVoiceOver(VO_SCRIPTS[6],{autoStart:true,rate:1.0});

  useEffect(()=>{const id=setInterval(()=>setTime(t=>t+0.016),16);return()=>clearInterval(id);},[]);
  useEffect(()=>{
    if(!mic) return;
    let ctxA,src,an,data,raf;
    navigator.mediaDevices.getUserMedia({audio:true}).then(stream=>{
      ctxA=new(window.AudioContext||window.webkitAudioContext)();
      src=ctxA.createMediaStreamSource(stream);
      an=ctxA.createAnalyser();an.fftSize=256;data=new Uint8Array(an.frequencyBinCount);
      src.connect(an);
      const tick=()=>{
        an.getByteTimeDomainData(data);
        let sum=0;for(const v of data){const d=(v-128)/128;sum+=d*d;}
        setRms(Math.min(1,Math.sqrt(sum/data.length)*4));
        raf=requestAnimationFrame(tick);
      };
      raf=requestAnimationFrame(tick);
    });
    return ()=>{if(raf)cancelAnimationFrame(raf);if(ctxA)ctxA.close();};
  },[mic]);

  useEffect(()=>{
    const canvas=canvasRef.current;if(!canvas)return;
    const ctx=canvas.getContext("2d");
    const dpr=window.devicePixelRatio||1;const w=canvas.clientWidth,h=canvas.clientHeight;
    canvas.width=w*dpr;canvas.height=h*dpr;ctx.scale(dpr,dpr);
    const particles=Array.from({length:800},(_,i)=>({θ:i*Math.PI*2/800,r:Math.random()*0.9}));
    let raf;
    const draw=()=>{
      ctx.fillStyle="rgba(5,5,10,1)";ctx.fillRect(0,0,w,h);
      const cx=w/2,cy=h/2;const baseR=Math.min(w,h)/3;
      particles.forEach(p=>{
        const decay=0.3+0.7*p.r;
        const spin=time*0.05*(1+p.r*0.5);
        const x=cx+Math.cos(p.θ+spin)*baseR*p.r;
        const y=cy+Math.sin(p.θ+spin)*baseR*p.r;
        const pulse=0.3+0.7*Math.pow(1-p.r,2);
        ctx.fillStyle=`hsla(${200+50*Math.sin(p.r*10)},100%,${60*pulse}%,${0.4+0.3*Math.sin(time*2+p.r)})`;
        ctx.beginPath();ctx.arc(x,y,1.5+1.5*rms*pulse,0,Math.PI*2);ctx.fill();
      });
      ctx.font="16px sans-serif";ctx.fillStyle="rgba(255,255,255,0.7)";
      ctx.textAlign="center";
      ctx.fillText("As D → ∞, α → 0, Var(Y) ∝ D⁻¹",cx,cy+h*0.35);
      ctx.fillText("Law of Flat Curvature — Forever Balanced",cx,cy+h*0.42);
      raf=requestAnimationFrame(draw);
    };
    raf=requestAnimationFrame(draw);
    return()=>cancelAnimationFrame(raf);
  },[time,rms]);

  return (
    <div className="text-center">
      <h2 className="text-3xl font-bold">Legacy</h2>
      <p className="text-white/75 mt-2">Even as time fades, the law echoes across all dimensions.</p>
      <div className="mt-8 relative mx-auto max-w-3xl aspect-video rounded-3xl overflow-hidden border border-white/10">
        <canvas ref={canvasRef} className="w-full h-full" />
      </div>
      <button
        onClick={()=>setMic(m=>!m)}
        className="mt-5 text-sm px-4 py-2 rounded-full border border-white/20 hover:bg-white/10"
      >
        {mic?"Mic: ON":"Mic: OFF"}
      </button>
      <VoiceoverControls vo={vo} compact />
      <DOIBadge />
    </div>
  );
}


function PlaceholderAct({ index }) {
    return (
        <div className="max-w-2xl mx-auto text-center text-white/70">
            <h2 className="text-3xl font-bold mb-4">Act {index + 1} Coming Soon</h2>
            <p>
                This act has not been implemented yet. We are progressing through the storyboard one act at a time so the visuals stay focused and polished.
            </p>
        </div>
    );
}



// ---- Voice-over engine (Web Speech API) -----------------------------

const VO_SCRIPTS = {
  0: [
    "When geometry bends, information flows. When information flows, geometry flattens.",
    "At their intersection lives an invariant: Y equals square root of d effective minus one, times A squared over I."
  ],
  1: [
    "Structured systems carry memory and bias.",
    "As chaos increases, alpha flattens toward zero.",
    "Press and hold to twirl the world—symmetry emerges."
  ],
  2: [
    "Watch the law reveal itself, hands free.",
    "As dimension doubles, variance collapses exactly like D to the power negative one.",
    "This is universality, not a coincidence."
  ],
  3: [
    "The theorem assembles itself.",
    "Geometry, entropy, and dimension converge into a single identity.",
    "Y equals the square root of d effective minus one, times A squared over I."
  ],
  4: [
    "Your breath randomizes the circuit.",
    "Clifford twirling erases structure. Alpha approaches zero. Variance cools.",
    "Metrics synchronize as symmetry returns."
  ],
  5: [
    "From chaos to flatness, the universe preserves its curvature through information.",
    "A living heartbeat of universality."
  ],
  6: [
    "As dimension goes to infinity, alpha goes to zero, and variance decays like D to the minus one.",
    "Law of flat curvature. Forever balanced."
  ],
};

// Small helper to get available voices (async on some browsers)
function getVoicesAsync() {
  return new Promise((resolve) => {
    const synth = window.speechSynthesis;
    const voices = synth.getVoices();
    if (voices && voices.length) return resolve(voices);
    const id = setInterval(() => {
      const v = synth.getVoices();
      if (v && v.length) {
        clearInterval(id);
        resolve(v);
      }
    }, 200);
  });
}

// Hook: voice-over player for a list of lines
function useVoiceOver(lines = [], { autoStart = false, rate = 1.0, voiceName = null } = {}) {
  const [index, setIndex] = useState(0);
  const [speaking, setSpeaking] = useState(false);
  const [voices, setVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState(null);
  const [error, setError] = useState(null);

  // Load voices (once)
  useEffect(() => {
    if (!("speechSynthesis" in window)) {
      setError("Voice-over not supported in this browser.");
      return;
    }
    let mounted = true;
    getVoicesAsync().then((v) => {
      if (!mounted) return;
      setVoices(v);
      if (voiceName) {
        const match = v.find((vv) => vv.name.includes(voiceName));
        if (match) setSelectedVoice(match);
      }
    });
    return () => { mounted = false; };
  }, [voiceName]);

  // Auto-start when lines or autoStart changes
  useEffect(() => {
    if (!autoStart || !lines.length) return;
    play();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoStart, lines.join("|")]);

  function cancelAll() {
    try { window.speechSynthesis.cancel(); } catch {}
    setSpeaking(false);
  }

  function speakLine(i) {
    if (!("speechSynthesis" in window) || !lines[i]) return;
    cancelAll();

    const utter = new SpeechSynthesisUtterance(lines[i]);
    if (selectedVoice) utter.voice = selectedVoice;
    utter.rate = rate;
    utter.onend = () => {
      setSpeaking(false);
      // auto-advance if there are more lines
      if (i < lines.length - 1) {
        setTimeout(() => { setIndex(i + 1); speakLine(i + 1); }, 200);
      }
    };
    utter.onerror = () => setSpeaking(false);

    setSpeaking(true);
    try { window.speechSynthesis.speak(utter); } catch (e) { setError(String(e)); }
  }

  function play() {
    if (!lines.length) return;
    speakLine(index);
  }
  function pause() {
    try { window.speechSynthesis.pause(); } catch {}
  }
  function resume() {
    try { window.speechSynthesis.resume(); } catch {}
  }
  function stop() {
    cancelAll();
    setIndex(0);
  }
  function next() {
    cancelAll();
    const i = Math.min(lines.length - 1, index + 1);
    setIndex(i);
    speakLine(i);
  }
  function prev() {
    cancelAll();
    const i = Math.max(0, index - 1);
    setIndex(i);
    speakLine(i);
  }
  function setVoiceByName(name) {
    const v = voices.find((vv) => vv.name === name);
    if (v) setSelectedVoice(v);
  }

  // keyboard shortcuts
  useEffect(() => {
    const onKey = (e) => {
      if (e.target && ["INPUT", "TEXTAREA"].includes(e.target.tagName)) return;
      if (e.code === "Space") { e.preventDefault(); speaking ? pause() : resume(); }
      if (e.key === "n" || e.key === "N") next();
      if (e.key === "p" || e.key === "P") prev();
      if (e.key === "m" || e.key === "M") stop();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [speaking, next, prev]);

  return {
    speaking, index, line: lines[index] || "", voices, selectedVoice, error,
    controls: { play, pause, resume, stop, next, prev, setVoiceByName },
  };
}

// Optional ready-made UI controls and captions
function VoiceoverControls({ vo, compact=false }) {
  if (vo.error) {
    return <div className="text-xs text-red-300">{vo.error}</div>;
  }
  return (
    <div className={`mt-3 flex flex-wrap items-center gap-2 ${compact ? "text-xs" : "text-sm"}`}>
      <button onClick={vo.controls.play}   className="px-2 py-1 rounded bg-white/10 hover:bg-white/20">Play</button>
      <button onClick={vo.controls.pause}  className="px-2 py-1 rounded bg-white/10 hover:bg-white/20">Pause</button>
      <button onClick={vo.controls.resume} className="px-2 py-1 rounded bg-white/10 hover:bg-white/20">Resume</button>
      <button onClick={vo.controls.stop}   className="px-2 py-1 rounded bg-white/10 hover:bg-white/20">Stop</button>
      <button onClick={vo.controls.prev}   className="px-2 py-1 rounded bg-white/10 hover:bg-white/20">Prev</button>
      <button onClick={vo.controls.next}   className="px-2 py-1 rounded bg-white/10 hover:bg-white/20">Next</button>
      <select
        className="ml-2 bg-black/40 border border-white/20 rounded px-2 py-1"
        onChange={(e)=>vo.controls.setVoiceByName(e.target.value)}
        value={vo.selectedVoice?.name || ""}
        title="Voice"
      >
        <option value="">auto voice</option>
        {vo.voices.map(v => <option key={v.name} value={v.name}>{v.name}</option>)}
      </select>
      {/* Live caption bubble */}
      <div className="ml-auto max-w-[60ch] px-3 py-1 rounded-full bg-white/8 text-white/80">
        {vo.line}
      </div>
    </div>
  );
}
