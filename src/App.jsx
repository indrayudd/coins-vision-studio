import { AnimatePresence, motion } from "framer-motion";
import "katex/dist/katex.min.css";
import { BlockMath } from "react-katex";
import { useEffect, useState } from "react";

const stagger = {
  hidden: { opacity: 0, y: 18 },
  show: (index) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.45, delay: index * 0.06 }
  })
};

const STEP_PLAYBOOK = {
  original: {
    friendly: "Raw Camera Frame",
    focus: "This is the untouched photo: what a person sees before any computer cleanup.",
    substeps: ["Read RGB pixels", "Resize image for stable speed", "Keep geometry for final overlays"],
    math: {
      formulaTex: "I(x,y)=\\begin{bmatrix}R(x,y)\\\\G(x,y)\\\\B(x,y)\\end{bmatrix}",
      kernelTex: "",
      note: "Every pixel starts as three numbers: red, green, and blue brightness."
    },
    metrics: []
  },
  grayscale: {
    friendly: "Black-and-White Projection",
    focus: "Convert color into one brightness channel so boundaries become easier to compare.",
    substeps: ["Mix RGB channels into one brightness value", "Preserve contrast where rims change quickly", "Pass one-channel image to later steps"],
    math: {
      formulaTex: "Y=0.299R+0.587G+0.114B",
      kernelTex: "\\begin{bmatrix}0.299 & 0.587 & 0.114\\end{bmatrix}\\begin{bmatrix}R\\\\G\\\\B\\end{bmatrix}",
      note: "Green contributes most because human vision is most sensitive to green intensity."
    },
    metrics: []
  },
  contrast: {
    friendly: "Local Contrast Lift",
    focus: "Brighten dim local areas and tame over-bright zones so coin rims pop consistently.",
    substeps: ["Split image into small tiles", "Stretch local brightness ranges", "Clip extreme boosts to avoid noise explosion"],
    math: {
      formulaTex: "I'(x,y)=\\operatorname{clip}\\left(T_{\\text{tile}}(I(x,y))\\right)",
      kernelTex: "",
      note: "Each tile gets its own remapping curve, then neighboring tiles are blended smoothly."
    },
    metrics: ["texture_energy"]
  },
  deglare: {
    friendly: "Glare Repair",
    focus: "Mirror-like bright spots are repaired so circular coin borders are no longer broken.",
    substeps: ["Find very bright low-color pixels", "Create glare mask", "Fill masked pixels from nearby context"],
    math: {
      formulaTex: "I_{\\text{repaired}}=\\operatorname{inpaint}(I,M_{\\text{glare}})",
      kernelTex:
        "M_{\\text{glare}}(x,y)=\\begin{cases}1,&V\\text{ high and }S\\text{ low}\\\\0,&\\text{otherwise}\\end{cases}",
      note: "First detect likely reflections, then replace them with neighborhood-consistent values."
    },
    metrics: ["glare_fraction"]
  },
  lowpass: {
    friendly: "Background Smoothing",
    focus: "Fine background grain is softened so edges mostly belong to coins, not surface texture.",
    substeps: ["Apply edge-aware smoothing", "Apply frequency-domain low-pass", "Blend and gently blur for stability"],
    math: {
      formulaTex: "I_{\\text{smooth}}=K\\ast I",
      kernelTex: "K\\approx\\frac{1}{16}\\begin{bmatrix}1&2&1\\\\2&4&2\\\\1&2&1\\end{bmatrix}",
      note: "Convolution averages each pixel with neighbors, reducing high-frequency noise."
    },
    metrics: ["texture_energy"]
  },
  line_suppress: {
    friendly: "Stripe/Line Suppression",
    focus: "Long tablecloth or wood stripes are removed so circle detection is not distracted.",
    substeps: ["Find long straight line segments", "Build line-only mask", "Inpaint those line pixels"],
    math: {
      formulaTex: "\\rho=x\\cos(\\theta)+y\\sin(\\theta)",
      kernelTex: "",
      note: "That line equation is used by Hough voting to detect dominant straight structures."
    },
    metrics: ["line_artifact_fraction", "texture_energy"]
  },
  edges: {
    friendly: "Edge Extraction",
    focus: "Highlight strong object boundaries so circles can be detected from rim outlines.",
    substeps: ["Measure left-right and up-down intensity change", "Compute edge strength map", "Keep strong and connected boundary pixels"],
    math: {
      formulaTex: "\\lVert\\nabla I\\rVert=\\sqrt{G_x^2+G_y^2}",
      kernelTex:
        "G_x=\\begin{bmatrix}-1&0&1\\\\-2&0&2\\\\-1&0&1\\end{bmatrix}\\quad G_y=\\begin{bmatrix}-1&-2&-1\\\\0&0&0\\\\1&2&1\\end{bmatrix}",
      note: "Sobel kernels estimate slope in horizontal and vertical directions."
    },
    metrics: ["canny_low_threshold", "canny_high_threshold", "line_artifact_fraction"]
  },
  mask: {
    friendly: "Coin Region Mask",
    focus: "Convert the image into coin-like vs background areas to narrow where circles can exist.",
    substeps: ["Choose threshold automatically", "Clean tiny specks and holes", "Keep regions that match coin scale"],
    math: {
      formulaTex: "M(x,y)=\\begin{cases}1,&I(x,y)\\ge\\tau\\\\0,&I(x,y)<\\tau\\end{cases}",
      kernelTex: "",
      note: "A threshold τ splits pixels into foreground/background, then morphology refines the mask."
    },
    metrics: ["foreground_fraction", "glare_fraction", "line_artifact_fraction"]
  },
  watershed: {
    friendly: "Touching-Coin Splitter",
    focus: "When coins touch, we split merged blobs into separate regions before final circle fitting.",
    substeps: ["Build distance map from boundaries", "Place seeds at local peaks", "Grow regions until they meet"],
    math: {
      formulaTex: "\\text{labels}=\\operatorname{watershed}(-D,\\text{seeds})",
      kernelTex: "",
      note: "Think of flooding valleys: each seed claims nearby pixels until boundaries collide."
    },
    metrics: ["region_proposals", "hough_min_radius", "hough_max_radius"]
  },
  hough: {
    friendly: "Final Coin Circles",
    focus: "Multiple circle proposals are combined and de-duplicated to produce final coin counts.",
    substeps: [
      "Vote for possible circles at many scales",
      "Score each circle by edge and mask support",
      "Merge overlaps and keep one circle per coin"
    ],
    math: {
      formulaTex: "(x-a)^2+(y-b)^2=r^2",
      kernelTex: "",
      note: "Each edge pixel votes for circle centers/radii that could pass through it."
    },
    metrics: [
      "hough_pass_candidates",
      "hough_min_radius",
      "hough_max_radius",
      "hough_param2",
      "region_proposals",
      "dog_fallback_candidates",
      "component_rescued"
    ]
  }
};

const METRIC_META = {
  canny_low_threshold: {
    label: "Faint-edge sensitivity",
    max: 255,
    digits: 0,
    meaning: "Lower values keep weaker edges. Too low may keep noise."
  },
  canny_high_threshold: {
    label: "Strong-edge gate",
    max: 255,
    digits: 0,
    meaning: "Higher values require stronger boundaries."
  },
  hough_min_radius: { label: "Smallest allowed coin radius", max: 64, digits: 0, meaning: "Lower catches smaller coins." },
  hough_max_radius: { label: "Largest allowed coin radius", max: 160, digits: 0, meaning: "Upper size limit for circle search." },
  hough_param2: {
    label: "Circle vote strictness",
    max: 120,
    digits: 0,
    meaning: "Higher = stricter acceptance of circle candidates."
  },
  hough_pass_candidates: {
    label: "Total circle ideas",
    max: 260,
    digits: 0,
    meaning: "How many raw circle proposals were generated before filtering."
  },
  foreground_fraction: {
    label: "Coin-region coverage",
    max: 1,
    digits: 3,
    meaning: "Fraction of image currently treated as coin-like foreground."
  },
  line_artifact_fraction: {
    label: "Line-texture share",
    max: 0.08,
    digits: 3,
    meaning: "How much strong line pattern was detected and suppressed."
  },
  glare_fraction: {
    label: "Glare share",
    max: 0.06,
    digits: 3,
    meaning: "Fraction flagged as bright reflections."
  },
  texture_energy: {
    label: "Background roughness",
    max: 0.06,
    digits: 3,
    meaning: "Higher means busier background texture."
  },
  dog_fallback_candidates: {
    label: "Small-coin rescue ideas",
    max: 18,
    digits: 0,
    meaning: "Extra blob-based proposals added when initial recall is low."
  },
  region_proposals: {
    label: "Region proposals",
    max: 90,
    digits: 0,
    meaning: "Segments from region analysis used to guide local circle search."
  },
  component_rescued: {
    label: "Rescued circles",
    max: 10,
    digits: 0,
    meaning: "Conservative post-pass circles added to reduce undercount."
  }
};

function metricRowsForStep(step, parameters = {}) {
  const recipe = STEP_PLAYBOOK[step.id];
  if (!recipe?.metrics?.length) {
    return [];
  }

  return recipe.metrics
    .filter((key) => Object.prototype.hasOwnProperty.call(parameters, key))
    .map((key) => {
      const meta = METRIC_META[key] || { label: key, max: 1, digits: 2 };
      const raw = Number(parameters[key]);
      const value = Number.isFinite(raw) ? raw : 0;
      const progress = Math.max(0, Math.min(1, value / Math.max(meta.max, 1e-6)));
      return {
        key,
        label: meta.label,
        value,
        display: value.toFixed(meta.digits ?? 2),
        progress,
        meaning: meta.meaning || ""
      };
    });
}

function contextTriplet(steps, index) {
  if (!Array.isArray(steps) || steps.length === 0) {
    return [];
  }
  return [
    { kind: "Before", index: Math.max(0, index - 1) },
    { kind: "Current", index },
    { kind: "After", index: Math.min(steps.length - 1, index + 1) }
  ].map((item) => ({ ...item, step: steps[item.index] }));
}

function StageButton({ step, index, onOpen, isActive }) {
  const recipe = STEP_PLAYBOOK[step.id];
  return (
    <motion.button
      type="button"
      className={`stage-card${isActive ? " active" : ""}`}
      onClick={() => onOpen(index)}
      custom={index}
      variants={stagger}
      initial="hidden"
      animate="show"
    >
      <div className="step-tag">Step {index + 1}</div>
      <h3>{step.title}</h3>
      <p>{recipe?.focus || step.description}</p>
      <img src={step.image} alt={step.title} loading="lazy" />
      <span className="stage-card-foot">Open detailed inspector →</span>
    </motion.button>
  );
}

function SampleCard({ sample, index, isActive, onSelect }) {
  const hasGroundTruth = typeof sample.ground_truth_count === "number";

  return (
    <button type="button" className={`sample-card${isActive ? " active" : ""}`} onClick={() => onSelect(index)}>
      <div className="sample-meta">
        <span>Sample {index + 1}</span>
        <strong>{sample.coin_count} pred</strong>
      </div>
      <div className="sample-counts">
        <span>{hasGroundTruth ? `GT ${sample.ground_truth_count}` : "GT —"}</span>
        <span>{sample.split}</span>
      </div>
      <small>{sample.image_path}</small>
    </button>
  );
}

function StepPanel({ isOpen, step, index, total, sample, onClose }) {
  const metrics = step ? metricRowsForStep(step, sample?.parameters || {}) : [];
  const recipe = step ? STEP_PLAYBOOK[step.id] : null;
  const strip = sample?.steps ? contextTriplet(sample.steps, index) : [];
  const hasGroundTruth = typeof sample?.ground_truth_count === "number";
  const errorText =
    sample?.error == null ? "N/A" : `${sample.error > 0 ? "+" : ""}${sample.error} (${sample.absolute_error} abs)`;

  return (
    <AnimatePresence>
      {isOpen && step ? (
        <>
          <motion.button
            type="button"
            className="panel-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            aria-label="Close stage inspector"
            onClick={onClose}
          />
          <motion.aside
            className="step-panel"
            initial={{ x: 380, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 380, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 32 }}
            role="dialog"
            aria-modal="true"
            aria-label="Stage details"
          >
            <header className="step-panel-head">
              <div>
                <span className="step-tag">Step {index + 1}</span>
                <h3>{step.title}</h3>
                <p>{sample?.image_path}</p>
              </div>
              <button type="button" className="close-btn" onClick={onClose} aria-label="Close inspector">
                ✕
              </button>
            </header>
            <div className="step-panel-body">
              <p className="step-description">{recipe?.focus || step.description}</p>
              <img src={step.image} alt={step.title} loading="lazy" />
              {strip.length > 0 ? (
                <section className="inspector-card">
                  <div className="transform-strip">
                    {strip.map((item, itemIndex) => (
                      <div key={`${item.kind}-${itemIndex}`} className={`strip-card ${item.kind === "Current" ? "current" : ""}`}>
                        <span>{item.kind}</span>
                        <img src={item.step.image} alt={`${item.kind} stage`} loading="lazy" />
                        <small>{item.step.title}</small>
                      </div>
                    ))}
                  </div>
                </section>
              ) : null}
              <section className="inspector-card">
                <h4>Selected Sample Summary</h4>
                <div className="inspector-stats">
                  <div>
                    <span>Prediction</span>
                    <strong>{sample?.coin_count}</strong>
                  </div>
                  <div>
                    <span>Ground Truth</span>
                    <strong>{hasGroundTruth ? sample.ground_truth_count : "—"}</strong>
                  </div>
                  <div>
                    <span>Error</span>
                    <strong>{errorText}</strong>
                  </div>
                </div>
              </section>
              {recipe ? (
                <section className="inspector-card">
                  <h4>The Algorithm</h4>
                  <p>{recipe.focus}</p>
                  <ol className="substep-list">
                    {recipe.substeps.map((substep, subIndex) => (
                      <li key={`${step.id}-${subIndex}`}>
                        <span>{subIndex + 1}</span>
                        <p>{substep}</p>
                      </li>
                    ))}
                  </ol>
                </section>
              ) : null}
              {recipe?.math ? (
                <section className="inspector-card math-card">
                  <h4>Core numeric rule</h4>
                  <div className="formula-block">
                    <BlockMath math={recipe.math.formulaTex} />
                  </div>
                  {recipe.math.kernelTex ? (
                    <div className="matrix-block">
                      <BlockMath math={recipe.math.kernelTex} />
                    </div>
                  ) : null}
                  <p className="math-note">{recipe.math.note}</p>
                </section>
              ) : null}
              {metrics.length > 0 ? (
                <section className="inspector-card">
                  <h4>Signal Readout</h4>
                  <div className="metric-stack">
                    {metrics.map((metric) => (
                      <div key={metric.key} className="metric-row">
                        <div className="metric-top">
                          <span>{metric.label}</span>
                          <strong>{metric.display}</strong>
                        </div>
                        <div className="metric-bar">
                          <span style={{ width: `${Math.max(6, metric.progress * 100)}%` }} />
                        </div>
                        {metric.meaning ? <p className="metric-meaning">{metric.meaning}</p> : null}
                      </div>
                    ))}
                  </div>
                </section>
              ) : null}
              <div className="step-position">
                Stage {index + 1} of {total}
              </div>
            </div>
          </motion.aside>
        </>
      ) : null}
    </AnimatePresence>
  );
}

export default function App() {
  const [splits, setSplits] = useState(["all"]);
  const [selectedSplit, setSelectedSplit] = useState("all");
  const [results, setResults] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [selectedStepIndex, setSelectedStepIndex] = useState(0);
  const [isStepPanelOpen, setIsStepPanelOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isEvalLoading, setIsEvalLoading] = useState(true);
  const [error, setError] = useState("");
  const [evaluation, setEvaluation] = useState(null);
  const [evaluationError, setEvaluationError] = useState("");

  useEffect(() => {
    const loadSplits = async () => {
      try {
        const response = await fetch("/api/splits");
        if (!response.ok) {
          throw new Error(`Could not load dataset folders (${response.status}).`);
        }
        const payload = await response.json();
        setSplits(payload.splits || ["all"]);
      } catch (err) {
        setError(err.message || "Failed to load dataset metadata.");
      }
    };
    loadSplits();
  }, []);

  useEffect(() => {
    const loadEvaluation = async () => {
      setIsEvalLoading(true);
      setEvaluationError("");
      try {
        const response = await fetch("/api/evaluation?split=all");
        if (!response.ok) {
          const detail = await response.json().catch(() => ({}));
          throw new Error(detail.detail || `Evaluation failed (${response.status})`);
        }
        const payload = await response.json();
        setEvaluation(payload);
      } catch (err) {
        setEvaluationError(err.message || "Could not compute dataset-wide error.");
      } finally {
        setIsEvalLoading(false);
      }
    };
    loadEvaluation();
  }, []);

  useEffect(() => {
    const handler = (event) => {
      if (event.key === "Escape") {
        setIsStepPanelOpen(false);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  useEffect(() => {
    const sample = results[selectedIndex];
    if (!sample) {
      return;
    }
    setSelectedStepIndex(Math.max(0, sample.steps.length - 1));
    setIsStepPanelOpen(false);
  }, [results, selectedIndex]);

  const handleDraw = async () => {
    setIsLoading(true);
    setError("");
    try {
      const seed = Date.now();
      const response = await fetch(`/api/draw?split=${encodeURIComponent(selectedSplit)}&seed=${seed}&count=10`);
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || `Draw failed (${response.status})`);
      }
      const payload = await response.json();
      const drawn = payload.items || [];
      setResults(drawn);
      setSelectedIndex(0);
      setSelectedStepIndex(0);
      setIsStepPanelOpen(false);
    } catch (err) {
      setError(err.message || "Pipeline execution failed.");
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  const activeSample = results[selectedIndex] || null;
  const activeStep = activeSample?.steps?.[selectedStepIndex] || null;
  const averageCoins =
    results.length > 0 ? (results.reduce((sum, item) => sum + item.coin_count, 0) / results.length).toFixed(1) : "0.0";
  const batchErrors = results.map((item) => item.absolute_error).filter((value) => typeof value === "number");
  const averageAbsError =
    batchErrors.length > 0 ? (batchErrors.reduce((sum, value) => sum + value, 0) / batchErrors.length).toFixed(2) : "—";

  const openStep = (index) => {
    setSelectedStepIndex(index);
    setIsStepPanelOpen(true);
  };

  return (
    <div className="page-shell">
      <div className="bg-layer bg-a" />
      <div className="bg-layer bg-b" />
      <main className="layout">
        <section className="hero">
          <span className="eyebrow">Computer Vision Project</span>
          <h1>Coin Vision Studio</h1>
          <div className="controls">
            <label>
              Dataset Source
              <select value={selectedSplit} onChange={(e) => setSelectedSplit(e.target.value)}>
                {splits.map((split) => (
                  <option key={split} value={split}>
                    {split}
                  </option>
                ))}
              </select>
            </label>
            <button type="button" onClick={handleDraw} disabled={isLoading}>
              {isLoading ? "Drawing..." : "Draw"}
            </button>
          </div>
          {error ? <p className="error-text">{error}</p> : null}
        </section>

        <section className="evaluation-panel global-panel">
          <div className="eval-head">
            <h2>Global Dataset Benchmark</h2>
          </div>
          {isEvalLoading ? (
            <p className="eval-loading">Computing full-dataset metrics...</p>
          ) : evaluationError ? (
            <p className="error-text">{evaluationError}</p>
          ) : evaluation ? (
            <div className="eval-grid">
              <div className="stat">
                <span className="stat-label">Scored Images</span>
                <span className="stat-value">
                  {evaluation.num_scored}/{evaluation.num_images}
                </span>
              </div>
              <div className="stat">
                <span className="stat-label">MAE</span>
                <span className="coin-count">{evaluation.mae.toFixed(2)}</span>
              </div>
              <div className="stat">
                <span className="stat-label">RMSE</span>
                <span className="stat-value">{evaluation.rmse.toFixed(2)}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Net Error</span>
                <span className="stat-value">
                  {evaluation.total_error > 0 ? "+" : ""}
                  {evaluation.total_error}
                </span>
              </div>
              <div className="stat">
                <span className="stat-label">Total Absolute Error</span>
                <span className="stat-value">{evaluation.total_absolute_error}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Exact Match Rate</span>
                <span className="stat-value">{(evaluation.exact_match_rate * 100).toFixed(1)}%</span>
              </div>
            </div>
          ) : null}
        </section>

        {activeSample ? (
          <>
            <section className="score-panel batch-panel">
              <div className="panel-heading panel-heading-batch">
                <h2>Current Draw Batch</h2>
              </div>
              <div className="stat">
                <span className="stat-label">Batch Size</span>
                <span className="stat-value">{results.length} samples</span>
              </div>
              <div className="stat">
                <span className="stat-label">Average Count</span>
                <span className="coin-count">{averageCoins}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Batch Avg Abs Error</span>
                <span className="stat-value">{averageAbsError}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Selected Sample Path</span>
                <span className="stat-value">{activeSample.image_path}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Selected Counts</span>
                <span className="stat-value">
                  Pred {activeSample.coin_count}, GT {activeSample.ground_truth_count ?? "—"}, Err{" "}
                  {activeSample.error == null
                    ? "—"
                    : `${activeSample.error > 0 ? "+" : ""}${activeSample.error}`}
                </span>
              </div>
            </section>

            <section className="sample-grid">
              <div className="samples-head">
                <h2>Drawn Samples</h2>
                <p>Choose a sample to inspect its pipeline stages.</p>
              </div>
              {results.map((sample, index) => (
                <SampleCard
                  key={`${sample.image_path}-${index}`}
                  sample={sample}
                  index={index}
                  isActive={index === selectedIndex}
                  onSelect={setSelectedIndex}
                />
              ))}
            </section>

            <section className="steps-grid">
              <div className="steps-head">
                <h2>Steps</h2>
              </div>
              {activeSample.steps.map((step, index) => (
                <StageButton
                  key={step.id}
                  step={step}
                  index={index}
                  onOpen={openStep}
                  isActive={selectedStepIndex === index && isStepPanelOpen}
                />
              ))}
            </section>

            <StepPanel
              isOpen={isStepPanelOpen}
              step={activeStep}
              index={selectedStepIndex}
              total={activeSample.steps.length}
              sample={activeSample}
              onClose={() => setIsStepPanelOpen(false)}
            />
          </>
        ) : (
          <section className="placeholder">
            <h2>Transformation storyboard will appear here</h2>
            <p>Use Draw to sample 10 images and inspect each detection result plus full intermediate pipeline.</p>
          </section>
        )}
      </main>
    </div>
  );
}
