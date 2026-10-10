// MobileFineTuner Dashboard Logic

const MODEL_SPECS = {
  gpt2_small: {
    title: "GPT-2 Small (124M)",
    params: "124,439,808",
    modules: "Q, V Attention Projection",
    loraParams: "768 (~0.001%)",
    precision: "FP32 / Native Graph",
    recommendedBatch: "4",
    recommendedLr: "0.0005"
  },
  gpt2_medium: {
    title: "GPT-2 Medium (355M)",
    params: "354,823,680",
    modules: "Q, V Attention Projection",
    loraParams: "1,536 (~0.001%)",
    precision: "FP32 / Native Graph",
    recommendedBatch: "2",
    recommendedLr: "0.0003"
  },
  gemma_270m: {
    title: "Gemma 3 (270M)",
    params: "268,142,592",
    modules: "14 Injected Modules (2 Layers)",
    loraParams: "28 LoRA Tensors",
    precision: "FP32 / RMSNorm Autograd",
    recommendedBatch: "4",
    recommendedLr: "0.0002"
  },
  gemma_1b_pt: {
    title: "Gemma 3 (1B-PT)",
    params: "1,000,000,000+",
    modules: "14 Injected Modules (2 Layers)",
    loraParams: "28 LoRA Tensors",
    precision: "FP32 / RMSNorm Autograd",
    recommendedBatch: "2",
    recommendedLr: "0.0002"
  },
  qwen: {
    title: "Qwen 2.5 (0.5B)",
    params: "494,032,896",
    modules: "RoPE / Attention LoRA",
    loraParams: "Rank=8, Alpha=16",
    precision: "FP32 / Topo-Engine",
    recommendedBatch: "1",
    recommendedLr: "0.0002"
  }
};

const USE_CASE_PRESETS = {
  personal_assistant: {
    model: "gpt2_small",
    steps: 5,
    lr: "0.0005",
    rank: "8",
    alpha: "16",
    batch: "4",
    desc: "Use Case: On-Device Personal Assistant. Adapts user tone & style without private data leaving the device."
  },
  domain_knowledge: {
    model: "qwen",
    steps: 5,
    lr: "0.0002",
    rank: "8",
    alpha: "16",
    batch: "1",
    desc: "Use Case: Domain Knowledge Adaptation. Adapts Qwen 2.5 on technical corpora (WikiText-2) for domain terms."
  },
  reasoning_qa: {
    model: "gemma_270m",
    steps: 5,
    lr: "0.0002",
    rank: "8",
    alpha: "16",
    batch: "4",
    desc: "Use Case: On-Phone Reasoning & QA. Adapts Gemma 3 (270M) on MMLU / ARC-C multiple choice benchmarks."
  },
  privacy_medical: {
    model: "gemma_1b_pt",
    steps: 5,
    lr: "0.0002",
    rank: "16",
    alpha: "32",
    batch: "2",
    desc: "Use Case: Confidential Health / Forms. Zero cloud communication ensures 100% compliance with GDPR/HIPAA."
  },
  ultra_low_ram: {
    model: "gemma_270m",
    steps: 5,
    lr: "0.0002",
    rank: "4",
    alpha: "8",
    batch: "1",
    desc: "Use Case: Ultra-Low RAM Embedded. Minimal Rank=4 configuration tailored for constrained RAM smartphones."
  }
};

let activeModel = "gpt2_small";
let isTraining = false;
let pollTimer = null;
let lossHistory = [];
let renderedLogCount = 0;

// Elements
const stepRange = document.getElementById("stepRange");
const stepsValue = document.getElementById("stepsValue");
const lrInput = document.getElementById("lrInput");
const rankInput = document.getElementById("rankInput");
const alphaInput = document.getElementById("alphaInput");
const batchInput = document.getElementById("batchInput");
const trainForm = document.getElementById("trainForm");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const startBtnText = document.getElementById("startBtnText");
const btnSpinner = document.getElementById("btnSpinner");
const engineStatusPill = document.getElementById("engineStatusPill");
const engineStatusText = document.getElementById("engineStatusText");
const terminalOutput = document.getElementById("terminalOutput");
const clearLogBtn = document.getElementById("clearLogBtn");
const copyLogBtn = document.getElementById("copyLogBtn");
const lossCanvas = document.getElementById("lossCanvas");
const canvasEmpty = document.getElementById("canvasEmpty");
const runTestsBtn = document.getElementById("runTestsBtn");
const testList = document.getElementById("testList");
const testsBadge = document.getElementById("testsBadge");
const artifactsList = document.getElementById("artifactsList");

// Telemetry Elements
const telLoss = document.getElementById("telLoss");
const telLossDelta = document.getElementById("telLossDelta");
const telInitLoss = document.getElementById("telInitLoss");
const telGradNorm = document.getElementById("telGradNorm");
const telStep = document.getElementById("telStep");
const progressFill = document.getElementById("progressFill");

// Canvas context
const ctx = lossCanvas.getContext("2d");

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  updateModelSpecs(activeModel);
  resizeCanvas();
  window.addEventListener("resize", () => {
    resizeCanvas();
    drawChart();
  });
  fetchArtifacts();
  fetchInitialStatus();
});

function setupEventListeners() {
  // Step Range Slider
  stepRange.addEventListener("input", (e) => {
    stepsValue.textContent = e.target.value;
  });

  // Preset Buttons
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      if (isTraining) return;
      document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const presetKey = btn.getAttribute("data-preset");
      const preset = USE_CASE_PRESETS[presetKey];
      if (preset) {
        // Switch model tab
        activeModel = preset.model;
        document.querySelectorAll(".model-tab").forEach(t => {
          t.classList.toggle("active", t.getAttribute("data-model") === preset.model);
        });
        updateModelSpecs(activeModel);

        // Update inputs
        stepRange.value = preset.steps;
        stepsValue.textContent = preset.steps;
        lrInput.value = preset.lr;
        rankInput.value = preset.rank;
        alphaInput.value = preset.alpha;
        batchInput.value = preset.batch;

        appendTerminalLine(`> [Preset Activated] ${preset.desc}`);
      }
    });
  });

  // Model Tabs
  document.querySelectorAll(".model-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      if (isTraining) return;
      document.querySelectorAll(".model-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      activeModel = tab.getAttribute("data-model");
      updateModelSpecs(activeModel);
    });
  });

  // Train Form Submit
  trainForm.addEventListener("submit", (e) => {
    e.preventDefault();
    startTraining();
  });

  // Stop Button
  stopBtn.addEventListener("click", () => {
    stopTraining();
  });

  // Terminal Controls
  clearLogBtn.addEventListener("click", () => {
    terminalOutput.innerHTML = "";
    renderedLogCount = 0;
  });

  copyLogBtn.addEventListener("click", () => {
    const text = terminalOutput.innerText;
    navigator.clipboard.writeText(text).then(() => {
      copyLogBtn.textContent = "Copied!";
      setTimeout(() => copyLogBtn.textContent = "Copy", 1500);
    });
  });

  // Run Tests Button
  runTestsBtn.addEventListener("click", () => {
    runAutogradTests();
  });
}

function updateModelSpecs(modelKey) {
  const specs = MODEL_SPECS[modelKey] || MODEL_SPECS.gpt2_small;
  document.getElementById("specModelTitle").textContent = specs.title;
  document.getElementById("specParams").textContent = specs.params;
  document.getElementById("specModules").textContent = specs.modules;
  document.getElementById("specLoraParams").textContent = specs.loraParams;

  if (specs.recommendedBatch) {
    batchInput.value = specs.recommendedBatch;
  }
  if (specs.recommendedLr) {
    lrInput.value = specs.recommendedLr;
  }
}

function resizeCanvas() {
  const rect = lossCanvas.parentElement.getBoundingClientRect();
  lossCanvas.width = rect.width * window.devicePixelRatio;
  lossCanvas.height = 260 * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
}

function fetchInitialStatus() {
  fetch("/api/status")
    .then(res => res.json())
    .then(data => {
      if (data.training && data.training.status === "running") {
        setRunningState(true);
        startPolling();
      }
    })
    .catch(() => {});
}

function startTraining() {
  const payload = {
    model: activeModel,
    steps: parseInt(stepRange.value, 10),
    lr: parseFloat(lrInput.value),
    rank: parseInt(rankInput.value, 10),
    alpha: parseInt(alphaInput.value, 10),
    batch_size: parseInt(batchInput.value, 10),
    seq_len: 128
  };

  // Reset UI State
  lossHistory = [];
  renderedLogCount = 0;
  terminalOutput.innerHTML = `<div class="log-line system-msg">> Starting ${MODEL_SPECS[activeModel].title} fine-tuning run...</div>`;
  canvasEmpty.style.display = "none";
  drawChart();

  telLoss.textContent = "--";
  telLossDelta.textContent = "Initializing autograd graph...";
  telInitLoss.textContent = "--";
  telGradNorm.textContent = "--";
  telStep.textContent = `0 / ${payload.steps}`;
  progressFill.style.width = "0%";

  fetch("/api/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  })
  .then(res => res.json())
  .then(data => {
    if (data.status === "started") {
      setRunningState(true);
      startPolling();
    } else {
      appendTerminalLine(`Error: ${data.error || 'Failed to start'}`);
    }
  })
  .catch(err => {
    appendTerminalLine(`Network Error: ${err.message}`);
  });
}

function stopTraining() {
  fetch("/api/stop", { method: "POST" })
    .then(res => res.json())
    .then(() => {
      setRunningState(false);
      appendTerminalLine(">>> Stop signal sent.");
    });
}

function setRunningState(running) {
  isTraining = running;
  if (running) {
    startBtn.disabled = true;
    stopBtn.disabled = false;
    btnSpinner.style.display = "inline-block";
    startBtnText.textContent = "Training in Progress...";
    engineStatusPill.style.background = "rgba(139, 92, 246, 0.15)";
    engineStatusPill.style.borderColor = "rgba(139, 92, 246, 0.4)";
    engineStatusPill.style.color = "#c084fc";
    engineStatusText.textContent = "AUTOGRAD ENGAGED";
  } else {
    startBtn.disabled = false;
    stopBtn.disabled = true;
    btnSpinner.style.display = "none";
    startBtnText.textContent = "⚡ Launch Fine-Tuning";
    engineStatusPill.style.background = "rgba(16, 185, 129, 0.1)";
    engineStatusPill.style.borderColor = "rgba(16, 185, 129, 0.25)";
    engineStatusPill.style.color = "#34d399";
    engineStatusText.textContent = "ENGINE READY";
  }
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollTrainingState, 350);
}

function pollTrainingState() {
  fetch("/api/logs")
    .then(res => res.json())
    .then(data => {
      // Append logs
      if (data.logs && data.logs.length > renderedLogCount) {
        const newLogs = data.logs.slice(renderedLogCount);
        newLogs.forEach(l => appendTerminalLine(l));
        renderedLogCount = data.logs.length;
      }

      // Update telemetry
      if (data.steps && data.steps.length > 0) {
        lossHistory = data.steps;
        const latest = data.steps[data.steps.length - 1];
        telLoss.textContent = latest.loss.toFixed(5);
        telGradNorm.textContent = latest.grad_norm.toFixed(6);
        telStep.textContent = `${latest.step} / ${data.total_steps}`;

        const pct = Math.min(100, Math.round((latest.step / data.total_steps) * 100));
        progressFill.style.width = `${pct}%`;

        if (data.initial_loss !== null) {
          telInitLoss.textContent = data.initial_loss.toFixed(5);
          const drop = data.initial_loss - latest.loss;
          const dropPct = ((drop / data.initial_loss) * 100).toFixed(2);
          telLossDelta.textContent = `Δ -${drop.toFixed(4)} (-${dropPct}%)`;
          telLossDelta.style.color = "#34d399";
        }

        drawChart();
      }

      // Check Completion
      if (data.status === "completed" || data.status === "error") {
        clearInterval(pollTimer);
        setRunningState(false);
        fetchArtifacts();

        if (data.status === "completed") {
          appendTerminalLine(`[SUCCESS] Training session completed cleanly in ${data.elapsed_sec}s!`);
          if (data.param_delta) {
            appendTerminalLine(`[LoRA Optimization] Total Weight Displacement Delta: ${data.param_delta}`);
          }
        } else if (data.status === "error") {
          appendTerminalLine(`[ERROR] ${data.error_message || 'Training failed.'}`);
        }
      }
    })
    .catch(() => {});
}

function appendTerminalLine(line) {
  const div = document.createElement("div");
  div.className = "log-line";

  if (line.includes("smoke step")) {
    div.className += " loss-step";
  } else if (line.includes("[PASS]") || line.includes("SUCCESS")) {
    div.className += " pass-msg";
  } else if (line.startsWith(">") || line.startsWith("==")) {
    div.className += " system-msg";
  }

  div.textContent = line;
  terminalOutput.appendChild(div);
  terminalOutput.scrollTop = terminalOutput.scrollHeight;
}

// Chart Rendering
function drawChart() {
  const rect = lossCanvas.parentElement.getBoundingClientRect();
  const width = rect.width;
  const height = 260;

  ctx.clearRect(0, 0, width, height);

  if (!lossHistory || lossHistory.length === 0) {
    canvasEmpty.style.display = "flex";
    return;
  }
  canvasEmpty.style.display = "none";

  const paddingLeft = 55;
  const paddingRight = 30;
  const paddingTop = 30;
  const paddingBottom = 40;

  const chartWidth = width - paddingLeft - paddingRight;
  const chartHeight = height - paddingTop - paddingBottom;

  const losses = lossHistory.map(s => s.loss);
  const minLoss = Math.min(...losses) * 0.98;
  const maxLoss = Math.max(...losses) * 1.02;
  const lossSpan = maxLoss - minLoss || 0.1;

  // Grid lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
  ctx.lineWidth = 1;
  const gridRows = 4;
  ctx.fillStyle = "#64748b";
  ctx.font = "10px JetBrains Mono";
  ctx.textAlign = "right";

  for (let i = 0; i <= gridRows; i++) {
    const y = paddingTop + (chartHeight / gridRows) * i;
    ctx.beginPath();
    ctx.moveTo(paddingLeft, y);
    ctx.lineTo(width - paddingRight, y);
    ctx.stroke();

    const val = (maxLoss - (lossSpan / gridRows) * i).toFixed(3);
    ctx.fillText(val, paddingLeft - 10, y + 3);
  }

  // Points computation
  const points = lossHistory.map((item, idx) => {
    const x = paddingLeft + (chartWidth / (lossHistory.length > 1 ? lossHistory.length - 1 : 1)) * idx;
    const y = paddingTop + chartHeight - ((item.loss - minLoss) / lossSpan) * chartHeight;
    return { x, y, loss: item.loss, step: item.step };
  });

  // Area Gradient
  const grad = ctx.createLinearGradient(0, paddingTop, 0, height - paddingBottom);
  grad.addColorStop(0, "rgba(6, 182, 212, 0.35)");
  grad.addColorStop(1, "rgba(6, 182, 212, 0.0)");

  ctx.beginPath();
  ctx.moveTo(points[0].x, height - paddingBottom);
  ctx.lineTo(points[0].x, points[0].y);

  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1];
    const curr = points[i];
    const cpX = (prev.x + curr.x) / 2;
    ctx.bezierCurveTo(cpX, prev.y, cpX, curr.y, curr.x, curr.y);
  }

  ctx.lineTo(points[points.length - 1].x, height - paddingBottom);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Curve line
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1];
    const curr = points[i];
    const cpX = (prev.x + curr.x) / 2;
    ctx.bezierCurveTo(cpX, prev.y, cpX, curr.y, curr.x, curr.y);
  }
  ctx.strokeStyle = "#06b6d4";
  ctx.lineWidth = 3;
  ctx.shadowColor = "#06b6d4";
  ctx.shadowBlur = 12;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Dots & Labels
  points.forEach((p, idx) => {
    // Circle
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.strokeStyle = "#06b6d4";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Value label
    ctx.fillStyle = "#38bdf8";
    ctx.textAlign = "center";
    ctx.font = "10px JetBrains Mono";
    ctx.fillText(p.loss.toFixed(4), p.x, p.y - 10);

    // X axis step label
    ctx.fillStyle = "#94a3b8";
    ctx.fillText(`Step ${p.step}`, p.x, height - paddingBottom + 18);
  });
}

// Unit Tests Runner
function runAutogradTests() {
  testList.innerHTML = `<div class="test-item-placeholder">Executing CTest unit suite (19 targets)...</div>`;
  testsBadge.textContent = "RUNNING...";
  testsBadge.className = "badge-pill";

  fetch("/api/run-tests", { method: "POST" })
    .then(res => res.json())
    .then(() => {
      const poll = setInterval(() => {
        fetch("/api/tests")
          .then(r => r.json())
          .then(data => {
            if (data.status === "completed" || data.status === "failed") {
              clearInterval(poll);
              renderTests(data);
            }
          });
      }, 500);
    });
}

function renderTests(data) {
  testList.innerHTML = "";
  testsBadge.textContent = `${data.total_passed} / ${data.total_tests} PASSED`;
  testsBadge.className = data.total_passed === data.total_tests ? "badge-success" : "badge-pill";

  data.tests.forEach(t => {
    const item = document.createElement("div");
    item.className = "test-item";
    item.innerHTML = `
      <span class="test-name">#${t.id} ${t.name}</span>
      <span class="test-status">${t.status} (${t.time_sec}s)</span>
    `;
    testList.appendChild(item);
  });
}

// Artifacts Scanner
function fetchArtifacts() {
  fetch("/api/artifacts")
    .then(res => res.json())
    .then(data => {
      artifactsList.innerHTML = "";
      if (!data.artifacts || data.artifacts.length === 0) {
        artifactsList.innerHTML = `<tr><td colspan="5" style="text-align:center; color:#64748b; padding:16px;">No exported SafeTensors adapters found yet.</td></tr>`;
        return;
      }

      data.artifacts.forEach(art => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${art.model}</strong></td>
          <td><code style="color:#a78bfa">${art.filename}</code></td>
          <td>${art.size_kb} KB</td>
          <td style="color:#64748b; font-size:0.75rem">${art.modified}</td>
          <td><span class="badge-pill">READY</span></td>
        `;
        artifactsList.appendChild(tr);
      });
    })
    .catch(() => {});
}
