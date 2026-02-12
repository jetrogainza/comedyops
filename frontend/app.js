// ComedyOps Frontend
// - Calls POST /goal to generate
// - Stores run_id so feedback can be sent without copy/paste
// - Calls POST /feedback to log rating + notes

const API_BASE = ""; // same origin (because FastAPI serves /ui). If you host separately, set e.g. "http://localhost:8000"

const els = {
  apiStatus: document.getElementById("apiStatus"),
  premise: document.getElementById("premise"),
  persona: document.getElementById("persona"),
  generateBtn: document.getElementById("generateBtn"),
  clearBtn: document.getElementById("clearBtn"),

  outputBox: document.getElementById("outputBox"),
  metaBox: document.getElementById("metaBox"),
  metaTask: document.getElementById("metaTask"),
  metaModel: document.getElementById("metaModel"),
  metaPersona: document.getElementById("metaPersona"),

  stars: document.getElementById("stars"),
  wouldUse: document.getElementById("wouldUse"),
  notes: document.getElementById("notes"),
  sendFeedbackBtn: document.getElementById("sendFeedbackBtn"),
  feedbackHint: document.getElementById("feedbackHint"),

  toast: document.getElementById("toast"),
};

let lastRunId = null;
let lastRating = null;

// ---------- Small UI helpers ----------
function setStatus(ok, text) {
  const dot = els.apiStatus.querySelector(".dot");
  const label = els.apiStatus.querySelector(".text");
  label.textContent = text;

  if (ok === true) dot.style.background = "rgba(46, 242, 160, 0.85)";
  else if (ok === false) dot.style.background = "rgba(255, 91, 122, 0.85)";
  else dot.style.background = "rgba(255, 255, 255, 0.35)";
}

function toast(msg) {
  els.toast.hidden = false;
  els.toast.textContent = msg;
  window.clearTimeout(toast._t);
  toast._t = window.setTimeout(() => (els.toast.hidden = true), 2400);
}

function setOutput(text) {
  els.outputBox.textContent = text;
}

function setMeta({ task, model, persona }) {
  els.metaBox.hidden = false;
  els.metaTask.textContent = task || "—";
  els.metaModel.textContent = model || "—";
  els.metaPersona.textContent = persona || "—";
}

function resetFeedbackUI() {
  lastRating = null;
  els.wouldUse.checked = false;
  els.notes.value = "";
  updateStarsUI();
  updateFeedbackButtonState();
}

function updateFeedbackButtonState() {
  const canSend = Boolean(lastRunId) && Boolean(lastRating);
  els.sendFeedbackBtn.disabled = !canSend;

  if (!lastRunId) {
    els.feedbackHint.textContent = "Generate something first.";
  } else if (!lastRating) {
    els.feedbackHint.textContent = "Pick a star rating to enable sending.";
  } else {
    els.feedbackHint.textContent = "Ready to send.";
  }
}

// ---------- Stars ----------
function buildStars() {
  els.stars.innerHTML = "";
  for (let i = 1; i <= 5; i++) {
    const btn = document.createElement("button");
    btn.className = "starBtn";
    btn.type = "button";
    btn.setAttribute("aria-label", `${i} star`);
    btn.textContent = "★";
    btn.addEventListener("click", () => {
      lastRating = i;
      updateStarsUI();
      updateFeedbackButtonState();
    });
    els.stars.appendChild(btn);
  }
}

function updateStarsUI() {
  const buttons = els.stars.querySelectorAll(".starBtn");
  buttons.forEach((b, idx) => {
    const starValue = idx + 1;
    b.classList.toggle("selected", lastRating !== null && starValue <= lastRating);
  });
}

// ---------- API calls ----------
async function checkHealth() {
  try {
    const r = await fetch(`${API_BASE}/health`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    setStatus(true, "API connected");
  } catch (e) {
    setStatus(false, "API not reachable");
  }
}

function buildGoalText(premise, persona) {
  // This is the key “glue”: /goal only accepts a single string.
  // We help the router by writing something unambiguous.
  const p = (premise || "").trim();
  const per = (persona || "").trim();

  if (!per) {
    return `Rewrite this joke to make it funnier and tighter:\n\n${p}`;
  }
  return `Rewrite this joke as a ${per}:\n\n${p}`;
}

async function callGoal(goalText) {
  const resp = await fetch(`${API_BASE}/goal`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ goal: goalText }),
  });

  if (!resp.ok) {
    const t = await resp.text();
    throw new Error(`Goal failed (HTTP ${resp.status}): ${t}`);
  }
  return resp.json();
}

async function sendFeedback(payload) {
  const resp = await fetch(`${API_BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!resp.ok) {
    const t = await resp.text();
    throw new Error(`Feedback failed (HTTP ${resp.status}): ${t}`);
  }
  return resp.json();
}

// ---------- Wire up events ----------
els.generateBtn.addEventListener("click", async () => {
  const premise = els.premise.value.trim();
  if (!premise) {
    toast("Add a joke/premise first 🙂");
    els.premise.focus();
    return;
  }

  els.generateBtn.disabled = true;
  setOutput("Generating…");
  els.metaBox.hidden = true;

  try {
    const goalText = buildGoalText(premise, els.persona.value);
    const data = await callGoal(goalText);

    // Store run_id so feedback can be sent later
    lastRunId = data.run_id;

    setOutput(data.rewritten);
    setMeta({ task: data.task, model: data.model, persona: data.persona });

    resetFeedbackUI();
    toast("Generated. Rate it when you’re ready.");
  } catch (e) {
    setOutput("Something went wrong.\n\n" + String(e.message || e));
    lastRunId = null;
    resetFeedbackUI();
    toast("Generation failed.");
  } finally {
    els.generateBtn.disabled = false;
    updateFeedbackButtonState();
  }
});

els.clearBtn.addEventListener("click", () => {
  els.premise.value = "";
  els.persona.value = "";
  setOutput("");
  els.metaBox.hidden = true;
  lastRunId = null;
  resetFeedbackUI();
  updateFeedbackButtonState();
  toast("Cleared.");
});

els.sendFeedbackBtn.addEventListener("click", async () => {
  if (!lastRunId || !lastRating) return;

  els.sendFeedbackBtn.disabled = true;

  const payload = {
    run_id: lastRunId,
    human_rating: lastRating,
    would_use_on_stage: Boolean(els.wouldUse.checked),
    notes: (els.notes.value || "").trim() || null,
  };

  try {
    await sendFeedback(payload);
    toast("Feedback sent ✅");
  } catch (e) {
    toast("Feedback failed ❌");
  } finally {
    updateFeedbackButtonState();
  }
});

// ---------- Init ----------
buildStars();
updateStarsUI();
updateFeedbackButtonState();
checkHealth();