/**
 * MediDocs AI / Vero Pro — Full-Featured Healthcare Document Intelligence Client
 */

let currentEncounterId = "enc-01";
let activeTab = "soap";
let currentUserRole = "attending_physician";
let encountersList = [];
let documentsList = [];
let currentEncounter = null;
let timerSeconds = 310; // 5 mins 10 secs
let autoSaveTimeout = null;
let isDictatingEditor = false;
let isDictatingChat = false;
let recognition = null;
let currentAuthUser = null;
let authToken = localStorage.getItem("medidocs_token") || sessionStorage.getItem("medidocs_token") || null;

// DOM Elements
const encountersContainer = document.getElementById("encounters-container");
const currentEncounterTitle = document.getElementById("current-encounter-title");
const editorBody = document.getElementById("editor-body");
const sessionTimerText = document.getElementById("session-timer-text");
const currentRoleLabel = document.getElementById("current-role-label");
const chatDialogueList = document.getElementById("chat-dialogue-list");
const assistantHeroView = document.getElementById("assistant-hero-view");
const assistantInput = document.getElementById("assistant-input");
const sendQueryBtn = document.getElementById("send-query-btn");
const saveStatusIndicator = document.getElementById("save-status-indicator");
const charWordCounter = document.getElementById("char-word-counter");
const themeToggleBtn = document.getElementById("theme-toggle-btn");

// Application Initialization
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initTimer();
  initSpeechRecognition();
  initAuth();
  fetchRoles();
  fetchEncounters();
  fetchDocuments();
  setupToolbarActions();
  setupEventListeners();
  setupModals();
});

// ==================== AUTHENTICATION & LOGIN MANAGEMENT ====================
async function initAuth() {
  const loginForm = document.getElementById("clinical-login-form");
  const logoutBtn = document.getElementById("logout-btn");
  const modalLogoutBtn = document.getElementById("modal-logout-btn");
  const userProfileHeaderChip = document.getElementById("user-profile-header-chip");
  const forgotLink = document.getElementById("auth-forgot-link");

  // Always bind interactive event listeners unconditionally
  if (loginForm) {
    loginForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const username = document.getElementById("login-username").value.trim();
      const password = document.getElementById("login-password").value.trim();
      await executeLogin(username, password);
    });
  }

  if (logoutBtn) {
    logoutBtn.addEventListener("click", (e) => {
      e.preventDefault();
      executeLogout();
    });
  }

  if (modalLogoutBtn) {
    modalLogoutBtn.addEventListener("click", (e) => {
      e.preventDefault();
      executeLogout();
    });
  }

  if (userProfileHeaderChip) {
    userProfileHeaderChip.addEventListener("click", () => {
      openModal("modal-roles");
    });
  }

  if (forgotLink) {
    forgotLink.addEventListener("click", (e) => {
      e.preventDefault();
      alert("Institutional Security Notice:\nTo reset your clinical workstation PIN or request RBAC clearance elevation, contact Hospital IT & Security Ops at extension 4357 or clinical-it@hospital.org.");
    });
  }

  await fetchDemoUsers();

  if (authToken) {
    try {
      const res = await fetch("/api/auth/me", {
        headers: { "Authorization": `Bearer ${authToken}` }
      });
      if (res.ok) {
        const data = await res.json();
        if (data.authenticated && data.user) {
          setAuthenticatedUser(data.user);
          return;
        }
      }
    } catch (e) {
      console.warn("Auth check failed:", e);
    }
  }

  // If no valid session, ensure token is cleared and display login portal
  authToken = null;
  localStorage.removeItem("medidocs_token");
  showAuthPortal();
}

async function fetchDemoUsers() {
  const grid = document.getElementById("demo-personas-grid");
  if (!grid) return;
  try {
    const res = await fetch("/api/auth/demo-users");
    const users = await res.json();
    grid.innerHTML = "";
    users.forEach(u => {
      const card = document.createElement("div");
      card.className = "persona-btn-card";
      card.title = `Sign in as ${u.name} (${u.role_label})`;
      card.innerHTML = `
        <span class="persona-avatar-chip">${u.avatar_initials}</span>
        <div class="persona-info-meta">
          <span class="persona-name-title">${u.name}</span>
          <span class="persona-role-sub">${u.role_label}</span>
        </div>
      `;
      card.addEventListener("click", () => {
        document.getElementById("login-username").value = u.email;
        document.getElementById("login-password").value = "password123";
        executeLogin(u.email, "password123");
      });
      grid.appendChild(card);
    });
  } catch (err) {
    console.error("Failed to load demo users:", err);
  }
}

async function executeLogin(username, password) {
  const alertBox = document.getElementById("login-error-alert");
  const submitBtn = document.getElementById("login-submit-btn");
  if (alertBox) alertBox.style.display = "none";
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<span>Verifying credentials...</span>`;
  }

  try {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username_or_email: username,
        password: password,
        workstation_id: "WS-CLINIC-401"
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Authentication failed");
    }

    const data = await res.json();
    authToken = data.access_token;
    const rememberCheckbox = document.getElementById("login-remember");
    const shouldRemember = rememberCheckbox ? rememberCheckbox.checked : true;
    if (shouldRemember) {
      localStorage.setItem("medidocs_token", authToken);
      sessionStorage.removeItem("medidocs_token");
    } else {
      sessionStorage.setItem("medidocs_token", authToken);
      localStorage.removeItem("medidocs_token");
    }
    setAuthenticatedUser(data.user);
    showToast(`Authenticated: Welcome, ${data.user.name}`);
  } catch (err) {
    if (alertBox) {
      alertBox.innerText = `🔒 Access Denied: ${err.message}`;
      alertBox.style.display = "block";
    }
  } finally {
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `
        <span>Sign In to Clinical Workspace</span>
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
      `;
    }
  }
}

async function executeLogout() {
  const tokenToRevoke = authToken;
  authToken = null;
  currentAuthUser = null;
  localStorage.removeItem("medidocs_token");
  sessionStorage.removeItem("medidocs_token");

  // Dismiss any open modal overlays
  document.querySelectorAll(".modal-overlay").forEach(overlay => {
    overlay.classList.remove("active");
  });

  // Immediately display authentication portal
  showAuthPortal();
  showToast("Signed out of clinical workstation");

  // Asynchronously notify backend session store
  if (tokenToRevoke) {
    try {
      await fetch("/api/auth/logout", {
        method: "POST",
        headers: { "Authorization": `Bearer ${tokenToRevoke}` }
      });
    } catch (e) {
      console.warn("Logout notification error:", e);
    }
  }
}
window.executeLogout = executeLogout;

function setAuthenticatedUser(user) {
  currentAuthUser = user;
  currentUserRole = user.role_id;

  // Update header profile chip
  const avatarEl = document.getElementById("user-avatar-initials");
  const nameEl = document.getElementById("user-display-name");
  const deptEl = document.getElementById("user-dept-label");
  const roleLabelEl = document.getElementById("current-role-label");

  if (avatarEl) avatarEl.innerText = user.avatar_initials;
  if (nameEl) nameEl.innerText = user.name;
  if (deptEl) deptEl.innerText = user.department;
  if (roleLabelEl) roleLabelEl.innerText = user.role_label.split("(")[0].trim();

  // Hide auth portal, show workspace app
  const authPortal = document.getElementById("auth-portal-view");
  const appContainer = document.getElementById("app");
  if (authPortal) authPortal.style.display = "none";
  if (appContainer) appContainer.style.display = "flex";

  fetchRoles();
}

function showAuthPortal() {
  const authPortal = document.getElementById("auth-portal-view");
  const appContainer = document.getElementById("app");
  const alertBox = document.getElementById("login-error-alert");
  if (alertBox) {
    alertBox.style.display = "none";
    alertBox.innerText = "";
  }
  if (authPortal) authPortal.style.display = "flex";
  if (appContainer) appContainer.style.display = "none";
}

// Theme Management
function updateThemeIcon(isDark) {
  const icon = document.getElementById("theme-moon-icon");
  if (!icon) return;
  if (isDark) {
    icon.innerHTML = `<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>`;
  } else {
    icon.innerHTML = `<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>`;
  }
}

function initTheme() {
  const savedTheme = localStorage.getItem("medidocs_theme") || "light";
  const isDark = savedTheme === "dark";
  if (isDark) {
    document.body.classList.add("dark-theme");
  }
  updateThemeIcon(isDark);

  themeToggleBtn.addEventListener("click", () => {
    document.body.classList.toggle("dark-theme");
    const nowDark = document.body.classList.contains("dark-theme");
    localStorage.setItem("medidocs_theme", nowDark ? "dark" : "light");
    updateThemeIcon(nowDark);
    showToast(nowDark ? "Switched to Dark Mode" : "Switched to Clinical Light Mode");
  });
}

// Session Timer
function initTimer() {
  setInterval(() => {
    timerSeconds++;
    const m = Math.floor(timerSeconds / 60);
    const s = timerSeconds % 60;
    sessionTimerText.innerHTML = `${m} mins, ${s < 10 ? '0' : ''}${s} secs &nbsp;•&nbsp; 1:31 - 1:36 PM`;
  }, 1000);
}

// Speech Recognition (Web Speech API)
function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      if (isDictatingEditor) {
        document.execCommand("insertText", false, " " + transcript);
        showToast(`Dictated: "${transcript}"`);
        stopDictation();
      } else if (isDictatingChat) {
        assistantInput.value = transcript;
        showToast(`Query captured: "${transcript}"`);
        stopDictation();
      }
    };

    recognition.onerror = () => stopDictation();
    recognition.onend = () => stopDictation();
  }
}

function startDictation(target) {
  if (!recognition) {
    showToast("Web Speech API not supported in this browser. Using simulation.");
    if (target === "chat") {
      assistantInput.value = "What is the procedure for reporting a medication storage temperature excursion?";
    } else {
      document.execCommand("insertText", false, " Patient reports mild improvement after spacer administration.");
    }
    return;
  }
  if (target === "editor") {
    isDictatingEditor = true;
    document.getElementById("tool-mic").classList.add("recording-active");
  } else {
    isDictatingChat = true;
    document.getElementById("voice-input-btn").classList.add("recording-active");
  }
  recognition.start();
  showToast("🎙️ Listening... speak clinical query or note.");
}

function stopDictation() {
  isDictatingEditor = false;
  isDictatingChat = false;
  const toolMic = document.getElementById("tool-mic");
  const voiceBtn = document.getElementById("voice-input-btn");
  if (toolMic) toolMic.classList.remove("recording-active");
  if (voiceBtn) voiceBtn.classList.remove("recording-active");
}

// Encounters Management
async function fetchEncounters() {
  try {
    const res = await fetch("/api/encounters");
    encountersList = await res.json();
    renderEncounters();
    selectEncounter(currentEncounterId);
  } catch (err) {
    console.error("Failed to fetch encounters:", err);
  }
}

function renderEncounters() {
  encountersContainer.innerHTML = "";
  const groups = {};
  encountersList.forEach(enc => {
    if (!groups[enc.category]) groups[enc.category] = [];
    groups[enc.category].push(enc);
  });

  for (const [category, encs] of Object.entries(groups)) {
    const groupEl = document.createElement("div");
    groupEl.className = "encounter-date-group";
    groupEl.innerHTML = `<div class="date-group-title">${category}</div>`;

    encs.forEach(enc => {
      const itemEl = document.createElement("div");
      itemEl.className = `encounter-item ${enc.id === currentEncounterId ? 'active' : ''}`;
      itemEl.dataset.id = enc.id;
      itemEl.innerHTML = `
        <span class="enc-time">${enc.time}</span>
        <span class="enc-name" title="${enc.title}">${enc.title}</span>
      `;
      itemEl.addEventListener("click", () => selectEncounter(enc.id));
      groupEl.appendChild(itemEl);
    });

    encountersContainer.appendChild(groupEl);
  }
}

async function selectEncounter(encId) {
  currentEncounterId = encId;
  document.querySelectorAll(".encounter-item").forEach(el => {
    el.classList.toggle("active", el.dataset.id === encId);
  });

  try {
    const res = await fetch(`/api/encounters/${encId}`);
    currentEncounter = await res.json();
    currentEncounterTitle.innerText = currentEncounter.title;
    if (sessionTimerText && currentEncounter.duration) {
      sessionTimerText.innerHTML = `${currentEncounter.duration}`;
    }
    renderEditorContent();
  } catch (err) {
    console.error("Error loading encounter:", err);
  }
}

function formatClinicalSection(headerTitle, content) {
  if (!content) return `<h3>${headerTitle}:</h3><p>Pending clinical notes...</p>`;
  let clean = content.trim();
  clean = clean.replace(new RegExp(`^${headerTitle}:?\\s*`, 'i'), '').trim();

  const lines = clean.split("\n");
  let listItems = [];
  let otherLines = [];

  lines.forEach(l => {
    let t = l.trim();
    if (!t) return;
    if (t.startsWith("- ") || t.startsWith("• ")) {
      let itemText = t.substring(2);
      if (itemText.includes("SOP-RES-04") || itemText.includes("Asthma")) {
        itemText = itemText.replace("Asthma exacerbation", 'Asthma exacerbation <span class="citation-chip" data-code="CLIN-FORM-022" title="Click to view policy citation">SOP-RES-04 §3.2</span>');
      }
      if (itemText.includes("Symbicort")) {
        itemText = itemText.replace("with spacer", 'with spacer <span class="citation-chip" data-code="CLIN-FORM-022" title="Click to view dosage guidelines">FORM-ASTHMA-v2</span>');
      }
      if (itemText.includes("Metformin") || itemText.includes("HbA1c")) {
        itemText += ' <span class="citation-chip" data-code="SOP-CLIN-012" title="Click to view metabolic guideline">SOP-CLIN-012</span>';
      }
      if (itemText.includes("Troponin") || itemText.includes("Coronary") || itemText.includes("Aspirin")) {
        itemText += ' <span class="citation-chip" data-code="SOP-ICU-088" title="Click to view cardiac protocol">SOP-ICU-088 §1.1</span>';
      }
      listItems.push(`<li>${itemText}</li>`);
    } else {
      otherLines.push(`<p>${t}</p>`);
    }
  });

  let inner = "";
  if (listItems.length > 0) inner += `<ul>${listItems.join("")}</ul>`;
  if (otherLines.length > 0) inner += otherLines.join("");
  if (!inner) inner = `<p>${clean}</p>`;

  return `<h3>${headerTitle}:</h3>\n${inner}`;
}

function renderEditorContent() {
  if (!currentEncounter) return;

  if (activeTab === "soap") {
    const s = formatClinicalSection("Subjective", currentEncounter.subjective);
    const o = formatClinicalSection("Objective", currentEncounter.objective);
    const a = formatClinicalSection("Assessment", currentEncounter.assessment);
    const p = formatClinicalSection("Plan", currentEncounter.plan);
    editorBody.innerHTML = `${s}\n${o}\n${a}\n${p}`;
    attachCitationChips();
  } else if (activeTab === "context") {
    editorBody.innerHTML = `
      <h3>Patient Electronic Health Record Summary:</h3>
      <p><strong>Patient Name:</strong> ${currentEncounter.patient_name} (${currentEncounter.patient_id})</p>
      <p><strong>Encounter Timestamp:</strong> ${currentEncounter.time} | Duration: ${currentEncounter.duration || '0 mins, 30 secs'}</p>
      <p><strong>Clinical Category:</strong> ${currentEncounter.category || 'TODAY'}</p>
      <p><strong>Chief Concern:</strong> ${currentEncounter.title}</p>
      <p><strong>Insurance Clearance:</strong> Blue Cross Blue Shield PPO (Verified)</p>
      <p><strong>Status:</strong> Active Clinical Workstation Session</p>
    `;
  } else if (activeTab === "handout") {
    editorBody.innerHTML = `
      <h3>Patient Education Handout</h3>
      <div style="background:var(--bg-subtle); padding:16px; border:1px solid var(--border-light); border-radius:8px; white-space:pre-wrap; font-family:var(--font-sans); font-size:13px;">${currentEncounter.patient_handout || 'No patient handout recorded for this encounter.'}</div>
    `;
  } else if (activeTab === "sicknote") {
    editorBody.innerHTML = `
      <h3>Formal Medical Certificate / Sick Note</h3>
      <div style="background:var(--bg-subtle); padding:16px; border:1px solid var(--border-light); border-radius:8px; white-space:pre-wrap; font-family:var(--font-mono); font-size:12.5px;">${currentEncounter.sick_note || 'No sick note issued.'}</div>
    `;
  }
  updateWordCount();
}

function updateWordCount() {
  const text = editorBody.innerText.trim();
  const words = text ? text.split(/\s+/).length : 0;
  const chars = text.length;
  charWordCounter.innerText = `${words} words • ${chars} chars`;
}

function attachCitationChips() {
  document.querySelectorAll(".citation-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const docCode = chip.dataset.code || "CLIN-FORM-022";
      openDocumentViewerByCode(docCode);
    });
  });
}

// Auto-save clinical note to backend
function scheduleAutoSave() {
  saveStatusIndicator.innerText = "Saving...";
  clearTimeout(autoSaveTimeout);
  autoSaveTimeout = setTimeout(async () => {
    if (!currentEncounter) return;
    try {
      const payload = {
        title: currentEncounterTitle.innerText.trim()
      };
      if (activeTab === "soap") {
        payload.subjective = editorBody.innerText;
      }
      await fetch(`/api/encounters/${currentEncounter.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      saveStatusIndicator.innerText = "✓ Saved to EHR";
    } catch (err) {
      saveStatusIndicator.innerText = "Offline (Local)";
    }
  }, 1000);
}

// Rich Toolbar Actions
function setupToolbarActions() {
  document.getElementById("tool-bold").addEventListener("click", () => document.execCommand("bold"));
  document.getElementById("tool-italic").addEventListener("click", () => document.execCommand("italic"));
  document.getElementById("tool-underline").addEventListener("click", () => document.execCommand("underline"));
  document.getElementById("tool-strike").addEventListener("click", () => document.execCommand("strikeThrough"));
  document.getElementById("tool-undo").addEventListener("click", () => document.execCommand("undo"));
  document.getElementById("tool-redo").addEventListener("click", () => document.execCommand("redo"));

  document.getElementById("tool-mic").addEventListener("click", () => {
    if (isDictatingEditor) stopDictation();
    else startDictation("editor");
  });

  document.getElementById("tool-patient-vars").addEventListener("click", () => {
    if (!currentEncounter) return;
    const varsText = `[Patient: ${currentEncounter.patient_name} | ID: ${currentEncounter.patient_id} | Date: Today]`;
    document.execCommand("insertText", false, varsText);
    showToast("Inserted patient demographic variables");
  });

  document.getElementById("tool-font-size").addEventListener("click", () => {
    document.execCommand("formatBlock", false, "<h3>");
    showToast("Formatted selection as section heading");
  });

  document.getElementById("tool-copy-btn").addEventListener("click", () => {
    navigator.clipboard.writeText(editorBody.innerText);
    showToast("Clinical note copied to clipboard!");
  });

  document.getElementById("tool-export").addEventListener("click", () => {
    const filename = `${currentEncounter ? currentEncounter.title.replace(/\s+/g, '_') : 'Clinical_Note'}.txt`;
    const blob = new Blob([editorBody.innerText], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    showToast(`Exported: ${filename}`);
  });

  document.getElementById("tool-evidence-lamp").addEventListener("click", () => {
    const query = "What hospital policies apply to this asthma exacerbation and Symbicort prescription?";
    assistantInput.value = query;
    sendQuery(query);
  });

  document.getElementById("tool-code-toggle").addEventListener("click", () => {
    showToast("Viewing rendered clinical documentation");
  });

  editorBody.addEventListener("input", () => {
    updateWordCount();
    scheduleAutoSave();
  });

  currentEncounterTitle.addEventListener("input", () => scheduleAutoSave());
}

// Fetch approved documents library
async function fetchDocuments() {
  try {
    const res = await fetch("/api/documents");
    documentsList = await res.json();
    renderDocumentsTable();
  } catch (err) {
    console.error("Failed to fetch documents:", err);
  }
}

function renderDocumentsTable() {
  const tbody = document.getElementById("documents-table-body");
  if (!tbody) return;
  tbody.innerHTML = "";

  documentsList.forEach(doc => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong style="color:var(--primary-cyan);">${doc.code}</strong></td>
      <td><strong>${doc.title}</strong></td>
      <td>${doc.department}</td>
      <td><span class="badge-status badge-active">${doc.version}</span></td>
      <td>${doc.effective_date}</td>
      <td><span class="badge-status badge-rbac">${doc.access_level}</span></td>
      <td><span class="badge-status ${doc.status === 'active' ? 'badge-active' : 'badge-superseded'}">${doc.status.toUpperCase()}</span></td>
      <td><button class="btn-primary-upload" style="padding:4px 8px; font-size:11px;" onclick="viewDocumentDetail('${doc.id}')">View</button></td>
    `;
    tbody.appendChild(tr);
  });
}

window.viewDocumentDetail = async function(docId) {
  try {
    const res = await fetch(`/api/documents/${docId}`);
    const data = await res.json();
    const container = document.getElementById("doc-viewer-container");
    document.getElementById("viewer-doc-code").innerText = data.metadata.code;
    document.getElementById("viewer-doc-title").innerText = `${data.metadata.title} (${data.metadata.version}, ${data.metadata.effective_date})`;
    document.getElementById("viewer-doc-content").innerText = data.full_text;
    container.style.display = "block";
    container.scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    console.error("Failed to load document:", err);
  }
};

function openDocumentViewerByCode(docCode) {
  openModal("modal-evidence");
  const doc = documentsList.find(d => d.code === docCode || docCode.includes(d.code));
  if (doc) {
    window.viewDocumentDetail(doc.id);
  }
}

// Fetch User Roles
async function fetchRoles() {
  try {
    const res = await fetch("/api/roles");
    const roles = await res.json();
    renderRolesList(roles);
  } catch (err) {
    console.error("Failed to fetch roles:", err);
  }
}

function renderRolesList(roles) {
  const listEl = document.getElementById("roles-selection-list");
  if (!listEl) return;
  listEl.innerHTML = "";

  roles.forEach(r => {
    const card = document.createElement("div");
    card.className = `role-card ${r.id === currentUserRole ? 'selected' : ''}`;
    card.innerHTML = `
      <div class="role-info">
        <h4>${r.label}</h4>
        <p>Permissions: ${r.permissions.join(", ")}</p>
      </div>
      <span style="font-size: 11px; font-weight:700; color:var(--primary-cyan);">${r.id === currentUserRole ? 'ACTIVE' : 'SELECT'}</span>
    `;
    card.addEventListener("click", () => {
      currentUserRole = r.id;
      currentRoleLabel.innerText = r.label.split("(")[0].trim();
      closeModal("modal-roles");
      fetchRoles();
      showToast(`Security clearance updated: ${r.label}`);
    });
    listEl.appendChild(card);
  });
}

// Send Grounded RAG Query
async function sendQuery(queryText) {
  const query = queryText || assistantInput.value.trim();
  if (!query) return;

  assistantHeroView.style.display = "none";
  assistantInput.value = "";

  appendChatBubble("user", query);

  const loadingBubble = appendChatBubble("assistant", `
    <div style="display:flex; align-items:center; gap:8px;">
      <span class="role-dot" style="background:#0284c7; animation: orb-breathe 1s infinite alternate;"></span>
      <span>Retrieving hospital policies with ${currentUserRole.replace('_', ' ')} clearance...</span>
    </div>
  `);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query, user_role: currentUserRole })
    });
    const data = await res.json();
    loadingBubble.remove();
    renderAssistantResponse(data);
  } catch (err) {
    loadingBubble.innerHTML = `<span style="color:var(--accent-red);">Network request failed. Please check backend.</span>`;
  }
}

function appendChatBubble(sender, contentHtml) {
  const bubble = document.createElement("div");
  bubble.className = sender === "user" ? "chat-bubble-user" : "chat-bubble-assistant";
  bubble.innerHTML = contentHtml;
  chatDialogueList.appendChild(bubble);
  document.getElementById("chat-messages").scrollTop = document.getElementById("chat-messages").scrollHeight;
  return bubble;
}

function renderAssistantResponse(resp) {
  let innerHtml = "";

  if (resp.conflict_warning) {
    innerHtml += `
      <div class="conflict-alert-box">
        <span style="font-size:16px;">⚠️</span>
        <div><strong>Policy Conflict / Version Alert:</strong><br>${resp.conflict_warning}</div>
      </div>
    `;
  }

  if (resp.access_blocked) {
    innerHtml += `
      <div class="access-denied-box">
        <strong>🔒 SECURITY ACCESS DENIED (RBAC ENFORCED):</strong><br>
        ${resp.answer}
      </div>
    `;
  } else {
    innerHtml += `<div>${formatAnswerMarkdown(resp.answer)}</div>`;
  }

  if (resp.citations && resp.citations.length > 0) {
    innerHtml += `
      <div class="citations-panel">
        <div class="citations-title">Verified Source Citations (${resp.citations.length})</div>
    `;
    resp.citations.forEach(c => {
      innerHtml += `
        <div class="citation-card" onclick="openDocumentViewerByCode('${c.doc_code}')" title="Click to view full passage in Document Viewer">
          <div class="citation-header">
            <span class="cit-code">${c.doc_code} (${c.version}) — p.${c.page_number}</span>
            <span class="cit-conf">${Math.round(c.confidence * 100)}% match</span>
          </div>
          <div style="font-weight:600; color:var(--text-primary); margin-bottom:2px;">${c.section}</div>
          <div class="cit-excerpt">${c.passage_excerpt}</div>
        </div>
      `;
    });
    innerHtml += `</div>`;
  }

  innerHtml += `
    <div style="font-size:11px; color:var(--text-faint); margin-top:8px; display:flex; justify-content:space-between;">
      <span>Mode: ${resp.model_mode}</span>
      <span>Latency: ${resp.latency_ms} ms</span>
    </div>
  `;

  appendChatBubble("assistant", innerHtml);
}

function formatAnswerMarkdown(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n• /g, '<br>• ')
    .replace(/\n1\. /g, '<br>1. ')
    .replace(/\n2\. /g, '<br>2. ')
    .replace(/\n3\. /g, '<br>3. ')
    .replace(/\n4\. /g, '<br>4. ')
    .replace(/\n5\. /g, '<br>5. ')
    .replace(/\n6\. /g, '<br>6. ');
}

// Templates & Snippets Fetching
async function fetchTemplates() {
  try {
    const res = await fetch("/api/templates");
    const templates = await res.json();
    const grid = document.getElementById("templates-grid");
    grid.innerHTML = "";
    templates.forEach(t => {
      const card = document.createElement("div");
      card.className = "template-card";
      card.innerHTML = `
        <div class="template-title">${t.title}</div>
        <div class="template-dept">${t.department}</div>
        <div class="template-preview">${t.content.slice(0, 150)}...</div>
        <button class="btn-primary-action" style="padding:6px 10px; font-size:12px; margin-top:auto;" onclick="insertTemplate('${t.id}')">Insert into Note</button>
      `;
      grid.appendChild(card);
    });
  } catch (err) {
    console.error("Templates error:", err);
  }
}

window.insertTemplate = async function(templateId) {
  try {
    const res = await fetch("/api/templates");
    const templates = await res.json();
    const tpl = templates.find(t => t.id === templateId);
    if (tpl) {
      editorBody.innerText = tpl.content;
      closeModal("modal-templates");
      updateWordCount();
      scheduleAutoSave();
      showToast(`Inserted template: ${tpl.title}`);
    }
  } catch (err) {
    console.error(err);
  }
};

async function fetchSnippets() {
  try {
    const res = await fetch("/api/snippets");
    const snippets = await res.json();
    const list = document.getElementById("snippets-list");
    list.innerHTML = "";
    snippets.forEach(s => {
      const item = document.createElement("div");
      item.className = "snippet-item";
      item.innerHTML = `
        <div class="snippet-title">${s.title}</div>
        <div class="snippet-text">${s.text}</div>
      `;
      item.addEventListener("click", () => {
        document.execCommand("insertText", false, " " + s.text);
        closeModal("modal-snippets");
        showToast(`Inserted: ${s.title}`);
      });
      list.appendChild(item);
    });
  } catch (err) {
    console.error("Snippets error:", err);
  }
}

async function fetchPatients() {
  try {
    const res = await fetch("/api/patients");
    const patients = await res.json();
    const tbody = document.getElementById("patients-table-body");
    tbody.innerHTML = "";
    patients.forEach(p => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong style="color:var(--primary-cyan);">${p.id}</strong></td>
        <td><strong>${p.name}</strong></td>
        <td>${p.gender}, ${p.dob}</td>
        <td>${p.department}</td>
        <td>${p.primary_dx}</td>
        <td><span style="color:#dc2626; font-weight:600;">${p.allergies}</span></td>
        <td><button class="btn-primary-upload" style="padding:4px 8px; font-size:11px;" onclick="startPatientEncounter('${p.name}', '${p.primary_dx}')">+ Encounter</button></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Patients error:", err);
  }
}

window.startPatientEncounter = async function(patientName, chiefComplaint) {
  closeModal("modal-patients");
  const formData = new FormData();
  formData.append("title", chiefComplaint);
  formData.append("patient_name", patientName);

  try {
    const res = await fetch("/api/encounters", { method: "POST", body: formData });
    const newEnc = await res.json();
    await fetchEncounters();
    selectEncounter(newEnc.id);
    showToast(`Created new encounter for ${patientName}`);
  } catch (err) {
    console.error(err);
  }
};

// Event Listeners & Modals
function setupEventListeners() {
  // Tabs
  document.querySelectorAll(".doc-tab[data-tab]").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".doc-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      activeTab = tab.dataset.tab;
      renderEditorContent();
    });
  });

  // Add custom note tab
  document.getElementById("add-custom-tab-btn").addEventListener("click", () => {
    const tabName = prompt("Enter title for new note tab (e.g. Procedure Note, Referral):", "Procedure Note");
    if (!tabName) return;
    const bar = document.getElementById("editor-tabs-bar");
    const addBtn = document.getElementById("add-custom-tab-btn");
    const newTab = document.createElement("button");
    newTab.className = "doc-tab";
    newTab.innerText = tabName;
    newTab.dataset.tab = "custom_" + Date.now();
    newTab.addEventListener("click", () => {
      document.querySelectorAll(".doc-tab").forEach(t => t.classList.remove("active"));
      newTab.classList.add("active");
      editorBody.innerHTML = `<h3>${tabName}</h3><p>Enter clinical notes...</p>`;
    });
    bar.insertBefore(newTab, addBtn);
    newTab.click();
  });

  // Prompt chips
  document.querySelectorAll(".prompt-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      sendQuery(chip.dataset.query);
    });
  });

  // Send query button & Enter key
  sendQueryBtn.addEventListener("click", () => sendQuery());
  assistantInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuery();
    }
  });

  // Voice Query in assistant dock
  document.getElementById("voice-input-btn").addEventListener("click", () => {
    if (isDictatingChat) stopDictation();
    else startDictation("chat");
  });

  // Close doc viewer in evidence modal
  document.getElementById("close-doc-viewer-btn").addEventListener("click", () => {
    document.getElementById("doc-viewer-container").style.display = "none";
  });

  // File Upload in evidence modal
  const fileInput = document.getElementById("policy-file-input");
  fileInput.addEventListener("change", async (e) => {
    if (e.target.files.length === 0) return;
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append("file", file);
    formData.append("title", file.name.replace(/\.[^/.]+$/, ""));
    formData.append("code", "POL-HOSP-" + Math.floor(100 + Math.random() * 900));
    formData.append("department", "Hospital Operations");
    formData.append("version", "v1.0");
    formData.append("access_level", "CLINICAL");
    formData.append("effective_date", "2026-04-01");

    try {
      showToast("Uploading and indexing document pages...");
      const res = await fetch("/api/documents/upload", { method: "POST", body: formData });
      await res.json();
      showToast("Document indexed successfully!");
      fetchDocuments();
    } catch (err) {
      alert("Failed to upload document: " + err.message);
    }
  });

  // Top Header Navigation & Panel Toggles
  const brandHomeBtn = document.getElementById("brand-home-btn");
  if (brandHomeBtn) {
    brandHomeBtn.addEventListener("click", () => {
      selectEncounter("enc-01");
      showToast("MediDocs AI / Vero Pro: Reset to Primary Encounter");
    });
  }

  const toggleSidebarBtn = document.getElementById("toggle-sidebar-btn");
  const sidebarPanel = document.getElementById("sidebar-panel");
  if (toggleSidebarBtn && sidebarPanel) {
    toggleSidebarBtn.addEventListener("click", () => {
      sidebarPanel.classList.toggle("collapsed");
      const isCol = sidebarPanel.classList.contains("collapsed");
      showToast(isCol ? "Sidebar Collapsed" : "Sidebar Expanded");
    });
  }

  const searchGlobalBtn = document.getElementById("search-global-btn");
  if (searchGlobalBtn) {
    searchGlobalBtn.addEventListener("click", () => {
      openModal("modal-evidence");
      fetchDocuments();
      showToast("Searching Approved Hospital Policies");
    });
  }

  const toggleAssistantBtn = document.getElementById("toggle-assistant-btn");
  const closeAssistantBtn = document.getElementById("close-assistant-btn");
  const assistantPanel = document.getElementById("assistant-panel");
  if (toggleAssistantBtn && assistantPanel) {
    toggleAssistantBtn.addEventListener("click", () => {
      assistantPanel.classList.toggle("collapsed");
      const isCol = assistantPanel.classList.contains("collapsed");
      showToast(isCol ? "AI Assistant Panel Hidden" : "AI Assistant Panel Visible");
    });
  }
  if (closeAssistantBtn && assistantPanel) {
    closeAssistantBtn.addEventListener("click", () => {
      assistantPanel.classList.add("collapsed");
      showToast("AI Assistant Panel Hidden");
    });
  }

  // New encounter form submission
  document.getElementById("new-encounter-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const patientName = document.getElementById("new-enc-patient").value;
    const title = document.getElementById("new-enc-title").value;
    const categorySelect = document.getElementById("new-enc-category");
    const category = categorySelect ? categorySelect.value : "TODAY";
    closeModal("modal-new-encounter");

    const formData = new FormData();
    formData.append("title", title);
    formData.append("patient_name", patientName);
    formData.append("category", category);

    try {
      const res = await fetch("/api/encounters", { method: "POST", body: formData });
      const newEnc = await res.json();
      await fetchEncounters();
      selectEncounter(newEnc.id);
      showToast(`Created encounter: ${title} (${category})`);
    } catch (err) {
      console.error(err);
    }
  });

  // Quick lightning action button
  document.getElementById("quick-lightning-btn").addEventListener("click", () => {
    openModal("modal-snippets");
    fetchSnippets();
  });

  // Policy scope filter dropdown
  document.getElementById("dock-policy-filter").addEventListener("click", () => {
    const scopes = ["All Approved Policies", "Pharmacy Only", "ICU & Resuscitation", "Infection Control"];
    const current = document.getElementById("active-policy-filter-label").innerText;
    const nextIdx = (scopes.indexOf(current) + 1) % scopes.length;
    document.getElementById("active-policy-filter-label").innerText = scopes[nextIdx];
    showToast(`Filter set to: ${scopes[nextIdx]}`);
  });

  // Thumbs up / down feedback
  document.getElementById("thumbs-up-btn").addEventListener("click", () => showToast("Feedback recorded: Helpful clinical draft"));
  document.getElementById("thumbs-down-btn").addEventListener("click", () => showToast("Feedback recorded: Suggested note revision"));

  // MobileFineTuner Hyperparameter Sliders
  const rankSlider = document.getElementById("mft-rank-slider");
  const alphaSlider = document.getElementById("mft-alpha-slider");
  const stepsSlider = document.getElementById("mft-steps-slider");

  rankSlider.addEventListener("input", (e) => { document.getElementById("val-rank").innerText = e.target.value; });
  alphaSlider.addEventListener("input", (e) => { document.getElementById("val-alpha").innerText = e.target.value; });
  stepsSlider.addEventListener("input", (e) => { document.getElementById("val-steps").innerText = e.target.value; });
}

// Modal handling
function setupModals() {
  document.getElementById("nav-evidence").addEventListener("click", () => { openModal("modal-evidence"); fetchDocuments(); });
  document.getElementById("nav-templates").addEventListener("click", () => { openModal("modal-templates"); fetchTemplates(); });
  document.getElementById("nav-snippets").addEventListener("click", () => { openModal("modal-snippets"); fetchSnippets(); });
  document.getElementById("nav-patients").addEventListener("click", () => { openModal("modal-patients"); fetchPatients(); });
  document.getElementById("new-encounter-btn").addEventListener("click", () => openModal("modal-new-encounter"));
  document.getElementById("open-role-modal-btn").addEventListener("click", () => openModal("modal-roles"));

  document.getElementById("nav-eval").addEventListener("click", () => {
    openModal("modal-eval");
    triggerBenchmarkSuite();
  });

  document.getElementById("nav-mft").addEventListener("click", () => {
    openModal("modal-mft");
    fetchMFTStatus();
  });

  document.getElementById("nav-audit").addEventListener("click", () => {
    openModal("modal-audit");
    fetchAuditLogs();
  });

  document.getElementById("upload-doc-dock-btn").addEventListener("click", () => openModal("modal-evidence"));

  // Close buttons
  document.querySelectorAll(".modal-close-btn").forEach(btn => {
    btn.addEventListener("click", () => closeModal(btn.dataset.close));
  });

  document.querySelectorAll(".modal-overlay").forEach(overlay => {
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) closeModal(overlay.id);
    });
  });

  // Run benchmark button
  document.getElementById("run-benchmark-btn").addEventListener("click", () => triggerBenchmarkSuite());

  // Category filter chips in evaluation modal
  const evalChipsContainer = document.getElementById("eval-filter-chips");
  const evalChips = evalChipsContainer ? evalChipsContainer.querySelectorAll(".filter-chip") : document.querySelectorAll(".filter-chip");
  evalChips.forEach(chip => {
    chip.addEventListener("click", () => {
      evalChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      filterBenchmarkTable(chip.dataset.cat);
    });
  });

  // Trigger MFT train button
  document.getElementById("trigger-mft-train-btn").addEventListener("click", async () => {
    const btn = document.getElementById("trigger-mft-train-btn");
    const model = document.getElementById("mft-model-select").value;
    const rank = document.getElementById("mft-rank-slider").value;
    const alpha = document.getElementById("mft-alpha-slider").value;
    const steps = document.getElementById("mft-steps-slider").value;

    btn.innerText = "⏳ Running C++ LoRA Forward/Backward Passes...";
    btn.disabled = true;

    try {
      const res = await fetch(`/api/finetuning/train?model_key=${model}&steps=${steps}&rank=${rank}&alpha=${alpha}`, { method: "POST" });
      const data = await res.json();
      btn.innerText = "✓ LoRA Smoke Completed!";
      showToast(`LoRA Checkpoint Saved: ${model}_clinical_lora.safetensors`);

      // Render Loss Badges
      const canvasContainer = document.getElementById("mft-loss-canvas-container");
      const badgesRow = document.getElementById("loss-badges-row");
      badgesRow.innerHTML = "";
      data.training_loss.forEach((loss, i) => {
        badgesRow.innerHTML += `<div class="loss-step-badge">Step ${i+1}: <strong>${loss.toFixed(2)}</strong></div>`;
      });
      canvasContainer.style.display = "block";

      fetchMFTStatus();
    } catch (err) {
      btn.innerText = "Failed";
    } finally {
      setTimeout(() => { btn.disabled = false; btn.innerText = "⚡ Run LoRA Adaptation Smoke"; }, 4000);
    }
  });
}

function openModal(id) {
  const modalEl = document.getElementById(id);
  if (modalEl) modalEl.classList.add("active");
}

function closeModal(id) {
  const modalEl = document.getElementById(id);
  if (modalEl) modalEl.classList.remove("active");
}

let lastBenchmarkResults = [];

async function triggerBenchmarkSuite() {
  const btn = document.getElementById("run-benchmark-btn");
  const kpiMetricsGrid = document.getElementById("kpi-metrics-grid");
  if (btn) {
    btn.innerText = "⏳ Running 50 Tests...";
    btn.disabled = true;
  }
  if (kpiMetricsGrid) {
    kpiMetricsGrid.style.transition = "opacity 0.2s";
    kpiMetricsGrid.style.opacity = "0.7";
  }

  try {
    const res = await fetch("/api/evaluation/run", { method: "POST" });
    const data = await res.json();
    lastBenchmarkResults = data.test_results;

    if (kpiMetricsGrid) kpiMetricsGrid.style.opacity = "1";

    document.getElementById("kpi-recall").innerText = data.summary.retrieval_recall_at_5 + "%";
    document.getElementById("kpi-citation").innerText = data.summary.citation_correctness + "%";
    document.getElementById("kpi-answer").innerText = data.summary.answer_correctness + "%";
    document.getElementById("kpi-unsupported").innerText = data.summary.unsupported_answer_rate + "%";
    document.getElementById("kpi-abstain").innerText = data.summary.abstention_f1 + "%";
    document.getElementById("kpi-rbac").innerText = data.summary.unauthorized_disclosures;

    renderBenchmarkTable(lastBenchmarkResults);
  } catch (err) {
    console.error("Benchmark failed:", err);
  } finally {
    btn.innerText = "▶ Run 50-Test Suite";
    btn.disabled = false;
  }
}

function filterBenchmarkTable(category) {
  if (category === "ALL") {
    renderBenchmarkTable(lastBenchmarkResults);
  } else {
    const filtered = lastBenchmarkResults.filter(t => t.category === category);
    renderBenchmarkTable(filtered);
  }
}

function renderBenchmarkTable(tests) {
  const tbody = document.getElementById("eval-tests-table-body");
  tbody.innerHTML = "";
  tests.forEach(t => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${t.id}</strong></td>
      <td>${t.category}</td>
      <td>${t.query}</td>
      <td><span class="badge-status badge-rbac">${t.user_role}</span></td>
      <td><span class="badge-status ${t.passed ? 'badge-active' : 'badge-superseded'}">${t.passed ? 'PASSED' : 'FAILED'}</span></td>
      <td>${t.latency_ms} ms</td>
      <td>${t.notes}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function fetchMFTStatus() {
  try {
    const res = await fetch("/api/finetuning/status");
    const data = await res.json();

    const coreStatus = document.getElementById("mft-core-status-text");
    if (coreStatus && data.experiment) {
      coreStatus.innerText = `C++17 Engine Ready (liboperators.a) • Base: ${data.experiment.base_model} • ${data.experiment.adapter_type}`;
    }

    const grid = document.getElementById("three-way-comparison-grid");
    grid.innerHTML = "";

    data.comparison.evaluations.forEach((item, idx) => {
      const card = document.createElement("div");
      card.className = `comparison-card ${idx === 2 ? 'highlight-card' : ''}`;
      card.innerHTML = `
        <div class="comp-mode">${item.mode}</div>
        <div class="comp-stats">
          Hallucination: <strong>${item.hallucination_rate}</strong> • Format Adherence: <strong>${item.format_adherence}</strong> • Citation Rate: <strong>${item.citation_rate}</strong>
        </div>
        <div class="comp-quote">"${item.sample_response}"</div>
        <div class="comp-verdict">${item.pros_cons}</div>
      `;
      grid.appendChild(card);
    });
  } catch (err) {
    console.error("Error fetching MFT status:", err);
  }
}

async function fetchAuditLogs() {
  try {
    const res = await fetch("/api/audit");
    const logs = await res.json();
    const tbody = document.getElementById("audit-table-body");
    tbody.innerHTML = "";
    logs.forEach(l => {
      const tr = document.createElement("tr");
      const badgeCls = l.status === "ALLOWED" ? "badge-active" : (l.status === "BLOCKED" ? "badge-superseded" : "badge-rbac");
      tr.innerHTML = `
        <td>${l.timestamp}</td>
        <td><span class="badge-status badge-rbac">${l.user_role}</span></td>
        <td>${l.query}</td>
        <td><span class="badge-status ${badgeCls}">${l.status}</span></td>
        <td>${l.documents_accessed.join(", ") || "None"}</td>
        <td>${l.latency_ms} ms</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error fetching audit logs:", err);
  }
}

function showToast(msg) {
  let toast = document.getElementById("global-toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "global-toast";
    toast.style.cssText = `
      position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
      background: #0f172a; color: #ffffff; padding: 8px 18px; border-radius: 9999px;
      font-size: 13px; font-weight: 500; z-index: 1000; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      transition: opacity 0.2s; opacity: 0; pointer-events: none;
    `;
    document.body.appendChild(toast);
  }
  toast.innerText = msg;
  toast.style.opacity = "1";
  setTimeout(() => { toast.style.opacity = "0"; }, 2500);
}
