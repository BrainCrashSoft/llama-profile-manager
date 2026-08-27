/* ==========================================================================
   Llama Profile Manager - frontend app
   Plain vanilla JS, no build step. Talks to the FastAPI backend over
   fetch() for CRUD/scan/settings, and a WebSocket for live server logs.
   ========================================================================== */

const API = {
  async get(path) { return handle(await fetch(path)); },
  async post(path, body) { return handle(await fetch(path, { method: "POST", headers: jsonHeaders(), body: JSON.stringify(body || {}) })); },
  async put(path, body) { return handle(await fetch(path, { method: "PUT", headers: jsonHeaders(), body: JSON.stringify(body || {}) })); },
  async del(path) { return handle(await fetch(path, { method: "DELETE" })); },
};
function jsonHeaders() { return { "Content-Type": "application/json" }; }
async function handle(resp) {
  let data = null;
  try { data = await resp.json(); } catch (e) { /* no body */ }
  if (!resp.ok) {
    const message = (data && (data.detail || data.error)) || `Request failed (${resp.status})`;
    throw new Error(message);
  }
  return data;
}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

const state = {
  view: "library",
  schema: null,               // { categories, parameters }
  schemaByKey: {},
  settings: null,
  libraryData: null,          // { models: [...] }
  profiles: [],               // all profiles (MRU order from backend)
  profileTab: "models",       // active tab on the profile page: "models" | "profiles"
  selectedProfileId: null,
  editingProfile: null,       // profile object currently being edited (or a draft)
  collapsedCategories: new Set(),
  logSocket: null,
  statusPollTimer: null,
  serverStatus: null,          // last /api/server/status response (incl. profile_id)
  hfResult: null,              // last resolved Hugging Face repo lookup
  hfTab: "search",             // active tab on the download page: "search" | "owned"
  hfSelectedRepo: null,        // org/repo to highlight in the "My Models" list (set by loadRepoDetail)
  // HF namespace (org/user) avatars: namespace -> in-flight Promise while the
  // request is out, then {url: string|null, name: string|null}. The backend
  // keeps its own 30d/24h persistent cache; this L1 just stops re-renders and
  // concurrent rows from piling duplicate requests onto it.
  hfAvatars: {},
  downloads: {},               // key (repo_id::group) -> download status; finished cards stay until dismissed
  dismissedDownloads: {},      // keys the user dismissed; the backend keeps reporting
                               // finished jobs for a while, so these must not resurrect
  activeDownloadKeys: [],      // keys currently queued/downloading (group rows show "downloading…")
  downloadMaxConcurrent: 3,    // from /ws/downloads frames
  renderHfGroups: null,        // latest group-list renderer (for live download updates)
  presets: [],                 // router presets
  allProfiles: [],             // every profile (for preset chips / picker)
  routerModels: [],            // last GET /models from the running router
  routerPollTimer: null,
  routerDirs: [],              // router-dir (--models-dir) profiles
  editingMode: "profile",      // "profile" | "router_dir" - which entity the shared editor is editing
  // GGUF facts for the model the editor is currently on (block_count sets the
  // --n-cpu-moe slider's maximum; context_length / chat_template are provided
  // too). Successful reads are cached per model path; failures aren't, so a
  // re-downloaded / repaired model isn't stuck on a stale error.
  ggufFacts: null,             // facts for the editor's current model, or null = unknown
  ggufFactsByPath: {},         // model_path → facts (from GET /api/gguf/facts)
  metaLoadPending: null,       // model_path with a cold GGUF read in flight (modal candidate)
  metaLoadModal: null,         // model_path of the open "reading metadata" modal
  metaLoadTimer: null,         // delay timer so fast (disk-cached) reads never flash the modal
  // ---- benchmarks ----
  benchmarks: [],              // /api/benchmarks records (newest first)
  benchSort: { key: "started_at", dir: "desc" },
  benchPage: 1,                // current history-table page (1-based)
  benchPerPage: 25,            // row limit per page ("Per page" select)
  benchCompareIds: [],         // ids checked for the ⚖ compare modal (max 2;
                               // check order = A, then B)
  benchSelectedId: null,
  benchFilterText: "",
  activeBenchmarkId: null,     // id of the in-flight run (we poll it)
  benchPollTimer: null,
  lspPollTimer: null,          // llama-server (llama.cpp) download progress poll
  lspLatest: null,             // last /api/llama-server/latest payload (for the card)
  benchLoadSeq: 0,             // monotonic counter: discards stale /api/benchmarks
                               // responses so a slow GET can't clobber a newer one
  // Model-folder file pickers (--mmproj, --chat-template-file):
  // model_path → Promise<Array<{path,filename}>> while the folder scan is in
  // flight, plain Array once resolved. Cached so editor re-renders don't
  // re-scan the folder.
  mmprojSuggestions: {},
  chatTemplateSuggestions: {},
};

function modelId(m) { return `${m.org}::${m.repo}::${m.model_name}`; }

// Space-separated search terms are ANDed together - every term must appear
// somewhere in the haystack, in any order. Used by every local (client-side)
// filter/search box: library filter, parameter search, copy-to-model picker.
function matchesAllTerms(haystack, query) {
  const q = (query || "").trim().toLowerCase();
  if (!q) return true;
  const terms = q.split(/\s+/).filter(Boolean);
  const hay = (haystack || "").toLowerCase();
  return terms.every(t => hay.includes(t));
}

// Order a profile's param keys the same way the editor form does (schema
// order, see state.paramOrder) instead of the order the values were last
// entered. Keys not in the schema are appended at the end, keeping their
// original relative order (sort is stable).
function orderedParamKeys(params) {
  const keys = Object.keys(params || {});
  const order = state.paramOrder;
  if (!order) return keys;
  return keys.sort((a, b) => {
    const oa = order[a] !== undefined ? order[a] : Number.MAX_SAFE_INTEGER;
    const ob = order[b] !== undefined ? order[b] : Number.MAX_SAFE_INTEGER;
    return oa - ob;
  });
}

// Last segment of a path-like string, for display only. Strings without a
// path separator are returned unchanged.
function pathBasename(s) {
  const t = String(s);
  const i = Math.max(t.lastIndexOf("/"), t.lastIndexOf("\\"));
  return i >= 0 ? t.slice(i + 1) : t;
}

// How a param value is shown in the Parameters lists: path-like values are
// shortened to their file name, because a long unbroken path would squash
// the flag out of the row. The command preview and the actual command still
// use the full stored value.
function shortParamValue(v) {
  if (v === true) return "";
  const t = String(v);
  return (t.includes("/") || t.includes("\\")) ? pathBasename(t) : t;
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", init);

async function init() {
  wireNav();
  wireLibrary();
  wireDownloadPage();
  wireEditor();
  wireServer();
  wireRouter();
  wireRouterDir();
  wireBenchmarks();
  wireSettings();

  try {
    state.schema = await API.get("/api/schema");
    state.schemaByKey = {};
    state.schema.parameters.forEach(p => state.schemaByKey[p.key] = p);
    // Display order of param keys, exactly as the editor form lists them:
    // schema categories in order, then parameters within each category.
    // Used to show a profile's overridden params in that order instead of
    // the order they were last entered.
    state.paramOrder = {};
    let orderIdx = 0;
    state.schema.categories.forEach(cat => {
      state.schema.parameters
        .filter(p => p.category === cat.id)
        .forEach(p => { if (!(p.key in state.paramOrder)) state.paramOrder[p.key] = orderIdx++; });
    });
    // Safety net: a parameter whose category isn't in the category list
    // still gets an order (appended) instead of drifting to the end.
    state.schema.parameters.forEach(p => {
      if (!(p.key in state.paramOrder)) state.paramOrder[p.key] = orderIdx++;
    });
  } catch (e) { toast(e.message, "error"); }

  try {
    state.settings = await API.get("/api/settings");
    applyTheme(state.settings.theme);
    populateSettingsForm();
    populateBenchFormFromSettings();
  } catch (e) { toast(e.message, "error"); }

  await loadLibrary(false);
  await loadProfiles();
  await refreshServerStatus();
  startStatusPolling();
  connectLogSocket();
  loadPresets();
  loadRouterDirs();
  refreshRouterCapability();
  refreshRouterDirCapability();
}

function applyTheme(theme) {
  document.body.classList.toggle("theme-light", theme === "light");
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

function wireNav() {
  document.querySelectorAll(".nav-item").forEach(btn => {
    btn.addEventListener("click", () => showView(btn.dataset.view));
  });
  document.querySelectorAll("[data-goto]").forEach(btn => {
    btn.addEventListener("click", () => showView(btn.dataset.goto));
  });
}

function showView(view) {
  state.view = view;
  // Leaving the editor: the in-flight read keeps running (its cache entry
  // still fills), but the "reading metadata" modal belongs to the editor -
  // don't leave it floating over another page.
  if (view !== "editor") {
    if (state.metaLoadTimer) { clearTimeout(state.metaLoadTimer); state.metaLoadTimer = null; }
    state.metaLoadPending = null;
    closeMetaLoadModal();
  }
  document.querySelectorAll(".view").forEach(el => el.hidden = true);
  document.getElementById(`view-${view}`).hidden = false;
  document.querySelectorAll(".nav-item").forEach(el => el.classList.toggle("is-active", el.dataset.view === view));

  // Returning to the profile page, refresh the list so rows ("last used")
  // reflect anything that happened while on another view (e.g. a server
  // start updating last_used_at).
  if (view === "library") loadProfiles();
  // Same for the router presets page - presets/profiles may have changed
  // while the user was elsewhere (e.g. a preset started or edited from
  // another window).
  if (view === "router") loadPresets();
  if (view === "router_dir") loadRouterDirs();
  if (view === "benchmarks") loadBenchmarks();
  // Settings: refresh the llama.cpp auto-download card (resumes the progress
  // poll if a job is in flight); leaving it clears the poll (same hygiene as
  // the benchmark poll timers - the job keeps running server-side).
  if (view === "settings") { refreshLlamaServerCard(); refreshAppVersionCard(); }
  else if (state.lspPollTimer) { clearInterval(state.lspPollTimer); state.lspPollTimer = null; }
  if (view === "download" && state.hfTab === "owned") {
    loadLibrary(false).then(() => {
      if (state.view === "download" && state.hfTab === "owned") {
        renderOwnedModels(document.getElementById("hf-search-query").value.trim());
      }
    });
  }
}

// ---------------------------------------------------------------------------
// Library view
// ---------------------------------------------------------------------------

function wireLibrary() {
  document.getElementById("btn-rescan").addEventListener("click", () => loadLibrary(true));
  document.getElementById("profile-search").addEventListener("input", debounce(renderProfileList, 150));
  document.getElementById("btn-new-profile-global").addEventListener("click", openNewProfileModal);
  document.getElementById("btn-import-profiles").addEventListener("click", () => importProfilesFromFile());
  document.getElementById("btn-export-all-profiles").addEventListener("click", exportAllProfiles);
  document.querySelectorAll(".profile-tab").forEach(tab => {
    tab.addEventListener("click", () => setProfileTab(tab.dataset.profileTab));
  });
}

function setProfileTab(tab) {
  state.profileTab = tab;
  document.querySelectorAll(".profile-tab").forEach(el =>
    el.classList.toggle("is-active", el.dataset.profileTab === tab));
  const isModels = tab === "models";
  document.getElementById("profile-models-list").hidden = !isModels;
  document.getElementById("profile-list").hidden = isModels;
}

async function loadLibrary(rescan) {
  try {
    state.libraryData = await API.get(`/api/models${rescan ? "?rescan=true" : ""}`);
  } catch (e) {
    toast(e.message, "error");
    state.libraryData = { models: [], roots: [], errors: [] };
  }
  updateProfileSub();
  renderProfileList();
  if (state.selectedProfileId) renderProfileDetail();
}

async function loadProfiles() {
  try {
    state.profiles = await API.get("/api/profiles");
  } catch (e) {
    toast(e.message, "error");
    state.profiles = [];
  }
  if (state.selectedProfileId && !state.profiles.some(p => p.id === state.selectedProfileId)) {
    state.selectedProfileId = null;
  }
  updateProfileSub();
  renderProfileList();
  renderProfileDetail();
  resumeBenchmarkPollingIfNeeded();
}

function modelForProfile(p) {
  return ((state.libraryData && state.libraryData.models) || []).find(m => modelId(m) === p.model_id);
}

// Groups that contain non-model-weight files (vision projections, draft
// models, imatrices) can't get a launch profile - they are hidden from the
// Models tab and their "create a new profile" button is removed on the
// download page.
function hasProfileExcludedFiles(files) {
  return (files || []).some(f => /mmproj|dflash|imatrix/i.test(f.filename || ""));
}

function updateProfileSub() {
  const d = state.libraryData;
  const sub = document.getElementById("profile-sub");
  if (!sub) return;
  if (!d || !d.roots || d.roots.length === 0) {
    sub.textContent = "No folders scanned yet - add one in Settings.";
    return;
  }
  const scannedAt = d.scanned_at ? new Date(d.scanned_at * 1000).toLocaleString() : "never";
  const modelCount = new Set((state.profiles || []).map(p => p.model_id)).size;
  sub.textContent = `${(state.profiles || []).length} profile(s) across ${modelCount} model(s) · last scan ${scannedAt}`;
  if (d.errors && d.errors.length) {
    d.errors.forEach(err => toast(`${err.root}: ${err.error}`, "error"));
  }
}

function renderProfileList() {
  // Both tabs share the filter box, and both lists are cheap to render
  // (local data), so keep both up to date - switching tabs is instant.
  renderFlatProfileList();
  renderModelGroupList();
}

// "Profiles" tab: one flat row per profile (the classic layout).
function renderFlatProfileList() {
  const container = document.getElementById("profile-list");
  const query = document.getElementById("profile-search").value;

  const visible = (state.profiles || []).filter(p => {
    const m = modelForProfile(p);
    const hay = m
      ? `${p.name} ${m.org} ${m.repo} ${m.model_name} ${m.quant || ""}`
      : `${p.name} ${p.model_id}`;
    return matchesAllTerms(hay, query);
  });

  container.innerHTML = "";
  if (visible.length === 0) {
    if ((state.profiles || []).length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <p>No profiles yet.</p>
          <p class="muted">Create one from a model in your library.</p>
          <button class="btn btn-primary" id="profile-list-new">+ New Profile</button>
        </div>
      `;
      container.querySelector("#profile-list-new").addEventListener("click", openNewProfileModal);
    } else {
      container.innerHTML = `
        <div class="empty-state">
          <p>No profiles match "${escapeHtml(query)}".</p>
        </div>
      `;
    }
    return;
  }

  visible.forEach(p => container.appendChild(buildProfileRowEl(p)));
}

// One clickable profile row (Start button, running badge, benchmark badge).
// Shared by the flat "Profiles" tab and the nested rows of the "Models" tab.
function buildProfileRowEl(p) {
  const m = modelForProfile(p);
  const stale = !m;
  const row = document.createElement("div");
  row.className = "list-row" + (state.selectedProfileId === p.id ? " is-active" : "");
  row.dataset.profileId = p.id;
  row.dataset.stale = stale ? "1" : "";
  const isOwner = runningProfileId() === p.id;
  const ownerState = isOwner && state.serverStatus ? state.serverStatus.state : null;
  const startLabel = isOwner
    ? (ownerState === "starting" ? "⏳ Loading…" : ownerState === "stopping" ? "■ Stopping…" : "■ Stop")
    : "▶ Start";
  const startCls = isOwner && ownerState === "running" ? "btn btn-danger btn-tiny" : "btn btn-primary btn-tiny";
  row.innerHTML = `
    <div class="list-row-main">
      <div class="list-row-title">${escapeHtml(p.name)}<span class="running-badge" ${isOwner && ownerState === "running" ? "" : "hidden"}>● running</span>${benchBadgeHtml(p)}</div>
      <div class="list-row-meta">
        <span>${escapeHtml(modelNameFromId(p.model_id))}</span>
        <span>${p.last_used_at ? "used " + new Date(p.last_used_at * 1000).toLocaleString() : "never used"}</span>
        ${stale ? `<span>⚠ model not found</span>` : ""}
      </div>
    </div>
    <div class="list-row-actions">
      <button class="${startCls}" data-act="start" ${stale || (isOwner && ownerState !== "running") ? "disabled" : ""}>${startLabel}</button>
    </div>
  `;
  row.addEventListener("click", () => selectProfile(p.id));
  if (!stale) {
    row.querySelector('[data-act="start"]').addEventListener("click", e => {
      e.stopPropagation();
      if (runningProfileId() === p.id) stopServer();
      else startServerWithProfile(p.id);
    });
  }
  // Benchmark badge → jumps to that record on the Benchmarks page.
  row.querySelectorAll(".bench-badge[data-bench-id]").forEach(el => {
    el.addEventListener("click", e => { e.stopPropagation(); openBenchmarkRecord(el.dataset.benchId); });
  });
  return row;
}

// ---------------------------------------------------------------------------
// Hugging Face namespace avatars
//
// The icon shown next to a model row is the *namespace's* (org/user) avatar -
// what huggingface.co itself shows next to a repo; there is no per-repo icon
// on the Hub, so every repo under one org shares the org's avatar.
// ---------------------------------------------------------------------------

function isAvatarNamespace(ns) {
  return typeof ns === "string" && ns.length > 0 && ns !== "(ungrouped)" && /^[\w.\-]+$/.test(ns);
}

// ~6 preset hues; a small string hash of the namespace picks one, so the same
// org/user always gets the same badge color across reloads (CSS defines hue-0…5).
function avatarHueClass(ns) {
  let h = 0;
  for (let i = 0; i < ns.length; i++) h = (h * 31 + ns.charCodeAt(i)) >>> 0;
  return `hf-avatar-hue-${h % 6}`;
}

// HF CDN: loaded by the webview directly, no proxy.
function avatarImgEl(url) {
  const img = document.createElement("img");
  img.src = url;
  img.alt = "";
  img.loading = "lazy";
  return img;
}

// Async path only: the row may already be gone by the time the request
// resolves, so verify the span is still in the document before touching it.
function applyAvatarEl(el, info) {
  if (!el.isConnected || !info || !info.url) return;
  el.appendChild(avatarImgEl(info.url));
  if (info.name) el.title = info.name;
}

// A 24×24 .hf-avatar box: the initial-letter fallback badge is ALWAYS in the
// DOM (so ungrouped models, offline, and unknown namespaces all degrade
// gracefully), and the <img> is appended only once a URL is known - a broken
// image can therefore never flash a broken-image icon.
function modelAvatarEl(namespace, label) {
  const span = document.createElement("span");
  span.className = "hf-avatar";
  const base = isAvatarNamespace(namespace) ? namespace : (label || namespace || "?");
  const fb = document.createElement("span");
  fb.className = "hf-avatar-fallback " + avatarHueClass(base);
  fb.textContent = (base.trim().charAt(0) || "?").toUpperCase();
  span.appendChild(fb);

  if (!isAvatarNamespace(namespace)) {
    if (base) span.title = base;
    return span;
  }

  const cached = state.hfAvatars[namespace];
  if (cached instanceof Promise) {
    // Another row already has this namespace in flight - piggyback on it.
    cached.then(info => applyAvatarEl(span, info)).catch(() => {});
    span.title = namespace;
    return span;
  }
  if (cached) {
    span.title = cached.name || namespace;
    // Sync path: the caller appends the span a few lines later, so the span
    // is NOT in the document yet and applyAvatarEl's isConnected guard would
    // silently skip it (this is what dropped icons on every warm re-render,
    // e.g. after Rescan). Build the <img> directly; a span that never makes
    // it into the DOM is discarded with its children, which is fine.
    if (cached.url) span.appendChild(avatarImgEl(cached.url));
    return span;
  }

  const promise = API.get(`/api/hf/avatar?namespace=${encodeURIComponent(namespace)}`)
    .then(data => ({ url: data.avatarUrl || null, name: data.fullName || null }))
    .catch(() => null);
  state.hfAvatars[namespace] = promise;
  promise.then(info => {
    // Settle the cache entry (null result = "unknown", also cached in memory
    // so we don't re-poll it for the session).
    state.hfAvatars[namespace] = info || { url: null, name: null };
    applyAvatarEl(span, info);
  }).catch(() => {});
  span.title = namespace;
  return span;
}

// "Models" tab: library models grouped by repo - a repo header row (avatar,
// "More Q."), then one row per quantization with a "+ Profile" button, and the
// profiles nested below their quantization. Stale profiles (model gone from
// the library) only appear in the flat "Profiles" tab, since they have no
// model group to sit in.
function renderModelGroupList() {
  const container = document.getElementById("profile-models-list");
  if (!container) return;
  const query = document.getElementById("profile-search").value;
  const models = (state.libraryData && state.libraryData.models) || [];
  const profiles = state.profiles || [];

  const profilesByModel = {};
  profiles.forEach(p => {
    (profilesByModel[p.model_id] = profilesByModel[p.model_id] || []).push(p);
  });

  const visible = models
    // mmproj / dflash / imatrix groups hold no model weights - no profiles.
    .filter(m => !hasProfileExcludedFiles(m.files))
    .filter(m => {
      const groupProfiles = profilesByModel[modelId(m)] || [];
      const hay = `${m.org}/${m.repo} ${m.model_name} ${m.quant || ""} ${groupProfiles.map(p => p.name).join(" ")}`;
      return matchesAllTerms(hay, query);
    })
    .sort((a, b) =>
      `${a.org}/${a.repo}/${a.model_name}`.localeCompare(`${b.org}/${b.repo}/${b.model_name}`));

  container.innerHTML = "";
  if (models.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <p>No models in your library yet.</p>
        <p class="muted">Rescan, or download a model from the Library page.</p>
      </div>
    `;
    return;
  }
  if (visible.length === 0) {
    container.innerHTML = `<div class="empty-state"><p>No models match "${escapeHtml(query)}".</p></div>`;
    return;
  }

  // Group the visible models by repo, so each repo renders once with its
  // quantizations as a second level below (visible is already sorted by
  // org/repo/model, so insertion order keeps everything sorted).
  const byRepo = new Map();
  visible.forEach(m => {
    const repoId = `${m.org}/${m.repo}`;
    if (!byRepo.has(repoId)) byRepo.set(repoId, { org: m.org, repo: m.repo, models: [] });
    byRepo.get(repoId).models.push(m);
  });

  byRepo.forEach(r => {
    const repoId = `${r.org}/${r.repo}`;
    const totalBytes = r.models.reduce((sum, m) => sum + (m.total_size_bytes || 0), 0);
    const group = document.createElement("div");
    group.className = "model-group";
    group.innerHTML = `
      <div class="list-row is-static model-group-head">
        <div class="list-row-main">
          <div class="list-row-title">${escapeHtml(repoId)}</div>
          <div class="list-row-meta">
            <span>${r.models.length} quant${r.models.length === 1 ? "" : "s"}</span>
            <span>${formatBytes(totalBytes)}</span>
          </div>
        </div>
        <div class="list-row-actions">
          ${isAvatarNamespace(r.org) && isAvatarNamespace(r.repo)
            ? `<button class="btn btn-tiny" data-act="more-quants" title="Browse all quantizations of this repo on Hugging Face">More Q.</button>`
            : ""}
        </div>
      </div>
      <div class="model-group-children"></div>
    `;
    group.querySelector(".model-group-head").prepend(modelAvatarEl(r.org, r.repo));

    const children = group.querySelector(".model-group-children");
    r.models.forEach(m => {
      const groupProfiles = profilesByModel[modelId(m)] || [];
      const branch = document.createElement("div");
      branch.className = "model-branch";
      branch.innerHTML = `
        <div class="list-row is-static model-branch-head">
          <div class="list-row-main">
            <div class="list-row-title">${escapeHtml(m.model_name)}</div>
            <div class="list-row-meta">
              <span>${formatBytes(m.total_size_bytes)}</span>
              <span>${groupProfiles.length} profile${groupProfiles.length === 1 ? "" : "s"}</span>
            </div>
          </div>
          <div class="list-row-actions">
            <button class="icon-btn" data-act="add-profile" title="Add a profile for this model">+</button>
          </div>
        </div>
        ${groupProfiles.length ? `<div class="model-branch-children"></div>` : ""}
      `;
      // Models with no profiles show no children section at all (the + Profile
      // button is the entry point).
      const branchChildren = branch.querySelector(".model-branch-children");
      if (branchChildren) {
        groupProfiles.forEach(p => {
          const child = buildProfileRowEl(p);
          child.classList.add("model-group-child");
          branchChildren.appendChild(child);
        });
      }
      branch.querySelector('[data-act="add-profile"]').addEventListener("click", e => {
        e.stopPropagation();
        openEditorForNewProfile(m, m.files[0].path);
      });
      children.appendChild(branch);
    });

    const moreQuantsBtn = group.querySelector('[data-act="more-quants"]');
    if (moreQuantsBtn) {
      moreQuantsBtn.addEventListener("click", async e => {
        e.stopPropagation();
        showView("download");
        // "My Models" tab with this repo's row selected, so the list on the
        // left matches the quantizations loaded on the right.
        await setHfTab("owned");
        selectOwnedRepo(repoId);
      });
    }
    container.appendChild(group);
  });
}

function selectProfile(id) {
  state.selectedProfileId = id;
  renderProfileList();
  renderProfileDetail();
}

async function renderProfileDetail() {
  const el = document.getElementById("profile-detail");
  const p = (state.profiles || []).find(x => x.id === state.selectedProfileId);
  if (!p) {
    el.innerHTML = `
      <div class="empty-state">
        <p>Select a profile to view details, or create a new one.</p>
        <button class="btn btn-primary" id="profile-detail-new">+ New Profile</button>
      </div>
    `;
    el.querySelector("#profile-detail-new").addEventListener("click", openNewProfileModal);
    return;
  }

  const m = modelForProfile(p);
  const isUngrouped = m && (m.org === "(ungrouped)" || m.repo === "(ungrouped)");
  const paramRows = orderedParamKeys(p.params)
    .map(k => [k, p.params[k]])
    .filter(([, v]) => v !== undefined && v !== null && v !== "" && v !== false)
    .map(([k, v]) => {
      const sp = state.schemaByKey[k];
      const flag = (sp && sp.flag) || ("--" + k);
      return `<div class="file-row"><span>${escapeHtml(flag)}</span><span>${escapeHtml(shortParamValue(v))}</span></div>`;
    }).join("");

  el.innerHTML = `
    <div class="detail-header">
      <div style="width:100%">
        <h2 class="detail-title">${escapeHtml(p.name)}</h2>
        <div class="detail-path">${escapeHtml(modelLabelFromId(p.model_id))}</div>
      </div>
      <div class="detail-header-actions">
        <button class="icon-btn" id="pd-open-folder" title="Open containing folder" ${m ? "" : "disabled"}>
          <svg viewBox="0 0 20 20" width="16" height="16"><path d="M2 5.5A1.5 1.5 0 0 1 3.5 4h4l1.5 2H16.5A1.5 1.5 0 0 1 18 7.5v7A1.5 1.5 0 0 1 16.5 16h-13A1.5 1.5 0 0 1 2 14.5v-9Z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/></svg>
        </button>
        <button class="icon-btn" id="pd-open-hf" title="Open on Hugging Face" ${m && !isUngrouped ? "" : "disabled"}>
          <span aria-hidden="true">🤗</span>
        </button>
      </div>
    </div>
    ${m ? `
    <div class="detail-stats">
      <div class="stat-block"><span class="stat-value">${formatBytes(m.total_size_bytes)}</span><span class="stat-label">Model size</span></div>
      <div class="stat-block"><span class="stat-value">${m.files.length}</span><span class="stat-label">File(s)</span></div>
      <div class="stat-block"><span class="stat-value">${new Date(m.latest_modified * 1000).toLocaleDateString()}</span><span class="stat-label">Modified</span></div>
    </div>
    ` : `
    <div class="preview-warn">⚠ This profile's model is not in your library anymore. Rescan (or re-download) before starting.</div>
    `}
    <div class="profile-row-actions" style="margin-bottom:6px; flex-wrap:wrap;">
      <button class="btn btn-primary btn-tiny" id="pd-start" ${m ? "" : "disabled"}>▶ Start</button>
      <button class="btn btn-tiny" id="pd-benchmark" ${m ? "" : "disabled"} title="Measure prefill/generation speed with this exact configuration">⚡ Benchmark</button>
      <button class="btn btn-tiny" id="pd-edit">Edit</button>
      <button class="btn btn-tiny" id="pd-duplicate">Duplicate</button>
      <button class="btn btn-tiny" id="pd-copy">Copy to model…</button>
      <button class="btn btn-tiny" id="pd-export">Export</button>
      <button class="btn btn-tiny" id="pd-delete">Delete</button>
    </div>
    ${profileBadgeBlock(p)}
    <div class="profile-list-head"><strong>Parameters</strong></div>
    <div class="file-list">
      ${paramRows || `<p class="muted small">Default parameters (none overridden).</p>`}
    </div>
    ${p.custom_flags && p.custom_flags.trim() ? `
    <div class="profile-list-head"><strong>Advanced: Custom flags</strong></div>
    <pre class="preview-code">${escapeHtml(p.custom_flags.trim())}</pre>
    ` : ""}
    ${p.notes ? `
    <div class="profile-list-head"><strong>Notes</strong></div>
    <p class="muted small">${escapeHtml(p.notes)}</p>
    ` : ""}
    <div class="profile-list-head"><strong>Command</strong>
      <button class="btn btn-tiny" id="pd-copy-command">Copy</button>
    </div>
    <pre class="preview-code" id="pd-command">…</pre>
  `;

  // Namespace avatar (badge for ungrouped / unknown / offline) next to the title.
  el.querySelector(".detail-header").prepend(modelAvatarEl(m ? m.org : null, m ? m.model_name : p.name));

  el.querySelector("#pd-start").addEventListener("click", () => {
    if (runningProfileId() === p.id) stopServer();
    else startServerWithProfile(p.id);
  });
  el.querySelector("#pd-benchmark").addEventListener("click", () => {
    const sel = document.getElementById("bench-profile");
    if (sel && !sel.disabled) sel.value = p.id;
    startBenchmarkFromForm(p.id);
  });
  const pdBenchOpen = el.querySelector("#pd-bench-open");
  if (pdBenchOpen) pdBenchOpen.addEventListener("click", () => openBenchmarkRecord(pdBenchOpen.dataset.benchId));
  el.querySelector("#pd-edit").addEventListener("click", () => openEditorForProfile(p));
  el.querySelector("#pd-duplicate").addEventListener("click", async () => {
    try {
      const dup = await API.post(`/api/profiles/${p.id}/duplicate`, {});
      toast("Profile duplicated", "ok");
      await loadProfiles();
      if (dup && dup.id) selectProfile(dup.id);
    } catch (e) { toast(e.message, "error"); }
  });
  el.querySelector("#pd-copy").addEventListener("click", () => openCopyToModelModal(p));
  el.querySelector("#pd-export").addEventListener("click", () => exportProfile(p));
  el.querySelector("#pd-delete").addEventListener("click", async () => {
    if (!await confirmModal({ title: `Delete profile "${p.name}"?`, message: "This can't be undone." })) return;
    try {
      await API.del(`/api/profiles/${p.id}`);
      toast("Profile deleted", "ok");
      state.selectedProfileId = null;
      await loadProfiles();
    } catch (e) { toast(e.message, "error"); }
  });
  if (m) {
    el.querySelector("#pd-open-folder").addEventListener("click", () => openModelFolder(m.files[0].path));
    if (!isUngrouped) el.querySelector("#pd-open-hf").addEventListener("click", () => openModelOnHuggingFace(m));
  }
  el.querySelector("#pd-copy-command").addEventListener("click", () => {
    const pre = document.getElementById("pd-command");
    if (!pre || pre.textContent === "…") { toast("Command not available yet.", "error"); return; }
    navigator.clipboard.writeText(pre.textContent).then(() => toast("Command copied", "ok")).catch(() => {});
  });
  updateProfileRunningChrome(); // reflect Loading…/Stop immediately if this profile owns the server

  try {
    const data = await API.get(`/api/profiles/${p.id}/command-preview`);
    const pre = document.getElementById("pd-command");
    if (pre && state.selectedProfileId === p.id) pre.textContent = data.command;
  } catch (e) {
    const pre = document.getElementById("pd-command");
    if (pre && state.selectedProfileId === p.id) pre.textContent = "(command preview unavailable)";
  }
}

function openNewProfileModal() {
  const models = ((state.libraryData && state.libraryData.models) || [])
    .filter(m => m.files && m.files.length > 0);
  if (models.length === 0) {
    toast("No models in your library yet - download one first.", "error");
    return;
  }
  openModelPickerModal({
    title: "New profile - pick a model",
    subtitle: "A profile wraps one specific model file. Pick the model this profile should launch.",
    models,
    confirmLabel: "Create",
    onConfirm: (m) => openEditorForNewProfile(m, m.files[0].path),
  });
}

async function exportProfile(profile) {
  try {
    const data = await API.get(`/api/profiles/${profile.id}/export`);
    const filename = `${profile.name.replace(/\s+/g, "_")}.json`;
    const content = JSON.stringify(data, null, 2);

    if (hasNativeDialogs()) {
      // Inside the pywebview window, a plain <a download> click has no
      // browser download manager to catch it, so it silently does nothing.
      // Use the native save-file dialog exposed via the Python side instead.
      const savedPath = await window.pywebview.api.save_text_file(filename, content);
      if (savedPath) toast(`Saved to ${savedPath}`, "ok");
      return;
    }
    downloadJson(filename, data);
  } catch (e) {
    toast(e.message, "error");
  }
}

// In-app confirmation dialog for destructive actions (replaces the browser's
// window.confirm). Resolves true when the user confirms, false otherwise
// (cancel button, backdrop click, or Escape).
function confirmModal({ title, message = "", confirmLabel = "Delete" }) {
  return new Promise(resolve => {
    const backdrop = document.createElement("div");
    backdrop.className = "modal-backdrop";
    backdrop.innerHTML = `
      <div class="modal-card modal-card-confirm">
        <h3>${escapeHtml(title)}</h3>
        <div class="confirm-message">${escapeHtml(message)}</div>
        <div class="modal-actions">
          <button class="btn btn-ghost" data-act="cancel">Cancel</button>
          <button class="btn btn-danger" data-act="confirm">${escapeHtml(confirmLabel)}</button>
        </div>
      </div>
    `;
    document.body.appendChild(backdrop);
    let done = false;
    const close = answer => {
      if (done) return;
      done = true;
      document.removeEventListener("keydown", onKey);
      backdrop.remove();
      resolve(answer);
    };
    const onKey = e => { if (e.key === "Escape") close(false); };
    document.addEventListener("keydown", onKey);
    backdrop.addEventListener("click", e => { if (e.target === backdrop) close(false); });
    backdrop.querySelector('[data-act="cancel"]').addEventListener("click", () => close(false));
    backdrop.querySelector('[data-act="confirm"]').addEventListener("click", () => close(true));
    // Focus Cancel (the safe choice) by default.
    backdrop.querySelector('[data-act="cancel"]').focus();
  });
}

function openModelPickerModal({ title, subtitle, models, confirmLabel, onConfirm, preselectModelId = null }) {
  const sortedModels = models
    .slice()
    .sort((a, b) => `${a.org}/${a.repo}/${a.model_name}`.localeCompare(`${b.org}/${b.repo}/${b.model_name}`));

  // Optional preselection (import flow): no behavior change when omitted.
  let selected = preselectModelId ? models.find(m => modelId(m) === preselectModelId) || null : null;
  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  backdrop.innerHTML = `
    <div class="modal-card">
      <h3>${escapeHtml(title)}</h3>
      <p class="muted small">${escapeHtml(subtitle)}</p>
      <div class="modal-filter"><input type="search" id="model-picker-filter" class="input" placeholder="Filter by name, org, or quant…" /></div>
      <div class="modal-list" id="model-picker-list"></div>
      <div class="modal-actions">
        <button class="btn btn-ghost" id="model-picker-cancel">Cancel</button>
        <button class="btn btn-primary" id="model-picker-confirm" disabled>${escapeHtml(confirmLabel)}</button>
      </div>
    </div>
  `;
  document.body.appendChild(backdrop);

  const listEl = backdrop.querySelector("#model-picker-list");
  const confirmBtn = backdrop.querySelector("#model-picker-confirm");

  function renderList(filterText) {
    const visible = sortedModels.filter(m => matchesAllTerms(`${m.org} ${m.repo} ${m.model_name} ${m.quant || ""}`, filterText));

    listEl.innerHTML = "";
    if (visible.length === 0) {
      listEl.innerHTML = `<p class="muted small" style="padding:10px;">No models match "${escapeHtml(filterText)}".</p>`;
      return;
    }
    visible.forEach(m => {
      const item = document.createElement("div");
      const isSelected = selected && modelId(selected) === modelId(m);
      item.className = "list-row" + (isSelected ? " is-active" : "");
      item.innerHTML = `
        <div class="list-row-main">
          <div class="list-row-title">${escapeHtml(m.model_name)}</div>
          <div class="list-row-meta"><span>${escapeHtml(m.org)} / ${escapeHtml(m.repo)}</span></div>
        </div>
      `;
      item.addEventListener("click", () => {
        selected = m;
        listEl.querySelectorAll(".list-row").forEach(el => el.classList.remove("is-active"));
        item.classList.add("is-active");
        confirmBtn.disabled = false;
      });
      listEl.appendChild(item);
    });
  }
  renderList("");
  if (selected) confirmBtn.disabled = false;

  backdrop.querySelector("#model-picker-filter").addEventListener("input", e => renderList(e.target.value));

  const close = () => backdrop.remove();
  backdrop.addEventListener("click", e => { if (e.target === backdrop) close(); });
  backdrop.querySelector("#model-picker-cancel").addEventListener("click", close);
  confirmBtn.addEventListener("click", async () => {
    if (!selected) return;
    close();
    try {
      await onConfirm(selected);
    } catch (e) {
      toast(e.message, "error");
    }
  });
}

async function exportAllProfiles() {
  try {
    const data = await API.get("/api/profiles/export-all");
    const n = (data.profiles || []).length;
    const d = new Date();
    const pad = x => String(x).padStart(2, "0");
    const stamp = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
    const filename = `llama-profile-manager-${stamp}.json`;

    if (hasNativeDialogs()) {
      // Same rationale as exportProfile: inside the pywebview window a
      // plain <a download> click has no download manager to catch it.
      const savedPath = await window.pywebview.api.save_text_file(filename, JSON.stringify(data, null, 2));
      if (savedPath) toast(`Exported ${n} profiles to ${savedPath}`, "ok");
      return;
    }
    downloadJson(filename, data);
    toast(`Exported ${n} profiles`, "ok");
  } catch (e) {
    toast(e.message, "error");
  }
}

// ---------------------------------------------------------------------------
// Profile import (single + all) - the collection-level complement to the
// per-profile Export button. Accepts, from one JSON file:
//   * a single canonical profile object (also a benchmark snapshot), or
//   * the export-all wrapper { app, version, profiles: [...] }, or
//   * a bare array of canonical profile objects.
// Target model(s) are picked locally; model-folder params are re-rooted by
// the backend (_store_model_folder_paths via the import endpoints).
// ---------------------------------------------------------------------------

// Reads the import file: native OPEN dialog in the desktop window, the hidden
// <input type=file> (plus FileReader via File.text) in browser-fallback mode.
// Resolves to the file's text, or null when the user cancels.
async function readImportFile() {
  if (hasNativeDialogs()) {
    const res = await window.pywebview.api.open_text_file();
    if (!res) return null;                 // cancelled
    if (res.error) throw new Error(res.error);
    return res.content;
  }
  const input = document.getElementById("file-import-profiles");
  return await new Promise((resolve, reject) => {
    input.value = "";
    let settled = false;
    const done = (fn, v) => {
      if (settled) return;
      settled = true;
      window.removeEventListener("focus", onFocus, true);
      fn(v);
    };
    const onFocus = () => {
      // The dialog's cancel button fires no `change`/`cancel` anywhere near
      // reliably, so on window refocus give `change` a beat; if nothing was
      // picked by then, treat it as a cancel.
      setTimeout(() => {
        if (!settled && (!input.files || input.files.length === 0)) done(resolve, null);
      }, 250);
    };
    input.addEventListener("change", () => {
      const f = input.files && input.files[0];
      if (!f) { done(resolve, null); return; }
      f.text().then(t => done(resolve, t), e => done(reject, e));
    }, { once: true });
    window.addEventListener("focus", onFocus, true);
    input.click();
  });
}

async function importProfilesFromFile() {
  let text;
  try {
    text = await readImportFile();
  } catch (e) {
    toast(e.message, "error");
    return;
  }
  if (text == null) return;               // user cancelled the picker
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (e) {
    toast("That file isn't valid JSON.", "error");
    return;
  }
  if (Array.isArray(parsed)) {
    startBatchImport(parsed);
  } else if (parsed && typeof parsed === "object" && Array.isArray(parsed.profiles)) {
    startBatchImport(parsed.profiles);    // export-all wrapper
  } else if (parsed && typeof parsed === "object") {
    startSingleImport(parsed);            // single canonical profile / snapshot
  } else {
    toast("Unrecognized file - expected a profile export, a list of them, or an Export-all file.", "error");
  }
}

// Best-effort match of a file entry to a local library model, in order:
//   1. exact model_id (path-derived; works when the library layout matches)
//   2. same model file name ("same model, different folder")
//   3. same HF org/repo (any quantization in that repo)
// Returns the model or null - never a reason to block the import.
function matchImportModel(data, models) {
  if (!data || typeof data !== "object" || Array.isArray(data)) return null;
  if (typeof data.model_id === "string" && data.model_id) {
    const m = models.find(m => modelId(m) === data.model_id);
    if (m) return m;
  }
  const srcBase = data.model_path ? pathBasename(String(data.model_path)) : "";
  if (srcBase) {
    const m = models.find(m => m.files && m.files.length > 0 && pathBasename(m.files[0].path) === srcBase);
    if (m) return m;
  }
  const parts = typeof data.model_id === "string" ? data.model_id.split("::") : [];
  if (parts.length >= 2 && parts[0] && parts[1]) {
    const m = models.find(m => m.org === parts[0] && m.repo === parts[1]);
    if (m) return m;
  }
  return null;
}

// Single-object file: pick the target model (preselected when the source
// model exists locally), then POST to the existing single-import endpoint.
function startSingleImport(data) {
  const name = (typeof data.name === "string" ? data.name : "").trim();
  if (!name || typeof data.model_id !== "string" || !data.model_id) {
    toast("This file doesn't look like a profile export - it's missing name or model_id.", "error");
    return;
  }
  const models = ((state.libraryData && state.libraryData.models) || [])
    .filter(m => m.files && m.files.length > 0);
  if (models.length === 0) {
    toast("No models in your library yet - add one first.", "error");
    return;
  }
  const match = matchImportModel(data, models);
  openModelPickerModal({
    title: `Import "${name}"`,
    subtitle: "Pick the local model to import this profile to. Parameters, custom flags and notes come from the file.",
    models,
    confirmLabel: "Import",
    preselectModelId: match ? modelId(match) : null,
    onConfirm: async (m) => {
      await API.post("/api/profiles/import", {
        model_id: modelId(m),
        model_path: m.files[0].path,
        data,
      });
      toast(`Imported "${name}" to ${m.model_name}`, "ok");
      await loadProfiles();
    },
  });
}

// Multi-entry file: confirmation modal with a per-row target-model select.
// Rows that matched a local model are pre-filled; unmatched rows default to
// the first library model and are marked, so the user consciously picks.
function startBatchImport(entries) {
  if (!Array.isArray(entries) || entries.length === 0) {
    toast("Nothing to import - the file contains no profiles.", "error");
    return;
  }
  const models = ((state.libraryData && state.libraryData.models) || [])
    .filter(m => m.files && m.files.length > 0);
  if (models.length === 0) {
    toast("No models in your library yet - add one first.", "error");
    return;
  }
  const sorted = models.slice().sort((a, b) =>
    `${a.org}/${a.repo}/${a.model_name}`.localeCompare(`${b.org}/${b.repo}/${b.model_name}`));

  const rows = entries.map((data, i) => {
    const isObj = !!data && typeof data === "object" && !Array.isArray(data);
    const name = isObj && typeof data.name === "string" && data.name.trim() ? data.name.trim() : "(unnamed)";
    const src = isObj
      ? (data.model_id || (data.model_path ? pathBasename(String(data.model_path)) : "no model info"))
      : "invalid entry";
    const match = isObj ? matchImportModel(data, models) : null;
    const targetId = match ? modelId(match) : modelId(sorted[0]);
    const options = sorted.map(m =>
      `<option value="${escapeHtml(modelId(m))}"${modelId(m) === targetId ? " selected" : ""}>`
      + `${escapeHtml(m.model_name)} - ${escapeHtml(m.org)} / ${escapeHtml(m.repo)}</option>`).join("");
    return {
      data, isObj, name, src, targetId,
      html: `
      <div class="import-all-row${!match ? " is-unmatched" : ""}">        
        <div class="import-all-cell import-all-cell-name">
          <div class="import-all-name">
            <span class="import-all-name-text">${escapeHtml(name)}</span>
            ${!match ? '<span class="import-all-flag" title="No local model matched - pick the target below">⚠ no match</span>' : ""}
            ${!isObj ? '<span class="import-all-flag">⚠ invalid entry</span>' : ""}
          </div>
          <div class="import-all-src">${escapeHtml(src)}</div>
        </div>
        <div class="import-all-cell import-all-cell-target">
          <select class="input import-all-select" data-idx="${i}"${isObj ? "" : " disabled"}>${options}</select>
        </div>
      </div>`,
    };
  });

  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  backdrop.innerHTML = `
    <div class="modal-card">
      <h3>Import ${entries.length} profile${entries.length === 1 ? "" : "s"}</h3>
      <p class="muted small">Pick the local model for each entry. Rows without a matching local model are
        marked ⚠ and default to the first model - change them if needed. Name collisions are auto-suffixed.</p>
      <div class="import-all-list">
        <div class="import-all-head">
          <div class="import-all-cell import-all-cell-name">Profile (from file)</div>
          <div class="import-all-cell import-all-cell-target">Import to model</div>
        </div>
        ${rows.map(r => r.html).join("")}
      </div>
      <div class="modal-actions">
        <button class="btn btn-ghost" id="import-all-cancel">Cancel</button>
        <button class="btn btn-primary" id="import-all-confirm">Import ${entries.length}</button>
      </div>
    </div>
  `;
  document.body.appendChild(backdrop);

  const close = () => backdrop.remove();
  backdrop.addEventListener("click", e => { if (e.target === backdrop) close(); });
  backdrop.querySelector("#import-all-cancel").addEventListener("click", close);
  backdrop.querySelector("#import-all-confirm").addEventListener("click", async () => {
    const items = [...backdrop.querySelectorAll(".import-all-row")].map((rowEl, i) => {
      const sel = rowEl.querySelector(".import-all-select");
      const m = models.find(x => modelId(x) === sel.value);
      return { model_id: sel.value, model_path: m.files[0].path, data: entries[i] };
    });
    close();
    let res;
    try {
      res = await API.post("/api/profiles/import-all", { profiles: items });
    } catch (e) { toast(e.message, "error"); return; }
    const nOk = (res.imported || []).length;
    const nErr = (res.errors || []).length;
    if (nErr === 0) {
      toast(`Imported ${nOk} profile${nOk === 1 ? "" : "s"}`, "ok");
    } else {
      const first = res.errors[0] || {};
      const why = first.error ? ` - ${first.name}: ${first.error}` : "";
      toast(`Imported ${nOk} - ${nErr} failed${why}`, "error");
    }
    await loadProfiles();
  });
}

function openCopyToModelModal(profile) {
  const otherModels = ((state.libraryData && state.libraryData.models) || []).filter(m => modelId(m) !== profile.model_id);
  if (otherModels.length === 0) {
    toast("No other models found to copy this profile to.", "error");
    return;
  }
  openModelPickerModal({
    title: `Copy "${profile.name}" to…`,
    subtitle: "Pick the model to copy this profile's parameters to. A new profile is created there - this one is left untouched.",
    models: otherModels,
    confirmLabel: "Copy",
    onConfirm: async (m) => {
      const exportData = await API.get(`/api/profiles/${profile.id}/export`);
      // The copy is a sibling, not a duplicate - give it a distinct name
      // (create_profile/_unique_name still suffixes if this collides too).
      exportData.name = `Copy of ${profile.name}`;
      await API.post("/api/profiles/import", {
        model_id: modelId(m),
        model_path: m.files[0].path,
        data: exportData,
      });
      toast(`Copied to ${m.model_name}`, "ok");
      await loadProfiles();
    },
  });
}

async function openModelFolder(filePath) {
  try {
    await API.post("/api/system/open-folder", { path: filePath });
  } catch (e) {
    toast(e.message, "error");
  }
}

async function openModelOnHuggingFace(m) {
  const url = `https://huggingface.co/${encodeURIComponent(m.org)}/${encodeURIComponent(m.repo)}`;
  if (hasNativeDialogs()) {
    // Route through the backend so it opens in the real default browser
    // instead of navigating inside the app's own window.
    try {
      await API.post("/api/system/open-url", { url });
      return;
    } catch (e) {
      toast(e.message, "error");
      return;
    }
  }
  window.open(url, "_blank", "noopener");
}

function openDownloadPageForModel(m) {
  showView("download");
  const input = document.getElementById("hf-input");
  input.value = `${m.org}/${m.repo}`;
  resolveHfInput();
}

// ---------------------------------------------------------------------------
// Download page (Hugging Face lookup + download)
// ---------------------------------------------------------------------------

function wireDownloadPage() {
  document.getElementById("btn-hf-resolve").addEventListener("click", resolveHfInput);
  document.getElementById("hf-input").addEventListener("keydown", e => {
    if (e.key === "Enter") resolveHfInput();
  });
  document.getElementById("btn-hf-search").addEventListener("click", searchHuggingFace);
  document.getElementById("hf-search-query").addEventListener("keydown", e => {
    if (e.key !== "Enter") return;
    if (state.hfTab === "owned") renderOwnedModels(e.target.value.trim());
    else searchHuggingFace();
  });
  const debouncedAutoSearch = debounce(() => {
    const q = document.getElementById("hf-search-query").value.trim();
    if (state.hfTab === "owned") {
      renderOwnedModels(q);
      return;
    }
    if (q) searchHuggingFace();
    else resetHfSearchResults();
  }, 400);
  document.getElementById("hf-search-query").addEventListener("input", debouncedAutoSearch);
  document.getElementById("hf-search-sort").addEventListener("change", () => {
    if (state.hfTab === "search") searchHuggingFace();
  });
  connectDownloadSocket();

  document.querySelectorAll(".hf-tab").forEach(tab => {
    tab.addEventListener("click", () => setHfTab(tab.dataset.hfTab));
  });
}

function setHfTab(tab) {
  state.hfTab = tab;
  const isOwned = tab === "owned";
  document.querySelectorAll(".hf-tab").forEach(el =>
    el.classList.toggle("is-active", el.dataset.hfTab === tab));
  document.getElementById("hf-search-results").hidden = isOwned;
  document.getElementById("hf-owned-results").hidden = !isOwned;
  // The search box doubles as a live filter for "My Models"; the limit and
  // Search button only make sense for the Hugging Face API search.
  const queryEl = document.getElementById("hf-search-query");
  queryEl.value = "";
  queryEl.placeholder = isOwned
    ? "Filter my models…"
    : "Search Hugging Face for GGUF models…";
  document.getElementById("hf-search-limit").hidden = isOwned;
  document.getElementById("hf-search-sort").hidden = isOwned;
  document.getElementById("btn-hf-search").hidden = isOwned;
  if (isOwned) {
    // Refresh (cheap, cached) so the list reflects models downloaded since
    // the last scan, then render. The promise is returned so callers can act
    // once the list is up (e.g. "More Q." selecting a repo in it).
    return loadLibrary(false).then(() => {
      if (state.hfTab === "owned") renderOwnedModels("");
    });
  }
  return Promise.resolve();
}

const BIN_ICON_SVG = `<svg viewBox="0 0 20 20" width="15" height="15"><path d="M4 6h12M8.2 6V4.6A1.6 1.6 0 0 1 9.8 3h.4a1.6 1.6 0 0 1 1.6 1.6V6M5.6 6l.6 9.1a1.6 1.6 0 0 0 1.6 1.5h4.4a1.6 1.6 0 0 0 1.6-1.5L14.4 6M8.3 9.2v4.2M11.7 9.2v4.2" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

// Ask, then delete model files from disk. The backend refuses any path
// outside the configured root folders and refreshes the scan cache.
async function deleteModelFiles(label, paths, onDone) {
  paths = (paths || []).filter(Boolean);
  if (paths.length === 0) { toast("No local files to delete.", "error"); return; }
  if (!await confirmModal({
    title: `Delete "${label}" from disk?`,
    message: `This permanently deletes ${paths.length} file(s) of "${label}" from disk.\n\nThis can't be undone.`,
  })) return;
  try {
    const res = await API.post("/api/models/delete", { paths });
    toast(`Deleted ${res.deleted} file(s)`, "ok");
    await loadLibrary(false);
    if (onDone) await onDone();
  } catch (e) { toast(e.message, "error"); }
}

// A quantization group from /api/hf/resolve carries HF repo-relative file
// paths, not local disk paths. Resolve the real ones: prefer the matching
// local model group (absolute paths from the scan); fall back to the
// {root}/{org}/{repo}/ layout that mark_downloaded checks (candidates from
// every root - the backend only deletes the ones that exist).
function localPathsForQuantGroup(repoId, g) {
  const [org, repo] = repoId.split("/");
  const matches = ((state.libraryData && state.libraryData.models) || [])
    .filter(m => m.org === org && m.repo === repo && m.model_name === g.group_name);
  const scanned = matches.flatMap(m => (m.files || []).map(f => f.path));
  if (scanned.length > 0) return scanned;
  const roots = ((state.settings && state.settings.model_root_folders) || []);
  return roots.flatMap(root =>
    (g.files || []).map(f => `${String(root).replace(/\\/g, "/")}/${org}/${repo}/${f.filename}`));
}

function renderOwnedModels(query) {
  const resultsEl = document.getElementById("hf-owned-results");
  const models = (state.libraryData && state.libraryData.models) || [];
  if (models.length === 0) {
    resultsEl.innerHTML = `
      <div class="empty-state">
        <p>No models in your library yet.</p>
        <p class="muted">Download one from the Search tab, or add a root folder in Settings.</p>
      </div>
    `;
    return;
  }

  // One row per org/repo - the individual quants show up in the detail
  // panel on the right when a repo is clicked.
  const byRepo = {};
  models.forEach(m => {
    const key = `${m.org}/${m.repo}`;
    const entry = byRepo[key] || (byRepo[key] = { org: m.org, repo: m.repo, models: [] });
    entry.models.push(m);
  });
  const repos = Object.values(byRepo)
    .filter(r => matchesAllTerms(
      `${r.org} ${r.repo} ${r.org}/${r.repo} ${r.models.map(m => `${m.model_name} ${m.quant || ""}`).join(" ")}`,
      query || ""))
    .sort((a, b) => `${a.org}/${a.repo}`.localeCompare(`${b.org}/${b.repo}`));

  if (repos.length === 0) {
    resultsEl.innerHTML = `<div class="empty-state"><p>No models match "${escapeHtml(query)}".</p></div>`;
    return;
  }

  resultsEl.innerHTML = "";
  // Selected row: the repo the user most recently picked (loadRepoDetail sets
  // it synchronously, so this also covers "More Q." selections made from the
  // profile page before the lookup resolves).
  const activeRepo = state.hfSelectedRepo || (state.hfResult && state.hfResult.repo_id);
  repos.forEach(r => {
    const repoId = `${r.org}/${r.repo}`;
    const totalBytes = r.models.reduce((sum, m) => sum + (m.total_size_bytes || 0), 0);
    const row = document.createElement("div");
    row.dataset.repo = repoId;
    row.className = "list-row" + (activeRepo === repoId ? " is-active" : "");
    row.innerHTML = `
      <div class="list-row-main">
        <div class="list-row-title">${escapeHtml(repoId)}</div>
        <div class="list-row-meta">
          <span>${r.models.length} quant${r.models.length === 1 ? "" : "s"}</span>
          <span>${formatBytes(totalBytes)}</span>
        </div>
      </div>
      <div class="list-row-actions">
        <button class="icon-btn" data-act="delete" title="Delete downloaded files">${BIN_ICON_SVG}</button>
      </div>
    `;
    row.prepend(modelAvatarEl(r.org, r.repo));
    row.addEventListener("click", () => {
      document.getElementById("hf-input").value = repoId;
      resultsEl.querySelectorAll(".list-row").forEach(el => el.classList.remove("is-active"));
      row.classList.add("is-active");
      loadRepoDetail(repoId, null);
    });
    row.querySelector('[data-act="delete"]').addEventListener("click", e => {
      e.stopPropagation();
      deleteModelFiles(repoId, r.models.flatMap(m => (m.files || []).map(f => f.path)), async () => {
        renderOwnedModels(document.getElementById("hf-search-query").value.trim());
        if (state.hfResult && state.hfResult.repo_id === repoId) await resolveHfInput();
      });
    });
    resultsEl.appendChild(row);
  });
}

// Highlight the given org/repo in the "My Models" list and load its
// quantizations on the right - the same effect as clicking the row. Used by
// the profile page's "More Q." button.
function selectOwnedRepo(repoId) {
  document.getElementById("hf-input").value = repoId;
  const resultsEl = document.getElementById("hf-owned-results");
  resultsEl.querySelectorAll(".list-row").forEach(el =>
    el.classList.toggle("is-active", el.dataset.repo === repoId));
  loadRepoDetail(repoId, null);
}

async function searchHuggingFace() {
  const query = document.getElementById("hf-search-query").value.trim();
  const limitInput = document.getElementById("hf-search-limit");
  const limit = parseInt(limitInput.value, 10) || 10;
  const sort = document.getElementById("hf-search-sort").value;

  const resultsEl = document.getElementById("hf-search-results");
  resultsEl.innerHTML = `<p class="muted small">Searching…</p>`;
  try {
    const result = await API.post("/api/hf/search", { query, limit, sort });
    renderHfSearchResults(result.results);
  } catch (e) {
    resultsEl.innerHTML = "";
    toast(e.message, "error");
  }
}

function resetHfSearchResults() {
  document.getElementById("hf-search-results").innerHTML = `
    <div class="empty-state">
      <p>Search above, or paste a repo to browse its quantizations.</p>
      <p class="muted">Limited to repos tagged <code>gguf</code>.</p>
    </div>
  `;
}

function renderHfSearchResults(results) {
  const resultsEl = document.getElementById("hf-search-results");
  if (!results || results.length === 0) {
    resultsEl.innerHTML = `<div class="empty-state"><p>No GGUF repos found for that search.</p></div>`;
    return;
  }
  resultsEl.innerHTML = "";
  results.forEach(r => {
    const row = document.createElement("div");
    row.className = "list-row" + (state.hfResult && state.hfResult.repo_id === r.repo_id ? " is-active" : "");
    row.innerHTML = `
      <div class="list-row-main">
        <div class="list-row-title">${escapeHtml(r.repo_id)}</div>
        <div class="list-row-meta">
          <span>⬇ ${formatCount(r.downloads)}</span>
          <span>♥ ${formatCount(r.likes)}</span>
          ${r.pipeline_tag ? `<span>${escapeHtml(r.pipeline_tag)}</span>` : ""}
          ${r.gated ? `<span class="flag-tag flag-tag-warn">gated</span>` : ""}
        </div>
      </div>
    `;
    row.prepend(modelAvatarEl(r.repo_id.split("/")[0], r.repo_id));
    row.addEventListener("click", () => {
      document.getElementById("hf-input").value = r.repo_id;
      resultsEl.querySelectorAll(".list-row").forEach(el => el.classList.remove("is-active"));
      row.classList.add("is-active");
      loadRepoDetail(r.repo_id, r);
    });
    resultsEl.appendChild(row);
  });
}

function formatCount(n) {
  if (n === null || n === undefined) return "0";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

async function resolveHfInput() {
  const input = document.getElementById("hf-input").value.trim();
  if (!input) return;
  await loadRepoDetail(input, null);
}

async function loadRepoDetail(repoId, searchMeta) {
  const roots = (state.settings && state.settings.model_root_folders) || [];
  if (roots.length === 0) {
    toast("Add a model root folder in Settings first - that's where downloads will be saved.", "error");
    return;
  }

  // Highlight target for the "My Models" list - set synchronously so the
  // selection survives re-renders that land while the lookup is in flight.
  state.hfSelectedRepo = repoId;

  const detailEl = document.getElementById("hf-detail");
  detailEl.innerHTML = `<p class="muted small">Looking up repository…</p>`;
  try {
    const result = await API.post("/api/hf/resolve", { input: repoId });
    state.hfResult = result;
    await renderHfDetail(result, searchMeta);
  } catch (e) {
    detailEl.innerHTML = `<div class="empty-state"><p>${escapeHtml(e.message)}</p></div>`;
  }
}

async function renderHfDetail(result, searchMeta) {
  const detailEl = document.getElementById("hf-detail");
  const roots = (state.settings && state.settings.model_root_folders) || [];

  // How many profiles exist per local model, so the rows can show
  // "Profiles N" for downloaded models that already have some.
  const profileCounts = {};
  try {
    (await API.get("/api/profiles")).forEach(p => {
      profileCounts[p.model_id] = (profileCounts[p.model_id] || 0) + 1;
    });
  } catch (e) { /* leave counts empty - the label simply won't show */ }
  const [orgId, repoId] = result.repo_id.split("/");

  const metaLine = searchMeta
    ? `⬇ ${formatCount(searchMeta.downloads)} downloads · ♥ ${formatCount(searchMeta.likes)} likes${searchMeta.pipeline_tag ? " · " + escapeHtml(searchMeta.pipeline_tag) : ""}`
    : `${result.groups.length} quantization(s) available`;

  detailEl.innerHTML = `
    <div class="detail-header">
      <div style="width:100%">
        <h2 class="detail-title">${escapeHtml(result.repo_id)}</h2>
        <div class="detail-path">${metaLine}</div>
      </div>
      <div class="detail-header-actions">
        <button class="icon-btn" id="btn-hf-detail-folder" title="Open local folder">
          <svg viewBox="0 0 20 20" width="16" height="16"><path d="M2 5.5A1.5 1.5 0 0 1 3.5 4h4l1.5 2H16.5A1.5 1.5 0 0 1 18 7.5v7A1.5 1.5 0 0 1 16.5 16h-13A1.5 1.5 0 0 1 2 14.5v-9Z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/></svg>
        </button>
        <button class="icon-btn" id="btn-hf-detail-open" title="Open on Hugging Face">
          <span aria-hidden="true">🤗</span>
        </button>
      </div>
    </div>
    ${roots.length > 1 ? `
      <div class="download-target-row">
        <label for="download-target-root">Save into</label>
        <select id="download-target-root" class="input">
          ${roots.map(r => `<option value="${escapeHtml(r)}">${escapeHtml(r)}</option>`).join("")}
        </select>
      </div>
    ` : ""}
    <div class="hf-group-filter">
      <label class="param-toggle"><input type="checkbox" id="hf-downloaded-only"> Downloaded only</label>
    </div>
    <div class="list-group" id="hf-group-list"></div>
  `;

  const downloadedOnlyEl = document.getElementById("hf-downloaded-only");
  downloadedOnlyEl.checked = !!state.hfDownloadedOnly;
  const renderHfGroups = () => {
    state.renderHfGroups = renderHfGroups;
    const listEl = document.getElementById("hf-group-list");
    listEl.innerHTML = "";
    const visible = result.groups.filter(g => !downloadedOnlyEl.checked || g.already_downloaded);
    if (visible.length === 0) {
      listEl.innerHTML = `<p class="muted small" style="padding:10px;">No quantizations match the current filter.</p>`;
      return;
    }
    visible.forEach(g => {
    const row = document.createElement("div");
    row.className = "list-row is-static" + (g.already_downloaded ? " is-flagged-ok" : "");
    const isNoProfile = hasProfileExcludedFiles(g.files);
    const dlState = state.downloads[`${result.repo_id}::${g.group_name}`]?.state;
    const isDownloading = dlState === "downloading" || dlState === "queued";
    row.innerHTML = `
      <div class="list-row-main">
        <div class="list-row-title">${escapeHtml(g.group_name)}</div>
        <div class="list-row-meta">
          <span>${formatBytes(g.total_size_bytes)}</span>
          ${g.is_multipart ? `<span>${g.files.length} parts</span>` : ""}
        </div>
      </div>
      <div class="list-row-actions">
        ${g.already_downloaded
          ? (profileCounts[`${orgId}::${repoId}::${g.group_name}`]
            ? `<span class="flag-tag flag-tag-ok" title="Downloaded - ${profileCounts[`${orgId}::${repoId}::${g.group_name}`]} profile(s)">Profiles ${profileCounts[`${orgId}::${repoId}::${g.group_name}`]}</span>`
            : `<span class="flag-tag flag-tag-ok">✓ Downloaded</span>`)
          : isDownloading
            ? `<button class="btn btn-tiny" disabled>${dlState === "queued" ? "queued…" : "downloading…"}</button>`
            : `<button class="btn btn-primary btn-tiny" data-act="download">⇩ Download</button>`}
        ${g.already_downloaded && !isNoProfile
          ? `<button class="icon-btn" data-act="new-profile" title="Create a new profile for this model">
              <svg viewBox="0 0 20 20" width="15" height="15"><path d="M10 4.5v11M4.5 10h11" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
            </button>`
          : ""}
        ${g.already_downloaded ? `<button class="icon-btn" data-act="delete" title="Delete from disk">${BIN_ICON_SVG}</button>` : ""}
      </div>
    `;
    if (!g.already_downloaded && !isDownloading) {
      row.querySelector('[data-act="download"]').addEventListener("click", () => startDownload(result.repo_id, g));
    }
    if (g.already_downloaded && !isNoProfile) {
      row.querySelector('[data-act="new-profile"]').addEventListener("click", () => {
        const [org, repo] = result.repo_id.split("/");
        const m = ((state.libraryData && state.libraryData.models) || [])
          .find(x => x.org === org && x.repo === repo && x.model_name === g.group_name && x.files && x.files.length > 0);
        if (!m) {
          toast("Model not found in the local scan - rescan first.", "error");
          return;
        }
        openEditorForNewProfile(m, m.files[0].path);
      });
    }
    if (g.already_downloaded) {
      row.querySelector('[data-act="delete"]').addEventListener("click", e => {
        e.stopPropagation();
        deleteModelFiles(g.group_name, localPathsForQuantGroup(result.repo_id, g), async () => {
          // Re-resolve so the group's already-downloaded flag refreshes (the
          // Download button comes back), and refresh the "My Models" list in
          // case this was the repo's last local quant - it is built from the
          // (now refreshed) library scan.
          renderOwnedModels(document.getElementById("hf-search-query").value.trim());
          await resolveHfInput();
        });
      });
    }
    listEl.appendChild(row);
    });
  };
  downloadedOnlyEl.addEventListener("change", () => {
    state.hfDownloadedOnly = downloadedOnlyEl.checked;
    renderHfGroups();
  });
  renderHfGroups();

  document.getElementById("btn-hf-detail-open").addEventListener("click", () => {
    const [org, repo] = result.repo_id.split("/");
    openModelOnHuggingFace({ org, repo });
  });

  const folderBtn = document.getElementById("btn-hf-detail-folder");
  const localModels = ((state.libraryData && state.libraryData.models) || [])
    .filter(m => `${m.org}/${m.repo}` === result.repo_id);
  if (localModels.length > 0 && localModels[0].files.length > 0) {
    folderBtn.addEventListener("click", () => openModelFolder(localModels[0].files[0].path));
  } else {
    folderBtn.disabled = true;
    folderBtn.title = "No local files for this repo yet";
  }

}

async function startDownload(repoId, group) {
  const roots = (state.settings && state.settings.model_root_folders) || [];
  const selectEl = document.getElementById("download-target-root");
  const targetRoot = selectEl ? selectEl.value : roots[0];
  if (!targetRoot) {
    toast("Add a model root folder in Settings first.", "error");
    return;
  }
  try {
    await API.post("/api/downloads/start", {
      repo_id: repoId,
      group_name: group.group_name,
      target_root: targetRoot,
      files: group.files.map(f => ({ path: f.path, filename: f.filename, size_bytes: f.size_bytes })),
    });
    // Optimistic card so the UI reacts before the first WebSocket frame.
    const key = `${repoId}::${group.group_name}`;
    delete state.dismissedDownloads[key];   // re-download of a dismissed group
    state.downloads[key] = Object.assign(state.downloads[key] || {}, {
      repo_id: repoId, group_name: group.group_name, state: "queued",
      total_files: (group.files || []).length,
      bytes_total_overall: group.total_size_bytes || 0,
      bytes_done_overall: 0,
    });
    updateActiveDownloadKeys();
    if (state.renderHfGroups) state.renderHfGroups();
    syncDownloadCards();
    toast(`Downloading ${group.group_name}…`, "ok");
  } catch (e) {
    toast(e.message, "error");
  }
}

async function cancelDownload(key) {
  try { await API.post("/api/downloads/cancel", { key }); }
  catch (e) { toast(e.message, "error"); }
}

function connectDownloadSocket() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${location.host}/ws/downloads`;
  try {
    const socket = new WebSocket(url);
    socket.onmessage = evt => {
      try { renderDownloadFrame(JSON.parse(evt.data)); } catch (e) { /* ignore malformed frame */ }
    };
    socket.onclose = () => setTimeout(connectDownloadSocket, 3000);
    socket.onerror = () => {};
  } catch (e) { /* older browsers / pywebview: skip */ }
}

// ---- download progress (up to 3 concurrent, one card each) ----

function updateActiveDownloadKeys() {
  state.activeDownloadKeys = Object.entries(state.downloads)
    .filter(([, d]) => d.state === "queued" || d.state === "downloading")
    .map(([key]) => key);
}

function renderDownloadFrame(frame) {
  if (frame && frame.max_concurrent) state.downloadMaxConcurrent = frame.max_concurrent;
  const frameDownloads = (frame && frame.downloads) || {};
  // The backend keeps reporting terminal jobs for up to 10 min after they
  // finish; drop dismiss markers once the backend has pruned the job itself.
  for (const k of Object.keys(state.dismissedDownloads || {})) {
    if (!frameDownloads[k]) delete state.dismissedDownloads[k];
  }
  const prevActive = new Set(state.activeDownloadKeys || []);
  let stateTransitioned = false;
  for (const [key, d] of Object.entries(frameDownloads)) {
    if (state.dismissedDownloads[key]) continue;  // user dismissed it; keep it gone
    const prev = state.downloads[key];
    state.downloads[key] = Object.assign({}, prev, d);
    // Toast + side effects fire exactly once, on the transition into a terminal state.
    if (prev && prev.state !== d.state) {
      stateTransitioned = true;
      if (d.state === "done") {
        toast(`Finished downloading ${d.repo_id} - ${d.group_name}`, "ok");
        if (state.hfResult && state.hfResult.repo_id === d.repo_id) resolveHfInput();
        loadLibrary(true);
      } else if (d.state === "error") {
        toast(d.error_message || "Download failed.", "error");
      } else if (d.state === "cancelled") {
        toast("Download cancelled.", "ok");
      }
    }
  }
  updateActiveDownloadKeys();
  syncDownloadCards();
  const nextActive = new Set(state.activeDownloadKeys);
  // Re-render the group rows when a row appears/disappears OR when an active
  // download's state transitions - queued -> downloading keeps the same key
  // active (the set is unchanged) but flips the row's button label.
  const rowsChanged = prevActive.size !== nextActive.size ||
    [...nextActive].some(k => !prevActive.has(k)) ||
    stateTransitioned;
  if (rowsChanged && state.renderHfGroups) state.renderHfGroups();
}

// Reconcile the panel with state.downloads WITHOUT rebuilding cards that are
// unchanged: frames arrive 4x/sec per active download, and a full innerHTML
// rewrite would constantly destroy/recreate buttons under the cursor (hover
// flicker, and ✕-dismiss clicks on finished cards never landing while other
// downloads are still running).
function syncDownloadCards() {
  const panel = document.getElementById("download-panel");
  if (!panel) return;
  const entries = Object.entries(state.downloads || {});
  if (entries.length === 0) { panel.hidden = true; panel.innerHTML = ""; return; }
  panel.hidden = false;
  const seen = new Set();
  entries.forEach(([key, d]) => {
    seen.add(key);
    let card = panel.querySelector(`.download-card[data-key="${CSS.escape(key)}"]`);
    if (!card) {
      card = buildDownloadCard(key, d);
      panel.appendChild(card);
    } else if (card.dataset.state !== d.state) {
      // Structural change (queued -> downloading -> terminal): swap the card.
      card.replaceWith(buildDownloadCard(key, d));
    } else {
      updateDownloadCard(card, d);
    }
  });
  // Drop cards for keys no longer tracked (dismissed or pruned by the backend).
  panel.querySelectorAll(".download-card").forEach(c => {
    if (!seen.has(c.dataset.key)) c.remove();
  });
}

function updateDownloadCard(card, d) {
  // Same state - only refresh the bits that change every frame.
  const fill = card.querySelector(".progress-fill");
  if (fill) {
    const pct = d.bytes_total_overall ? Math.min(100, (d.bytes_done_overall / d.bytes_total_overall) * 100) : 0;
    fill.style.width = `${pct.toFixed(1)}%`;
  }
  const meta = card.querySelector(".progress-meta");
  if (meta && meta.children.length === 2 && d.state === "downloading") {
    const speed = d.speed_bytes_per_sec ? ` · ${formatBytes(d.speed_bytes_per_sec)}/s` : "";
    meta.children[0].textContent = `${formatBytes(d.bytes_done_overall || 0)} / ${formatBytes(d.bytes_total_overall || 0)}${speed}`;
    meta.children[1].textContent = d.current_file ? `${d.file_index}/${d.total_files} - ${d.current_file}` : "starting…";
  }
}

function buildDownloadCard(key, d) {
  const card = document.createElement("div");
  const terminal = d.state === "done" || d.state === "error" || d.state === "cancelled";
  const title = `${d.repo_id || key} - ${d.group_name || ""}`.replace(/\s*-\s*$/, "");
  const pct = d.bytes_total_overall ? Math.min(100, (d.bytes_done_overall / d.bytes_total_overall) * 100) : 0;
  const speed = d.speed_bytes_per_sec ? ` · ${formatBytes(d.speed_bytes_per_sec)}/s` : "";

  let actions, body;
  if (terminal) {
    const word = d.state === "done" ? "✓ Done"
      : d.state === "error" ? "✗ Failed"
      : "⊘ Cancelled";
    actions = `<span class="dl-chip dl-${d.state}">${word}</span>
      <button class="icon-btn dl-dismiss" title="Dismiss">✕</button>`;
    body = `<div class="progress-meta">
      ${d.state === "error" ? escapeHtml(d.error_message || "Download failed.")
        : d.state === "done" ? `All ${d.total_files || ""} file(s) saved.`
        : "Stopped before completion - partial files were removed."}
    </div>`;
  } else if (d.state === "queued") {
    actions = `<button class="btn btn-danger btn-tiny dl-cancel">Cancel</button>`;
    body = `<div class="progress-meta">Queued - waiting for a free download slot (max ${frameMaxConcurrent()} concurrent).</div>`;
  } else {
    actions = `<button class="btn btn-danger btn-tiny dl-cancel">Cancel</button>`;
    body = `
      <div class="progress-track"><div class="progress-fill" style="width:${pct.toFixed(1)}%"></div></div>
      <div class="progress-meta">
        <span>${formatBytes(d.bytes_done_overall || 0)} / ${formatBytes(d.bytes_total_overall || 0)}${speed}</span>
        <span>${d.current_file ? `${d.file_index}/${d.total_files} - ${d.current_file}` : "starting…"}</span>
      </div>`;
  }

  card.className = "download-card" + (d.state === "error" ? " is-error" : "");
  card.dataset.key = key;
  card.dataset.state = d.state;
  card.innerHTML = `
    <div class="download-card-head">
      <span class="download-card-title" title="${escapeHtml(title)}">${escapeHtml(title)}</span>
      <span class="download-card-actions">${actions}</span>
    </div>
    ${body}`;
  card.querySelector(".dl-cancel")?.addEventListener("click", () => cancelDownload(key));
  card.querySelector(".dl-dismiss")?.addEventListener("click", () => dismissDownload(key));
  return card;
}

function frameMaxConcurrent() {
  return state.downloadMaxConcurrent || 3;
}

function dismissDownload(key) {
  delete state.downloads[key];
  state.dismissedDownloads[key] = true;
  updateActiveDownloadKeys();
  syncDownloadCards();
  if (state.renderHfGroups) state.renderHfGroups();
}

// ---------------------------------------------------------------------------
// Profile editor / parameter configurator
// ---------------------------------------------------------------------------

function wireEditor() {
  document.getElementById("btn-back-to-library").addEventListener("click", () => {
    // Return to whichever page the editor was opened from (Profiles vs Router Dir).
    if (state.editingMode === "router_dir") {
      showView("router_dir");
      loadRouterDirs();
    } else {
      showView("library");
      loadProfiles();
    }
  });
  document.getElementById("btn-save-profile").addEventListener("click", saveEditingProfile);
  document.getElementById("toggle-advanced").addEventListener("change", e => {
    document.getElementById("advanced-panel").hidden = !e.target.checked;
  });
  document.getElementById("param-search").addEventListener("input", debounce(renderParamCategories, 120));
  document.getElementById("btn-copy-command").addEventListener("click", () => {
    const text = document.getElementById("command-preview").textContent;
    navigator.clipboard.writeText(text).then(() => toast("Command copied", "ok")).catch(() => {});
  });
  document.getElementById("btn-start-from-editor").addEventListener("click", async () => {
    if (state.editingMode === "router_dir") {
      // If this router dir owns the running server, the button acts as Stop.
      if (state.editingProfile && state.editingProfile.id && runningRouterDirId() === state.editingProfile.id) {
        await stopServer();
        return;
      }
      const saved = await saveEditingProfile();
      if (saved) await startRouterDir(saved.id);
      return;
    }
    // If the profile being edited is the one running the server, this
    // button acts as Stop (see updateProfileRunningChrome).
    if (state.editingProfile && state.editingProfile.id && runningProfileId() === state.editingProfile.id) {
      await stopServer();
      return;
    }
    const saved = await saveEditingProfile();
    // Awaited so the start request (whose mark_used rewrites profiles.json)
    // has completed before the user can fire the next request.
    if (saved) await startServerWithProfile(saved.id);
  });
  ["profile-name", "profile-notes", "custom-flags", "router-dir-path",
    "router-dir-models-max", "router-dir-autoload"].forEach(id => {
    document.getElementById(id).addEventListener("input", updateCommandPreview);
  });
  document.getElementById("router-dir-autoload").addEventListener("change", updateCommandPreview);
  document.getElementById("btn-browse-router-dir").addEventListener("click", async () => {
    if (!hasNativeDialogs()) {
      toast("Folder picker needs the desktop app window - type the path manually here.", "error");
      return;
    }
    try {
      const path = await window.pywebview.api.pick_folder();
      if (path) {
        document.getElementById("router-dir-path").value = path;
        updateCommandPreview();
      }
    } catch (e) { toast("Could not open the folder picker.", "error"); }
  });
}

function defaultProfileName(model, modelPath) {
  // The model's display name already excludes the .gguf extension (and the
  // multipart -NNNNN-of-MMMMM suffix), so prefer it; fall back to the file
  // basename with the extension stripped.
  if (model && model.model_name) return model.model_name;
  const base = (modelPath || "").split(/[\\/]/).pop() || "";
  return base.replace(/\.gguf$/i, "");
}

function openEditorForNewProfile(model, modelPath) {
  state.editingMode = "profile";
  state.editingProfile = {
    id: null,
    model_id: modelId(model),
    model_path: modelPath,
    name: defaultProfileName(model, modelPath),
    params: {},   // starts empty - only params the user explicitly sets get added to the command
    custom_flags: "",
    notes: "",
  };
  openEditor(`New profile - ${model.model_name}`);
}

async function openEditorForProfile(profile) {
  // Re-fetch from the backend instead of trusting state.profiles:
  // the cache can be stale (another window edited the profile, a server
  // start touched it, …) and editing from a stale copy would let a plain
  // Save overwrite newer on-disk values with old ones.
  let fresh = profile;
  try {
    fresh = await API.get(`/api/profiles/${profile.id}`);
  } catch (e) {
    toast("Couldn't load the latest profile - editing the cached copy.", "error");
  }
  state.editingMode = "profile";
  state.editingProfile = JSON.parse(JSON.stringify(fresh));
  // The profile's own name is already shown in the "Profile name" field below;
  // the subtitle identifies the *model* being configured instead.
  openEditor(`Editing - ${modelLabelFromId(fresh.model_id)}`);
}

function openEditor(subtitle) {
  const isRouterDir = state.editingMode === "router_dir";
  document.getElementById("editor-title").textContent = isRouterDir ? "Router Dir" : "Launch Profile";
  document.getElementById("editor-sub").textContent = subtitle;
  document.getElementById("profile-name").value = state.editingProfile.name || "";
  document.getElementById("profile-notes").value = state.editingProfile.notes || "";
  document.getElementById("custom-flags").value = state.editingProfile.custom_flags || "";
  document.getElementById("router-dir-fields").hidden = !isRouterDir;
  document.getElementById("btn-save-profile").textContent = isRouterDir ? "Save Router Dir" : "Save Profile";
  document.getElementById("btn-start-from-editor").textContent =
    isRouterDir ? "▶ Start Server With This Router Dir" : "▶ Start Server With This Profile";
  document.getElementById("editor-launch-note").textContent = isRouterDir
    ? "Starts a router that auto-discovers every model in the directory, using the parameters above as shared defaults."
    : "Runs this profile against the configured llama-server binary.";
  if (isRouterDir) {
    document.getElementById("router-dir-path").value = state.editingProfile.models_dir || "";
    document.getElementById("router-dir-models-max").value = (state.editingProfile.models_max ?? 4);
    document.getElementById("router-dir-autoload").checked = state.editingProfile.autoload !== false;
  }
  // Advanced mode's expanded/collapsed state isn't separately persisted -
  // it's derived from whether the profile actually has custom flags, so
  // reopening a profile with existing advanced content shows it expanded
  // instead of always resetting to collapsed.
  const hasCustomFlags = !!(state.editingProfile.custom_flags && state.editingProfile.custom_flags.trim());
  document.getElementById("toggle-advanced").checked = hasCustomFlags;
  document.getElementById("advanced-panel").hidden = !hasCustomFlags;
  document.getElementById("param-search").value = "";
  // GGUF facts for the model being edited (block_count is the --n-cpu-moe
  // slider's max). Set synchronously when cached; a fresh read re-renders
  // the param list when it lands. Router dirs have no single model file.
  if (!isRouterDir && state.editingProfile.model_path) {
    loadGgufFacts(state.editingProfile.model_path);
  } else {
    state.ggufFacts = null;
  }
  renderParamCategories();
  updateCommandPreview();
  showView("editor");
}

// "Reading model metadata" modal. A cold GGUF read (first open of a model
// after an app restart, or a newly downloaded file) can take a few seconds
// on large models, during which the editor shows with its metadata-driven
// controls disabled - the modal explains why. Shown only when the read
// takes more than ~350 ms, so fast (disk-cached) reads never flash it.
// Dismissible: the read keeps running in the background and the param list
// re-renders when it lands (see loadGgufFacts).
function requestMetaLoadModal(modelPath) {
  if (state.metaLoadTimer) { clearTimeout(state.metaLoadTimer); state.metaLoadTimer = null; }
  if (state.metaLoadModal && state.metaLoadModal !== modelPath) closeMetaLoadModal();
  state.metaLoadPending = modelPath;
  state.metaLoadTimer = setTimeout(() => {
    state.metaLoadTimer = null;
    if (state.metaLoadPending === modelPath) openMetaLoadModal(modelPath);
  }, 350);
}

function openMetaLoadModal(modelPath) {
  const name = modelPath.split(/[\\/]/).pop() || modelPath;
  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop meta-load";
  backdrop.innerHTML = `
    <div class="modal-card">
      <h3>Reading model metadata</h3>
      <div class="meta-load-row">
        <span class="spinner"></span>
        <p class="muted small" style="margin:0">First-time read of <strong>${escapeHtml(name)}</strong> - this can take a few seconds for large models.</p>
      </div>
      <p class="muted small">You can close this and keep using the editor; the values that depend on the metadata fill in when the read finishes.</p>
      <div class="modal-actions"><button class="btn btn-ghost" id="meta-load-close">Close</button></div>
    </div>
  `;
  document.body.appendChild(backdrop);
  state.metaLoadModal = modelPath;
  backdrop.addEventListener("click", e => { if (e.target === backdrop) closeMetaLoadModal(); });
  backdrop.querySelector("#meta-load-close").addEventListener("click", closeMetaLoadModal);
}

function closeMetaLoadModal() {
  const el = document.querySelector(".modal-backdrop.meta-load");
  if (el) el.remove();
  state.metaLoadModal = null;
}

// Called when the read settles (success or error). Safe to call for a path
// that has been superseded by a newer read - everything is path-checked.
function settleMetaLoadModal(modelPath) {
  if (state.metaLoadTimer && state.metaLoadPending === modelPath) {
    clearTimeout(state.metaLoadTimer);
    state.metaLoadTimer = null;
  }
  if (state.metaLoadPending === modelPath) state.metaLoadPending = null;
  if (state.metaLoadModal === modelPath) closeMetaLoadModal();
}

// Fetch (or reuse the cached) GGUF facts for a model file. Sets
// state.ggufFacts synchronously when the result is cached; otherwise it
// stays null while the read is in flight and this function re-renders the
// param list when the answer arrives - but only if the editor is still
// open on the same model.
async function loadGgufFacts(modelPath) {
  const cached = (state.ggufFactsByPath || {})[modelPath];
  if (cached) {
    state.ggufFacts = cached;
    return;
  }
  state.ggufFacts = null;
  requestMetaLoadModal(modelPath);
  let facts = null, error = null;
  try {
    facts = await API.get("/api/gguf/facts?path=" + encodeURIComponent(modelPath));
    state.ggufFactsByPath[modelPath] = facts;
  } catch (e) {
    error = e.message;
  }
  settleMetaLoadModal(modelPath);
  if (!(state.editingProfile && state.editingProfile.model_path === modelPath)) return;
  state.ggufFacts = facts || (error ? { error } : null);
  if (state.view === "editor") {
    renderParamCategories();
    updateCommandPreview();
  }
}

// Slider maximum from the model's GGUF facts (schema: slider_max_from,
// with an optional slider_max_offset - e.g. --n-cpu-moe caps at
// block_count - 1). null = unavailable → the control renders disabled.
function sliderMaxFor(p) {
  const facts = state.ggufFacts || {};
  let v = Number(facts[p.slider_max_from]);
  if (!Number.isInteger(v) || v <= 0) return null;
  if (p.slider_max_offset) v += p.slider_max_offset;
  return v >= 0 ? v : null;
}

function sliderUnavailableReason(p) {
  const facts = state.ggufFacts || {};
  const why = facts.error
    ? `couldn't read the model's GGUF metadata (${facts.error})`
    : `no ${p.slider_max_from} found in the model's GGUF metadata`;
  return `${p.flag} is disabled - ${why}.`;
}

// A parameter can require a model GGUF fact to be shown at all - e.g.
// --n-cpu-moe is only offered on MoE models (expert_count > 0). While the
// facts are still loading (or unreadable / router-dir mode) the fact is
// unknown, so such rows stay hidden; loadGgufFacts re-renders when they
// arrive.
function paramVisible(p) {
  if (!p.requires_model_fact) return true;
  const v = Number((state.ggufFacts || {})[p.requires_model_fact]);
  return Number.isInteger(v) && v > 0;
}

function renderParamCategories() {
  const container = document.getElementById("param-categories");
  container.innerHTML = "";
  if (!state.schema) return;
  const query = document.getElementById("param-search").value;

  state.schema.categories.forEach(cat => {
    const paramsInCat = state.schema.parameters.filter(p => p.category === cat.id && matchesQuery(p, query) && paramVisible(p));
    if (paramsInCat.length === 0) return;

    const catEl = document.createElement("div");
    catEl.className = "param-category" + (state.collapsedCategories.has(cat.id) ? " is-collapsed" : "");
    catEl.innerHTML = `
      <div class="param-category-head">
        <span class="param-category-title">${escapeHtml(cat.label)}</span>
        <span class="param-category-count">${paramsInCat.length}</span>
      </div>
      <div class="param-category-body"></div>
    `;
    catEl.querySelector(".param-category-head").addEventListener("click", () => {
      catEl.classList.toggle("is-collapsed");
      if (catEl.classList.contains("is-collapsed")) state.collapsedCategories.add(cat.id);
      else state.collapsedCategories.delete(cat.id);
    });

    const body = catEl.querySelector(".param-category-body");
    paramsInCat.forEach(p => body.appendChild(renderParamRow(p)));
    container.appendChild(catEl);
  });
}

function matchesQuery(p, query) {
  return matchesAllTerms(`${p.label} ${p.flag} ${p.help}`, query);
}

function renderParamRow(p) {
  const isSet = Object.prototype.hasOwnProperty.call(state.editingProfile.params, p.key);
  const currentValue = isSet ? state.editingProfile.params[p.key] : undefined;

  const row = document.createElement("div");
  row.className = "param-row" + (isSet ? " is-set" : "");
  row.dataset.key = p.key;
  row.innerHTML = `
    <div class="param-label-wrap">
      <div>
        <span class="param-label">${escapeHtml(p.label)}</span>
        <span class="param-flag">${escapeHtml(p.short_flag ? p.short_flag + ", " + p.flag : p.flag)}</span>
      </div>
      <span class="param-info" tabindex="0">?<span class="tooltip">${escapeHtml(p.help)}</span></span>
    </div>
    <div class="param-control"></div>
    <div class="param-row-actions">
      <button class="param-action" data-action="default" title="Add this parameter at its default value">Set default</button>
      <button class="param-action param-action-clear" data-action="clear" title="Remove - this parameter won't be passed on the command line" ${isSet ? "" : "disabled"}>✕</button>
    </div>
  `;

  const controlWrap = row.querySelector(".param-control");
  controlWrap.appendChild(buildControl(p, currentValue, isSet));

  row.querySelector('[data-action="default"]').addEventListener("click", () => {
    state.editingProfile.params[p.key] = p.type === "bool"
      ? (p.default === null || p.default === undefined ? true : !!p.default)
      : (p.default !== null && p.default !== undefined ? p.default : (p.type === "enum" ? (p.options && p.options[0]) : ""));
    renderParamCategories();
    updateCommandPreview();
  });

  row.querySelector('[data-action="clear"]').addEventListener("click", () => {
    delete state.editingProfile.params[p.key];
    renderParamCategories();
    updateCommandPreview();
  });

  // A slider whose maximum can't be read from the model's metadata is
  // unavailable wholesale (no point setting a value you can't range-check).
  if (p.widget === "slider" && sliderMaxFor(p) === null) {
    const defBtn = row.querySelector('[data-action="default"]');
    defBtn.disabled = true;
    defBtn.title = sliderUnavailableReason(p);
  }

  return row;
}

function buildControl(p, currentValue, isSet) {
  const wrap = document.createElement("div");

  if (p.type === "bool") {
    wrap.className = "param-toggle";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = isSet && !!currentValue;
    input.addEventListener("change", () => {
      if (input.checked) state.editingProfile.params[p.key] = true;
      else delete state.editingProfile.params[p.key]; // false === absent for a store-true CLI flag
      updateCommandPreview();
      refreshParamRowChrome(p.key, input.checked);
    });
    wrap.appendChild(input);
    return wrap;
  }

  if (p.type === "enum") {
    const select = document.createElement("select");
    select.className = "input";
    if (!isSet) {
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = `- not set (default: ${p.default ?? "model default"}) -`;
      placeholder.selected = true;
      select.appendChild(placeholder);
    }
    (p.options || []).forEach(opt => {
      const o = document.createElement("option");
      o.value = opt; o.textContent = opt;
      if (isSet && currentValue === opt) o.selected = true;
      select.appendChild(o);
    });
    select.addEventListener("change", () => {
      state.editingProfile.params[p.key] = select.value;
      updateCommandPreview();
      renderParamCategories();
    });
    wrap.appendChild(select);
    return wrap;
  }

  if (p.widget === "slider") {
    // Range control whose maximum comes from the model's own GGUF metadata
    // (--ctx-size: 0 … context_length, --n-gpu-layers / --n-cpu-moe:
    // 0 … block_count). Dragging always sets an explicit numeric value;
    // "not set" shows at the minimum until the user touches it. Disabled -
    // with a reason in the value slot - when the max is unknown.
    const minVal = (p.min !== null && p.min !== undefined) ? p.min : 0;
    const maxFromModel = sliderMaxFor(p);
    const maxVal = maxFromModel !== null ? maxFromModel
      : ((p.max !== null && p.max !== undefined) ? p.max : minVal);

    const wrapEl = document.createElement("div");
    wrapEl.className = "param-slider" + (maxFromModel === null ? " is-unavailable" : "");
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = minVal;
    slider.max = maxVal;
    slider.step = (p.step !== null && p.step !== undefined) ? p.step : 1;

    // Stored value → slider position. Legacy string values on a numeric
    // slider (n_gpu_layers: "auto"/"all" from before it was a slider) map
    // to the ends of the range; numeric strings are parsed. The value chip
    // shows the stored value verbatim, so saved data is never misreported.
    let stored;
    if (!isSet) {
      stored = minVal;
    } else if (typeof currentValue === "string") {
      const s = currentValue.trim().toLowerCase();
      if (s === "all") stored = maxVal;
      else if (s === "auto") stored = minVal;
      else {
        const n = parseInt(s, 10);
        stored = Number.isNaN(n) ? minVal : n;
      }
    } else {
      stored = Number(currentValue);
      if (!Number.isFinite(stored)) stored = minVal;
    }
    slider.value = clamp(stored, minVal, maxVal);

    const valueEl = document.createElement("span");
    valueEl.className = "param-slider-value";
    if (maxFromModel === null) {
      slider.disabled = true;
      valueEl.textContent = "-";
      slider.title = sliderUnavailableReason(p);
      valueEl.title = sliderUnavailableReason(p);
    } else {
      valueEl.textContent = isSet
        ? (typeof currentValue === "string" ? currentValue : stored)
        : (p.unset_label || "not set");
      slider.title = `0 … ${maxVal} - the model's ${p.slider_max_from} from its GGUF metadata`;
    }
    slider.addEventListener("input", () => {
      state.editingProfile.params[p.key] = p.type === "float"
        ? parseFloat(slider.value)
        : parseInt(slider.value, 10);
      valueEl.textContent = slider.value;
      updateCommandPreview();
      refreshParamRowChrome(p.key, true);
    });
    wrapEl.appendChild(slider);
    wrapEl.appendChild(valueEl);
    return wrapEl;
  }

  const input = document.createElement("input");
  input.className = "input";
  input.type = (p.type === "int" || p.type === "float") ? "number" : "text";
  // Spinner step: the schema's value where defined (ctx-size +2048, temp +0.1, …);
  // floats without a schema step stay free-typed ("any").
  if (p.step !== null && p.step !== undefined) input.step = p.step;
  else if (p.type === "float") input.step = "any";
  if (p.min !== null && p.min !== undefined) input.min = p.min;
  if (p.max !== null && p.max !== undefined) input.max = p.max;
  const isModelFile = !!MODEL_FOLDER_FILE_PARAMS[p.key];
  if (isModelFile) {
    // The field only shows the file name; the full path (what the command
    // uses) is resolved from the model folder's file list. A custom full
    // path can still be typed - it's stored and shown shortened too.
    input.value = isSet ? pathBasename(currentValue) : "";
    input.placeholder = "file in the model folder, or a full path";
  } else {
    input.value = isSet ? currentValue : "";
    input.placeholder = p.default !== null && p.default !== undefined ? String(p.default) : "not set";
  }
  input.addEventListener("input", () => {
    const raw = input.value;
    if (raw === "") { delete state.editingProfile.params[p.key]; }
    else if (p.type === "int") {
      const n = parseInt(raw, 10);
      state.editingProfile.params[p.key] = Number.isNaN(n) ? raw : clamp(n, p.min, p.max);
    } else if (p.type === "float") {
      const n = parseFloat(raw);
      state.editingProfile.params[p.key] = Number.isNaN(n) ? raw : clamp(n, p.min, p.max);
    } else {
      state.editingProfile.params[p.key] = isModelFile ? resolveModelFolderFileValue(p.key, raw) : raw;
    }
    updateCommandPreview();
    refreshParamRowChrome(p.key, raw !== "");
  });
  wrap.appendChild(input);

  // Params with suggested values get a datalist dropdown: pick a common
  // value from the list, or type any custom one - the input stays editable.
  if (p.suggestions && p.suggestions.length) {
    const dl = document.createElement("datalist");
    dl.id = `suggestions-${p.key}`;
    p.suggestions.forEach(v => {
      const o = document.createElement("option");
      o.value = v;
      dl.appendChild(o);
    });
    input.setAttribute("list", dl.id);
    wrap.appendChild(dl);
  }

  // Model-folder file params (--mmproj, --chat-template-file) get a datalist
  // of candidate files found in this model's own folder (fetched from the
  // backend, cached per model) - pick one from the list or type any custom
  // path.
  if (isModelFile) {
    attachModelFolderFileSuggestions(p, input, wrap);
  }
  return wrap;
}

// Params whose value is a file in the model's own folder: the field shows a
// bare file name, the command gets the full path. cache holds the per-model
// scan results (see the state fields above).
const MODEL_FOLDER_FILE_PARAMS = {
  mmproj: { endpoint: "/api/mmproj/files", cache: state.mmprojSuggestions },
  chat_template_file: { endpoint: "/api/chat-template/files", cache: state.chatTemplateSuggestions },
};

// Resolve what the user typed in a model-folder file field to the value the
// command should actually use: a bare file name that matches a candidate in
// the current model's folder becomes that file's full path, so the field can
// show the short name while the command gets the proper path. Anything else
// (custom relative/absolute path) is kept as typed.
function resolveModelFolderFileValue(key, raw) {
  const modelPath = state.editingProfile && state.editingProfile.model_path;
  if (!modelPath) return raw;
  const files = MODEL_FOLDER_FILE_PARAMS[key].cache[modelPath];
  if (!Array.isArray(files)) return raw; // folder scan not finished yet
  const lower = raw.trim().toLowerCase();
  const hit = files.find(f => f.filename.toLowerCase() === lower || f.path.toLowerCase() === lower);
  return hit ? hit.path : raw;
}

// Datalist source for a model-folder file field: candidate files in the
// current model's folder. Results are cached per model path; while the scan
// is in flight the cache holds the promise itself so concurrent renders
// don't trigger duplicate requests.
function modelFolderFilesFor(key, modelPath) {
  const cfg = MODEL_FOLDER_FILE_PARAMS[key];
  let entry = cfg.cache[modelPath];
  if (!entry) {
    entry = API.get(`${cfg.endpoint}?model_path=${encodeURIComponent(modelPath)}`)
      .then(data => (data && data.files) || [])
      .catch(() => []);
    cfg.cache[modelPath] = entry;
  }
  return entry;
}

function attachModelFolderFileSuggestions(p, input, wrap) {
  const modelPath = state.editingProfile && state.editingProfile.model_path;
  if (!modelPath) return;
  const profile = state.editingProfile;

  const dl = document.createElement("datalist");
  dl.id = `suggestions-${p.key}`;
  input.setAttribute("list", dl.id);
  wrap.appendChild(dl);

  const fill = (files) => {
    dl.innerHTML = "";
    files.forEach(f => {
      const o = document.createElement("option");
      o.value = f.filename;       // selecting keeps the bare name in the field
      o.textContent = f.filename; // what's shown in the dropdown
      dl.appendChild(o);
    });
  };

  // If a bare file name was stored while the list was unavailable (typed
  // before the scan finished, or carried over from an older profile),
  // upgrade it to the full path now that we know where the file lives.
  const upgradeStoredValue = (files) => {
    if (state.editingProfile !== profile) return; // editor moved on
    const cur = profile.params && profile.params[p.key];
    if (typeof cur !== "string" || cur === "" || cur.includes("/") || cur.includes("\\")) return;
    const hit = files.find(f => f.filename.toLowerCase() === cur.toLowerCase());
    if (hit && hit.path !== cur) {
      profile.params[p.key] = hit.path;
      updateCommandPreview();
    }
  };

  const entry = modelFolderFilesFor(p.key, modelPath);
  if (Array.isArray(entry)) {
    fill(entry);
    upgradeStoredValue(entry);
  } else {
    // Still scanning. Patch the list once it lands if this row (and its
    // datalist) is still on the page - otherwise a later render will pick
    // the cached result up on its own.
    entry.then(files => {
      if (dl.isConnected) fill(files);
      upgradeStoredValue(files);
    });
  }
}

// Lightweight chrome update (the "set" dot + clear-button enabled state) that
// avoids a full re-render on every keystroke, which would steal input focus.
function refreshParamRowChrome(key, isSet) {
  const row = document.querySelector(`.param-row[data-key="${cssEscape(key)}"]`);
  if (!row) return;
  row.classList.toggle("is-set", isSet);
  const clearBtn = row.querySelector('[data-action="clear"]');
  if (clearBtn) clearBtn.disabled = !isSet;
}

function cssEscape(str) {
  return window.CSS && CSS.escape ? CSS.escape(str) : str.replace(/[^a-zA-Z0-9_-]/g, "\\$&");
}

function clamp(n, min, max) {
  if (min !== null && min !== undefined && n < min) return min;
  if (max !== null && max !== undefined && n > max) return max;
  return n;
}


function updateCommandPreview() {
  if (!state.editingProfile || !state.schema) return;
  const p = state.editingProfile;
  p.name = document.getElementById("profile-name").value;
  p.notes = document.getElementById("profile-notes").value;
  p.custom_flags = document.getElementById("custom-flags").value;

  const isRouterDir = state.editingMode === "router_dir";
  if (isRouterDir) {
    p.models_dir = document.getElementById("router-dir-path").value.trim();
    p.models_max = Math.max(0, parseInt(document.getElementById("router-dir-models-max").value, 10) || 0);
    p.autoload = document.getElementById("router-dir-autoload").checked;
  }

  const binaryName = (state.settings && state.settings.llama_server_path)
    ? state.settings.llama_server_path.split(/[\\/]/).pop()
    : "llama-server";

  const pushParams = (skipHostPort) => {
    orderedParamKeys(p.params).forEach(key => {
      if (skipHostPort && (key === "host" || key === "port")) return; // router owns host/port
      const value = p.params[key];
      const schemaP = state.schemaByKey[key];
      if (!schemaP || value === undefined || value === null || value === "") return;
      if (schemaP.type === "bool") {
        if (value) args.push(schemaP.flag);
      } else {
        args.push(schemaP.flag, String(value));
      }
    });
  };

  let args;
  if (isRouterDir) {
    const s = state.settings || {};
    const host = s.default_host || "127.0.0.1";
    const port = s.default_port || 8080;
    args = ["--models-dir", p.models_dir || "…", "--models-max", String(p.models_max ?? 4)];
    if (p.autoload === false) args.push("--no-models-autoload");
    pushParams(true);
    args.push("--host", String(host), "--port", String(port));
  } else {
    args = ["--model", p.model_path];
    pushParams(false);
  }

  if (p.custom_flags && p.custom_flags.trim()) {
    args.push(...p.custom_flags.trim().split(/\s+/));
  }

  const quoted = args.map(a => (a === "" || /\s/.test(a)) ? `"${a}"` : a);
  document.getElementById("command-preview").textContent = `${binaryName} ${quoted.join(" ")}`;

  // Keep the "Parameters" summary card in sync (same rows as the profile page).
  const paramsEl = document.getElementById("editor-params-list");
  if (paramsEl) {
    const rows = orderedParamKeys(p.params)
      .map(k => [k, p.params[k]])
      .filter(([, v]) => v !== undefined && v !== null && v !== "" && v !== false)
      .map(([k, v]) => {
        const sp = state.schemaByKey[k];
        const flag = (sp && sp.flag) || ("--" + k);
        return `<div class="file-row"><span>${escapeHtml(flag)}</span><span>${escapeHtml(shortParamValue(v))}</span></div>`;
      }).join("");
    paramsEl.innerHTML = rows || `<p class="muted small">Default parameters (none overridden).</p>`;
  }

  // Mirror the profile-detail pane: surface the raw custom flags in the
  // preview too, so what you see here matches what the profile page shows.
  const advBlock = document.getElementById("editor-advanced-block");
  const advPreview = document.getElementById("editor-advanced-preview");
  if (advBlock && advPreview) {
    const advText = (p.custom_flags || "").trim();
    advBlock.hidden = !advText;
    if (advText) advPreview.textContent = advText;
  }
}

async function saveEditingProfile() {
  const p = state.editingProfile;
  p.name = document.getElementById("profile-name").value.trim();
  p.notes = document.getElementById("profile-notes").value;
  p.custom_flags = document.getElementById("custom-flags").value;

  if (!p.name) { toast("Give this a name first.", "error"); return null; }

  try {
    let saved;
    if (state.editingMode === "router_dir") {
      const payload = {
        name: p.name,
        models_dir: document.getElementById("router-dir-path").value.trim(),
        models_max: Math.max(0, parseInt(document.getElementById("router-dir-models-max").value, 10) || 0),
        autoload: document.getElementById("router-dir-autoload").checked,
        params: p.params,
        custom_flags: p.custom_flags,
        notes: p.notes,
      };
      if (!payload.models_dir) { toast("Set the models directory first.", "error"); return null; }
      if (p.id) saved = await API.put(`/api/router-dirs/${p.id}`, payload);
      else saved = await API.post("/api/router-dirs", payload);
      toast("Router dir saved", "ok");
      state.editingProfile = saved;
      document.getElementById("profile-name").value = saved.name;
      await loadRouterDirs();
      return saved;
    }
    if (p.id) {
      saved = await API.put(`/api/profiles/${p.id}`, { name: p.name, params: p.params, custom_flags: p.custom_flags, notes: p.notes });
    } else {
      saved = await API.post("/api/profiles", { model_id: p.model_id, model_path: p.model_path, name: p.name, params: p.params, custom_flags: p.custom_flags, notes: p.notes });
    }
    toast("Profile saved", "ok");
    state.editingProfile = saved;
    // The backend may have adjusted the name (per-model uniqueness), so
    // sync the input to what was actually stored.
    document.getElementById("profile-name").value = saved.name;
    await loadProfiles();
    return saved;
  } catch (e) {
    toast(e.message, "error");
    return null;
  }
}

// ---------------------------------------------------------------------------
// Server console
// ---------------------------------------------------------------------------

function wireServer() {
  document.getElementById("btn-server-stop").addEventListener("click", stopServer);
  document.getElementById("btn-server-start").addEventListener("click", async () => {
    // Quick start: resume the most recently used profile. (Starting with a
    // specific profile/preset is still done from the Profiles page / presets.)
    await loadProfiles();
    const p = state.profiles && state.profiles[0];
    if (!p) { toast("No profiles yet - create one on the Profiles page first.", "error"); return; }
    await startServerWithProfile(p.id);
  });
  document.getElementById("btn-clear-log").addEventListener("click", () => {
    document.getElementById("log-output").textContent = "";
  });
}

// The profile (if any) that owns the current server process, per the
// backend's status (it stores the profile_id at launch). null while
// nothing server-side is up, or when a router preset is running.
function runningProfileId() {
  const s = state.serverStatus;
  if (!s || !s.profile_id) return null;
  if (!["running", "starting", "stopping"].includes(s.state)) return null;
  return s.profile_id;
}

// The router preset (if any) that owns the current server process, per the
// backend's status (it stores the preset_id at router launch). null while
// nothing is up or a single-profile server is running.
function runningRouterPresetId() {
  const s = state.serverStatus;
  if (!s || !s.preset_id || s.mode !== "router") return null;
  if (!["running", "starting", "stopping"].includes(s.state)) return null;
  return s.preset_id;
}

// The router dir (if any) that owns the current server process. null while
// nothing is up or a single-profile / preset router is running.
function runningRouterDirId() {
  const s = state.serverStatus;
  if (!s || !s.router_dir_id || s.mode !== "router") return null;
  if (!["running", "starting", "stopping"].includes(s.state)) return null;
  return s.router_dir_id;
}

// Keep every profile Start button (list rows, detail pane, editor launch)
// in sync with which profile owns the running server: the owner's button
// shows ⏳ Loading… / ■ Stopping… / ■ Stop, everyone else's is disabled
// while any server is up. Runs on every status poll.
function updateProfileRunningChrome() {
  const s = state.serverStatus;
  const ownerId = runningProfileId();
  const presetOwnerId = runningRouterPresetId();
  const anyServer = !!s && ["running", "starting", "stopping"].includes(s.state);

  document.querySelectorAll("#profile-list .list-row").forEach(row => {
    const btn = row.querySelector('[data-act="start"]');
    if (btn) applyProfileStartButton(btn, row.dataset.profileId, ownerId, anyServer, row.dataset.stale === "1");
    const badge = row.querySelector(".running-badge");
    if (badge) badge.hidden = !(row.dataset.profileId === ownerId && s && s.state === "running");
  });

  // Router presets: same owner/Loading…/Stop treatment as profile rows.
  document.querySelectorAll("#preset-rows .preset-row").forEach(row => {
    const btn = row.querySelector('[data-act="start"]');
    if (btn) applyProfileStartButton(btn, row.dataset.presetId, presetOwnerId, anyServer, false);
    const badge = row.querySelector(".running-badge");
    if (badge) badge.hidden = !(row.dataset.presetId === presetOwnerId && s && s.state === "running");
  });

  // Router dirs: same treatment as preset rows.
  const routerDirOwnerId = runningRouterDirId();
  document.querySelectorAll("#router-dir-rows .preset-row").forEach(row => {
    const btn = row.querySelector('[data-act="start"]');
    if (btn) applyProfileStartButton(btn, row.dataset.routerDirId, routerDirOwnerId, anyServer, false);
    const badge = row.querySelector(".running-badge");
    if (badge) badge.hidden = !(row.dataset.routerDirId === routerDirOwnerId && s && s.state === "running");
  });

  if (state.selectedProfileId) {
    const sel = (state.profiles || []).find(x => x.id === state.selectedProfileId);
    const pdBtn = document.getElementById("pd-start");
    if (pdBtn) applyProfileStartButton(pdBtn, state.selectedProfileId, ownerId, anyServer, !!sel && !modelForProfile(sel));
  }

  const edBtn = document.getElementById("btn-start-from-editor");
  if (edBtn && state.editingProfile && state.editingProfile.id) {
    const edOwnerId = state.editingMode === "router_dir" ? runningRouterDirId() : ownerId;
    applyProfileStartButton(edBtn, state.editingProfile.id, edOwnerId, anyServer, false);
  }
}

function applyProfileStartButton(btn, profileId, ownerId, anyServer, forceDisabled) {
  const s = state.serverStatus;
  const stopMode = ownerId === profileId && s.state === "running";
  btn.classList.toggle("btn-primary", !stopMode);
  btn.classList.toggle("btn-danger", stopMode);
  if (ownerId === profileId) {
    if (s.state === "starting") {
      btn.textContent = "⏳ Loading…";
      btn.disabled = true;
      btn.title = "Loading model…";
    } else if (s.state === "stopping") {
      btn.textContent = "■ Stopping…";
      btn.disabled = true;
      btn.title = "Waiting for the server to stop";
    } else {
      btn.textContent = "■ Stop";
      btn.disabled = !!forceDisabled;
      btn.title = "Stop the server started by this profile";
    }
  } else {
    btn.textContent = "▶ Start";
    btn.disabled = !!forceDisabled || anyServer;
    btn.title = forceDisabled
      ? "Model not found"
      : (anyServer
        ? (s.state === "stopping" ? "Wait for the server to stop" : "Stop the running server first")
        : "Start the server with this profile");
  }
}

async function startServerWithProfile(profileId) {
  try {
    await API.post("/api/server/start", { profile_id: profileId });
    toast("Server starting…", "ok");
    showView("server");
    await refreshServerStatus();
  } catch (e) {
    toast(e.message, "error");
  }
}

async function stopServer() {
  try {
    await API.post("/api/server/stop", {});
    toast("Stopping server…", "ok");
    await refreshServerStatus();
  } catch (e) {
    toast(e.message, "error");
  }
}

function startStatusPolling() {
  if (state.statusPollTimer) clearInterval(state.statusPollTimer);
  state.statusPollTimer = setInterval(refreshServerStatus, 2000);
}

async function refreshServerStatus() {
  let status;
  try { status = await API.get("/api/server/status"); }
  catch (e) { return; }
  state.serverStatus = status;

  const dot = document.getElementById("status-dot");
  const label = document.getElementById("status-label");
  const meta = document.getElementById("status-meta");

  dot.dataset.state = status.state;
  label.textContent = {
    stopped: "Server stopped",
    starting: "Server starting…",
    running: "Server running",
    stopping: "Stopping…",
    error: "Server error",
  }[status.state] || status.state;

  meta.textContent = status.pid ? `pid ${status.pid}${status.uptime_seconds ? " · " + formatUptime(status.uptime_seconds) : ""}` : "";

  const ownerName =
    (status.profile_id && (state.profiles || []).find(x => x.id === status.profile_id)?.name)
    || (status.preset_id && (state.presets || []).find(x => x.id === status.preset_id)?.name)
    || (status.router_dir_id && (state.routerDirs || []).find(x => x.id === status.router_dir_id)?.name)
    || null;
  document.getElementById("server-sub").textContent =
    status.state === "running"
      ? `Running on ${status.host}:${status.port}${ownerName ? ` - ${ownerName}` : ""}`
      : status.state === "starting"
        ? `Loading${ownerName ? ` - ${ownerName}` : ""}…`
        : (status.error_message || "No server running.");

  // Both buttons stay visible at all times; which one is actionable makes
  // the server state obvious without reading the status line.
  const startBtn = document.getElementById("btn-server-start");
  const stopBtn = document.getElementById("btn-server-stop");
  const busy = status.state === "running" || status.state === "starting";
  startBtn.hidden = false;
  startBtn.disabled = busy || status.state === "stopping";
  startBtn.textContent = status.state === "starting" ? "⏳ Loading…" : "▶ Start";
  startBtn.title = busy
    ? "Server already running"
    : (status.state === "stopping" ? "Wait for the server to stop" : "Start the most recently used profile");
  stopBtn.hidden = false;
  stopBtn.disabled = !busy;
  stopBtn.textContent = status.state === "running" ? "■ Stop Server" : "■ Stop";
  stopBtn.title = busy ? "Stop the running server" : "Server is not running";

  // "Open Web UI" = this app's own UI (always available); "Open llama UI"
  // = the llama-server web UI (only while a server is running).
  const appLink = document.getElementById("link-open-ui");
  appLink.href = location.origin;

  const llamaLink = document.getElementById("link-open-llama");
  if (status.state === "running") {
    llamaLink.hidden = false;
    const h = status.host === "0.0.0.0" ? "127.0.0.1" : status.host;
    llamaLink.href = `http://${h}:${status.port}/`;
  } else {
    llamaLink.hidden = true;
  }

  updateRouterPanel(status);
  updateProfileRunningChrome();
  updateNavServerDots(status, ownerName);

  const grid = document.getElementById("server-meta-grid");
  grid.innerHTML = `
    <div class="meta-card"><div class="meta-card-label">State</div><div class="meta-card-value">${status.state}</div></div>
    <div class="meta-card"><div class="meta-card-label">PID</div><div class="meta-card-value">${status.pid ?? "-"}</div></div>
    <div class="meta-card"><div class="meta-card-label">Uptime</div><div class="meta-card-value">${status.uptime_seconds ? formatUptime(status.uptime_seconds) : "-"}</div></div>
    <div class="meta-card"><div class="meta-card-label">Endpoint</div><div class="meta-card-value">${status.state === "running" ? status.host + ":" + status.port : "-"}</div></div>
  `;

  if (status.state === "error" && status.error_message) {
    // surface once per transition; simplest approach is a toast each poll would be noisy,
    // so only show it if the server console view is active or on first detection.
  }
}

// Small dot on the Server Console / Router Presets / Router Dir nav items,
// so from ANY page you can see at a glance that a server is up and which
// context started it (single profile → console, preset router, or dir router).
function updateNavServerDots(status, ownerName) {
  const s = status.state;
  const on = ["starting", "running", "stopping", "error"].includes(s);
  const targets = [
    ["nav-dot-server", on && !status.preset_id && !status.router_dir_id],
    ["nav-dot-preset", on && !!status.preset_id],
    ["nav-dot-dir", on && !!status.router_dir_id],
  ];
  targets.forEach(([id, active]) => {
    const d = document.getElementById(id);
    if (!d) return;
    d.classList.toggle("on", active);
    if (active) {
      d.dataset.state = s;
      d.title = `Server ${s}${ownerName ? ` - ${ownerName}` : ""}`;
    } else {
      d.title = "";
    }
  });
}

function connectLogSocket() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${location.host}/ws/logs`;
  try {
    state.logSocket = new WebSocket(url);
    state.logSocket.onmessage = (evt) => appendLogLine(evt.data);
    state.logSocket.onclose = () => setTimeout(connectLogSocket, 3000);
    state.logSocket.onerror = () => {};
  } catch (e) { /* pywebview / older browsers: silently skip */ }
}

function appendLogLine(line) {
  const out = document.getElementById("log-output");
  const span = document.createElement("div");
  const lower = line.toLowerCase();
  if (lower.includes("error")) span.className = "log-error";
  else if (lower.includes("warn")) span.className = "log-warn";
  else if (line.startsWith("[model-manager]")) span.className = "log-system";
  span.textContent = line;
  out.appendChild(span);
  out.scrollTop = out.scrollHeight;
}

// ---------------------------------------------------------------------------
// Router presets (--models-preset)
// ---------------------------------------------------------------------------

function wireRouter() {
  document.getElementById("btn-new-preset").addEventListener("click", () => openPresetModal(null)); // async: refreshes profile list first
  document.getElementById("btn-router-reload").addEventListener("click", () => fetchRouterModels(true));
}

async function loadPresets() {
  try {
    const [presets, allProfiles] = await Promise.all([
      API.get("/api/presets"),
      API.get("/api/profiles"),
    ]);
    state.presets = presets;
    state.allProfiles = allProfiles;
  } catch (e) {
    state.presets = [];
    toast(e.message, "error");
  }
  renderPresets();
}

function profileNameById(id) {
  const p = (state.allProfiles || []).find(x => x.id === id);
  return p ? p.name : "(missing profile)";
}

function renderPresets() {
  const el = document.getElementById("preset-rows");
  if (!el) return;
  if (!state.presets.length) {
    el.innerHTML = `<p class="muted small">No router presets yet. Create one from two or more profiles (usually different models).</p>`;
    return;
  }
  el.innerHTML = "";
  state.presets.forEach(preset => {
    const rows = document.createElement("div");
    rows.className = "preset-row";
    rows.dataset.presetId = preset.id;
    const isOwner = runningRouterPresetId() === preset.id;
    const ownerState = isOwner && state.serverStatus ? state.serverStatus.state : null;
    const startLabel = isOwner
      ? (ownerState === "starting" ? "⏳ Loading…" : ownerState === "stopping" ? "■ Stopping…" : "■ Stop")
      : "▶ Start";
    const startCls = isOwner && ownerState === "running" ? "btn btn-danger btn-tiny" : "btn btn-primary btn-tiny";

    const chips = (preset.profile_ids || []).map(id => {
      const name = profileNameById(id);
      const prof = (state.allProfiles || []).find(x => x.id === id);
      const missing = !prof;
      return `<span class="preset-chip" ${prof ? `data-profile-id="${id}"` : `data-missing-id="${id}"`} title="${missing ? "This profile no longer exists - click to remove it from the preset." : "Open profile in editor"}">${escapeHtml(name)}${missing ? " ✕" : ""}</span>`;
    }).join("");

    rows.innerHTML = `
      <div style="min-width:0;">
        <div class="preset-row-name">${escapeHtml(preset.name)}<span class="running-badge" ${isOwner && ownerState === "running" ? "" : "hidden"}>● running</span></div>
        <div class="preset-row-meta">${(preset.profile_ids || []).length} model(s) · max ${preset.models_max === 0 ? "unlimited" : preset.models_max} concurrent · autoload ${preset.autoload ? "on" : "off"}</div>
        <div class="preset-chips">${chips}</div>
      </div>
      <div class="preset-row-actions">
        <button class="${startCls}" data-act="start" ${isOwner && ownerState !== "running" ? "disabled" : ""}>${startLabel}</button>
        <button class="btn btn-tiny" data-act="ini">📄 INI</button>
        <button class="btn btn-tiny" data-act="edit">✎</button>
        <button class="btn btn-tiny" data-act="delete">✕</button>
      </div>
    `;
    rows.querySelectorAll(".preset-chip[data-profile-id]").forEach(chip => {
      chip.addEventListener("click", () => {
        const prof = (state.allProfiles || []).find(x => x.id === chip.dataset.profileId);
        if (prof) openProfileById(prof);
      });
    });
    // A chip for a deleted profile is dead weight the editor can't clear
    // (its id is invisible in the picker) - so the chip itself removes it.
    rows.querySelectorAll(".preset-chip[data-missing-id]").forEach(chip => {
      chip.addEventListener("click", async () => {
        const updated = (preset.profile_ids || []).filter(pid => pid !== chip.dataset.missingId);
        try {
          await API.put(`/api/presets/${preset.id}`, { profile_ids: updated });
          toast("Removed missing profile from preset", "ok");
          await loadPresets();
        } catch (e) { toast(e.message, "error"); }
      });
    });
    rows.querySelector('[data-act="start"]').addEventListener("click", () => {
      if (runningRouterPresetId() === preset.id) stopServer();
      else startPreset(preset.id);
    });
    rows.querySelector('[data-act="ini"]').addEventListener("click", () => showPresetIni(preset));
    rows.querySelector('[data-act="edit"]').addEventListener("click", () => openPresetModal(preset)); // async: refreshes profile list first
    rows.querySelector('[data-act="delete"]').addEventListener("click", async () => {
      if (!await confirmModal({ title: `Delete preset "${preset.name}"?`, message: "This can't be undone." })) return;
      try { await API.del(`/api/presets/${preset.id}`); toast("Preset deleted", "ok"); await loadPresets(); }
      catch (e) { toast(e.message, "error"); }
    });
    el.appendChild(rows);
  });
  updateProfileRunningChrome(); // disable non-owner Start buttons if a server is already up
}

async function startPreset(presetId) {
  try {
    await API.post("/api/server/start", { preset_id: presetId });
    toast("Router starting…", "ok");
    await refreshServerStatus();
  } catch (e) {
    toast(e.message, "error");
  }
}

async function showPresetIni(preset) {
  let data;
  try {
    data = await API.get(`/api/presets/${preset.id}/preview`);
  } catch (e) {
    toast(e.message, "error");
    return;
  }
  const binaryName = (state.settings && state.settings.llama_server_path)
    ? state.settings.llama_server_path.split(/[\\/]/).pop()
    : "llama-server";
  const quoted = (data.args || []).map(a => (a === "" || /\s/.test(a)) ? `"${a}"` : a);
  const command = [binaryName, ...quoted].join(" ");

  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  backdrop.innerHTML = `
    <div class="modal-card">
      <h3>Router preset - ${escapeHtml(preset.name)}</h3>
      <p class="muted small">Regenerated from the profiles each time you start the router. The section names below are the model ids clients request.</p>
      ${(data.warnings || []).map(w => `<div class="preview-warn">⚠ ${escapeHtml(w)}</div>`).join("")}
      ${data.models.length
        ? `<div class="preview-section-names">API model names: ${data.models.map(m => escapeHtml(m.section)).join(", ")}</div>`
        : `<div class="preview-warn">⚠ This preset has no usable profiles.</div>`}
      <div class="profile-list-head"><strong>INI</strong>
        <button class="btn btn-tiny" id="preset-ini-copy-ini">Copy</button>
      </div>
      <div class="preview-ini">${escapeHtml(data.ini || "(empty)")}</div>
      <div class="profile-list-head"><strong>Command</strong>
        <button class="btn btn-tiny" id="preset-ini-copy-cmd">Copy</button>
      </div>
      <div class="preview-ini">${escapeHtml(command)}</div>
      <div class="modal-actions">
        <button class="btn btn-ghost" id="preset-ini-close">Close</button>
      </div>
    </div>
  `;
  document.body.appendChild(backdrop);
  backdrop.addEventListener("click", e => { if (e.target === backdrop) backdrop.remove(); });
  backdrop.querySelector("#preset-ini-close").addEventListener("click", () => backdrop.remove());
  backdrop.querySelector("#preset-ini-copy-ini").addEventListener("click", () => {
    navigator.clipboard.writeText(data.ini || "").then(() => toast("INI copied", "ok")).catch(() => {});
  });
  backdrop.querySelector("#preset-ini-copy-cmd").addEventListener("click", () => {
    navigator.clipboard.writeText(command).then(() => toast("Command copied", "ok")).catch(() => {});
  });
}

function modelLabelFromId(modelIdStr) {
  const parts = (modelIdStr || "").split("::");
  return parts.length >= 3 ? `${parts[0]}/${parts[1]}/${parts[2]}` : modelIdStr;
}

// Just the model (model_name) part of a model_id - for the profile rows,
// where the org/repo prefix is redundant (it's in the detail pane / group
// header already).
function modelNameFromId(modelIdStr) {
  const parts = (modelIdStr || "").split("::");
  return parts.length >= 3 ? parts[2] : modelIdStr;
}

async function openProfileById(profile) {
  // Refresh the profile list first so returning to the profile page reflects
  // the state this profile was seen in, then open the fresh editor.
  await loadProfiles();
  openEditorForProfile(profile);
}

async function openPresetModal(preset) {
  // Refresh the profile list first - profiles created since the app opened
  // (or in another window) wouldn't be in state.allProfiles yet.
  await loadPresets();
  const isEdit = !!preset;
  let selectedIds = new Set((preset && preset.profile_ids) || []);

  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  backdrop.innerHTML = `
    <div class="modal-card">
      <h3>${isEdit ? `Edit "${escapeHtml(preset.name)}"` : "New router preset"}</h3>
      <p class="muted small">Pick the profiles this router should serve. Each profile becomes one model (its name is the API model id).</p>
      <div style="margin-bottom:10px;">
        <input class="input" id="preset-name" placeholder="Preset name" value="${isEdit ? escapeHtml(preset.name) : ""}" />
      </div>
      <div class="modal-filter"><input type="search" id="preset-pick-filter" class="input" placeholder="Filter profiles…" /></div>
      <div class="modal-list" id="preset-pick-list"></div>
      <div class="preset-opt-row">
        <label>Max concurrent models <input class="input" type="number" id="preset-models-max" min="0" max="64" value="${isEdit ? preset.models_max : 4}" title="0 = unlimited"></label>
        <label><input type="checkbox" id="preset-autoload" ${isEdit ? (preset.autoload ? "checked" : "") : "checked"}> load on request (autoload)</label>
        <label><input type="checkbox" id="preset-load-on-startup" ${isEdit ? (preset.load_on_startup ? "checked" : "") : "checked"}> load at startup</label>
      </div>
      <div style="margin: 0 0 14px;">
        <div style="font-size: 12.5px; color: var(--text-secondary); margin-bottom: 4px;">
          Defaults for all models <span class="muted small">- optional <code>key = value</code> lines (the INI <code>[*]</code> section); each model's own values override these</span>
        </div>
        <textarea id="preset-defaults" class="input" rows="4" placeholder="c = 8192&#10;n-gpu-layers = 8" style="width: 100%; font-family: var(--font-mono); font-size: 12px; resize: vertical; box-sizing: border-box;">${isEdit ? escapeHtml(preset.defaults || "") : ""}</textarea>
      </div>
      <div class="modal-actions">
        <button class="btn btn-ghost" id="preset-modal-cancel">Cancel</button>
        <button class="btn btn-primary" id="preset-modal-confirm" ${isEdit ? "" : "disabled"}>${isEdit ? "Save" : "Create"}</button>
      </div>
    </div>
  `;
  document.body.appendChild(backdrop);

  const listEl = backdrop.querySelector("#preset-pick-list");
  const filterEl = backdrop.querySelector("#preset-pick-filter");
  const confirmBtn = backdrop.querySelector("#preset-modal-confirm");
  const nameEl = backdrop.querySelector("#preset-name");

  // group all profiles by model for readable ordering
  const profiles = [...(state.allProfiles || [])];
  const byModel = {};
  profiles.forEach(p => {
    (byModel[p.model_id] = byModel[p.model_id] || []).push(p);
  });
  const ordered = Object.keys(byModel).sort().flatMap(mid =>
    byModel[mid].sort((a, b) => a.name.localeCompare(b.name)).map(p => ({ ...p, _modelLabel: modelLabelFromId(mid) }))
  );

  function renderList(filterText) {
    const q = (filterText || "").trim().toLowerCase();
    const visible = ordered.filter(p => !q || `${p.name} ${p._modelLabel}`.toLowerCase().includes(q));
    listEl.innerHTML = "";
    if (visible.length === 0) {
      listEl.innerHTML = ordered.length === 0
        ? `<p class="muted small" style="padding:10px;">No profiles exist yet - create one in the Library first.</p>`
        : `<p class="muted small" style="padding:10px;">No profiles match "${escapeHtml(filterText)}".</p>`;
      return;
    }
    visible.forEach(p => {
      const row = document.createElement("label");
      row.className = "preset-pick-row";
      row.innerHTML = `
        <input type="checkbox" ${selectedIds.has(p.id) ? "checked" : ""} />
        <span style="min-width:0;">
          <span class="preset-pick-name">${escapeHtml(p.name)}</span>
          <div class="preset-pick-model">${escapeHtml(p._modelLabel)}</div>
        </span>
      `;
      row.querySelector("input").addEventListener("change", e => {
        if (e.target.checked) selectedIds.add(p.id);
        else selectedIds.delete(p.id);
        confirmBtn.disabled = selectedIds.size === 0;
      });
      listEl.appendChild(row);
    });
  }
  renderList("");
  filterEl.addEventListener("input", e => renderList(e.target.value));
  confirmBtn.disabled = selectedIds.size === 0;

  const close = () => backdrop.remove();
  backdrop.addEventListener("click", e => { if (e.target === backdrop) close(); });
  backdrop.querySelector("#preset-modal-cancel").addEventListener("click", close);

  confirmBtn.addEventListener("click", async () => {
    const payload = {
      name: nameEl.value.trim(),
      profile_ids: [...selectedIds],
      models_max: Math.max(0, parseInt(backdrop.querySelector("#preset-models-max").value, 10) || 0),
      autoload: backdrop.querySelector("#preset-autoload").checked,
      load_on_startup: backdrop.querySelector("#preset-load-on-startup").checked,
      defaults: backdrop.querySelector("#preset-defaults").value,
    };
    if (!payload.name) { toast("Give the preset a name.", "error"); return; }
    try {
      if (isEdit) await API.put(`/api/presets/${preset.id}`, payload);
      else await API.post("/api/presets", payload);
      toast(isEdit ? "Preset saved" : "Preset created", "ok");
      close();
      await loadPresets();
    } catch (e) { toast(e.message, "error"); }
  });
}

// ---- capability badge ----

async function refreshRouterCapability() {
  const el = document.getElementById("router-cap-badge");
  if (!el) return;
  try {
    const cap = await API.get("/api/presets/capability");
    if (!cap.binary) {
      el.textContent = "llama-server not configured";
      el.className = "router-cap warn";
    } else if (cap.supported === true) {
      el.textContent = "router supported";
      el.className = "router-cap ok";
    } else if (cap.supported === false) {
      el.textContent = "llama-server too old for router mode";
      el.className = "router-cap err";
    } else {
      el.textContent = "";
      el.className = "router-cap";
    }
  } catch (e) {
    el.textContent = "";
    el.className = "router-cap";
  }
}

// ---- live model status of the running router ----

function updateRouterPanel(status) {
  // Two pages can host a live router (Router Presets via --models-preset,
  // Router Dir via --models-dir); only the one whose entity owns the running
  // server shows its "models" panel.
  const active = status.mode === "router" && (status.state === "running" || status.state === "starting");
  const presetCard = document.getElementById("router-models-card");
  if (presetCard) presetCard.hidden = !(active && !!status.preset_id);
  const dirCard = document.getElementById("router-dir-models-card");
  if (dirCard) dirCard.hidden = !(active && !!status.router_dir_id);

  if (active && status.state === "running" && !state.routerPollTimer) {
    state.routerPollTimer = setInterval(() => fetchRouterModels(false), 3000);
    fetchRouterModels(false);
  } else if (!active && state.routerPollTimer) {
    clearInterval(state.routerPollTimer);
    state.routerPollTimer = null;
    state.routerModels = [];
    renderRouterModels();
  }
}

async function fetchRouterModels(reload) {
  try {
    const data = await API.get(`/api/server/router/models${reload ? "?reload=true" : ""}`);
    state.routerModels = (data && data.data) || [];
    renderRouterModels();
  } catch (e) {
    // router not answering yet (still starting) or not in router mode - ignore
  }
}

function renderRouterModels() {
  // Both the Router Presets and Router Dir pages host the same live-router
  // model list (only one router runs at a time); render into whichever
  // panels exist so the right page shows it.
  const targets = ["router-model-rows", "router-dir-model-rows"]
    .map(id => document.getElementById(id)).filter(Boolean);
  if (!targets.length) return;
  targets.forEach(el => {
    if (!state.routerModels.length) {
      el.innerHTML = `<p class="muted small">Waiting for the router to report its models…</p>`;
      return;
    }
    el.innerHTML = "";
    state.routerModels.forEach(m => el.appendChild(buildRouterModelRow(m)));
  });
}

function buildRouterModelRow(m) {
  const status = (m.status && m.status.value) || "unknown";
  const row = document.createElement("div");
  row.className = "router-model-row";
  const canLoad = !["loaded", "loading", "downloading"].includes(status);
  const canUnload = ["loaded", "loading", "sleeping", "unloaded"].includes(status) && !!m.id;
  const failed = m.status && m.status.failed
    ? ` <span class="muted small">(exit ${m.status.exit_code ?? "?"})</span>` : "";
  row.innerHTML = `
    <span class="router-model-id">${escapeHtml(m.id || "?")}</span>
    <span class="router-model-actions">
      <span class="status-badge ${escapeHtml(status)}">${escapeHtml(status)}${failed}</span>
      <button class="btn btn-tiny" data-act="load" ${canLoad ? "" : "disabled"}>Load</button>
      <button class="btn btn-tiny" data-act="unload" ${canUnload ? "" : "disabled"}>Unload</button>
    </span>
  `;
  row.querySelector('[data-act="load"]').addEventListener("click", () => routerModelAction("load", m.id));
  row.querySelector('[data-act="unload"]').addEventListener("click", () => routerModelAction("unload", m.id));
  return row;
}

async function routerModelAction(action, modelIdStr) {
  try {
    await API.post(`/api/server/router/models/${action}`, { model: modelIdStr });
    toast(`Model ${action} requested`, "ok");
    await fetchRouterModels(false);
  } catch (e) {
    toast(e.message, "error");
  }
}

// ---------------------------------------------------------------------------
// Router dirs (--models-dir)
// ---------------------------------------------------------------------------

function wireRouterDir() {
  document.getElementById("btn-new-router-dir").addEventListener("click", () => openRouterDirEditor(null));
  document.getElementById("btn-router-dir-reload").addEventListener("click", () => fetchRouterModels(true));
}

async function loadRouterDirs() {
  try {
    state.routerDirs = await API.get("/api/router-dirs");
  } catch (e) {
    state.routerDirs = [];
    toast(e.message, "error");
  }
  renderRouterDirList();
}

function renderRouterDirList() {
  const el = document.getElementById("router-dir-rows");
  if (!el) return;
  if (!state.routerDirs.length) {
    el.innerHTML = `<p class="muted small">No router dirs yet. Create one to serve every model in a folder with shared default parameters.</p>`;
    return;
  }
  el.innerHTML = "";
  state.routerDirs.forEach(rd => {
    const rows = document.createElement("div");
    rows.className = "preset-row";
    rows.dataset.routerDirId = rd.id;
    const isOwner = runningRouterDirId() === rd.id;
    const ownerState = isOwner && state.serverStatus ? state.serverStatus.state : null;
    const startLabel = isOwner
      ? (ownerState === "starting" ? "⏳ Loading…" : ownerState === "stopping" ? "■ Stopping…" : "■ Stop")
      : "▶ Start";
    const startCls = isOwner && ownerState === "running" ? "btn btn-danger btn-tiny" : "btn btn-primary btn-tiny";
    rows.innerHTML = `
      <div style="min-width:0;">
        <div class="preset-row-name">${escapeHtml(rd.name)}<span class="running-badge" ${isOwner && ownerState === "running" ? "" : "hidden"}>● running</span></div>
        <div class="preset-row-meta">${escapeHtml(rd.models_dir || "(no directory set)")} · max ${rd.models_max === 0 ? "unlimited" : rd.models_max} concurrent · autoload ${rd.autoload ? "on" : "off"}</div>
      </div>
      <div class="preset-row-actions">
        <button class="${startCls}" data-act="start" ${isOwner && ownerState !== "running" ? "disabled" : ""}>${startLabel}</button>
        <button class="btn btn-tiny" data-act="cmd">⌨ cmd</button>
        <button class="btn btn-tiny" data-act="edit">✎</button>
        <button class="btn btn-tiny" data-act="delete">✕</button>
      </div>
    `;
    rows.querySelector('[data-act="start"]').addEventListener("click", () => {
      if (runningRouterDirId() === rd.id) stopServer();
      else startRouterDir(rd.id);
    });
    rows.querySelector('[data-act="cmd"]').addEventListener("click", () => showRouterDirCommand(rd));
    rows.querySelector('[data-act="edit"]').addEventListener("click", () => openRouterDirEditor(rd));
    rows.querySelector('[data-act="delete"]').addEventListener("click", async () => {
      if (runningRouterDirId() === rd.id) { toast("Stop the server before deleting this router dir.", "error"); return; }
      if (!await confirmModal({ title: `Delete router dir "${rd.name}"?`, message: "This can't be undone." })) return;
      try { await API.del(`/api/router-dirs/${rd.id}`); toast("Router dir deleted", "ok"); await loadRouterDirs(); }
      catch (e) { toast(e.message, "error"); }
    });
    el.appendChild(rows);
  });
  updateProfileRunningChrome();
}

function openRouterDirEditor(rd) {
  state.editingMode = "router_dir";
  state.editingProfile = rd
    ? {
      id: rd.id, name: rd.name, models_dir: rd.models_dir, models_max: rd.models_max, autoload: rd.autoload,
      params: JSON.parse(JSON.stringify(rd.params || {})), custom_flags: rd.custom_flags || "", notes: rd.notes || "",
    }
    : { id: null, name: "", models_dir: "", models_max: 4, autoload: true, params: {}, custom_flags: "", notes: "" };
  openEditor(rd ? `Editing - ${rd.name}` : "New router dir");
}

async function startRouterDir(routerDirId) {
  try {
    await API.post("/api/server/start", { router_dir_id: routerDirId });
    toast("Router starting…", "ok");
    await refreshServerStatus();
  } catch (e) {
    toast(e.message, "error");
  }
}

async function showRouterDirCommand(rd) {
  let data;
  try { data = await API.get(`/api/router-dirs/${rd.id}/command-preview`); }
  catch (e) { toast(e.message, "error"); return; }
  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  backdrop.innerHTML = `
    <div class="modal-card">
      <h3>Router dir - ${escapeHtml(rd.name)}</h3>
      <p class="muted small">The command that would be run. host/port come from your app defaults.</p>
      <div class="profile-list-head"><strong>Command</strong>
        <button class="btn btn-tiny" id="router-dir-cmd-copy">Copy</button>
      </div>
      <div class="preview-ini">${escapeHtml(data.command || "(empty)")}</div>
      <div class="modal-actions"><button class="btn btn-ghost" id="router-dir-cmd-close">Close</button></div>
    </div>
  `;
  document.body.appendChild(backdrop);
  backdrop.addEventListener("click", e => { if (e.target === backdrop) backdrop.remove(); });
  backdrop.querySelector("#router-dir-cmd-close").addEventListener("click", () => backdrop.remove());
  backdrop.querySelector("#router-dir-cmd-copy").addEventListener("click", () => {
    navigator.clipboard.writeText(data.command || "").then(() => toast("Command copied", "ok")).catch(() => {});
  });
}

async function refreshRouterDirCapability() {
  const el = document.getElementById("router-dir-cap-badge");
  if (!el) return;
  try {
    const cap = await API.get("/api/router-dirs/capability");
    if (!cap.binary) { el.textContent = "llama-server not configured"; el.className = "router-cap warn"; }
    else if (cap.supported === true) { el.textContent = "--models-dir supported"; el.className = "router-cap ok"; }
    else if (cap.supported === false) { el.textContent = "llama-server too old for --models-dir"; el.className = "router-cap err"; }
    else { el.textContent = ""; el.className = "router-cap"; }
  } catch (e) {
    el.textContent = "";
    el.className = "router-cap";
  }
}

// ---------------------------------------------------------------------------
// Settings (autosaves - no explicit Save button)
// ---------------------------------------------------------------------------

function hasNativeDialogs() {
  return !!(window.pywebview && window.pywebview.api);
}

function wireSettings() {
  document.getElementById("lsp-download").addEventListener("click", lspStartDownload);
  document.getElementById("lsp-cancel").addEventListener("click", lspCancelDownload);
  document.getElementById("lsp-variant").addEventListener("change", lspUpdateDownloadButton);
  document.getElementById("btn-add-llama-version").addEventListener("click", addLlamaVersion);
  document.getElementById("btn-add-folder").addEventListener("click", addFolder);
  document.getElementById("settings-new-folder").addEventListener("keydown", e => {
    if (e.key === "Enter") addFolder();
  });

  const debouncedSave = debounce(saveSettingsForm, 500);
  document.getElementById("settings-host").addEventListener("input", debouncedSave);
  document.getElementById("settings-port").addEventListener("input", debouncedSave);
  document.getElementById("settings-theme").addEventListener("change", saveSettingsForm);
  document.getElementById("settings-verbose").addEventListener("change", saveSettingsForm);
  document.getElementById("settings-allow-lan").addEventListener("change", async () => {
    await saveSettingsForm();
    refreshLanAccessStatus();
  });

  document.getElementById("btn-browse-folder").addEventListener("click", async () => {
    if (!hasNativeDialogs()) {
      toast("Folder picker needs the desktop app window - type the path manually here.", "error");
      return;
    }
    try {
      const path = await window.pywebview.api.pick_folder();
      if (path) {
        document.getElementById("settings-new-folder").value = path;
        addFolder();
      }
    } catch (e) { toast("Could not open the folder picker.", "error"); }
  });

  refreshLanAccessStatus();
}

async function refreshLanAccessStatus() {
  const statusEl = document.getElementById("lan-access-status");
  try {
    const info = await API.get("/api/system/network-info");
    if (info.currently_lan_reachable && info.lan_url) {
      statusEl.textContent = `Currently reachable from other devices at ${info.lan_url}`;
      statusEl.className = "field-hint ok";
    } else if (info.allow_lan_access && !info.currently_lan_reachable) {
      statusEl.textContent = "Enabled, but not active yet - restart the app for this to take effect.";
      statusEl.className = "field-hint";
    } else {
      statusEl.textContent = "Currently only accessible from this computer.";
      statusEl.className = "field-hint";
    }
  } catch (e) {
    statusEl.textContent = "";
  }
}

function populateSettingsForm() {
  const s = state.settings;
  renderLlamaVersions();
  document.getElementById("settings-host").value = s.default_host || "127.0.0.1";
  document.getElementById("settings-port").value = s.default_port || 8080;
  document.getElementById("settings-theme").value = s.theme || "dark";
  document.getElementById("settings-allow-lan").checked = !!s.allow_lan_access;
  document.getElementById("settings-verbose").checked = !!s.verbose;
  renderFolderList();
}

function renderFolderList() {
  const list = document.getElementById("folder-list");
  list.innerHTML = "";
  (state.settings.model_root_folders || []).forEach((folder, idx) => {
    const li = document.createElement("li");
    li.innerHTML = `<span>${escapeHtml(folder)}</span><button data-idx="${idx}">✕ remove</button>`;
    li.querySelector("button").addEventListener("click", () => {
      state.settings.model_root_folders.splice(idx, 1);
      renderFolderList();
      saveSettingsForm();
    });
    list.appendChild(li);
  });
}

function addFolder() {
  const input = document.getElementById("settings-new-folder");
  const val = input.value.trim();
  if (!val) return;
  state.settings.model_root_folders = state.settings.model_root_folders || [];
  if (!state.settings.model_root_folders.includes(val)) {
    state.settings.model_root_folders.push(val);
  }
  input.value = "";
  renderFolderList();
  saveSettingsForm();
}

// ---- llama-server versions (multiple builds, one active) ----

// Derive a sensible name from a binary path: file name without extension
// (llama-server-b6120.exe -> llama-server-b6120).
function llamaStem(path) {
  const base = (path || "").split(/[\\/]/).pop() || "";
  return base.replace(/\.(exe|bin)$/i, "");
}

function activeLlamaEntry() {
  const s = state.settings;
  return (s.llama_servers || []).find(e => e.name && e.name === s.active_llama_server) || null;
}

function renderLlamaVersions() {
  const wrap = document.getElementById("llama-versions");
  const hint = document.getElementById("llama-active-hint");
  if (!wrap) return;
  const servers = state.settings.llama_servers || [];
  wrap.innerHTML = "";
  if (!servers.length) {
    wrap.innerHTML = `<p class="muted small">No versions added yet - use “Add version” below, or
      <a href="#" id="lsp-autolink">download the latest official build above</a>.</p>`;
    const link = document.getElementById("lsp-autolink");
    if (link) link.addEventListener("click", e => {
      e.preventDefault();
      const card = document.getElementById("llama-versions-card");
      if (card) card.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    if (hint) { hint.textContent = "No llama-server configured - servers cannot be started."; hint.className = "field-hint error"; }
    return;
  }

  const debouncedSave = debounce(saveSettingsForm, 500);
  servers.forEach((entry, idx) => {
    const isActive = !!(entry.name && entry.name === state.settings.active_llama_server);
    const row = document.createElement("div");
    row.className = "llama-version-row" + (isActive ? " is-active" : "");
    row.innerHTML = `
      <label class="lv-active" title="Use this version to start servers">
        <input type="radio" name="active-llama-server" ${entry.path ? "" : "disabled"} ${isActive ? "checked" : ""} />
        <span>use</span>
      </label>
      <input type="text" class="input lv-name" placeholder="Name / version (e.g. b6120)" value="${escapeHtml(entry.name || "")}" />
      <input type="text" class="input lv-path" placeholder="/path/to/llama-server" value="${escapeHtml(entry.path || "")}" />
      <button class="btn btn-ghost lv-browse" title="Browse for the executable">Browse…</button>
      <button class="btn btn-ghost lv-check" title="Check that this file exists">Check</button>
      <button class="btn btn-ghost lv-remove" title="Remove this version">✕</button>
    `;

    row.querySelector("input[type=radio]").addEventListener("change", () => {
      // A version needs a name to be referenceable; derive one from the file if missing.
      if (!entry.name) entry.name = llamaStem(entry.path) || `version-${idx + 1}`;
      state.settings.active_llama_server = entry.name;
      renderLlamaVersions();
      saveSettingsForm();
    });

    row.querySelector(".lv-name").addEventListener("input", e => {
      const wasActive = entry.name === state.settings.active_llama_server;
      entry.name = e.target.value;
      if (wasActive) state.settings.active_llama_server = entry.name.trim();
      debouncedSave();
    });

    row.querySelector(".lv-path").addEventListener("input", e => {
      entry.path = e.target.value;
      debouncedSave();
    });

    row.querySelector(".lv-browse").addEventListener("click", async () => {
      if (!hasNativeDialogs()) {
        toast("File picker needs the desktop app window - type the path manually here.", "error");
        return;
      }
      try {
        const path = await window.pywebview.api.pick_executable();
        if (!path) return;
        entry.path = path;
        if (!entry.name) entry.name = llamaStem(path);
        renderLlamaVersions();
        await saveSettingsForm();
        checkLlamaVersionRow(entry, row);
      } catch (e) { toast("Could not open the file picker.", "error"); }
    });

    row.querySelector(".lv-check").addEventListener("click", () => checkLlamaVersionRow(entry, row));

    row.querySelector(".lv-remove").addEventListener("click", async () => {
      const label = entry.name || llamaStem(entry.path) || "this version";
      const pathStr = (entry.path || "").trim();
      if (pathStr) {
        if (!await confirmModal({
          title: `Remove "${label}"?`,
          message: `Removes it from the version list and deletes its files from disk:\n\n${pathStr}\n\nThis can't be undone.`,
          confirmLabel: "Remove",
        })) return;
        try {
          const res = await API.post("/api/llama-server/remove", { path: pathStr });
          // res.deleted === false: not on disk (already moved/renamed) - the
          // stale list entry is still dropped below.
          if (res.deleted) toast(res.message, "ok");
        } catch (e) {
          toast(e.message, "error");
          return;  // deletion failed (e.g. build in use) - keep the entry
        }
      }
      state.settings.llama_servers.splice(idx, 1);
      if (isActive) {
        const first = (state.settings.llama_servers || []).find(e => e.path);
        state.settings.active_llama_server = first ? first.name : "";
      }
      renderLlamaVersions();
      saveSettingsForm();
    });

    wrap.appendChild(row);
  });

  const active = activeLlamaEntry();
  if (hint) {
    if (active && active.path) {
      hint.textContent = `Using: ${active.name} - ${active.path}`;
      hint.className = "field-hint ok";
    } else {
      hint.textContent = "No usable active version - servers cannot be started.";
      hint.className = "field-hint error";
    }
  }
}

function addLlamaVersion() {
  state.settings.llama_servers = state.settings.llama_servers || [];
  state.settings.llama_servers.push({ name: "", path: "" });
  renderLlamaVersions();
  const rows = document.querySelectorAll("#llama-versions .llama-version-row");
  const last = rows[rows.length - 1];
  if (last) last.querySelector(".lv-name").focus();
}

async function checkLlamaVersionRow(entry, row) {
  if (!entry.path) { toast("Set the path first.", "error"); return; }
  row.classList.remove("lv-ok", "lv-bad");
  const btn = row.querySelector(".lv-check");
  btn.disabled = true;
  btn.textContent = "…";
  try {
    const result = await API.post("/api/settings/validate-binary", { path: entry.path });
    row.classList.add(result.valid ? "lv-ok" : "lv-bad");
    toast(result.message, result.valid ? "ok" : "error");
  } catch (e) {
    row.classList.add("lv-bad");
    toast(e.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Check";
  }
}

// ---- llama-server (llama.cpp): latest-build check + one-click install ----

// Show the running app's version in its own card on the Settings page and
// whether a newer release exists (the "new version available" badge).
// The version comes from GET /api/system/info (single source of truth: the
// __version__ in backend/__init__.py); the latest release from
// GET /api/app/latest, which already carries the backend-computed
// `is_newer` flag (same semver rules the backend uses everywhere).
// Called once per Settings visit, alongside refreshLlamaServerCard() in
// showView(). The update check is deliberately quiet: when the latest check
// fails (offline, or no releases yet on the repo) the badge just stays
// hidden - no toast, no error text.
async function refreshAppVersionCard() {
  const lineEl = document.getElementById("lpm-version-line");
  const badgeEl = document.getElementById("lpm-new-badge");
  const openBtn = document.getElementById("btn-lpm-open-release");
  if (!lineEl) return;
  if (badgeEl) badgeEl.hidden = true;
  if (openBtn) openBtn.hidden = true;

  lineEl.textContent = "LPM version …";
  try {
    const info = await API.get("/api/system/info");
    lineEl.textContent = `LPM version v${info.version}`;
  } catch (e) {
    lineEl.textContent = "LPM version (unknown)";
    return;
  }

  let latest = null;
  try {
    latest = await API.get("/api/app/latest");
  } catch (e) {
    // Offline / no releases on the repo yet: stay quiet (same posture as
    // the lsp error path, minus the error line - this is not actionable).
    return;
  }
  if (latest && latest.is_newer) {
    if (badgeEl) badgeEl.hidden = false;
    if (openBtn) {
      openBtn.hidden = false;
      openBtn.onclick = () => openReleaseUrl(latest.html_url);
    }
  }
}

// Open the release page in the user's real default browser (not inside the
// pywebview window) - same bridge as the Hugging Face links.
async function openReleaseUrl(url) {
  try {
    await API.post("/api/system/open-url", { url });
  } catch (e) {
    toast(e.message, "error");
  }
}

const LSP_ACTIVE_STATES = ["downloading", "extracting", "registering"];

// Refresh the "llama-server (llama.cpp)" settings card: latest release info
// + current job/active-build state. Called on entering the Settings view
// and after job transitions. Never spams the console: all failures land in
// the card's status line (or a toast) instead.
async function refreshLlamaServerCard() {
  const statusEl = document.getElementById("lsp-status");
  const badgeEl = document.getElementById("lsp-new-badge");
  const dlBtn = document.getElementById("lsp-download");
  const cancelBtn = document.getElementById("lsp-cancel");
  if (!statusEl) return;

  let latest = null, status = null;
  try {
    [latest, status] = await Promise.all([
      API.get("/api/llama-server/latest"),
      API.get("/api/llama-server/status"),
    ]);
  } catch (e) {
    // Offline / unsupported platform / API hiccup: one clean error line.
    statusEl.textContent = e.message || "Could not check for the latest llama.cpp build.";
    statusEl.className = "field-hint error";
    if (badgeEl) badgeEl.hidden = true;
    if (dlBtn) { dlBtn.textContent = "Download latest"; dlBtn.disabled = true; }
    if (cancelBtn) cancelBtn.hidden = true;
    document.getElementById("lsp-progress").hidden = true;
    lspStopPolling();
    return;
  }
  state.lspLatest = latest;
  lspFillVariantSelect(latest);

  const cur = status.current || {};
  const date = latest.published_at
    ? new Date(latest.published_at).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })
    : "";
  const active = cur.build
    ? `Active: b${cur.build}`
    : (cur.path ? "Active: build unknown" : "Not installed yet");
  statusEl.textContent = `${active} · Latest: b${latest.build}${date ? ` (${date})` : ""}`;
  statusEl.className = "field-hint";
  if (badgeEl) badgeEl.hidden = !(cur.build && latest.build > cur.build);
  lspUpdateDownloadButton();
  if (dlBtn) dlBtn.disabled = LSP_ACTIVE_STATES.includes(status.state);
  if (cancelBtn) cancelBtn.hidden = !LSP_ACTIVE_STATES.includes(status.state);
  lspUpdateProgress(status);
  if (LSP_ACTIVE_STATES.includes(status.state)) lspStartPolling();
  else lspStopPolling();
}

// Fill the build-variant <select> (CPU / CUDA / Vulkan / ...) from the
// /latest payload. Keeps the current selection when it still exists,
// otherwise defaults to the CPU build. Hidden when there's only one choice.
function lspFillVariantSelect(latest) {
  const sel = document.getElementById("lsp-variant");
  if (!sel) return;
  const variants = latest.variants || [];
  const keep = sel.value;
  sel.innerHTML = "";
  variants.forEach(v => {
    const o = document.createElement("option");
    o.value = v.name;
    o.textContent = `${v.label} (${formatBytes(v.size || 0)})`;
    sel.appendChild(o);
  });
  sel.hidden = variants.length <= 1;
  if (variants.some(v => v.name === keep)) {
    sel.value = keep;
  } else {
    const cpu = variants.find(v => v.is_cpu) || variants[0];
    if (cpu) sel.value = cpu.name;
  }
}

function lspChosenVariant() {
  const sel = document.getElementById("lsp-variant");
  const variants = (state.lspLatest && state.lspLatest.variants) || [];
  return variants.find(v => v.name === (sel && sel.value)) || null;
}

// Re-labels the download button from the chosen variant. Called on card
// refresh AND on the <select>'s change event, so picking a variant updates
// the button immediately. (Only touches the label - the disabled state is
// owned by the job lifecycle, so a selection mid-download doesn't re-enable.)
function lspUpdateDownloadButton() {
  const dlBtn = document.getElementById("lsp-download");
  const latest = state.lspLatest;
  if (!dlBtn || !latest) return;
  const chosen = lspChosenVariant();
  dlBtn.textContent = chosen
    ? `Download b${latest.build} - ${chosen.label} (${formatBytes(chosen.size || 0)})`
    : `Download b${latest.build}`;
}

function lspUpdateProgress(status) {
  const progEl = document.getElementById("lsp-progress");
  const fill = document.getElementById("lsp-progress-fill");
  const meta = document.getElementById("lsp-progress-meta");
  const dlBtn = document.getElementById("lsp-download");
  const cancelBtn = document.getElementById("lsp-cancel");
  if (!progEl) return;
  const active = LSP_ACTIVE_STATES.includes(status.state);
  progEl.hidden = !active;
  // Kept in sync on every poll tick, not just on card refresh, so the
  // buttons react the moment a job starts/stops.
  if (cancelBtn) cancelBtn.hidden = !active;
  if (dlBtn) dlBtn.disabled = active;
  if (!active) return;
  const pct = status.bytes_total ? Math.min(100, (status.bytes_done / status.bytes_total) * 100) : 0;
  const label = { downloading: "Downloading…", extracting: "Extracting…", registering: "Registering…" }[status.state] || "";
  if (fill) fill.style.width = `${pct}%`;
  if (meta) meta.textContent = status.bytes_total
    ? `${label} ${formatBytes(status.bytes_done || 0)} / ${formatBytes(status.bytes_total)}`
    : `${label} ${formatBytes(status.bytes_done || 0)}`;
}

// Poll /api/llama-server/status every 1.5 s while a job is in flight. The
// timer is cleared on view change (see showView) and on job completion.
function lspStartPolling() {
  if (state.lspPollTimer) return;
  state.lspPollTimer = setInterval(lspPollTick, 1500);
}

function lspStopPolling() {
  if (state.lspPollTimer) { clearInterval(state.lspPollTimer); state.lspPollTimer = null; }
}

async function lspPollTick() {
  let status;
  try {
    status = await API.get("/api/llama-server/status");
  } catch (e) {
    return;  // transient network blip - keep polling
  }
  lspUpdateProgress(status);
  if (LSP_ACTIVE_STATES.includes(status.state)) return;
  lspStopPolling();
  if (status.state === "done") {
    toast(`llama-server ${status.tag || ""} installed - it's in the versions list below.`, "ok");
    try {
      state.settings = await API.get("/api/settings");  // pick up the new entry
    } catch (e) { /* keep current settings; the card below still refreshes */ }
    populateSettingsForm();                             // re-renders the version rows
  } else if (status.state === "error") {
    toast(status.error || "The llama-server download failed.", "error");
  } else if (status.state === "cancelled") {
    toast("llama-server download cancelled.");
  }
  refreshLlamaServerCard();
}

async function lspStartDownload() {
  const dlBtn = document.getElementById("lsp-download");
  if (dlBtn) dlBtn.disabled = true;
  const chosen = lspChosenVariant();
  try {
    const status = await API.post("/api/llama-server/download", { asset: chosen ? chosen.name : null });
    lspUpdateProgress(status);
    lspStartPolling();
  } catch (e) {
    if (/already in progress/i.test(e.message || "")) {
      refreshLlamaServerCard();   // watch the existing job instead
    } else {
      toast(e.message, "error");
      refreshLlamaServerCard();
    }
  }
}

async function lspCancelDownload() {
  try {
    await API.post("/api/llama-server/download/cancel", {});
  } catch (e) {
    toast(e.message, "error");
  }
}

// All settings changes funnel through this queue so saves run strictly in
// order - an older in-flight save must not land after a newer one and clobber
// state (e.g. reverting a version selection that was just made).
let settingsSaveQueue = Promise.resolve();

function saveSettingsForm() {
  settingsSaveQueue = settingsSaveQueue.then(doSaveSettings).catch(() => {});
  return settingsSaveQueue;
}

async function doSaveSettings() {
  // Trim the version entries IN PLACE: the row inputs' listeners hold
  // references to these exact objects, so replacing them (map/filter or a
  // wholesale state swap) would detach the listeners and subsequent
  // keystrokes would be lost. Empty rows are only filtered out of the
  // payload, so their objects (and their row's listeners) stay live.
  const live = state.settings.llama_servers || (state.settings.llama_servers = []);
  live.forEach(e => { e.name = (e.name || "").trim(); e.path = (e.path || "").trim(); });
  const servers = live.filter(e => e.name || e.path);
  const activeName = (state.settings.active_llama_server || "").trim();
  state.settings.active_llama_server = activeName;
  const activeEntry = servers.find(e => e.name === activeName);
  const payload = {
    llama_servers: servers,
    active_llama_server: activeName,
    llama_server_path: activeEntry ? activeEntry.path : "",
    model_root_folders: state.settings.model_root_folders || [],
    default_host: document.getElementById("settings-host").value.trim() || "127.0.0.1",
    default_port: parseInt(document.getElementById("settings-port").value, 10) || 8080,
    theme: document.getElementById("settings-theme").value,
    allow_lan_access: document.getElementById("settings-allow-lan").checked,
    verbose: document.getElementById("settings-verbose").checked,
  };
  const statusEl = document.getElementById("settings-save-status");
  const rootsChanged = JSON.stringify(state.settings.model_root_folders || []) !== JSON.stringify(payload.model_root_folders);
  try {
    const saved = await API.put("/api/settings", payload);
    // Merge, don't replace: keep the live version-entry objects and the
    // active name we just sent. The server echoes them back as fresh
    // deserialized objects, which would detach the row inputs.
    const { llama_servers: _ls, active_llama_server: _aa, llama_server_path: _lp, ...rest } = saved;
    Object.assign(state.settings, rest);
    state.settings.llama_server_path = activeEntry ? activeEntry.path : "";
    applyTheme(state.settings.theme);
    statusEl.textContent = "Saved automatically.";
    statusEl.className = "field-hint ok";
    if (rootsChanged) await loadLibrary(true);
    refreshRouterCapability(); // the binary may have changed
    // The llama-server card ("Active: bNNNN" line + the "new build
    // available" badge) is derived from the active build too - e.g. after
    // downloading a newer build and switching to it via the "use" radio. The
    // view guard matters: the debounced save may fire after the user left
    // Settings, and there is nothing to refresh then.
    if (state.view === "settings") refreshLlamaServerCard();
  } catch (e) {
    statusEl.textContent = e.message;
    statusEl.className = "field-hint error";
  }
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

function wireBenchmarks() {
  // Remember the run form's values (prompt/gen tokens, repetitions, custom
  // prompt) as the user edits them, like the Settings page does.
  const debouncedBenchSave = debounce(saveBenchFormSettings, 500);
  ["bench-prompt-tokens", "bench-gen-tokens", "bench-reps", "bench-custom-prompt"].forEach(id => {
    document.getElementById(id).addEventListener("input", debouncedBenchSave);
  });
  document.getElementById("bench-filter").addEventListener("input", e => {
    state.benchFilterText = e.target.value;
    state.benchPage = 1;   // a new filter restarts pagination at the top
    renderBenchmarks();
  });
  document.getElementById("bench-per-page").addEventListener("change", e => {
    state.benchPerPage = parseInt(e.target.value, 10) || 25;
    state.benchPage = 1;
    renderBenchmarks();
  });
  document.getElementById("bench-prev-page").addEventListener("click", () => {
    if (state.benchPage > 1) { state.benchPage -= 1; renderBenchmarks(); }
  });
  document.getElementById("bench-next-page").addEventListener("click", () => {
    state.benchPage += 1;
    renderBenchmarks();     // renderBenchmarks clamps the page into range
  });
  document.getElementById("btn-bench-edit-profile").addEventListener("click", () => {
    const pid = document.getElementById("bench-profile").value;
    const p = (state.profiles || []).find(x => x.id === pid);
    if (!p) { toast("Select a profile first.", "error"); return; }
    openEditorForProfile(p);   // re-fetches the latest copy, then opens the editor
  });
  document.getElementById("btn-bench-compare").addEventListener("click", () => openBenchCompareModal());
  document.getElementById("btn-bench-run").addEventListener("click", () => startBenchmarkFromForm());
  document.getElementById("btn-bench-cancel").addEventListener("click", () => {
    if (state.activeBenchmarkId) cancelBenchmark(state.activeBenchmarkId);
  });
  document.querySelectorAll(".bench-table th[data-sort]").forEach(th => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      if (state.benchSort.key === key) {
        state.benchSort.dir = state.benchSort.dir === "desc" ? "asc" : "desc";
      } else {
        const textCols = ["profile_name", "model_name"];
        state.benchSort = { key, dir: textCols.includes(key) ? "asc" : "desc" };
      }
      renderBenchmarks();
    });
  });
}

async function loadBenchmarks() {
  const seq = ++state.benchLoadSeq;
  try {
    const data = await API.get("/api/benchmarks");
    if (seq !== state.benchLoadSeq) return; // a newer load finished; this response is stale
    state.benchmarks = data;
    // Prune compare selections whose record no longer exists (deleted from
    // another window, startup recovery, …) and refresh the button if so.
    const known = new Set(data.map(b => b.id));
    if (state.benchCompareIds.some(id => !known.has(id))) {
      state.benchCompareIds = state.benchCompareIds.filter(id => known.has(id));
      updateBenchCompareButton();
    }
  } catch (e) {
    if (seq !== state.benchLoadSeq) return;
    state.benchmarks = [];
    toast(e.message, "error");
  }
  populateBenchProfileSelect();
  renderBenchmarks();
  renderBenchDetail();
  setBenchRunningChrome(!!state.activeBenchmarkId);
  resumeBenchmarkPollingIfNeeded();
}

function populateBenchProfileSelect() {
  const sel = document.getElementById("bench-profile");
  if (!sel) return;
  const cur = sel.value;
  sel.innerHTML = "";
  const editBtn = document.getElementById("btn-bench-edit-profile");
  const profiles = state.profiles || [];
  if (profiles.length === 0) {
    sel.innerHTML = `<option value="">(no profiles yet)</option>`;
    sel.disabled = true;
    if (editBtn) editBtn.disabled = true;
    return;
  }
  sel.disabled = false;
  if (editBtn) editBtn.disabled = false;
  profiles.forEach(p => {
    const o = document.createElement("option");
    o.value = p.id;
    o.textContent = `${p.name} - ${modelLabelFromId(p.model_id)}`;
    sel.appendChild(o);
  });
  if (cur && profiles.some(p => p.id === cur)) sel.value = cur;
  else if (state.selectedProfileId && profiles.some(p => p.id === state.selectedProfileId)) sel.value = state.selectedProfileId;
}

// The token/repetition settings as they sit in the run form. Both a fresh
// run and a re-run send these, so "Re-run" honours whatever the user has in
// the UI (the model params still come from the profile / saved snapshot).
function readBenchOptions() {
  return {
    prompt_tokens: parseInt(document.getElementById("bench-prompt-tokens").value, 10) || 512,
    gen_tokens: parseInt(document.getElementById("bench-gen-tokens").value, 10) || 256,
    repetitions: parseInt(document.getElementById("bench-reps").value, 10) || 5,
    custom_prompt: document.getElementById("bench-custom-prompt").value.trim() || null,
  };
}

// The run form's values are remembered in the app settings (data/settings.json)
// so the Benchmarks page opens with the user's last choices. The form itself is
// the UI for these settings - no Settings-page row is shown for them.
function populateBenchFormFromSettings() {
  const s = state.settings || {};
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
  set("bench-prompt-tokens", s.bench_prompt_tokens ?? 512);
  set("bench-gen-tokens", s.bench_gen_tokens ?? 256);
  set("bench-reps", s.bench_repetitions ?? 5);
  set("bench-custom-prompt", s.bench_custom_prompt ?? "");
}

function saveBenchFormSettings() {
  const body = {
    bench_prompt_tokens: parseInt(document.getElementById("bench-prompt-tokens").value, 10) || 512,
    bench_gen_tokens: parseInt(document.getElementById("bench-gen-tokens").value, 10) || 256,
    bench_repetitions: parseInt(document.getElementById("bench-reps").value, 10) || 5,
    bench_custom_prompt: document.getElementById("bench-custom-prompt").value,
  };
  // Echo back the fields WE sent (not the whole response): a wholesale state
  // swap would replace state.settings.llama_servers and detach the version-row
  // inputs' listeners (see doSaveSettings).
  API.put("/api/settings", body)
    .then(() => { Object.assign(state.settings, body); })
    .catch(() => toast("Couldn't save benchmark settings.", "error"));
}

async function startBenchmarkFromForm(profileId) {
  const pid = profileId || document.getElementById("bench-profile").value;
  if (!pid) { toast("Pick a profile to benchmark first.", "error"); return null; }
  showView("benchmarks");
  if (state.activeBenchmarkId) {
    toast("A benchmark is already running - wait for it to finish or cancel it.", "error");
    return null;
  }
  const body = {
    profile_id: pid,
    ...readBenchOptions(),
  };
  setBenchRunningChrome(true, "starting…");
  try {
    const rec = await API.post("/api/benchmarks/run", body);
    state.activeBenchmarkId = rec.id;
    state.benchSelectedId = rec.id;
    startBenchPolling();
    toast("Benchmark started…", "ok");
    // Refresh the table NOW, not on the next poll: a load that was already
    // in flight when the POST went out (e.g. from showView above) resolves
    // WITHOUT this record, and polling only updates rows it already knows.
    await loadBenchmarks();
    // Refresh badges across the app (the profile now shows "running").
    loadProfiles();
    return rec.id;
  } catch (e) {
    setBenchRunningChrome(false);
    toast(e.message, "error");
    return null;
  }
}

async function reRunBenchmark(rec) {
  if (state.activeBenchmarkId) {
    toast("A benchmark is already running - wait for it to finish or cancel it.", "error");
    return;
  }
  setBenchRunningChrome(true, "starting…");
  try {
    const newRec = await API.post("/api/benchmarks/run", { benchmark_id: rec.id, ...readBenchOptions() });
    state.activeBenchmarkId = newRec.id;
    state.benchSelectedId = newRec.id;
    startBenchPolling();
    toast("Re-running with the saved model params and the form's token settings…", "ok");
    await loadBenchmarks();  // same in-flight-GET race as a fresh start
  } catch (e) {
    setBenchRunningChrome(false);
    toast(e.message, "error");
  }
}

async function cancelBenchmark(id) {
  try {
    await API.post(`/api/benchmarks/${id}/cancel`, {});
    toast("Cancelling benchmark…", "ok");
  } catch (e) { toast(e.message, "error"); }
}

// ---- live progress of the in-flight run ----

function startBenchPolling() {
  if (state.benchPollTimer) return;
  state.benchPollTimer = setInterval(pollActiveBenchmark, 2000);
  pollActiveBenchmark();
}

function stopBenchPolling() {
  if (state.benchPollTimer) { clearInterval(state.benchPollTimer); state.benchPollTimer = null; }
}

// Pick up a run that was started before this page/window saw it (badge
// showing "running" after a reload, or kicked off from the Profiles page).
function resumeBenchmarkPollingIfNeeded() {
  if (state.activeBenchmarkId) { startBenchPolling(); return; }
  const running = (state.profiles || []).find(
    p => p.benchmark_badge && p.benchmark_badge.state === "running" && p.benchmark_badge.benchmark_id);
  if (running) {
    state.activeBenchmarkId = running.benchmark_badge.benchmark_id;
    startBenchPolling();
  }
}

async function pollActiveBenchmark() {
  const id = state.activeBenchmarkId;
  if (!id) { stopBenchPolling(); return; }
  let rec;
  try { rec = await API.get(`/api/benchmarks/${id}`); } catch (e) { return; }

  if (state.view === "benchmarks") {
    updateBenchProgressCard(rec);
    // Update the row if present, otherwise INSERT it (a stale list load may
    // still not contain this record - the row must appear regardless).
    const idx = (state.benchmarks || []).findIndex(b => b.id === id);
    if (idx === -1) state.benchmarks = [rec, ...(state.benchmarks || [])];
    else state.benchmarks[idx] = rec;
    renderBenchmarks();
    if (state.benchSelectedId === id) renderBenchDetail();
  }

  if (["completed", "failed", "cancelled"].includes(rec.status)) {
    state.activeBenchmarkId = null;
    stopBenchPolling();
    setBenchRunningChrome(false);
    if (rec.status === "completed") toast(`Benchmark finished - ${rec.generation_tps} tok/s generation`, "ok");
    else if (rec.status === "cancelled") toast("Benchmark cancelled.", "ok");
    else toast(rec.error || "Benchmark failed.", "error");
    if (state.view === "benchmarks") {
      state.benchSelectedId = id;
      const idx = (state.benchmarks || []).findIndex(b => b.id === id);
      if (idx === -1) state.benchmarks = [rec, ...(state.benchmarks || [])];
      else state.benchmarks[idx] = rec;
      renderBenchmarks();
      renderBenchDetail();
    }
    loadProfiles();  // badges (and the list) need the finished badge state
  }
}

function setBenchRunningChrome(running, progressText) {
  const runBtn = document.getElementById("btn-bench-run");
  const cancelBtn = document.getElementById("btn-bench-cancel");
  const profileSel = document.getElementById("bench-profile");
  if (runBtn) {
    runBtn.disabled = running || !profileSel || profileSel.disabled;
    runBtn.textContent = running ? "⏳ Running…" : "▶ Run benchmark";
  }
  if (cancelBtn) cancelBtn.hidden = !running;
  if (!running) {
    const pc = document.getElementById("bench-progress");
    if (pc) pc.hidden = true;
  } else if (progressText) {
    updateBenchProgressCard({ progress: progressText });
  }
}

function updateBenchProgressCard(rec) {
  const el = document.getElementById("bench-progress");
  if (!el) return;
  el.hidden = false;
  el.textContent = `${rec.progress || "running…"} - ${rec.profile_name || ""}`;
}

// ---- history table ----

function renderBenchmarks() {
  const tbody = document.getElementById("bench-tbody");
  if (!tbody) return;
  const filter = (state.benchFilterText || "").trim().toLowerCase();
  let rows = (state.benchmarks || []).filter(b =>
    !filter || `${b.profile_name} ${b.model_name} ${b.model_path}`.toLowerCase().includes(filter));
  rows = rows.slice().sort(compareBenchmarks);

  // Pagination: clamp the current page into range, then render only this page.
  const perPage = state.benchPerPage;
  const totalPages = Math.max(1, Math.ceil(rows.length / perPage));
  state.benchPage = Math.min(Math.max(1, state.benchPage), totalPages);
  const startIdx = (state.benchPage - 1) * perPage;
  const pageRows = rows.slice(startIdx, startIdx + perPage);

  const countEl = document.getElementById("bench-count");
  if (countEl) {
    const total = (state.benchmarks || []).length;
    countEl.textContent = filter
      ? `${rows.length} of ${total} record${total === 1 ? "" : "s"} match`
      : `${total} record${total === 1 ? "" : "s"}`;
  }
  renderBenchPager(rows.length, totalPages);

  tbody.innerHTML = "";
  if (pageRows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9"><div class="empty-state" style="padding:18px;">
      <p>${(state.benchmarks || []).length === 0 ? "No benchmarks yet - run one above." : "No records match the filter."}</p>
      </div></td></tr>`;
    return;
  }
  pageRows.forEach(rec => tbody.appendChild(buildBenchRow(rec)));
}

function renderBenchPager(totalRows, totalPages) {
  const info = document.getElementById("bench-page-info");
  const pageNum = document.getElementById("bench-page-num");
  const prev = document.getElementById("bench-prev-page");
  const next = document.getElementById("bench-next-page");
  if (!info || !pageNum || !prev || !next) return;
  const start = totalRows === 0 ? 0 : (state.benchPage - 1) * state.benchPerPage + 1;
  const end = Math.min(totalRows, state.benchPage * state.benchPerPage);
  info.textContent = totalRows === 0 ? "No records" : `Showing ${start}–${end} of ${totalRows}`;
  pageNum.textContent = `Page ${state.benchPage} of ${totalPages}`;
  prev.disabled = state.benchPage <= 1;
  next.disabled = state.benchPage >= totalPages;
}

function compareBenchmarks(a, b) {
  const { key, dir } = state.benchSort;
  const m = dir === "asc" ? 1 : -1;
  const av = a[key], bv = b[key];
  if (av == null && bv == null) return 0;
  if (av == null) return 1;   // nulls last, regardless of direction
  if (bv == null) return -1;
  if (typeof av === "number" && typeof bv === "number") return (av - bv) * m;
  return String(av).localeCompare(String(bv)) * m;
}

function buildBenchRow(rec) {
  const tr = document.createElement("tr");
  tr.dataset.id = rec.id;
  if (state.benchSelectedId === rec.id) tr.classList.add("is-selected");
  const when = rec.timestamp || rec.started_at;
  const profileGone = rec.profile_id && !(state.profiles || []).some(p => p.id === rec.profile_id);
  tr.innerHTML = `
    <td title="${when ? new Date(when * 1000).toLocaleString() : ""}">${when ? new Date(when * 1000).toLocaleDateString() : "-"}</td>
    <td>${escapeHtml(rec.profile_name || "")}${profileGone ? ` <span class="muted small">(profile deleted)</span>` : ""}</td>
    <td title="${escapeHtml(rec.model_path || "")}">${escapeHtml(rec.model_name || "-")}</td>
    <td class="num">${rec.prefill_tps != null ? Number(rec.prefill_tps).toFixed(1) : "-"}</td>
    <td class="num">${rec.generation_tps != null ? Number(rec.generation_tps).toFixed(1) : "-"}</td>
    <td class="num">${(rec.prompt_tokens != null || rec.gen_tokens != null) ? `${rec.prompt_tokens ?? "?"} / ${rec.gen_tokens ?? "?"}` : "-"}</td>
    <td>${benchStatusChip(rec)}</td>
    <td class="row-actions-cell">
      <button class="btn btn-tiny" data-act="rerun" title="Re-run with this record's saved parameters">↻ Re-run</button>
      <button class="btn btn-tiny" data-act="delete" title="Delete record">✕</button>
    </td>
    <td class="bench-compare-cell" title="Select for compare (pick exactly two)">
      <input type="checkbox" class="bench-compare-cb" ${state.benchCompareIds.includes(rec.id) ? "checked" : ""}>
    </td>
  `;
  tr.addEventListener("click", () => { state.benchSelectedId = rec.id; renderBenchmarks(); renderBenchDetail(); });
  tr.querySelector('[data-act="rerun"]').addEventListener("click", e => { e.stopPropagation(); reRunBenchmark(rec); });
  tr.querySelector('[data-act="delete"]').addEventListener("click", e => { e.stopPropagation(); deleteBenchmarkRecord(rec); });
  const cb = tr.querySelector(".bench-compare-cb");
  cb.addEventListener("click", e => {
    e.stopPropagation();   // don't also select the row
    if (cb.checked) {
      if (state.benchCompareIds.length >= 2) {
        cb.checked = false;
        toast("Uncheck one of the selected records first", "error");
        return;
      }
      state.benchCompareIds.push(rec.id);
    } else {
      state.benchCompareIds = state.benchCompareIds.filter(x => x !== rec.id);
    }
    updateBenchCompareButton();
    renderBenchmarks();
  });
  return tr;
}

function benchStatusChip(rec) {
  if (rec.status === "completed") return `<span class="bench-chip bench-chip-ok">completed</span>`;
  if (rec.status === "failed") return `<span class="bench-chip bench-chip-err" title="${escapeHtml(rec.error || "")}">failed</span>`;
  if (rec.status === "cancelled") return `<span class="bench-chip bench-chip-mut">cancelled</span>`;
  return `<span class="bench-chip bench-chip-run">${escapeHtml(rec.progress || "running")}…</span>`;
}

// ---- compare selection (⚖ button + modal) ----

// The ⚖ Compare button mirrors the checkbox selection: enabled only when
// exactly two records are checked. A/B order = check order
// (state.benchCompareIds[0] is A, [1] is B).
function updateBenchCompareButton() {
  const btn = document.getElementById("btn-bench-compare");
  if (!btn) return;
  const n = state.benchCompareIds.length;
  btn.disabled = n !== 2;
  btn.textContent = `⚖ Compare (${n}/2)`;
}

// Deep equality for JSON values - the normalized benchmarkable objects from
// the API are JSON scalars / nested containers.
function benchDiffEq(x, y) {
  if (x === y) return true;
  if (typeof x !== typeof y) return false;
  if (x && y && typeof x === "object") {
    const kx = Object.keys(x), ky = Object.keys(y);
    if (kx.length !== ky.length) return false;
    return kx.every(k => benchDiffEq(x[k], y[k]));
  }
  return false;
}

// Side-by-side compare of two records' BENCHMARKABLE params (the normalized
// model_path / params / custom_flags subset that lands on the llama-server
// command line). The endpoint returns already-normalized objects plus the
// server-computed `same` verdict - the same source of truth as the staleness
// badge - so the structural diff here is plain per-key equality.
async function openBenchCompareModal() {
  const [idA, idB] = state.benchCompareIds;
  if (!idA || !idB) { toast("Select two records to compare first.", "error"); return; }

  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  backdrop.innerHTML = `
    <div class="modal-card">
      <h3>Compare parameter snapshots</h3>
      <p class="muted small">Benchmarkable parameters only - the model path, params, and custom flags
        that reach the llama-server command line. Same definition as the staleness badge.</p>
      <div id="bench-compare-body"><p class="muted small">Loading…</p></div>
      <div class="modal-actions">
        <button class="btn btn-ghost" id="bench-compare-close">Close</button>
      </div>
    </div>
  `;
  document.body.appendChild(backdrop);

  const close = () => backdrop.remove();
  backdrop.addEventListener("click", e => { if (e.target === backdrop) close(); });
  backdrop.querySelector("#bench-compare-close").addEventListener("click", close);

  let data;
  try {
    data = await API.post("/api/benchmarks/compare", { a: idA, b: idB });
  } catch (e) {
    toast(e.message, "error");
    close();
    return;
  }
  const bodyEl = backdrop.querySelector("#bench-compare-body");
  if (!bodyEl) return;   // closed while the request was in flight

  const { a, b, same } = data;
  const ba = a.benchmarkable, bb = b.benchmarkable;
  const fmtTps = v => v != null ? Number(v).toFixed(1) : "-";
  const fmtWhen = t => t ? new Date(t * 1000).toLocaleString() : "-";
  const sideContext = (label, s) => `
    <div class="bench-diff-side">
      <div class="bench-diff-side-title">${escapeHtml(label)} - ${escapeHtml(s.profile_name || "unknown profile")}</div>
      <div class="file-row"><span>When</span><span>${escapeHtml(fmtWhen(s.timestamp))}</span></div>
      <div class="file-row"><span>Server</span><span>${escapeHtml(s.server_version || "-")}</span></div>
      <div class="file-row"><span>Prefill</span><span>${fmtTps(s.prefill_tps)} t/s</span></div>
      <div class="file-row"><span>Gen</span><span>${fmtTps(s.generation_tps)} t/s</span></div>
    </div>`;

  // One row per key: model_path, then every key in the union of both params
  // objects in schema order (orderedParamKeys), then custom_flags. A/B values
  // are already server-normalized; a row is flagged when they differ.
  const rows = [];
  let diffCount = 0;
  const pushRow = (label, va, vb) => {
    const changed = !benchDiffEq(va, vb);
    if (changed) diffCount += 1;
    const cell = v => `<td class="bench-diff-val" title="${escapeHtml(String(v === undefined ? "" : v))}">${escapeHtml(v === undefined ? "-" : shortParamValue(v))}</td>`;
    rows.push(`<tr class="${changed ? "bench-diff-changed" : ""}">
      <td class="bench-diff-key">${escapeHtml(label)}</td>${cell(va)}${cell(vb)}</tr>`);
  };
  pushRow("model_path", ba.model_path, bb.model_path);
  for (const k of orderedParamKeys({ ...ba.params, ...bb.params })) {
    const sp = state.schemaByKey[k];
    const flag = (sp && sp.flag) || ("--" + k);
    pushRow(flag, ba.params[k], bb.params[k]);
  }
  pushRow("custom_flags", ba.custom_flags, bb.custom_flags);

  const headline = same
    ? `<div class="bench-diff-headline bench-diff-same">✓ Parameters identical</div>`
    : `<div class="bench-diff-headline bench-diff-diff">⚠ ${diffCount} parameter difference${diffCount === 1 ? "" : "s"}</div>`;

  bodyEl.innerHTML = `
    <div class="bench-diff-context">
      ${sideContext("A", a)}
      ${sideContext("B", b)}
    </div>
    ${headline}
    <div class="bench-diff-scroll">
      <table class="bench-diff-table">
        <thead><tr><th>Parameter</th><th>A</th><th>B</th></tr></thead>
        <tbody>${rows.join("")}</tbody>
      </table>
    </div>
  `;
}

// ---- detail pane ----

function renderBenchDetail() {
  const el = document.getElementById("bench-detail");
  if (!el) return;
  const rec = (state.benchmarks || []).find(b => b.id === state.benchSelectedId);
  if (!rec) {
    el.innerHTML = `<div class="bench-detail-card"><div class="empty-state">
      <p>Select a benchmark row to see its full parameter snapshot, raw output, and actions.</p>
      </div></div>`;
    return;
  }
  const snap = rec.profile_params_snapshot || {};
  const profileGone = rec.profile_id && !(state.profiles || []).some(p => p.id === rec.profile_id);
  // The visible param rows as data (flag + FULL stored value - the display
  // shortens paths, but a copy should carry what the server actually gets).
  const paramLines = orderedParamKeys(snap.params)
    .map(k => [k, (snap.params || {})[k]])
    .filter(([, v]) => v !== undefined && v !== null && v !== "" && v !== false)
    .map(([k, v]) => {
      const sp = state.schemaByKey[k];
      return { flag: (sp && sp.flag) || ("--" + k), value: v };
    });
  const paramRows = paramLines
    .map(({ flag, value }) => `<div class="file-row"><span>${escapeHtml(flag)}</span><span>${escapeHtml(shortParamValue(value))}</span></div>`)
    .join("");
  let raw = rec.raw_output || "";
  try { raw = JSON.stringify(JSON.parse(raw), null, 2); } catch (e) { /* keep as-is */ }
  const when = rec.timestamp || rec.started_at;

  el.innerHTML = `
    <div class="bench-detail-card">
      <div class="detail-header">
        <div style="width:100%">
          <h2 class="detail-title">Benchmark - ${escapeHtml(rec.profile_name || "unknown profile")}</h2>
          <div class="detail-path">${escapeHtml(rec.model_path || "(no model path)")}</div>
        </div>
        <div class="detail-header-actions">${benchStatusChip(rec)}</div>
      </div>
      ${profileGone ? `<div class="preview-warn">⚠ The profile this was benchmarked from no longer exists. The record keeps a full snapshot, so re-run and import still work.</div>` : ""}
      ${rec.error ? `<div class="preview-warn">⚠ ${escapeHtml(rec.error)}</div>` : ""}
      <div class="detail-stats">
        <div class="stat-block"><span class="stat-value">${rec.prefill_tps != null ? Number(rec.prefill_tps).toFixed(1) : "-"}</span><span class="stat-label">Prefill t/s</span></div>
        <div class="stat-block"><span class="stat-value">${rec.generation_tps != null ? Number(rec.generation_tps).toFixed(1) : "-"}</span><span class="stat-label">Generation t/s</span></div>
        <div class="stat-block"><span class="stat-value">${rec.prompt_tokens ?? "-"}</span><span class="stat-label">Prompt tokens</span></div>
        <div class="stat-block"><span class="stat-value">${rec.gen_tokens ?? "-"}</span><span class="stat-label">Generated tokens</span></div>
        <div class="stat-block"><span class="stat-value">${rec.duration_s != null ? Number(rec.duration_s).toFixed(1) + "s" : "-"}</span><span class="stat-label">Duration</span></div>
        <div class="stat-block"><span class="stat-value bench-version-val">${escapeHtml(rec.server_version || "-")}</span><span class="stat-label">Server version</span></div>
      </div>
      <div class="profile-row-actions" style="flex-wrap:wrap; margin-bottom:8px;">
        <button class="btn btn-primary btn-tiny" id="bd-rerun">↻ Re-run</button>
        <button class="btn btn-tiny" id="bd-import">Import as profile…</button>
        <button class="btn btn-tiny" id="bd-export">Export snapshot JSON</button>
        <button class="btn btn-tiny" id="bd-delete">Delete record</button>
      </div>
      <div class="profile-list-head"><strong>Parameters snapshot</strong>
        <button class="btn btn-tiny" id="bd-copy-params">Copy</button>
      </div>
      <div class="file-list">
        ${paramRows || `<p class="muted small">Default parameters (none overridden).</p>`}
      </div>
      ${snap.custom_flags && snap.custom_flags.trim() ? `
      <div class="profile-list-head"><strong>Custom flags</strong></div>
      <pre class="preview-code">${escapeHtml(snap.custom_flags.trim())}</pre>` : ""}
      <div class="profile-list-head"><strong>Run info</strong></div>
      <p class="muted small">Started ${when ? new Date(when * 1000).toLocaleString() : "-"}${rec.re_ran_from ? " · re-run of another record" : ""}</p>
      ${raw ? `<div class="profile-list-head"><strong>Raw output</strong></div><pre class="preview-code" style="max-height:300px;">${escapeHtml(raw)}</pre>` : ""}
    </div>
  `;
  el.querySelector("#bd-rerun").addEventListener("click", () => reRunBenchmark(rec));
  el.querySelector("#bd-import").addEventListener("click", () => openImportBenchmarkModal(rec));
  el.querySelector("#bd-export").addEventListener("click", () => exportBenchmarkSnapshot(rec));
  el.querySelector("#bd-delete").addEventListener("click", () => deleteBenchmarkRecord(rec));
  el.querySelector("#bd-copy-params").addEventListener("click", () => {
    if (!paramLines.length) { toast("No overridden parameters to copy.", "error"); return; }
    const text = paramLines.map(({ flag, value }) => value === true ? flag : `${flag} ${value}`).join("\n");
    navigator.clipboard.writeText(text).then(() => toast("Parameters copied", "ok")).catch(() => {});
  });
}

// ---- row actions ----

async function deleteBenchmarkRecord(rec) {
  const tps = rec.generation_tps != null ? rec.generation_tps.toFixed(1) + " t/s gen" : "n/a";
  if (!await confirmModal({
    title: "Delete this benchmark record?",
    message: `Result: ${tps}\n\nAny profile badges pointing at it will be cleared.`,
  })) return;
  try {
    await API.del(`/api/benchmarks/${rec.id}`);
    toast("Benchmark deleted", "ok");
    if (state.benchSelectedId === rec.id) state.benchSelectedId = null;
    state.benchCompareIds = state.benchCompareIds.filter(x => x !== rec.id);
    updateBenchCompareButton();
    await loadBenchmarks();
    await loadProfiles();
  } catch (e) { toast(e.message, "error"); }
}

async function exportBenchmarkSnapshot(rec) {
  try {
    const data = await API.get(`/api/benchmarks/${rec.id}/snapshot`);
    const when = (rec.timestamp || rec.started_at)
      ? new Date((rec.timestamp || rec.started_at) * 1000).toISOString().slice(0, 10)
      : "unknown-date";
    const filename = `benchmark_${(rec.profile_name || "profile").replace(/\s+/g, "_")}_${when}.json`;
    const content = JSON.stringify(data, null, 2);
    if (hasNativeDialogs()) {
      const savedPath = await window.pywebview.api.save_text_file(filename, content);
      if (savedPath) toast(`Saved to ${savedPath}`, "ok");
      return;
    }
    downloadJson(filename, data);
  } catch (e) { toast(e.message, "error"); }
}

function openImportBenchmarkModal(rec) {
  const profiles = state.profiles || [];
  const when = (rec.timestamp || rec.started_at)
    ? new Date((rec.timestamp || rec.started_at) * 1000).toISOString().slice(0, 10)
    : "";
  const defaultName = `${rec.profile_name || "Imported"} (from benchmark ${when})`;

  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  backdrop.innerHTML = `
    <div class="modal-card">
      <h3>Import benchmark as profile</h3>
      <p class="muted small">Builds a profile from this record's saved parameter snapshot. Its benchmark
        badge starts <b>fresh</b>, since the imported parameters match the record exactly.</p>
      <div class="preset-opt-row" style="flex-direction:column; align-items:flex-start; gap:8px;">
        <label><input type="radio" name="bench-import-mode" value="new" checked> Create a new profile</label>
        <label><input type="radio" name="bench-import-mode" value="overwrite" ${profiles.length ? "" : "disabled"}> Overwrite an existing profile</label>
      </div>
      <div id="bench-import-new" style="margin-bottom:10px;">
        <label class="muted small">New profile name</label>
        <input type="text" id="bench-import-name" class="input" value="${escapeHtml(defaultName)}">
      </div>
      <div id="bench-import-overwrite" hidden style="margin-bottom:10px;">
        <label class="muted small">Profile to overwrite (its model + parameters are replaced)</label>
        <select id="bench-import-target" class="input" style="width:100%;">${profiles.map(p =>
          `<option value="${p.id}">${escapeHtml(p.name)} - ${escapeHtml(modelNameFromId(p.model_id))}</option>`).join("")}</select>
      </div>
      <div class="modal-actions">
        <button class="btn btn-ghost" id="bench-import-cancel">Cancel</button>
        <button class="btn btn-primary" id="bench-import-confirm" ${profiles.length ? "" : "disabled"}>Import</button>
      </div>
    </div>
  `;
  document.body.appendChild(backdrop);

  const modeRadios = backdrop.querySelectorAll("input[name=bench-import-mode]");
  const sync = () => {
    const mode = [...modeRadios].find(r => r.checked).value;
    backdrop.querySelector("#bench-import-new").hidden = mode !== "new";
    backdrop.querySelector("#bench-import-overwrite").hidden = mode !== "overwrite";
  };
  modeRadios.forEach(r => r.addEventListener("change", sync));
  sync();

  const close = () => backdrop.remove();
  backdrop.addEventListener("click", e => { if (e.target === backdrop) close(); });
  backdrop.querySelector("#bench-import-cancel").addEventListener("click", close);
  backdrop.querySelector("#bench-import-confirm").addEventListener("click", async () => {
    const mode = [...modeRadios].find(r => r.checked).value;
    const body = { mode };
    if (mode === "new") body.name = backdrop.querySelector("#bench-import-name").value.trim();
    else body.profile_id = backdrop.querySelector("#bench-import-target").value;
    try {
      const p = await API.post(`/api/benchmarks/${rec.id}/import-as-profile`, body);
      toast(`Profile "${p.name}" created from benchmark`, "ok");
      close();
      showView("library");
      await loadProfiles();
      selectProfile(p.id);
    } catch (e2) { toast(e2.message, "error"); }
  });
}

// ---- profile badges (rendered everywhere profiles appear) ----

// The badge chip next to a profile's name. States & colors per BenchPlan §2.
function benchBadgeHtml(p) {
  const b = p.benchmark_badge;
  if (!b || !b.state || b.state === "none") return "";
  const id = b.benchmark_id || "";
  if (b.state === "running") {
    return `<span class="bench-badge bench-badge-running" data-bench-id="${id}" title="Benchmark in progress">⚡ benchmarking…</span>`;
  }
  const gen = b.generation_tps != null ? `${Number(b.generation_tps).toFixed(1)} t/s` : "";
  if (b.state === "fresh") {
    const pre = b.prefill_tps != null ? Number(b.prefill_tps).toFixed(1) + " t/s" : "-";
    return `<span class="bench-badge bench-badge-fresh" data-bench-id="${id}" title="Benchmarked with the current parameters · prefill ${pre} · click to open the record">⚡ ${gen}</span>`;
  }
  if (b.state === "stale") {
    return `<span class="bench-badge bench-badge-stale" data-bench-id="${id}" title="Benchmarked at a different configuration - parameters have changed since. Click to open the record.">⚡ ${gen} · stale</span>`;
  }
  if (b.state === "failed") {
    return `<span class="bench-badge bench-badge-failed" data-bench-id="${id}" title="The last benchmark failed - click to open the record">⚡ failed</span>`;
  }
  return "";
}

// The larger badge block in the profile detail pane.
function profileBadgeBlock(p) {
  const b = p.benchmark_badge;
  if (!b || !b.state || b.state === "none") {
    return `<div class="bench-inline">
      <span class="bench-badge bench-badge-none">⚡ not benchmarked</span>
      <span class="muted small">click “⚡ Benchmark” above to measure this configuration</span>
    </div>`;
  }
  const gen = b.generation_tps != null ? Number(b.generation_tps).toFixed(1) : "-";
  const pre = b.prefill_tps != null ? Number(b.prefill_tps).toFixed(1) : "-";
  const stateLabel = {
    fresh: "matches current parameters",
    stale: "parameters changed since this benchmark",
    running: "in progress…",
    failed: "last run failed",
  }[b.state] || b.state;
  return `<div class="bench-inline">
    <span class="bench-badge bench-badge-${b.state}">${b.state === "running" ? "⚡ benchmarking…" : `⚡ ${gen} t/s${b.state === "stale" ? " · stale" : ""}`}</span>
    <span class="muted small">prefill ${pre} t/s · ${escapeHtml(stateLabel)}</span>
    ${b.benchmark_id ? `<button class="btn btn-ghost btn-tiny" id="pd-bench-open" data-bench-id="${b.benchmark_id}">Open record</button>` : ""}
  </div>`;
}

// Jump to a benchmark record on the Benchmarks page (badge clicks land here).
function openBenchmarkRecord(id) {
  if (!id) return;
  state.benchSelectedId = id;
  showView("benchmarks");
  loadBenchmarks();
}

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0, n = bytes;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}

function formatUptime(seconds) {
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function debounce(fn, wait) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), wait); };
}

function downloadJson(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function toast(message, kind) {
  const stack = document.getElementById("toast-stack");
  const el = document.createElement("div");
  el.className = "toast" + (kind ? ` ${kind}` : "");
  el.textContent = message;
  stack.appendChild(el);
  setTimeout(() => el.remove(), 4500);
}
