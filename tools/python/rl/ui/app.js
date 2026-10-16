const state = {
	meta: null,
	activeBattleId: null,
	activeCursor: null,
	inspectPlayer: "p1",
	choiceRequestInFlight: false,
	replayFrame: null,
	replayBattleId: null,
	jobsTimer: null,
	battleTimer: null,
	completedJobIds: new Set(),
	hoverRegistry: new Map(),
	hoverCounter: 0,
};

const FORM_IDS = ["train-form", "eval-form", "battle-form"];

const $ = selector => document.querySelector(selector);

async function fetchJSON(url, options = {}) {
	const response = await fetch(url, {
		headers: {"Content-Type": "application/json"},
		...options,
	});
	const payload = await response.json().catch(() => ({}));
	if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
	return payload;
}

function setMetaSummary(meta) {
	$("#meta-summary").innerHTML = `
		<div><strong>${meta.teams.length}</strong> team files available</div>
		<div><strong>${meta.checkpoints.length}</strong> checkpoints discovered</div>
		<div class="muted">repo: ${meta.repo_root}</div>
	`;
}

function buildOptions(items, includeBlank = true) {
	const options = [];
	if (includeBlank) options.push(`<option value="">Use bridge default team</option>`);
	for (const item of items) {
		options.push(`<option value="${item.id}">${item.label} - ${item.headline || item.path}</option>`);
	}
	return options.join("");
}

function populateSelectors(meta) {
	const teamOptions = buildOptions(meta.teams, true);
	const checkpointOptions = [
		`<option value="">No checkpoint / first legal action fallback</option>`,
		...meta.checkpoints.map(item => `<option value="${item.id}">${item.label}</option>`),
	].join("");
	document.querySelectorAll(".team-select").forEach(select => {
		select.innerHTML = teamOptions;
	});
	document.querySelectorAll(".checkpoint-select").forEach(select => {
		select.innerHTML = checkpointOptions;
	});
}

function refreshCheckpointSelectors(meta) {
	const checkpointOptions = [
		`<option value="">No checkpoint / first legal action fallback</option>`,
		...meta.checkpoints.map(item => `<option value="${item.id}">${item.label}</option>`),
	].join("");
	document.querySelectorAll(".checkpoint-select").forEach(select => {
		const selectedValue = select.value;
		select.innerHTML = checkpointOptions;
		if (meta.checkpoints.some(item => item.id === selectedValue) || selectedValue === "") {
			select.value = selectedValue;
		}
	});
}

function formToObject(form) {
	const data = new FormData(form);
	const payload = {};
	for (const [key, value] of data.entries()) {
		payload[key] = typeof value === "string" ? value : String(value);
	}
	for (const key of [
		"iterations", "rollout_steps", "ppo_epochs", "minibatch_size", "hidden_dim", "max_actions",
		"seed", "episodes", "count", "opponent_refresh_interval", "planning_top_k", "planning_depth",
	]) {
		if (payload[key] !== undefined && payload[key] !== "") payload[key] = Number(payload[key]);
	}
	for (const key of ["gamma", "gae_lambda", "clip_ratio", "lr", "value_coef", "entropy_coef", "max_grad_norm", "shaping_scale", "planning_gamma"]) {
		if (payload[key] !== undefined && payload[key] !== "") payload[key] = Number(payload[key]);
	}
	for (const key of ["simple_state", "include_log"]) {
		if (payload[key] !== undefined) payload[key] = payload[key] === "true";
	}
	if (payload.human_player === "") payload.human_player = null;
	return payload;
}

function statusLabel(status) {
	if (status === "running") return `<span class="status-pill status-running">running</span>`;
	if (status === "queued") return `<span class="status-pill status-queued">queued</span>`;
	if (status === "completed") return `<span class="status-pill status-completed">completed</span>`;
	if (status === "failed") return `<span class="status-pill status-failed">failed</span>`;
	return `<span class="status-pill">${escapeHTML(status)}</span>`;
}

function formatNumber(value, digits = 4) {
	if (value === null || value === undefined) return "-";
	const num = Number(value);
	if (!Number.isFinite(num)) return "-";
	return num.toFixed(digits);
}

function compactDuration(seconds) {
	const value = Number(seconds);
	if (!Number.isFinite(value) || value < 0) return "-";
	if (value < 60) return `${value.toFixed(1)}s`;
	const mins = Math.floor(value / 60);
	const secs = Math.round(value % 60);
	return `${mins}m ${secs}s`;
}

function buildCurvePath(points, width = 320, height = 120) {
	if (!points.length) return "";
	const xs = points.map(point => Number(point.x));
	const ys = points.map(point => Number(point.y));
	const xMin = Math.min(...xs);
	const xMax = Math.max(...xs);
	const yMin = Math.min(...ys);
	const yMax = Math.max(...ys);
	const xSpan = Math.max(1e-9, xMax - xMin);
	const ySpan = Math.max(1e-9, yMax - yMin);
	return points.map((point, index) => {
		const px = ((point.x - xMin) / xSpan) * width;
		const py = height - ((point.y - yMin) / ySpan) * height;
		return `${index === 0 ? "M" : "L"}${px.toFixed(2)} ${py.toFixed(2)}`;
	}).join(" ");
}

function renderCurveCard({title, key, color}, curve) {
	const samples = curve
		.filter(point => Number.isFinite(Number(point.elapsed_seconds)) && Number.isFinite(Number(point[key])))
		.map(point => ({x: Number(point.elapsed_seconds), y: Number(point[key])}));
	if (samples.length < 2) return "";
	const width = 320;
	const height = 120;
	const path = buildCurvePath(samples, width, height);
	const first = samples[0].y;
	const last = samples[samples.length - 1].y;
	const delta = last - first;
	return `
		<div class="learning-card">
			<div class="learning-title">${escapeHTML(title)}</div>
			<svg class="learning-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="${escapeHTML(title)} over time">
				<path class="learning-grid" d="M0 ${height - 1} L${width} ${height - 1}" />
				<path class="learning-line" style="--line-color:${escapeHTML(color)}" d="${path}" />
			</svg>
			<div class="learning-stats muted">start ${formatNumber(first)} · end ${formatNumber(last)} · Δ ${formatNumber(delta)}</div>
		</div>
	`;
}

function renderLearningSection(result) {
	if (!result || !Array.isArray(result.learning_curve) || result.learning_curve.length < 2) return "";
	const curve = result.learning_curve;
	const summary = result.learning_summary || {};
	const charts = [
		{title: "Reward Mean", key: "reward_mean", color: "#4dd0a8"},
		{title: "Loss Mean", key: "loss_mean", color: "#f5b971"},
		{title: "Entropy Mean", key: "entropy_mean", color: "#8dd6ff"},
		{title: "Value Loss Mean", key: "value_loss_mean", color: "#ff9fa3"},
	].map(spec => renderCurveCard(spec, curve)).join("");
	const trendBlock = `
		<div class="learning-summary muted">
			<div>elapsed: <strong>${compactDuration(summary.elapsed_seconds)}</strong></div>
			<div>reward slope/s: <strong>${formatNumber(summary.reward_slope_per_second, 6)}</strong></div>
			<div>loss slope/s: <strong>${formatNumber(summary.loss_slope_per_second, 6)}</strong></div>
			<div>reward improvement: <strong>${formatNumber(summary.reward_improvement)}</strong></div>
		</div>
	`;
	return `
		<details class="learning-panel" open>
			<summary>Learning Curves</summary>
			${trendBlock}
			<div class="learning-grid-layout">${charts}</div>
		</details>
	`;
}

function setCookie(name, value, maxAgeSeconds = 60 * 60 * 24 * 30) {
	document.cookie = `${name}=${encodeURIComponent(value)}; path=/; max-age=${maxAgeSeconds}; SameSite=Lax`;
}

function getCookie(name) {
	const prefix = `${name}=`;
	for (const part of document.cookie.split(";")) {
		const trimmed = part.trim();
		if (trimmed.startsWith(prefix)) {
			return decodeURIComponent(trimmed.slice(prefix.length));
		}
	}
	return null;
}

function formSnapshot(form) {
	const snapshot = {};
	for (const [key, value] of new FormData(form).entries()) {
		snapshot[key] = typeof value === "string" ? value : String(value);
	}
	return snapshot;
}

function restoreFormSnapshot(form, snapshot) {
	for (const [key, value] of Object.entries(snapshot)) {
		const field = form.elements.namedItem(key);
		if (!field) continue;
		if (field instanceof RadioNodeList) {
			for (const item of field) {
				if (item.value === value) item.checked = true;
			}
			continue;
		}
		if ("value" in field) field.value = String(value);
	}
}

function persistForm(formId) {
	const form = document.getElementById(formId);
	if (!form) return;
	const snapshot = JSON.stringify(formSnapshot(form));
	const cookieKey = `rlui_${formId}`;
	const storageKey = `${cookieKey}_ls`;
	if (snapshot.length > 3500) {
		localStorage.setItem(storageKey, snapshot);
		setCookie(cookieKey, "__LOCAL_STORAGE__");
		return;
	}
	setCookie(cookieKey, snapshot);
}

function restoreForm(formId) {
	const form = document.getElementById(formId);
	if (!form) return;
	const cookieKey = `rlui_${formId}`;
	const storageKey = `${cookieKey}_ls`;
	let raw = getCookie(cookieKey);
	if (raw === "__LOCAL_STORAGE__") raw = localStorage.getItem(storageKey);
	if (!raw) return;
	try {
		restoreFormSnapshot(form, JSON.parse(raw));
	} catch {
		// Ignore malformed persisted values.
	}
}

function enableFormPersistence(formId) {
	const form = document.getElementById(formId);
	if (!form) return;
	form.addEventListener("input", () => persistForm(formId));
	form.addEventListener("change", () => persistForm(formId));
}

function renderJobCard(job) {
	const logMeta = job.log_file
		? `<div class="job-meta muted">Log file: ${escapeHTML(job.log_file)}</div>`
		: "";
	const logLinks = job.log_url
		? `<div class="job-meta muted">Logs: <a href="${escapeHTML(job.log_url)}" target="_blank" rel="noreferrer">view</a>${job.vscode_url ? ` · <a href="${escapeHTML(job.vscode_url)}">open in VS Code</a>` : ""}</div>`
		: "";
	return `
		<div class="job" data-job-id="${escapeHTML(job.id)}">
			<div class="job-header">
				<h4>${job.kind} · ${job.id}</h4>
				${statusLabel(job.status)}
			</div>
			<div class="job-meta muted">Started: ${job.started_at ? new Date(job.started_at * 1000).toLocaleString() : "waiting"}</div>
			<div class="job-meta muted">Finished: ${job.finished_at ? new Date(job.finished_at * 1000).toLocaleString() : "pending"}</div>
			${logMeta}
			${logLinks}
			${renderLearningSection(job.result)}
			${job.result ? `<details class="json-tree"><summary>Result</summary>${renderJsonTree(job.result)}</details>` : ""}
			${job.error ? `<div class="danger">${escapeHTML(job.error)}</div>` : ""}
		</div>
	`;
}

function ensureJobsBuckets(root) {
	if (!root.querySelector("[data-jobs-active]")) {
		root.innerHTML = `
			<div class="jobs-section">
				<h4 class="jobs-section-title">Active / Pending</h4>
				<div class="stack small" data-jobs-active></div>
			</div>
			<div class="jobs-section">
				<h4 class="jobs-section-title">Completed</h4>
				<div class="stack small" data-jobs-completed></div>
			</div>
		`;
	}
	return {
		activeRoot: root.querySelector("[data-jobs-active]"),
		completedRoot: root.querySelector("[data-jobs-completed]"),
	};
}

function renderJobs(jobs) {
	const root = $("#jobs-list");
	const summary = $("#jobs-summary");
	const counts = jobs.reduce((acc, job) => {
		acc[job.status] = (acc[job.status] || 0) + 1;
		return acc;
	}, {});
	if (summary) {
		const active = (counts.running || 0) + (counts.queued || 0);
		summary.innerHTML = jobs.length
			? `Active jobs: <strong>${active}</strong> · queued <strong>${counts.queued || 0}</strong> · running <strong>${counts.running || 0}</strong> · completed <strong>${counts.completed || 0}</strong> · failed <strong>${counts.failed || 0}</strong>`
			: `No jobs started yet.`;
	}
	if (!jobs.length) {
		state.completedJobIds.clear();
		root.innerHTML = `<div class="muted">No jobs started yet.</div>`;
		return;
	}

	const {activeRoot, completedRoot} = ensureJobsBuckets(root);
	const activeJobs = jobs.filter(job => job.status !== "completed");
	activeRoot.innerHTML = activeJobs.length
		? activeJobs.map(renderJobCard).join("")
		: `<div class="muted">No active jobs.</div>`;

	const newCompleted = jobs.filter(job => job.status === "completed" && !state.completedJobIds.has(job.id));
	if (newCompleted.length && !state.completedJobIds.size) {
		completedRoot.innerHTML = "";
	}
	for (const job of newCompleted) {
		state.completedJobIds.add(job.id);
		completedRoot.insertAdjacentHTML("beforeend", renderJobCard(job));
	}
	if (!state.completedJobIds.size) {
		completedRoot.innerHTML = `<div class="muted">No completed jobs yet.</div>`;
	}
}

function escapeHTML(text) {
	return String(text)
		.replaceAll("&", "&amp;")
		.replaceAll("<", "&lt;")
		.replaceAll(">", "&gt;")
		.replaceAll('"', "&quot;");
}

const SPARSE_EPSILON = 1e-9;

function formatSparseVector(vector) {
	if (!Array.isArray(vector)) return JSON.stringify(vector, null, 2);
	const entries = [];
	for (let i = 0; i < vector.length; i++) {
		const value = Number(vector[i]);
		if (!Number.isFinite(value) || Math.abs(value) <= SPARSE_EPSILON) continue;
		entries.push(`${i}:${value.toFixed(6)}`);
	}
	return `dim=${vector.length}, nnz=${entries.length} {${entries.join(", ")}}`;
}

function isNumericVector(value) {
	return Array.isArray(value) && value.every(item => Number.isFinite(Number(item)));
}

function formatVectorBlock(value) {
	if (isNumericVector(value)) {
		return formatSparseVector(value);
	}
	if (Array.isArray(value) && value.every(isNumericVector)) {
		return value
			.map((row, index) => `[${index}] ${formatSparseVector(row)}`)
			.join("\n");
	}
	return JSON.stringify(value ?? null, null, 2);
}

function summarizeJsonValue(value) {
	if (value === null) return "null";
	if (Array.isArray(value)) return `Array(${value.length})`;
	const type = typeof value;
	if (type === "string") return `string(${JSON.stringify(value)})`;
	if (type === "number") return Number.isFinite(value) ? String(value) : type;
	if (type === "boolean") return value ? "true" : "false";
	if (type === "undefined") return "undefined";
	if (type === "object") return `Object(${Object.keys(value).length})`;
	return type;
}

function renderJsonNode(label, value) {
	if (value === null || typeof value !== "object") {
		return `<div class="json-leaf"><span class="json-key">${escapeHTML(label)}:</span> <span class="json-value json-${typeof value}">${escapeHTML(summarizeJsonValue(value))}</span></div>`;
	}
	if (Array.isArray(value)) {
		return `
			<details class="json-tree-node">
				<summary><span class="json-key">${escapeHTML(label)}:</span> <span class="json-summary">${escapeHTML(summarizeJsonValue(value))}</span></summary>
				<div class="json-children">
					${value.length ? value.map((item, index) => renderJsonNode(String(index), item)).join("") : `<div class="json-leaf empty">(empty)</div>`}
				</div>
			</details>
		`;
	}
	const entries = Object.entries(value);
	return `
		<details class="json-tree-node">
			<summary><span class="json-key">${escapeHTML(label)}:</span> <span class="json-summary">${escapeHTML(summarizeJsonValue(value))}</span></summary>
			<div class="json-children">
				${entries.length ? entries.map(([key, child]) => renderJsonNode(key, child)).join("") : `<div class="json-leaf empty">(empty)</div>`}
			</div>
		</details>
	`;
}

function renderJsonTree(value) {
	if (value === null || typeof value !== "object") {
		return `<div class="json-leaf"><span class="json-value json-${typeof value}">${escapeHTML(summarizeJsonValue(value))}</span></div>`;
	}
	if (Array.isArray(value)) {
		return `
			<div class="json-children">
				${value.length ? value.map((item, index) => renderJsonNode(String(index), item)).join("") : `<div class="json-leaf empty">(empty)</div>`}
			</div>
		`;
	}
	const entries = Object.entries(value);
	return `
		<div class="json-children">
			${entries.length ? entries.map(([key, child]) => renderJsonNode(key, child)).join("") : `<div class="json-leaf empty">(empty)</div>`}
		</div>
	`;
}

function toID(text) {
	return String(text || "").toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function splitPokemonDetails(details) {
	const raw = String(details || "").split(",")[0].trim();
	return raw || "Unknown";
}

function pokemonDisplayName(entry) {
	const ident = String(entry?.ident || "");
	const nickname = ident.includes(":") ? ident.split(":").slice(1).join(":").trim() : ident.trim();
	const species = splitPokemonDetails(entry?.details);
	if (nickname && nickname !== species) return `${nickname} (${species})`;
	return nickname || species;
}

function pokemonShortName(entry) {
	const ident = String(entry?.ident || "");
	const nickname = ident.includes(":") ? ident.split(":").slice(1).join(":").trim() : ident.trim();
	return nickname || splitPokemonDetails(entry?.details);
}

function parseHPCondition(entry) {
	const condition = String(entry?.condition || "");
	if (!condition) return {percent: null, status: ""};
	const [hpPart = "", status = ""] = condition.split(" ");
	if (hpPart === "0" || status === "fnt" || condition.includes("fnt")) return {percent: 0, status: "fnt"};
	const fracMatch = hpPart.match(/^(\d+(?:\.\d+)?)\/(\d+(?:\.\d+)?)$/);
	if (fracMatch) {
		const current = Number(fracMatch[1]);
		const max = Number(fracMatch[2]);
		return {percent: max ? Math.max(0, Math.min(100, Math.round((current / max) * 100))) : null, status};
	}
	const pctMatch = hpPart.match(/^(\d+)%$/);
	if (pctMatch) return {percent: Number(pctMatch[1]), status};
	return {percent: null, status};
}

function statusLabelForCondition(status) {
	const map = {
		fnt: "fainted",
		slp: "sleep",
		par: "paralyzed",
		brn: "burned",
		psn: "poisoned",
		tox: "toxic",
		frz: "frozen",
	};
	return map[status] || status || "healthy";
}

function spriteUrl(entry, back = false) {
	const species = splitPokemonDetails(entry?.details);
	const spriteId = toID(species);
	const shiny = String(entry?.details || "").includes(", shiny");
	const folder = `${back ? "ani-back" : "ani"}${shiny ? "-shiny" : ""}`;
	return `https://play.pokemonshowdown.com/sprites/${folder}/${encodeURIComponent(spriteId)}.gif`;
}

function statLabel(stat) {
	return {atk: "Atk", def: "Def", spa: "SpA", spd: "SpD", spe: "Spe"}[stat] || stat;
}

function cleanChoiceDescription(description, command = "") {
	const text = String(description || command || "");
	return text
		.replace(/\s*\(auto-order rest as[^)]*\)/ig, "")
		.replace(/\s*auto-order rest as.*$/ig, "")
		.trim();
}

function prettyChoiceLabel(description, command) {
	const text = cleanChoiceDescription(description, command);
	return text
		.replace(/^mega evolve and use /i, "Mega: ")
		.replace(/^ultra burst and use /i, "Ultra Burst: ")
		.replace(/^terastallize and use /i, "Tera: ")
		.replace(/^dynamax and use /i, "Dynamax: ")
		.replace(/^use /i, "")
		.replace(/^switch slot \d+ to /i, "")
		.replace(/^switch to /i, "")
		.trim();
}

function stripChoiceModifier(command) {
	const parts = String(command || "").trim().split(/\s+/);
	if (parts.length > 2 && ["mega", "ultra", "terastallize", "max"].includes(parts[parts.length - 1])) {
		parts.pop();
	}
	return parts.join(" ");
}

function registerHover(info) {
	const id = `hover-${++state.hoverCounter}`;
	state.hoverRegistry.set(id, info);
	return id;
}

function renderHpMeter(percent, status = "") {
	const safePercent = percent === null || percent === undefined ? null : Math.max(0, Math.min(100, Number(percent)));
	const width = safePercent === null ? 0 : safePercent;
	const className = safePercent === null ? "hp-meter unknown" : `hp-meter hp-${width <= 25 ? "low" : width <= 50 ? "mid" : "high"}`;
	return `
		<div class="${className}">
			<div class="hp-meter-fill" style="width:${safePercent === null ? 0 : width}%;"></div>
		</div>
		<div class="hp-text">${safePercent === null ? "?" : `${width}%`}${status ? ` · ${escapeHTML(statusLabelForCondition(status))}` : ""}</div>
	`;
}

function renderPokemonTooltip(info) {
	if (info?.kind === "move") {
		return `
			<div class="tooltip-head">
				<div>
					<div class="tooltip-title">${escapeHTML(info.label || info.description || info.command || "Move")}</div>
					<div class="tooltip-subtitle">${escapeHTML(info.description || info.command || "")}</div>
				</div>
			</div>
			<div class="tooltip-section">
				<div class="tooltip-label">Policy / Value</div>
				<div class="tooltip-qv-list">
					<div class="tooltip-qv-row"><span>Policy</span><span>${info.policy !== null && info.policy !== undefined ? Number(info.policy).toFixed(4) : "-"}</span><span></span></div>
					<div class="tooltip-qv-row"><span>Q</span><span>${info.q !== null && info.q !== undefined ? Number(info.q).toFixed(4) : "-"}</span><span></span></div>
					<div class="tooltip-qv-row"><span>V(next)</span><span>${info.v !== null && info.v !== undefined ? Number(info.v).toFixed(4) : "-"}</span><span></span></div>
				</div>
			</div>
		`;
	}

	const stats = info.stats || {};
	const moves = info.moves || [];
	const qvRows = info.qvRows || [];
	const activeScoreRows = qvRows.length
		? `<div class="tooltip-section"><div class="tooltip-label">Policy / Q / V(next)</div><div class="tooltip-qv-list">${qvRows.map(row => `<div class="tooltip-qv-row"><span>${escapeHTML(row.label)}</span><span>${row.policy !== null && row.policy !== undefined ? Number(row.policy).toFixed(4) : "-"}</span><span>${row.q !== null && row.q !== undefined ? Number(row.q).toFixed(4) : "-"}</span><span>${row.v !== null && row.v !== undefined ? Number(row.v).toFixed(4) : "-"}</span></div>`).join("")}</div></div>`
		: "";
	return `
		<div class="tooltip-head">
			<img class="tooltip-sprite" src="${escapeHTML(info.sprite)}" alt="${escapeHTML(info.species)}" />
			<div>
				<div class="tooltip-title">${escapeHTML(info.name)}</div>
				<div class="tooltip-subtitle">${escapeHTML(info.species)}${info.teraType ? ` · Tera ${escapeHTML(info.teraType)}` : ""}${info.status ? ` · ${escapeHTML(statusLabelForCondition(info.status))}` : ""}</div>
				<div class="tooltip-subtitle">${escapeHTML(info.item || "no item")} · ${escapeHTML(info.ability || info.baseAbility || "unknown ability")}</div>
			</div>
		</div>
		<div class="tooltip-section">
			<div class="tooltip-label">HP</div>
			${renderHpMeter(info.percent, info.status)}
		</div>
		<div class="tooltip-section">
			<div class="tooltip-label">Stats</div>
			<div class="tooltip-stats-grid">${Object.entries(stats).map(([stat, value]) => `<div class="tooltip-stat"><span>${statLabel(stat)}</span><strong>${escapeHTML(String(value))}</strong></div>`).join("")}</div>
		</div>
		<div class="tooltip-section">
			<div class="tooltip-label">Moves</div>
			<div class="tooltip-moves">${moves.length ? moves.map(move => `<span class="tooltip-move">${escapeHTML(move)}</span>`).join("") : `<span class="muted">No moves available</span>`}</div>
		</div>
		${activeScoreRows}
	`;
}

function parseBattleToken(label) {
	const raw = String(label || "");
	return raw.includes(":") ? raw.split(":").slice(1).join(":").trim() : raw.trim();
}

function formatBattleLogLine(line) {
	if (!line) return "";
	if (!line.startsWith("|")) return escapeHTML(line);
	const parts = line.split("|");
	const type = parts[1] || "";
	const source = parseBattleToken(parts[2]);
	const details = parts[3] || "";
	const target = parseBattleToken(parts[4]);
	const hpStatus = parts[3] && /\d/.test(parts[3]) ? parts[3] : parts[4] || "";
	const sourceText = source ? escapeHTML(source) : "Unknown Pokemon";
	const targetText = target ? escapeHTML(target) : "";
	const moveText = details ? escapeHTML(details) : "";
	const renderHpChange = status => escapeHTML(status.replace(/fnt/i, "fainted"));
	switch (type) {
	case "turn": return `Turn ${escapeHTML(details)}`;
	case "switch":
	case "drag": {
		const species = splitPokemonDetails(parts[3]);
		const hpInfo = parts[4] ? ` at ${escapeHTML(parts[4])}` : "";
		return `${sourceText} sent out ${escapeHTML(species)}${hpInfo}.`;
	}
	case "move": {
		const targetSuffix = targetText ? ` on ${targetText}` : "";
		return `${sourceText} used ${moveText}${targetSuffix}.`;
	}
	case "faint": return `${sourceText} fainted!`;
	case "win": return `${sourceText} won the battle!`;
	case "tie": return `The battle ended in a tie.`;
	case "-damage": return `${sourceText} took damage and is now ${escapeHTML(hpStatus || "updated")} HP.`;
	case "-heal": return `${sourceText} healed and is now ${escapeHTML(hpStatus || "updated")} HP.`;
	case "-status": return `${sourceText} was ${renderHpChange(details)}.`;
	case "-curestatus": return `${sourceText} was cured of ${escapeHTML(statusLabelForCondition(details))}.`;
	case "-boost": return `${sourceText}'s ${escapeHTML(statLabel(details))} rose.`;
	case "-unboost": return `${sourceText}'s ${escapeHTML(statLabel(details))} fell.`;
	case "-ability": return `${sourceText}'s ability activated: ${escapeHTML(details)}.`;
	case "-item": return `${sourceText} obtained ${escapeHTML(details)}.`;
	case "-enditem": return `${sourceText} lost ${escapeHTML(details)}.`;
	case "-activate": return `${sourceText} activated ${escapeHTML(details)}.`;
	case "-sidestart": return `${escapeHTML(details)} started on ${sourceText || "a side"}.`;
	case "-sideend": return `${escapeHTML(details)} ended on ${sourceText || "a side"}.`;
	case "-fieldstart": return `${escapeHTML(details)} started.`;
	case "-fieldend": return `${escapeHTML(details)} ended.`;
	case "-weather": return `${escapeHTML(details)} weather began.`;
	case "-weatherend": return `${escapeHTML(details)} weather ended.`;
	case "-terastallize": return `${sourceText} terastallized${details ? ` into ${escapeHTML(details)}` : ""}.`;
	case "-start": return `${sourceText} gained ${escapeHTML(details)}.`;
	case "-end": return `${sourceText} lost ${escapeHTML(details)}.`;
	case "-crit": return `${sourceText} landed a critical hit.`;
	case "-miss": return `${sourceText} missed ${targetText || "the target"}.`;
	case "-fail": return `${sourceText} failed to use ${escapeHTML(details)}.`;
	case "-singleturn": return `${sourceText} is preparing ${escapeHTML(details)}.`;
	case "request": return `Decision requested.`;
	default:
		return escapeHTML(line);
	}
}

function renderBattleLog(logs) {
	const entries = (logs || []).flatMap(line => String(line).split("\n")).filter(Boolean);
	return `
		<div class="battle-log-list">
			${entries.length ? entries.slice(-80).map(line => `<div class="battle-log-entry">${formatBattleLogLine(line)}</div>`).join("") : `<div class="battle-log-entry muted">No battle log messages yet.</div>`}
		</div>
	`;
}

function normalizeReplayLines(lines) {
	if (!Array.isArray(lines)) return [];
	const filtered = [];
	for (const line of lines) {
		const text = String(line || "").trim();
		if (!text || !text.startsWith("|")) continue;
		filtered.push(text);
	}
	if (!filtered.length || !filtered.some(line => line.startsWith("|init|"))) {
		filtered.unshift("|init|battle");
	}
	return filtered;
}

function replayEmbedDocument(payload) {
	const replayLines = normalizeReplayLines(payload?.replay?.lines || payload?.state?.logs || []);
	const replayText = replayLines.join("\n");
	const safeReplayText = replayText.replaceAll("</script", "<\\/script");
	const safeReplayJSON = JSON.stringify(replayText).replaceAll("</script", "<\\/script");
	const initialTurn = Number(payload?.state?.turns || 0);
	return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="stylesheet" href="https://play.pokemonshowdown.com/style/replay.css" />
  <style>
    html, body { margin: 0; padding: 0; background: #f3f4f8; }
    .wrapper { max-width: none; margin: 0; }
    .replay-controls { position: sticky; top: 0; z-index: 3; }
  </style>
  <script src="https://play.pokemonshowdown.com/js/lib/jquery-1.11.0.min.js"></script>
  <script src="https://play.pokemonshowdown.com/js/lib/lodash.compat.js"></script>
  <script src="https://play.pokemonshowdown.com/js/lib/html-sanitizer-minified.js"></script>
  <script src="https://play.pokemonshowdown.com/js/lib/soundmanager2-nodebug-jsmin.js"></script>
  <script src="https://play.pokemonshowdown.com/js/config.js"></script>
  <script src="https://play.pokemonshowdown.com/js/battledata.js"></script>
</head>
<body>
  <div class="wrapper replay-wrapper">
    <div class="battle"></div>
    <div class="battle-log"></div>
		<div class="replay-controls"></div>
		<div class="replay-controls-2"></div>
		<script type="text/plain" class="battle-log-data">${safeReplayText}</script>
    <script>
      window.__RL_INITIAL_TURN__ = ${initialTurn};
      window.__RL_PENDING_UPDATE__ = {
        type: 'rl_replay_update',
				log: ${safeReplayJSON},
        turn: window.__RL_INITIAL_TURN__,
        autoplay: true,
      };

      function applyReplayUpdate(payload) {
        if (!payload || !window.Replays || !window.Replays.battle) return false;
        const rawLog = String(payload.log || '');
        const queue = rawLog ? rawLog.split('\n') : ['|init|battle'];
        window.Replays.battle.setQueue(queue);
        const targetTurn = Number(payload.turn);
				const autoplay = payload.autoplay !== false;
				let desiredTurn = Number.isFinite(targetTurn) ? targetTurn : Infinity;
				if (autoplay && desiredTurn <= 0) desiredTurn = Infinity;
        if (Number.isFinite(targetTurn)) {
					window.Replays.battle.seekTurn(desiredTurn);
        } else {
          window.Replays.battle.seekTurn(Infinity);
        }
				if (autoplay) {
					setTimeout(() => window.Replays.play(), 0);
        } else {
          window.Replays.pause();
        }
        return true;
      }

      window.addEventListener('message', event => {
        const data = event.data;
        if (!data || data.type !== 'rl_replay_update') return;
        if (!applyReplayUpdate(data)) {
          window.__RL_PENDING_UPDATE__ = data;
        }
      });

      window.__RL_UPDATE_READY_TIMER__ = setInterval(() => {
        if (!window.Replays || !window.Replays.battle) return;
        const pending = window.__RL_PENDING_UPDATE__;
        if (pending) applyReplayUpdate(pending);
        window.__RL_PENDING_UPDATE__ = null;
        clearInterval(window.__RL_UPDATE_READY_TIMER__);
      }, 50);
    </script>
  </div>
  <script src="https://play.pokemonshowdown.com/js/replay-embed.js"></script>
</body>
</html>`;
}

function renderStandardBattleUi(payload) {
	void payload;
	return `
		<div class="battle-standard-shell card">
			<h3>Pokemon Showdown Battle UI</h3>
			<div id="battle-standard-host"></div>
		</div>
	`;
}

function resizeReplayFrameToContent(iframe) {
	if (!iframe) return;
	try {
		const doc = iframe.contentDocument;
		if (!doc) return;
		const bodyHeight = doc.body ? doc.body.scrollHeight : 0;
		const docHeight = doc.documentElement ? doc.documentElement.scrollHeight : 0;
		const height = Math.max(bodyHeight, docHeight, 320);
		iframe.style.height = `${height}px`;
	} catch {}
}

function playReplaySoundOff(cw, replays) {
	if (!cw || !replays) return;
	const doc = cw.document;
	const soundOffButton = doc?.querySelector(".startsoundchooser button[value='off'], .startsoundchooser button:last-child");
	if (soundOffButton instanceof HTMLElement) {
		soundOffButton.click();
		return;
	}
	if (typeof replays.changeSetting === "function") {
		replays.changeSetting("sound", "off");
	}
	const playButton = doc?.querySelector(".replay-controls button[data-action='play']");
	if (playButton instanceof HTMLElement) {
		playButton.click();
		return;
	}
	if (typeof replays.play === "function") {
		replays.play();
	}
}

function applyReplayUpdateToFrame(iframe, update, attempts = 0) {
	if (!iframe) return;
	try {
		const cw = iframe.contentWindow;
		const replays = cw?.Replays;
		const battle = replays?.battle;
		if (!battle || typeof battle.setQueue !== "function") {
			if (attempts < 50) {
				setTimeout(() => applyReplayUpdateToFrame(iframe, update, attempts + 1), 40);
			}
			return;
		}

		const rawLog = String(update.log || "");
		const queue = rawLog ? rawLog.split("\n") : ["|init|battle"];
		battle.setQueue(queue);
		const targetTurn = Number(update.turn);
		const autoplay = update.autoplay !== false;
		let desiredTurn = Number.isFinite(targetTurn) ? targetTurn : Infinity;
		if (autoplay && desiredTurn <= 0) desiredTurn = Infinity;
		battle.seekTurn(desiredTurn);
		if (autoplay) {
			setTimeout(() => {
				if (battle.paused) playReplaySoundOff(cw, replays);
				resizeReplayFrameToContent(iframe);
			}, 0);
		} else if (!autoplay && typeof replays.pause === "function") {
			replays.pause();
			resizeReplayFrameToContent(iframe);
		}
		setTimeout(() => resizeReplayFrameToContent(iframe), 80);
	} catch {
		if (attempts < 50) {
			setTimeout(() => applyReplayUpdateToFrame(iframe, update, attempts + 1), 40);
		}
	}
}

function syncReplayFrame(payload) {
	const host = document.getElementById("battle-standard-host");
	if (!host) return;

	const replayLines = normalizeReplayLines(payload?.replay?.lines || payload?.state?.logs || []);
	const replayText = replayLines.join("\n");
	const update = {
		type: "rl_replay_update",
		log: replayText,
		turn: Number(payload?.state?.turns || 0),
		autoplay: true,
	};

	if (state.replayFrame && state.replayBattleId === payload.id && !host.contains(state.replayFrame)) {
		host.innerHTML = "";
		host.appendChild(state.replayFrame);
		applyReplayUpdateToFrame(state.replayFrame, update);
		return;
	}

	const needsNewFrame = !state.replayFrame || state.replayBattleId !== payload.id || !host.contains(state.replayFrame);
	if (needsNewFrame) {
		const iframe = document.createElement("iframe");
		iframe.className = "battle-standard-frame";
		iframe.title = "Pokemon Showdown Battle";
		iframe.setAttribute("sandbox", "allow-scripts allow-same-origin");
		iframe.srcdoc = replayEmbedDocument(payload);
		iframe.addEventListener("load", () => {
			applyReplayUpdateToFrame(iframe, update);
		});
		host.innerHTML = "";
		host.appendChild(iframe);
		state.replayFrame = iframe;
		state.replayBattleId = payload.id;
		return;
	}
	applyReplayUpdateToFrame(state.replayFrame, update);
}

function actionKind(command) {
	if (String(command).startsWith("switch ")) return "switch";
	if (String(command).startsWith("move ")) return "move";
	if (String(command).startsWith("team ")) return "team";
	return "other";
}

function buildChoiceMaps(actionRows) {
	const byCommand = new Map();
	const byBase = new Map();
	for (const row of actionRows) {
		byCommand.set(row.command, row);
		const base = stripChoiceModifier(row.command);
		if (!byBase.has(base)) byBase.set(base, row);
		if (String(row.command).endsWith(" terastallize")) byBase.set(base, row);
	}
	return {byCommand, byBase};
}

function renderBattleFieldPanel(title, playerKey, request, opponent = false) {
	const team = request?.side?.pokemon || [];
	const actives = request?.active || [];
	const activeTeam = team.filter(mon => mon?.active);
	return `
		<div class="battle-field-panel ${opponent ? "opponent" : "player"}">
			<div class="battle-field-title">${escapeHTML(title)}</div>
			<div class="battle-team-row">
				${(actives.length ? actives : activeTeam).map((active, index) => {
					const teamEntry = team[index] || active;
					if (!teamEntry) return "";
					const hp = parseHPCondition(teamEntry);
					const info = {
						kind: "pokemon",
						name: pokemonDisplayName(teamEntry),
						species: splitPokemonDetails(teamEntry.details),
						sprite: spriteUrl(teamEntry, opponent),
						item: teamEntry.item,
						ability: teamEntry.ability,
						baseAbility: teamEntry.baseAbility,
						teraType: teamEntry.teraType,
						status: hp.status,
						percent: hp.percent,
						stats: teamEntry.stats,
						moves: teamEntry.moves || [],
					};
					const hoverId = registerHover(info);
					return `
						<div class="battle-mon-card ${teamEntry.active ? "active" : "bench"}" data-hover-id="${hoverId}">
							<div class="battle-mon-sprite-wrap">
								<img class="battle-mon-sprite" src="${escapeHTML(spriteUrl(teamEntry, opponent))}" alt="${escapeHTML(info.species)}" />
								<div class="battle-mon-hp-mini">
									${renderHpMeter(hp.percent, hp.status)}
								</div>
							</div>
							<div class="battle-mon-name">${escapeHTML(pokemonShortName(teamEntry))}</div>
						</div>
					`;
				}).join("")}
			</div>
		</div>
	`;
}

function renderChoiceButtons(actionRows, playerRequest, isHuman, choiceMaps) {
	const team = playerRequest?.side?.pokemon || [];
	const actives = playerRequest?.active || [];
	const isSingleActive = actives.length <= 1;
	const teraEnabled = actives.some(active => active?.canTerastallize);
	const moveGroups = new Map();
	const switchRows = [];
	const teamRows = [];
	for (const row of actionRows) {
		if (actionKind(row.command) === "move") {
			const slot = Number(row.command.split(/\s+/)[1] || 0);
			if (!moveGroups.has(slot)) moveGroups.set(slot, []);
			moveGroups.get(slot).push(row);
		} else if (actionKind(row.command) === "switch") {
			switchRows.push(row);
		} else if (actionKind(row.command) === "team") {
			teamRows.push(row);
		}
	}

	const singleMoveRows = isSingleActive
		? (() => {
			const byBase = new Map();
			for (const row of actionRows) {
				if (actionKind(row.command) !== "move") continue;
				const base = stripChoiceModifier(row.command);
				const existing = byBase.get(base) || {normal: null, tera: null};
				if (String(row.command).endsWith(" terastallize")) {
					existing.tera = row;
				} else {
					existing.normal = row;
				}
				byBase.set(base, existing);
			}
			return Array.from(byBase.entries())
				.map(([, pair]) => pair.normal || pair.tera)
				.filter(Boolean)
				.sort((a, b) => {
					const aIndex = Number(stripChoiceModifier(a.command).split(/\s+/)[1] || 99);
					const bIndex = Number(stripChoiceModifier(b.command).split(/\s+/)[1] || 99);
					return aIndex - bIndex;
				});
		})()
		: [];

	if (!actives.length && teamRows.length) {
		const leadMap = new Map();
		for (const row of teamRows) {
			const order = String(row.command).slice(5).trim();
			const commaLead = order.split(",")[0]?.replace(/\D/g, "") || "";
			const leadToken = commaLead ? commaLead.charAt(0) : order.replace(/\D/g, "").charAt(0);
			const leadSlot = Number(leadToken);
			if (!Number.isFinite(leadSlot) || leadSlot < 1) continue;
			const existing = leadMap.get(leadSlot);
			if (!existing || Number(row.policy_probability ?? -Infinity) > Number(existing.policy_probability ?? -Infinity)) {
				leadMap.set(leadSlot, row);
			}
		}

		const leadButtons = Array.from(leadMap.entries())
			.sort((a, b) => a[0] - b[0])
			.map(([leadSlot, row]) => {
				const teamEntry = team[leadSlot - 1] || null;
				if (!teamEntry) return "";
				const hp = parseHPCondition(teamEntry);
				const label = prettyChoiceLabel(row.description, row.command) || `Lead slot ${leadSlot}`;
				const cleanDescription = cleanChoiceDescription(row.description, row.command);
				const hover = registerHover({
					kind: "pokemon",
					name: pokemonDisplayName(teamEntry),
					species: splitPokemonDetails(teamEntry.details),
					sprite: spriteUrl(teamEntry, false),
					item: teamEntry.item,
					ability: teamEntry.ability,
					baseAbility: teamEntry.baseAbility,
					teraType: teamEntry.teraType,
					status: hp.status,
					percent: hp.percent,
					stats: teamEntry.stats,
					moves: teamEntry.moves || [],
					qvRows: [{label, policy: row.policy_probability, q: row.one_step_q_estimate, v: row.next_state_value}],
				});
				return `
					<button class="battle-choice switch-choice" data-choice-command="${escapeHTML(row.command)}" data-hover-id="${hover}">
						<img src="${escapeHTML(spriteUrl(teamEntry, false))}" alt="${escapeHTML(splitPokemonDetails(teamEntry.details))}" />
						<span>
							<strong>${escapeHTML(pokemonShortName(teamEntry))}</strong>
							<small>${escapeHTML(cleanDescription || label)} · π ${row.policy_probability !== undefined ? Number(row.policy_probability).toFixed(4) : "-"} · Q ${row.one_step_q_estimate !== undefined && row.one_step_q_estimate !== null ? Number(row.one_step_q_estimate).toFixed(4) : "-"} · V ${row.next_state_value !== undefined && row.next_state_value !== null ? Number(row.next_state_value).toFixed(4) : "-"}</small>
						</span>
					</button>
				`;
			});

		return `
			<div class="battle-command-stack">
				<div class="battle-action-group">
					<div class="battle-action-group-title">Choose lead Pokemon</div>
					<div class="battle-action-group-subtitle">Selecting a lead submits the full team order command for that option.</div>
					<div class="battle-switch-grid">${leadButtons.join("") || `<div class="muted">No lead choices available.</div>`}</div>
				</div>
			</div>
		`;
	}

	const moveSections = (actives.length ? actives : [null]).map((active, index) => {
		const slot = index + 1;
		const moveRows = isSingleActive ? singleMoveRows.slice() : (moveGroups.get(slot) || []).slice();
		const canTera = Boolean(active?.canTerastallize);
		const species = team[index] ? splitPokemonDetails(team[index].details) : `Slot ${slot}`;
		const currentInfo = team[index] || active || {};
		const currentHp = parseHPCondition(currentInfo);
		const moveQvRows = (isSingleActive
			? actionRows.filter(row => actionKind(row.command) === "move")
			: actionRows.filter(row => String(row.command).startsWith(`move ${slot}`))
		).map(row => ({label: prettyChoiceLabel(row.description, row.command), policy: row.policy_probability, q: row.one_step_q_estimate, v: row.next_state_value}));
		const teraLabel = currentInfo?.teraType ? `Terastalize this choice (Tera ${escapeHTML(currentInfo.teraType)})` : "Terastalize this choice";
		const hoverId = registerHover({
			kind: "pokemon",
			name: pokemonDisplayName(currentInfo),
			species,
			sprite: spriteUrl(currentInfo, false),
			item: currentInfo.item,
			ability: currentInfo.ability,
			baseAbility: currentInfo.baseAbility,
			teraType: currentInfo.teraType,
			status: currentHp.status,
			percent: currentHp.percent,
			stats: currentInfo.stats,
			moves: currentInfo.moves || [],
			qvRows: moveQvRows,
		});
		return `
			<div class="battle-action-group">
				<div class="battle-action-group-head">
					<div>
						<div class="battle-action-group-title">${escapeHTML(pokemonShortName(currentInfo) || species)}</div>
						<div class="battle-action-group-subtitle">${escapeHTML(species)}</div>
					</div>
					<div class="battle-action-group-sprite" data-hover-id="${hoverId}">
						<img src="${escapeHTML(spriteUrl(currentInfo, false))}" alt="${escapeHTML(species)}" />
					</div>
				</div>
				${teraEnabled ? `<label class="tera-toggle"><input type="checkbox" data-tera-slot="${slot}" ${canTera ? "" : "disabled"} /> ${teraLabel}</label>` : ""}
				<div class="battle-move-grid">
					${moveRows.length ? moveRows.map(row => {
						const label = prettyChoiceLabel(row.description, row.command);
						const hover = registerHover({
							kind: "move",
							label,
							command: row.command,
							description: cleanChoiceDescription(row.description, row.command),
							q: row.one_step_q_estimate,
							v: row.next_state_value,
							policy: row.policy_probability,
						});
						const base = stripChoiceModifier(row.command);
						const exact = escapeHTML(row.command);
						return `<button class="battle-choice move-choice" data-choice-command="${exact}" data-choice-base="${escapeHTML(base)}" data-hover-id="${hover}"><span>${escapeHTML(label)}</span><small>${row.policy_probability !== undefined ? Number(row.policy_probability).toFixed(4) : "-"} · Q ${row.one_step_q_estimate !== undefined && row.one_step_q_estimate !== null ? Number(row.one_step_q_estimate).toFixed(4) : "-"} · V ${row.next_state_value !== undefined && row.next_state_value !== null ? Number(row.next_state_value).toFixed(4) : "-"}</small></button>`;
					}).join("") : `<div class="muted">No move choices available.</div>`}
				</div>
			</div>
		`;
	}).join("");

	const switchButtons = switchRows.map(row => {
		const slot = Number(String(row.command).split(/\s+/)[1] || 0) - 1;
		const teamEntry = team[slot] || null;
		if (!teamEntry) return "";
		const hp = parseHPCondition(teamEntry);
		const hover = registerHover({
			kind: "pokemon",
			name: pokemonDisplayName(teamEntry),
			species: splitPokemonDetails(teamEntry.details),
			sprite: spriteUrl(teamEntry, false),
			item: teamEntry.item,
			ability: teamEntry.ability,
			baseAbility: teamEntry.baseAbility,
			teraType: teamEntry.teraType,
			status: hp.status,
			percent: hp.percent,
			stats: teamEntry.stats,
			moves: teamEntry.moves || [],
			qvRows: actionRows.filter(choice => choice.command === row.command).map(choice => ({label: prettyChoiceLabel(choice.description, choice.command), policy: choice.policy_probability, q: choice.one_step_q_estimate, v: choice.next_state_value})),
		});
		return `
			<button class="battle-choice switch-choice" data-choice-command="${escapeHTML(row.command)}" data-hover-id="${hover}">
				<img src="${escapeHTML(spriteUrl(teamEntry, false))}" alt="${escapeHTML(splitPokemonDetails(teamEntry.details))}" />
				<span>
					<strong>${escapeHTML(pokemonShortName(teamEntry))}</strong>
					<small>${renderHpMeter(hp.percent, hp.status)}</small>
				</span>
			</button>
		`;
	}).join("");

	return `
		<div class="battle-command-stack">
			${moveSections}
			<div class="battle-action-group">
				<div class="battle-action-group-title">Switch</div>
				<div class="battle-switch-grid">${switchButtons || `<div class="muted">No switch choices available.</div>`}</div>
			</div>
		</div>
	`;
}

function renderBattle(payload) {
	state.hoverRegistry.clear();
	state.hoverCounter = 0;
	state.activeBattleId = payload.id;
	state.activeCursor = payload.cursor;
	$("#battle-toolbar").hidden = false;
	$("#battle-cursor").max = String(payload.max_cursor);
	$("#battle-cursor").value = String(payload.cursor);
	$("#battle-cursor-number").max = String(payload.max_cursor);
	$("#battle-cursor-number").value = String(payload.cursor);
	$("#inspect-player").value = state.inspectPlayer;

	const actionRows = payload.analysis?.action_scores || payload.analysis?.legal_actions || [];
	const playerRequest = payload.state.requests?.[state.inspectPlayer] || null;
	const humanCanMove = payload.human_player && payload.state.pending_players.includes(payload.human_player);
	const choiceMaps = buildChoiceMaps(actionRows);
	const standardBattleUi = renderStandardBattleUi(payload);
	const controlPanel = `
		<div class="battle-controls card">
			<h3>Choice menu</h3>
			<div class="muted" style="margin-bottom:0.5rem;font-size:0.8rem;">Move/switch/team actions are below the battle frame and update from the selected inspect player.</div>
			${humanCanMove ? `<div class="battle-controls-note">${escapeHTML(payload.human_player)} may act now.</div>` : `<div class="battle-controls-note muted">No human choice is pending; actions are still visible for inspection.</div>`}
			${renderChoiceButtons(actionRows, playerRequest, Boolean(humanCanMove), choiceMaps)}
		</div>
	`;
	const inspectionRows = actionRows.length ? actionRows : [];

	$("#battle-content").innerHTML = `
		<div class="stack">
			<div class="summary-grid">
				<div class="metric"><span class="metric-label">Battle</span><span class="metric-value">${escapeHTML(payload.state.battle_id)}</span></div>
				<div class="metric"><span class="metric-label">Turn</span><span class="metric-value">${payload.state.turns}</span></div>
				<div class="metric"><span class="metric-label">Winner</span><span class="metric-value">${escapeHTML(payload.state.winner || "ongoing")}</span></div>
				<div class="metric"><span class="metric-label">Cursor</span><span class="metric-value">${payload.cursor} / ${payload.max_cursor}</span></div>
				<div class="metric"><span class="metric-label">Pending</span><span class="metric-value">${escapeHTML(payload.state.pending_players.join(", ") || "none")}</span></div>
				<div class="metric"><span class="metric-label">Checkpoint</span><span class="metric-value">${escapeHTML(payload.checkpoint || "none")}</span></div>
			</div>
			${standardBattleUi}
			<div class="battle-workbench">
				${controlPanel}
				<div class="card battle-analysis-panel">
					<details open>
						<summary>Action values</summary>
						<table>
							<thead><tr><th>#</th><th>Command</th><th>Description</th><th>Policy</th><th>One-step Q</th><th>Next V</th></tr></thead>
							<tbody>
								${inspectionRows.map(row => `
									<tr>
										<td>${row.index}</td>
										<td>${escapeHTML(row.command)}</td>
										<td>${escapeHTML(cleanChoiceDescription(row.description, row.command))}</td>
										<td>${row.policy_probability !== undefined ? Number(row.policy_probability).toFixed(4) : "-"}</td>
										<td>${row.one_step_q_estimate !== undefined && row.one_step_q_estimate !== null ? Number(row.one_step_q_estimate).toFixed(4) : "-"}</td>
										<td>${row.next_state_value !== undefined && row.next_state_value !== null ? Number(row.next_state_value).toFixed(4) : "-"}</td>
									</tr>
								`).join("")}
							</tbody>
						</table>
						${payload.analysis?.state_value !== undefined ? `<p class="muted">State value: ${Number(payload.analysis.state_value).toFixed(4)}</p>` : ""}
					</details>
				</div>
			</div>
			<div class="battle-grid">
				<div class="card"><h3>Scalar features</h3><pre>${escapeHTML(formatVectorBlock(payload.analysis?.scalar_features || []))}</pre></div>
				<div class="card"><h3>Party features</h3><pre>${escapeHTML(formatVectorBlock(payload.analysis?.party_features || []))}</pre></div>
			</div>
			<div class="battle-grid">
				<div class="card"><h3>P1 request</h3><div class="json-tree">${renderJsonTree(payload.state.requests?.p1 || null)}</div></div>
				<div class="card"><h3>P2 request</h3><div class="json-tree">${renderJsonTree(payload.state.requests?.p2 || null)}</div></div>
			</div>
			<div id="battle-hovercard" class="battle-hovercard" hidden></div>
		</div>
	`;
	syncReplayFrame(payload);
	installBattleInteractionHandlers(payload, choiceMaps);
}

function installBattleInteractionHandlers(payload, choiceMaps) {
	const root = $("#battle-content");
	const hoverCard = $("#battle-hovercard");
	if (!root || !hoverCard) return;

	const showHover = hoverId => {
		const info = state.hoverRegistry.get(hoverId);
		if (!info) return;
		hoverCard.innerHTML = renderPokemonTooltip(info);
		hoverCard.hidden = false;
	};

	const hideHover = () => {
		hoverCard.hidden = true;
	};

	const findHoverTarget = event => {
		const origin = event.target;
		if (!(origin instanceof Element)) return null;
		return origin.closest("[data-hover-id]");
	};

	root.onmouseover = event => {
		const target = findHoverTarget(event);
		if (!target) return;
		showHover(target.dataset.hoverId);
	};
	root.onmouseout = event => {
		const related = event.relatedTarget;
		if (related instanceof Element && root.contains(related)) return;
		hideHover();
	};
	root.onfocusin = event => {
		const target = findHoverTarget(event);
		if (!target) return;
		showHover(target.dataset.hoverId);
	};
	root.onfocusout = event => {
		const related = event.relatedTarget;
		if (related instanceof Element && root.contains(related)) return;
		hideHover();
	};
	root.onclick = async event => {
		const origin = event.target;
		if (!(origin instanceof Element)) return;
		const button = origin.closest("[data-choice-command]");
		if (!button) return;
		event.preventDefault();
		const baseCommand = button.dataset.choiceBase || button.dataset.choiceCommand;
		const teraBox = button.closest(".battle-action-group")?.querySelector("input[data-tera-slot]");
		const teraChecked = Boolean(teraBox && teraBox.checked);
		let command = button.dataset.choiceCommand;
		if (teraChecked) {
			const modifier = command.endsWith(" terastallize") ? command : `${baseCommand} terastallize`;
			command = choiceMaps.byCommand.get(modifier)?.command || choiceMaps.byBase.get(baseCommand + " terastallize")?.command || modifier;
		}
		await chooseHumanAction(command);
	};
}

async function refreshJobs() {
	const meta = await fetchJSON("/api/meta");
	state.meta = meta;
	setMetaSummary(meta);
	refreshCheckpointSelectors(meta);
	const payload = await fetchJSON("/api/jobs");
	renderJobs(payload.jobs || []);
}

async function loadBattle(cursor = state.activeCursor, player = state.inspectPlayer) {
	if (!state.activeBattleId) return;
	const query = new URLSearchParams();
	if (cursor !== null && cursor !== undefined) query.set("cursor", String(cursor));
	if (player) query.set("player", player);
	const payload = await fetchJSON(`/api/battles/${state.activeBattleId}?${query.toString()}`);
	renderBattle(payload);
}

async function chooseHumanAction(command) {
	if (!state.activeBattleId) return;
	if (state.choiceRequestInFlight) return;
	state.choiceRequestInFlight = true;
	try {
		const payload = await fetchJSON(`/api/battles/${state.activeBattleId}/choose`, {
			method: "POST",
			body: JSON.stringify({action: command, player: state.inspectPlayer}),
		});
		renderBattle(payload);
	} finally {
		state.choiceRequestInFlight = false;
	}
}

async function advanceBattle(mode) {
	if (!state.activeBattleId) return;
	const payload = await fetchJSON(`/api/battles/${state.activeBattleId}/advance`, {
		method: "POST",
		body: JSON.stringify({mode, player: state.inspectPlayer}),
	});
	renderBattle(payload);
}

async function restoreBattleCursor(cursor) {
	if (!state.activeBattleId) return;
	const payload = await fetchJSON(`/api/battles/${state.activeBattleId}/restore`, {
		method: "POST",
		body: JSON.stringify({cursor: Number(cursor), player: state.inspectPlayer}),
	});
	renderBattle(payload);
}

async function undoBattle(steps = 1) {
	if (!state.activeBattleId) return;
	const payload = await fetchJSON(`/api/battles/${state.activeBattleId}/undo`, {
		method: "POST",
		body: JSON.stringify({steps, player: state.inspectPlayer}),
	});
	renderBattle(payload);
}

async function redoBattle(steps = 1) {
	if (!state.activeBattleId) return;
	const payload = await fetchJSON(`/api/battles/${state.activeBattleId}/redo`, {
		method: "POST",
		body: JSON.stringify({steps, player: state.inspectPlayer}),
	});
	renderBattle(payload);
}

async function submitTrain(event) {
	event.preventDefault();
	const payload = formToObject(event.currentTarget);
	const summary = $("#jobs-summary");
	if (summary) summary.textContent = "Submitting PPO training job...";
	await fetchJSON("/api/train/ppo", {method: "POST", body: JSON.stringify(payload)});
	await refreshJobs();
}

async function submitEval(event) {
	event.preventDefault();
	const payload = formToObject(event.currentTarget);
	const summary = $("#jobs-summary");
	if (summary) summary.textContent = "Submitting PPO evaluation job...";
	await fetchJSON("/api/eval/ppo", {method: "POST", body: JSON.stringify(payload)});
	await refreshJobs();
}

async function submitBattle(event) {
	event.preventDefault();
	const payload = formToObject(event.currentTarget);
	if (payload.mode === "human" && !payload.human_player) payload.human_player = "p1";
	if (payload.mode !== "human") payload.human_player = null;
	delete payload.mode;
	const result = await fetchJSON("/api/battles", {method: "POST", body: JSON.stringify(payload)});
	if (result?.human_player === "p1" || result?.human_player === "p2") {
		state.inspectPlayer = result.human_player;
		$("#inspect-player").value = state.inspectPlayer;
	}
	renderBattle(result);
	await loadBattle(null, state.inspectPlayer);
}

async function closeBattle() {
	if (!state.activeBattleId) return;
	await fetchJSON(`/api/battles/${state.activeBattleId}/close`, {method: "POST", body: JSON.stringify({})});
	state.activeBattleId = null;
	state.activeCursor = null;
	state.replayBattleId = null;
	state.replayFrame = null;
	$("#battle-toolbar").hidden = true;
	$("#battle-content").className = "battle-content empty";
	$("#battle-content").textContent = "No battle loaded yet.";
}

async function changeTeams() {
	if (state.activeBattleId) {
		await closeBattle();
	}
	const form = $("#battle-form");
	if (form && typeof form.scrollIntoView === "function") {
		form.scrollIntoView({behavior: "smooth", block: "start"});
	}
}

async function bootstrap() {
	state.meta = await fetchJSON("/api/meta");
	setMetaSummary(state.meta);
	populateSelectors(state.meta);
	FORM_IDS.forEach(restoreForm);
	FORM_IDS.forEach(enableFormPersistence);
	await refreshJobs();
	$("#train-form").addEventListener("submit", submitTrain);
	$("#eval-form").addEventListener("submit", submitEval);
	$("#battle-form").addEventListener("submit", submitBattle);
	$("#inspect-player").addEventListener("change", event => {
		state.inspectPlayer = event.target.value;
		loadBattle($("#battle-cursor-number").value, state.inspectPlayer).catch(showError);
	});
	$("#battle-cursor").addEventListener("input", event => {
		$("#battle-cursor-number").value = event.target.value;
	});
	$("#battle-cursor").addEventListener("change", event => {
		restoreBattleCursor(event.target.value).catch(showError);
	});
	$("#battle-cursor-number").addEventListener("change", event => {
		$("#battle-cursor").value = event.target.value;
		restoreBattleCursor(event.target.value).catch(showError);
	});
	$("#jump-live").addEventListener("click", () => loadBattle(null, state.inspectPlayer).catch(showError));
	$("#close-battle").addEventListener("click", () => closeBattle().catch(showError));
	$("#undo-action").addEventListener("click", () => undoBattle(1).catch(showError));
	$("#redo-action").addEventListener("click", () => redoBattle(1).catch(showError));
	$("#change-teams").addEventListener("click", () => changeTeams().catch(showError));
	document.querySelectorAll("[data-advance]").forEach(button => {
		button.addEventListener("click", () => advanceBattle(button.dataset.advance).catch(showError));
	});
	state.jobsTimer = setInterval(() => refreshJobs().catch(showError), 3000);
}

function showError(error) {
	console.error(error);
	alert(error.message || String(error));
}

bootstrap().catch(showError);