// Race-day dashboard — read-only views of the open track.
"use strict";

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const sec = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));
const delta = (x, d = 2) => (x == null ? "–" : (x > 0 ? "+" : "") + Number(x).toFixed(d));
const cls = (x, tol = 0.005) => (x == null ? "" : x < -tol ? "good" : x > tol ? "bad" : "ok");
const pct = (x) => (x == null ? "–" : (100 * x).toFixed(1) + "%");
const api = (p) => fetch(p, { cache: "no-store" }).then((r) => (r.ok ? r.json() : Promise.reject(r.status)));
const PILOT_COLORS = ["#4da3ff", "#ff9f43", "#3ecf8e", "#c56cf0", "#ff6b6b", "#f5c451", "#00d2d3", "#ff9ff3"];
const colorFor = (() => { const m = {}; return (n) => (m[n] ??= PILOT_COLORS[Object.keys(m).length % PILOT_COLORS.length]); })();

// A short "how to read it" line under a title or number
const hint = (t) => `<div class="hint">${t}</div>`;
// A number tile with its meaning underneath
const fact = (k, v, h, extra = "") => `<div class="fact"><div class="k">${k}</div>${v}${extra}${hint(h)}</div>`;

function steadiness(cv) {
  if (cv == null) return "";
  if (cv < 0.03) return '<span class="pill good">steady</span>';
  if (cv < 0.07) return '<span class="pill ok">fair</span>';
  return '<span class="pill bad">variable</span>';
}

// Colour a section time against the field median (green faster, red slower)
function heat(v, ref) {
  if (v == null || !ref) return "";
  const r = Math.max(-1, Math.min(1, (v - ref) / (ref * 0.15)));
  const c = r < 0 ? [62, 207, 142] : [255, 107, 107];
  return `background:rgba(${c[0]},${c[1]},${c[2]},${(0.15 + 0.55 * Math.abs(r)).toFixed(2)})`;
}

// ── header: status every 3 s ───────────────────────────────────
let STATUS = null;
async function refreshStatus() {
  try {
    STATUS = await api("/api/status");
    $("#track").textContent = STATUS.track;
    $("#mode").textContent = STATUS.mode === "live" ? "LIVE" : "REPLAYS";
    const ph = $("#phase");
    if (STATUS.phase === "learn") {
      ph.className = "badge learn";
      ph.textContent = `learning the track ${STATUS.learn_progress[0]}/${STATUS.learn_progress[1]}`;
    } else { ph.className = "badge result"; ph.textContent = "results"; }
    const ch = (STATUS.live && STATUS.live.channels) || {};
    const names = (STATUS.pilot_info && STATUS.pilot_info.pilots) || {};
    $("#live").innerHTML = (STATUS.live && STATUS.live.heat ? `<span class="chip">Heat ${STATUS.live.heat}</span>` : "") +
      (STATUS.pilot_info && STATUS.pilot_info.round != null && ["scheduled", "prestart", "running"].includes(STATUS.pilot_info.phase)
        ? `<span class="chip">${STATUS.pilot_info.stage ? esc(STATUS.pilot_info.stage) + " · " : ""}Round ${esc(STATUS.pilot_info.round)} · Race ${esc(STATUS.pilot_info.race ?? "–")} · ${esc(STATUS.pilot_info.phase)}</span>` : "") +
      Object.entries(ch).map(([c, s]) => {
        const on = s.state === "live", drop = s.state === "dropout";
        const who = names[String(c).replace(/\s/g, "").toUpperCase()];
        return `<span class="chip ${on ? "on" : drop ? "drop" : ""}">${esc(c)}${who ? " " + esc(who) : ""} ${on || drop ? sec(s.seconds, 0) + "s" : "waiting"}</span>`;
      }).join("") +
      (STATUS.queue.analysing + STATUS.queue.waiting ? `<span class="chip">analysing ${STATUS.queue.analysing + STATUS.queue.waiting}</span>` : "");
  } catch (e) { /* server restarting */ }
}

// ── router ─────────────────────────────────────────────────────
let current = { key: null, timer: null };
function route() {
  const h = decodeURIComponent(location.hash.slice(1) || "/");
  const [, page, arg] = h.split("/");
  const key = h;
  if (current.timer) clearInterval(current.timer);
  current = { key, timer: null };
  const views = { "": home, run: runPage, heat: heatPage, pilot: pilotPage, track: trackPage };
  const view = views[page || ""] || home;
  view(arg, true).catch(showErr);
  current.timer = setInterval(() => view(arg, false).catch(() => {}), 5000);
  window.scrollTo(0, 0);
}
function showErr(e) { $("#app").innerHTML = `<p class="muted">Couldn't load (${esc(e)}).</p>`; }
window.addEventListener("hashchange", route);

// ── lap bar: the track drawn as sections ───────────────────────
function lapBar(sections, times) {
  if (!sections || !times || !times.some((x) => x)) return "";
  const tot = times.reduce((a, b) => a + (b || 0), 0);
  return `<div class="lapbar">${sections.map((s, i) => {
    const w = (100 * (times[i] || 0)) / tot;
    return `<div style="width:${w}%;background:hsl(${(i * 360) / sections.length},45%,${i % 2 ? 26 : 32}%)" title="${esc(s)} ${sec(times[i])}s">${w > 6 ? esc(s.replace("→", "›")) : ""}</div>`;
  }).join("")}</div>`;
}

// ── home: learn progress or results ────────────────────────────
async function home(_, first) {
  const [st, data] = await Promise.all([api("/api/status"), api("/api/runs")]);
  const runs = data.runs, field = data.field || {};
  if (st.phase === "learn") return learnView(st, runs);
  const done = runs.filter((r) => r.status === "labelled");
  // leaderboard: best of each pilot
  const by = {};
  for (const r of done) {
    const p = (by[r.pilot] ??= { pilot: r.pilot, runs: 0, best_lap: null, best3: null, tb: null, cv: null, si: null });
    p.runs++;
    const min = (a, b) => (a == null ? b : b == null ? a : Math.min(a, b));
    p.best_lap = min(p.best_lap, r.best_lap); p.best3 = min(p.best3, r.best3); p.tb = min(p.tb, r.theoretical_best);
    p.cv = min(p.cv, r.lap_cv); p.si = p.si == null ? r.speed_index : Math.max(p.si, r.speed_index ?? -1);
  }
  const board = Object.values(by).sort((a, b) => (a.best_lap ?? 1e9) - (b.best_lap ?? 1e9));
  const fastest = (field.section_best || []).map((b, i) => b ? `<span class="pill">${esc(field.sections[i])} <b>${sec(b.time)}</b> ${esc(b.pilot)}</span>` : "").join(" ");
  $("#app").innerHTML = `
    <h1>Results</h1>
    ${field.sections ? hint("The bar below is the track, drawn as sections (G1→G2, G2→G3 …). Wider block = that stretch takes longer. Example: a wide G5→G6 block is the longest stretch of the lap.") : ""}
    ${lapBar(field.sections, field.section_median)}
    <h2>Leaderboard</h2>
    ${hint("Best lap = fastest single lap. Best 3 laps = fastest 3 laps in a row. Theoretical best = your best time in every section added up — the lap you could fly. Consistency 2.5% = your laps vary about 2.5% (smaller = steadier). Speed vs field ×1.05 = 5% faster than the typical pilot here. Tap a pilot for their page.")}
    <div class="panel scroll"><table id="board"><thead><tr>
      <th>Pilot</th><th>Best lap</th><th>Best 3 laps</th><th>Theoretical best</th><th>Consistency</th><th>Speed vs field</th><th>Runs</th></tr></thead>
      <tbody>${board.map((p, i) => `<tr class="click" onclick="location.hash='#/pilot/${encodeURIComponent(p.pilot)}'">
        <td><b>${i + 1}.</b> <span style="color:${colorFor(p.pilot)}">●</span> ${esc(p.pilot)}</td><td><b>${sec(p.best_lap)}</b></td><td>${sec(p.best3)}</td>
        <td>${sec(p.tb)}</td><td>${pct(p.cv)} ${steadiness(p.cv)}</td><td class="${p.si > 1.005 ? "good" : p.si < 0.995 ? "bad" : ""}">${p.si ? "×" + sec(p.si) : "–"}</td><td>${p.runs}</td></tr>`).join("") ||
      '<tr><td colspan="7" class="muted">No runs with laps yet.</td></tr>'}</tbody></table></div>
    ${fastest ? `<h2>Fastest sections</h2>${hint("Who holds the best time in each section of the track. Example: “G9→G10 0.48 Talper” — nobody flew from gate 9 to gate 10 faster than 0.48 s.")}<div class="panel">${fastest}</div>` : ""}
    <h2>Latest runs</h2>
    ${hint("One card per pilot run, newest first. Big number = best lap. “steady” = lap times close together, “variable” = they change a lot. Tap a card to see the run.")}
    <div class="grid cards">${runs.slice(0, 24).map(runCard).join("") || '<p class="muted">Waiting for the first run…</p>'}</div>`;
}

function roundRace(r) {
  if (!r || r.round == null) return "";
  return `${r.stage ? esc(r.stage) + " · " : ""}Round ${esc(r.round)} · Race ${esc(r.race ?? "–")} · `;
}

function runCard(r) {
  const ready = r.status === "labelled";
  const flags = (r.missed ? `<span class="pill bad">${r.missed} missed</span>` : "") + (r.ended_mid_lap ? '<span class="pill ok">ended mid-lap</span>' : "");
  return `<a class="card" href="#/run/${encodeURIComponent(r.stem)}">
    <div class="who"><span style="color:${colorFor(r.pilot)}">●</span> ${esc(r.pilot)}</div>
    <div class="meta">${roundRace(r)}Heat ${r.heat} · ${esc(r.channel)} · ${sec(r.live_s, 0)}s</div>
    ${ready ? `<div class="big">${sec(r.best_lap)}</div><div class="meta">best lap · ${r.laps} laps ${steadiness(r.lap_cv)}</div>${flags}`
            : `<div class="meta" style="margin-top:8px">${r.status === "failed" ? "analysis failed" : "analysing…"}</div>`}</a>`;
}

function learnView(st, runs) {
  const [n, need] = st.learn_progress;
  $("#app").innerHTML = `
    <h1>Learning the track</h1>
    ${hint("The system learns the gates and their timing from the first runs by itself. Once enough runs are recorded, results appear for every run, including these.")}
    <div class="panel"><div class="big">${n} / ${need} runs</div>
      <div class="progress" style="margin:10px 0"><div style="width:${Math.min(100, (100 * n) / need)}%"></div></div>
      <div class="muted">Results for every run appear once ${need} runs are recorded and the ${st.n_gates || ""} gates are learned.</div></div>
    <h2>Runs collected</h2>
    <div class="panel scroll"><table><thead><tr><th>Run</th><th>Pilot</th><th>Heat</th><th>Channel</th><th>Length</th><th>Status</th></tr></thead>
      <tbody>${runs.map((r) => `<tr><td>${esc(r.stem)}</td><td>${esc(r.pilot)}</td><td>${r.heat}</td><td>${esc(r.channel)}</td><td>${sec(r.live_s, 0)}s</td><td>${esc(r.status)}</td></tr>`).join("")}</tbody></table></div>`;
}

// ── run page ───────────────────────────────────────────────────
async function runPage(stem, first) {
  const r = await api("/api/run/" + encodeURIComponent(stem));
  const st = r.stats, cmp = r.compare || {}, field = r.field || {};
  if (!first) { if (st && !$("#lapTable")) return runPage(stem, true); return; }   // keep the video playing
  const head = `<h1><span style="color:${colorFor(r.pilot)}">●</span> ${esc(r.pilot)}</h1>
    <p class="muted">${roundRace(r)}<a href="#/heat/${r.heat}">Heat ${r.heat}</a> · ${esc(r.channel)} · <a href="#/pilot/${encodeURIComponent(r.pilot)}">all runs of ${esc(r.pilot)}</a></p>`;
  if (!st) { $("#app").innerHTML = head + `<div class="panel">This run is ${esc(r.status)} — results appear when it's analysed${STATUS && STATUS.phase === "learn" ? " and the track is learned" : ""}.</div>
    <video src="${r.video_url}" controls playsinline preload="metadata"></video>`; return; }
  const best = st.best_lap_index;
  const live0 = r.live_start_s || 0;
  $("#app").innerHTML = head + `
    <video id="vid" src="${r.video_url}" controls playsinline preload="metadata"></video>
    <div id="timeline" class="panel" style="padding:6px 10px"></div>
    ${hint("Timeline: thin blue lines = gates passed, yellow lines = start/finish (G1, a new lap), L1 L2 … = laps, red = where the video is now. Tap the bar to jump there.")}
    <div class="facts">
      ${fact("Best lap", `<div class="v good">${sec(st.best_lap)}</div>`, "Your fastest full lap (G1 to G1), in seconds.")}
      ${fact("Average lap", `<div class="v">${sec(st.mean_lap)}</div>`, "The mean of all your full laps in this run.")}
      ${fact("Laps", `<div class="v">${st.laps}</div>`, "Full laps flown (start/finish to start/finish).")}
      ${fact("Consistency", `<div class="v">±${sec(st.lap_sd)}</div>`, "±0.40 = your laps vary about 0.4 s around the average. Smaller = steadier.", steadiness(st.lap_cv))}
      ${fact("Theoretical best", `<div class="v">${sec(st.theoretical_best)}</div>`, "Your best time in each section added up — the lap you could fly if you put it all together.")}
      ${fact("Speed vs field", `<div class="v ${cmp.speed_index > 1.005 ? "good" : cmp.speed_index < 0.995 ? "bad" : ""}">${cmp.speed_index ? "×" + sec(cmp.speed_index) : "–"}</div>`, "×1.05 = 5% faster than the typical pilot on this track, ×0.97 = 3% slower.")}
      ${fact(`Holeshot (to G${st.first_gate || 1})`, `<div class="v">${sec(st.holeshot)}</div>`, "Seconds from take-off to the first gate.")}
      ${fact("Missed gates", `<div class="v ${st.missed_gates.length ? "bad" : ""}">${st.missed_gates.length}</div>`, "Gates skipped (or not seen) between two passes. Each one is listed below with a link to the moment in the video.")}
    </div>
    ${st.missed_gates.length ? `<p class="muted">Missed: ${st.missed_gates.map((m) => `<a href="#" onclick="seek(${m.t - 1});return false">G${m.gate} at ${sec(m.t - live0, 1)}s</a>`).join(", ")}${st.ended_mid_lap ? " · run ended mid-lap" : ""}</p>` : st.ended_mid_lap ? '<p class="muted">Run ended mid-lap.</p>' : ""}
    <h2>Laps</h2>
    ${hint("vs own best +0.35 = this lap was 0.35 s slower than your best lap. vs field median: green = faster than the typical lap on this track, red = slower. In the chart, the dashed line is the typical lap. Tap a lap to watch it.")}
    <div class="panel scroll"><table id="lapTable"><thead><tr><th>Lap</th><th>Time</th><th>vs own best</th><th>vs field median</th></tr></thead><tbody>
      ${st.lap_times.map((t, i) => `<tr class="click ${i === best ? "best" : ""}" onclick="seek(${r.lap_marks[i].t0 - 0.5})"><td>${i + 1}</td><td>${sec(t)}</td>
        <td>${delta(t - st.best_lap)}</td><td class="${cls(field.median_lap ? t - field.median_lap : null)}">${field.median_lap ? delta(t - field.median_lap) : "–"}</td></tr>`).join("")}
    </tbody></table>${lapChart(st.lap_times, field.median_lap)}</div>
    <h2>Sections</h2>
    ${hint("One row per lap, one column per section of the track. Green cell = faster than most pilots in that section, red = slower. Example: a red column means that section costs you time on every lap.")}
    <div class="panel scroll"><table><thead><tr><th>Lap</th>${st.sections.map((s) => `<th>${esc(s)}</th>`).join("")}</tr></thead><tbody>
      ${st.section_grid.map((row, i) => `<tr><td>${i + 1}</td>${row.map((v, j) => `<td><div class="cell" style="${heat(v, field.section_median?.[j])}">${sec(v)}</div></td>`).join("")}</tr>`).join("")}
      <tr class="best"><td>Best</td>${st.section_best.map((v) => `<td>${sec(v)}</td>`).join("")}</tr>
      <tr><td class="muted">Field median</td>${(field.section_median || []).map((v) => `<td class="muted">${sec(v)}</td>`).join("")}</tr>
      <tr><td class="muted">Field best</td>${(field.section_best || []).map((b) => `<td class="muted">${b ? sec(b.time) : "–"}</td>`).join("")}</tr>
    </tbody></table></div>
    <h2>Where time is gained or lost</h2>
    ${hint("Your best time in each section against the typical pilot. Green bar to the left = you gain time there (−0.20 = 0.2 s faster). Red bar to the right = you lose time there.")}
    <div class="panel">${sectionBars(st.sections, cmp.section_vs_median)}</div>`;
  drawTimeline(r);
}

function seek(t) { const v = $("#vid"); if (v) { v.currentTime = Math.max(0, t); v.play().catch(() => {}); } }

function drawTimeline(r) {
  const v = $("#vid"), box = $("#timeline");
  if (!v || !box) return;
  const draw = () => {
    const dur = v.duration || Math.max(1, ...r.passes.map((p) => p.t)) + 2;
    const X = (t) => (100 * t) / dur;
    // drawn in screen pixels (no stretched viewBox), so gate lines stay 1.5 px thin on any screen
    const laps = r.lap_marks.map((l, i) => `<rect x="${X(l.t0)}%" y="0" width="${Math.max(0, X(l.t1) - X(l.t0))}%" height="34" fill="${i % 2 ? "rgba(255,255,255,0.025)" : "rgba(255,255,255,0.05)"}"/>
      <text x="${X((l.t0 + l.t1) / 2)}%" y="11" fill="#8a96ad" font-size="10" text-anchor="middle">L${l.lap}</text>`).join("");
    const passes = r.passes.filter((p) => p.gate >= 1).map((p) => p.gate === 1
      ? `<line x1="${X(p.t)}%" x2="${X(p.t)}%" y1="2" y2="34" stroke="#f5c451" stroke-width="1.5"/>`
      : `<line x1="${X(p.t)}%" x2="${X(p.t)}%" y1="15" y2="34" stroke="#4da3ff" stroke-width="1.5"/>`).join("");
    box.innerHTML = `<svg width="100%" height="34" style="cursor:pointer;display:block">${laps}${passes}
      <line id="ph" x1="${X(v.currentTime)}%" x2="${X(v.currentTime)}%" y1="0" y2="34" stroke="#ff3b5c" stroke-width="2"/></svg>`;
    box.querySelector("svg").onclick = (e) => { const b = e.currentTarget.getBoundingClientRect(); seek(((e.clientX - b.left) / b.width) * dur); };
  };
  v.addEventListener("loadedmetadata", draw);
  v.addEventListener("timeupdate", () => { const ph = $("#ph"); if (ph && v.duration) { const x = (100 * v.currentTime) / v.duration + "%"; ph.setAttribute("x1", x); ph.setAttribute("x2", x); } });
  draw();
}

function lapChart(times, ref) {
  if (!times || times.length < 2) return "";
  const W = 600, H = 120, pad = 24, lo = Math.min(...times, ref ?? 1e9) * 0.97, hi = Math.max(...times, ref ?? 0) * 1.03;
  const x = (i) => pad + (i * (W - 2 * pad)) / (times.length - 1), y = (t) => H - pad - ((t - lo) / (hi - lo)) * (H - 2 * pad);
  const pts = times.map((t, i) => `${x(i)},${y(t)}`).join(" ");
  return `<svg viewBox="0 0 ${W} ${H}" style="margin-top:10px">
    ${ref ? `<line x1="${pad}" x2="${W - pad}" y1="${y(ref)}" y2="${y(ref)}" stroke="#8a96ad" stroke-dasharray="4 4"/><text x="${W - pad}" y="${y(ref) - 4}" fill="#8a96ad" font-size="11" text-anchor="end">field median</text>` : ""}
    <polyline points="${pts}" fill="none" stroke="#4da3ff" stroke-width="2.5"/>
    ${times.map((t, i) => `<circle cx="${x(i)}" cy="${y(t)}" r="4" fill="#4da3ff"/><text x="${x(i)}" y="${H - 6}" fill="#8a96ad" font-size="11" text-anchor="middle">L${i + 1}</text>`).join("")}</svg>`;
}

function sectionBars(sections, diffs) {
  if (!diffs || !diffs.some((d) => d != null)) return '<p class="muted">Needs more runs on the track.</p>';
  const m = Math.max(0.05, ...diffs.filter((d) => d != null).map(Math.abs));
  return sections.map((s, i) => {
    const d = diffs[i]; if (d == null) return "";
    const w = (50 * Math.abs(d)) / m;
    return `<div style="display:flex;align-items:center;gap:8px;margin:4px 0"><div style="width:76px;font-size:12px">${esc(s)}</div>
      <div style="flex:1;position:relative;height:16px;background:var(--panel2);border-radius:4px">
        <div style="position:absolute;top:0;bottom:0;${d < 0 ? `right:50%` : `left:50%`};width:${w}%;background:${d < 0 ? "var(--good)" : "var(--bad)"};border-radius:3px"></div>
        <div style="position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--muted)"></div></div>
      <div style="width:56px;text-align:right" class="${cls(d)}">${delta(d)}</div></div>`;
  }).join("");
}

// ── heat page: 4 videos in sync + gap chart ────────────────────
async function heatPage(n, first) {
  const h = await api("/api/heat/" + n);
  if (!first) return;
  const runs = h.runs;
  const L = Math.min(...runs.filter((r) => r.stats && r.stats.laps).map((r) => r.stats.laps));
  const order = runs.filter((r) => r.stats && r.stats.laps >= L && isFinite(L)).map((r) => {
    const lt = r.stats.lap_times.slice(0, L).reduce((a, b) => a + b, 0);
    return { r, t: lt };
  }).sort((a, b) => a.t - b.t);
  const rr = runs.find((r) => r.round != null);
  $("#app").innerHTML = `<h1>${rr ? `${rr.stage ? esc(rr.stage) + " · " : ""}Round ${esc(rr.round)} · Race ${esc(rr.race ?? "–")}` : `Heat ${n}`}</h1>
    <p><span class="btn" onclick="heatPlay(true)">▶ Play all</span> <span class="btn" onclick="heatPlay(false)">❚❚ Pause</span></p>
    ${hint("▶ Play all starts the pilots' videos together, as they flew. Tap a name to open that pilot's run.")}
    <div class="quad">${runs.map((r) => `<div><div class="meta"><span style="color:${colorFor(r.pilot)}">●</span> <a href="#/run/${encodeURIComponent(r.stem)}">${esc(r.pilot)}</a> · ${esc(r.channel)}</div>
      <video class="hv" data-off="${r.live_start_s}" src="${r.video_url}" playsinline muted preload="metadata"></video></div>`).join("")}</div>
    ${isFinite(L) && order.length ? `<h2>Order over the first ${L} lap${L > 1 ? "s" : ""}</h2>${hint("Total time for the laps everyone completed. Gap +1.20 = 1.2 s behind the leader.")}<div class="panel scroll"><table><thead><tr><th>Pilot</th><th>Time</th><th>Gap</th><th>Best lap</th></tr></thead><tbody>
      ${order.map((o, i) => `<tr><td><b>${i + 1}.</b> ${esc(o.r.pilot)}</td><td>${sec(o.t)}</td><td>${i ? "+" + sec(o.t - order[0].t) : "–"}</td><td>${sec(o.r.stats.best_lap)}</td></tr>`).join("")}</tbody></table></div>` : ""}
    <h2>Gap to the leader at every gate</h2>${hint("One line per pilot. The leader runs along the top. A line dropping lower = falling further behind; a line climbing = catching up. L1 L2 … mark the laps.")}<div class="panel">${gapChart(runs, h.runs[0]?.stats?.n_gates)}</div>`;
}

function heatPlay(on) {
  const vs = [...document.querySelectorAll(".hv")];
  if (!on) return vs.forEach((v) => v.pause());
  const t = Math.max(0, (vs[0]?.currentTime || 0) - (+vs[0]?.dataset.off || 0));
  vs.forEach((v) => { v.currentTime = t + (+v.dataset.off || 0); v.play().catch(() => {}); });
}

function gapChart(runs, n) {
  if (!n) return '<p class="muted">Results appear when the runs are analysed.</p>';
  // k = gates passed since the first G1: lap*n + gate
  const seqs = runs.map((r) => {
    let lap = -1, out = {};
    for (const p of r.passes) { if (p.gate === 1) lap++; if (lap >= 0) out[lap * n + p.gate - 1] = p.t; }
    return { r, out };
  });
  const ks = [...new Set(seqs.flatMap((s) => Object.keys(s.out).map(Number)))].sort((a, b) => a - b);
  const lead = {}; for (const k of ks) lead[k] = Math.min(...seqs.map((s) => s.out[k]).filter((x) => x != null));
  const maxGap = Math.max(1, ...seqs.flatMap((s) => ks.filter((k) => s.out[k] != null).map((k) => s.out[k] - lead[k])));
  const W = 700, H = 200, pad = 28, X = (k) => pad + ((k - ks[0]) * (W - 2 * pad)) / Math.max(1, ks[ks.length - 1] - ks[0]), Y = (g) => pad + (g / maxGap) * (H - 2 * pad);
  const lines = seqs.map((s) => {
    const pts = ks.filter((k) => s.out[k] != null).map((k) => `${X(k)},${Y(s.out[k] - lead[k])}`).join(" ");
    return `<polyline points="${pts}" fill="none" stroke="${colorFor(s.r.pilot)}" stroke-width="2.5"/>`;
  }).join("");
  const lapTicks = ks.filter((k) => k % n === 0).map((k) => `<line x1="${X(k)}" x2="${X(k)}" y1="${pad}" y2="${H - pad}" stroke="#2a3347"/><text x="${X(k)}" y="${H - 8}" fill="#8a96ad" font-size="11" text-anchor="middle">L${k / n + 1}</text>`).join("");
  return `<svg viewBox="0 0 ${W} ${H}">${lapTicks}<text x="${pad}" y="16" fill="#8a96ad" font-size="11">seconds behind the leader (down = further behind)</text>${lines}</svg>
    <div>${seqs.map((s) => `<span class="pill"><span style="color:${colorFor(s.r.pilot)}">●</span> ${esc(s.r.pilot)}</span>`).join("")}</div>`;
}

// ── pilot page ─────────────────────────────────────────────────
async function pilotPage(name, first) {
  const p = await api("/api/pilot/" + encodeURIComponent(name));
  if (!first) return;
  const runs = p.runs;
  $("#app").innerHTML = `<h1><span style="color:${colorFor(name)}">●</span> ${esc(name)}</h1>
    <div class="facts">
      ${fact("Best lap", `<div class="v good">${sec(Math.min(...runs.map((r) => r.best_lap ?? 1e9)))}</div>`, "Fastest lap over all your runs today.")}
      ${fact("Runs", `<div class="v">${runs.length}</div>`, "Runs recorded for you.")}
      ${fact("Laps", `<div class="v">${runs.reduce((a, r) => a + r.laps, 0)}</div>`, "Full laps flown, all runs together.")}
    </div>
    <h2>Runs</h2>${hint("Each of your runs — tap one to open it. The chart shows your best lap in each run through the day (R1, R2 …): going down = you got faster. Dashed line = the typical lap on this track.")}<div class="panel scroll"><table><thead><tr><th>Run</th><th>Laps</th><th>Best</th><th>Average</th><th>Consistency</th></tr></thead><tbody>
      ${runs.map((r) => `<tr class="click" onclick="location.hash='#/run/${encodeURIComponent(r.stem)}'"><td>Heat ${r.heat} · ${esc(r.channel)}</td><td>${r.laps}</td><td>${sec(r.best_lap)}</td><td>${sec(r.mean_lap)}</td><td>${pct(r.lap_cv)} ${steadiness(r.lap_cv)}</td></tr>`).join("")}</tbody></table>
      ${lapChart(runs.map((r) => r.best_lap).filter((x) => x), p.field.median_lap).replaceAll(">L", ">R")}</div>
    <h2>Strengths and weaknesses</h2>${hint("Your best time in each section, from all your runs, against the typical pilot. Green bar to the left = a strength (you're faster there), red bar to the right = where you lose time.")}<div class="panel">${sectionBars(p.sections || [], p.vs_field_median)}</div>`;
}

// ── track page ─────────────────────────────────────────────────
async function trackPage(_, first) {
  const t = await api("/api/track");
  if (!first) return;
  const f = t.field || {};
  const hard = (f.sections || []).map((s, i) => ({ s, sp: f.section_spread?.[i], med: f.section_median?.[i] }))
    .filter((x) => x.sp != null).sort((a, b) => b.sp - a.sp).slice(0, 4);
  $("#app").innerHTML = `<h1>Track — ${t.gates.length} gates</h1>
    ${hint("The track drawn as sections, each block as wide as the time it usually takes.")}
    ${lapBar(f.sections, f.section_median || t.legs_s)}
    ${hard.length ? `<h2>Hardest sections</h2>${hint("Where pilots differ most. Spread 0.90 s = the gap between fast and slow pilots in that section — where races are won or lost.")}<div class="panel">${hard.map((h) => `<span class="pill">${esc(h.s)}: median ${sec(h.med)}s, spread ${sec(h.sp)}s</span>`).join(" ")}</div>` : ""}
    <h2>Gates</h2>${hint("Pictures the system learned for each gate, in the order they're flown, with the typical time to the next gate.")}<div class="gates">${t.gates.map((g) => `<div class="gate panel"><b>G${g.gate}</b> <span class="muted">${esc(g.type)}</span>
      ${g.images.slice(0, 2).map((src) => `<img loading="lazy" src="${src}">`).join("")}
      <div class="muted" style="font-size:12px">G${g.gate}→G${g.gate % t.gates.length + 1}: ${sec(f.section_median?.[g.gate - 1] ?? t.legs_s?.[g.gate - 1])}s</div></div>`).join("") || '<p class="muted">The track isn\'t learned yet.</p>'}</div>`;
}

refreshStatus(); setInterval(refreshStatus, 3000); route();
