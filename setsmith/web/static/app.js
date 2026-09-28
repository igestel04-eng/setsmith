"use strict";
// Setsmith web UI. All text from the collection is inserted with textContent, never HTML.

const SVG = "http://www.w3.org/2000/svg";
const $ = (sel, root = document) => root.querySelector(sel);

// ---------------------------------------------------------------- helpers

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* not JSON */ }
    throw new Error(message);
  }
  return res.json();
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function svg(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

// replaceChildren would render null as the text "null"; drop empty slots first.
function setChildren(target, ...children) {
  target.replaceChildren(...children.flat().filter((c) => c !== null && c !== undefined && c !== false));
}

function status(target, message, isError = false) {
  target.textContent = message;
  target.classList.toggle("error", isError);
}

function mmss(seconds) {
  if (seconds === null || seconds === undefined) return "-";
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}` : `${m}:${String(r).padStart(2, "0")}`;
}

function fmt(v, digits = 0) {
  return v === null || v === undefined ? "-" : Number(v).toFixed(digits).replace(/\.0+$/, "");
}

// Camelot colours: one hue per wheel number; minor (A) deeper, major (B) lighter.
function keyColor(camelot) {
  const m = /^(\d{1,2})([AB])$/.exec(camelot || "");
  if (!m) return "hsl(220 10% 60%)";
  const hue = ((Number(m[1]) - 1) * 30 + 130) % 360;
  return m[2] === "A" ? `hsl(${hue} 65% 62%)` : `hsl(${hue} 75% 76%)`;
}

function keyChip(camelot) {
  return el("span", { class: "key", style: `background:${keyColor(camelot)}` }, camelot || "?");
}

function scoreCell(total) {
  if (total === null || total === undefined) return el("td", { class: "num" }, "");
  const cls = total >= 80 ? "good" : total >= 60 ? "ok" : "bad";
  return el("td", { class: "num" }, el("span", { class: `score ${cls}` }, Math.round(total)));
}

function trackName(t) {
  return t.artist ? `${t.artist} - ${t.title}` : t.title || `Track ${t.id}`;
}

// ---------------------------------------------------------------- tabs

function initTabs() {
  const buttons = document.querySelectorAll(".tabs button");
  buttons.forEach((btn) => btn.addEventListener("click", () => {
    buttons.forEach((b) => b.setAttribute("aria-selected", String(b === btn)));
    document.querySelectorAll(".tab").forEach((t) => { t.hidden = t.id !== `tab-${btn.dataset.tab}`; });
    if (btn.dataset.tab === "livesets") loadLivesets();
  }));
}

// ---------------------------------------------------------------- track search

function initSearch(container) {
  const input = $("input", container);
  const list = $(".results", container);
  let timer = null;
  container.selectedId = null;
  input.addEventListener("input", () => {
    container.selectedId = null;
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const q = input.value.trim();
      if (!q) { list.hidden = true; return; }
      try {
        const rows = await api(`/api/tracks?q=${encodeURIComponent(q)}&limit=10`);
        const anySong = el("li", {
          role: "option",
          class: "any-song",
          onmousedown: (e) => {
            e.preventDefault();
            container.selectedId = null;
            list.hidden = true;
          },
        }, el("span", {}, `Use "${q}" (any song)`), el("span", { class: "muted" }, "BPM/key looked up"));
        setChildren(list, ...rows.map((t) => el("li", {
          role: "option",
          onmousedown: (e) => {
            e.preventDefault();
            container.selectedId = t.id;
            input.value = trackName(t);
            list.hidden = true;
          },
        }, el("span", {}, trackName(t)), el("span", { class: "muted" }, `${fmt(t.bpm, 1)} · ${t.key || "?"}`))), anySong);
        list.hidden = false;
      } catch (err) { list.hidden = true; }
    }, 200);
  });
  input.addEventListener("blur", () => setTimeout(() => { list.hidden = true; }, 150));
}

// The picked library track, or else the typed text as any song (looked up by the server).
function seedOf(selector) {
  const container = $(selector);
  if (container.selectedId) return { track_id: container.selectedId };
  const text = $("input", container).value.trim();
  return text ? { song: text } : null;
}

const EXTERNAL_ID = "external:song";

function seedLine(seed) {
  const base = `${trackName(seed)} (${fmt(seed.bpm, 1)} BPM, ${seed.key || "?"})`;
  if (seed.in_library !== false) return base;
  const source = seed.bpm_key_source ? `BPM/key ${seed.bpm_key_source}` : "no BPM/key found";
  const notes = (seed.warnings || []).length ? ` Note: ${seed.warnings.join("; ")}` : "";
  return `${base}, not in your library; ${source}.${notes}`;
}

// ---------------------------------------------------------------- timeline

// items: [{start, end, lane, color, label, sub, onclick, id}]; overlaps: [{start, end, label}]
// energy: {points: [{t, value}], targets: [{t, value}]} on a 1-10 scale.
function renderTimeline(target, { items, overlaps = [], energy = null, duration }) {
  const width = Math.max(target.clientWidth || 900, 720);
  const left = 44, right = 12, energyH = energy ? 90 : 0, laneH = 46, gap = 18, axisH = 22;
  const lanes = Math.max(1, ...items.map((i) => i.lane + 1));
  const height = energyH + (energy ? gap : 0) + lanes * laneH + (lanes - 1) * gap + axisH + 6;
  const x = (t) => left + (t / duration) * (width - left - right);
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img", "aria-label": "Set timeline" });

  // time grid: the smallest round step giving at most ~10 ticks
  const step = [15, 30, 60, 120, 300, 600, 900, 1800].find((st) => duration / st <= 10) || 3600;
  const axis = svg("g", { class: "axis" });
  for (let t = 0; t <= duration; t += step) {
    axis.append(svg("line", { class: "grid", x1: x(t), x2: x(t), y1: 0, y2: height - axisH }));
    axis.append(svg("text", { x: x(t), y: height - 6, "text-anchor": "middle" }, mmss(t)));
  }
  root.append(axis);

  if (energy) {
    const y = (v) => 6 + (1 - (v - 1) / 9) * (energyH - 12);
    const line = (pts, cls) => {
      if (pts.length < 2) return;
      root.append(svg("path", { class: cls, d: pts.map((p, i) => `${i ? "L" : "M"}${x(p.t)},${y(p.value)}`).join(" ") }));
    };
    root.append(svg("text", { class: "lane-label", x: 4, y: 14 }, "E"));
    [1, 5, 10].forEach((v) => root.append(svg("text", { class: "lane-label", x: 22, y: y(v) + 4 }, v)));
    line(energy.targets, "target");
    line(energy.points, "energy");
    energy.points.forEach((p) => root.append(svg("circle", { class: "energy-dot", cx: x(p.t), cy: y(p.value), r: 3 })));
    const legend = svg("g", { class: "legend" });
    legend.append(svg("line", { class: "energy", x1: width - 210, x2: width - 190, y1: 10, y2: 10 }));
    legend.append(svg("text", { x: width - 185, y: 14 }, "energy"));
    if (energy.targets.length) {
      legend.append(svg("line", { class: "target", x1: width - 120, x2: width - 100, y1: 10, y2: 10 }));
      legend.append(svg("text", { x: width - 95, y: 14 }, "curve target"));
    }
    root.append(legend);
  }

  const laneTop = (lane) => energyH + (energy ? gap : 0) + lane * (laneH + gap);
  for (let lane = 0; lane < lanes; lane++) {
    root.append(svg("text", { class: "lane-label", x: 4, y: laneTop(lane) + laneH / 2 + 4 }, lanes > 1 ? (lane ? "B" : "A") : ""));
  }
  const top = laneTop(0), bottom = laneTop(lanes - 1) + laneH;
  // Transitions: shaded across both lanes, a short tag in the gap, details on hover.
  overlaps.forEach((o) => {
    if (o.end <= o.start) return;
    const g = svg("g", {});
    g.append(svg("rect", { class: "overlap", x: x(o.start), y: top, width: Math.max(1, x(o.end) - x(o.start)), height: bottom - top }));
    if (o.tag && lanes > 1) {
      g.append(svg("text", { class: "overlap-label", x: (x(o.start) + x(o.end)) / 2, y: top + laneH + gap / 2 + 4, "text-anchor": "middle" }, o.tag));
    }
    if (o.label) g.append(svg("title", {}, o.label));
    root.append(g);
  });

  items.forEach((item) => {
    const g = svg("g", { class: "block", tabindex: 0, "data-id": item.id });
    const w = Math.max(2, x(item.end) - x(item.start));
    const yTop = laneTop(item.lane);
    g.append(svg("rect", { x: x(item.start), y: yTop, width: w, height: laneH, rx: 6, fill: item.color }));
    const title = svg("title", {}, `${item.label}\n${item.sub}`);
    g.append(title);
    if (w > 60) {
      const clip = `clip-${item.id}-${Math.random().toString(36).slice(2, 7)}`;
      const cp = svg("clipPath", { id: clip });
      cp.append(svg("rect", { x: x(item.start) + 4, y: yTop, width: w - 8, height: laneH }));
      g.append(cp);
      g.append(svg("text", { x: x(item.start) + 7, y: yTop + 18, "clip-path": `url(#${clip})` }, item.label));
      g.append(svg("text", { class: "sub", x: x(item.start) + 7, y: yTop + 34, "clip-path": `url(#${clip})` }, item.sub));
    }
    if (item.onclick) {
      const activate = () => {
        root.querySelectorAll(".block.selected").forEach((b) => b.classList.remove("selected"));
        g.classList.add("selected");
        item.onclick();
      };
      g.addEventListener("click", activate);
      g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(); } });
    }
    root.append(g);
  });
  setChildren(target, root);
}

// ---------------------------------------------------------------- build

let lastBuild = null;

function buildBody(form) {
  const f = new FormData(form);
  const unit = f.get("unit");
  const length = Number(f.get("length"));
  const curve = f.get("curve") === "__custom" ? f.get("custom_curve") : f.get("curve");
  const num = (v) => (v === "" || v === null ? null : Number(v));
  return {
    minutes: unit === "minutes" ? length : null,
    tracks: unit === "tracks" ? Math.round(length) : null,
    curve: curve || null,
    style: f.get("style") || null,
    bpm_min: num(f.get("bpm_min")),
    bpm_max: num(f.get("bpm_max")),
    start_id: seedOf('[data-search="start"]')?.track_id || null,
    start_song: seedOf('[data-search="start"]')?.song || null,
    genres: String(f.get("genres") || "").split(",").map((g) => g.trim()).filter(Boolean),
    learned: f.get("learned") === "on",
    name: f.get("name") || null,
  };
}

function showSetDetail(position, data) {
  const t = position.track;
  const detail = $("#set-detail");
  const dl = el("dl", {},
    el("dt", {}, "BPM"), el("dd", {}, fmt(t.bpm, 2)),
    el("dt", {}, "Key"), el("dd", {}, keyChip(t.key), t.detected_key && t.detected_key !== t.key ? ` (audio: ${t.detected_key})` : ""),
    el("dt", {}, "Energy"), el("dd", {}, `${fmt(t.energy, 1)} (target ${fmt(position.target_energy, 1)})`),
    el("dt", {}, "Genre"), el("dd", {}, t.genre || "-"),
    t.intro_bars !== null && t.intro_bars !== undefined ? [el("dt", {}, "Intro / outro"), el("dd", {}, `${t.intro_bars} / ${t.outro_bars} bars`)] : null,
    t.my_tags && t.my_tags.length ? [el("dt", {}, "My Tags"), el("dd", {}, t.my_tags.join(", "))] : null,
  );
  const prev = data.positions[position.position - 2];
  const parts = [el("h3", {}, `${position.position}. ${trackName(t)}`), dl];
  if (t.id === EXTERNAL_ID) parts.push(el("p", { class: "muted" }, "Not in your library: add it in Rekordbox and put it first. The exported playlist starts at track 2; BPM and key here were looked up."));
  if (prev && prev.transition_to_next) {
    parts.push(el("h3", {}, `In from #${prev.position}: ${Math.round(prev.transition_to_next.total)}`), el("pre", {}, prev.explain.join("\n")));
  }
  if (position.transition_to_next) {
    parts.push(el("h3", {}, `Out to #${position.position + 1}: ${Math.round(position.transition_to_next.total)}`), el("pre", {}, position.explain.join("\n")));
  }
  if (position.alternates.length) {
    parts.push(el("h3", {}, "Alternates"), el("ul", {}, position.alternates.map((a) =>
      el("li", {}, `${trackName(a.track)} `, el("span", { class: "muted" }, `${fmt(a.track.bpm, 1)} · `), keyChip(a.track.key), el("span", { class: "muted" }, ` · E${fmt(a.track.energy, 1)}`)))));
  }
  setChildren(detail, ...parts);
}

function renderSet(result) {
  lastBuild = result;
  const data = result.set;
  $("#build-result").hidden = false;
  $("#set-name").textContent = data.name;
  $("#set-summary").textContent = result.summary;
  const stats = data.stats;
  $("#set-style").textContent = stats.style
    ? `Style ${stats.style} (a profile, not endorsed by the artists): mean fit ${fmt(stats.mean_style_fit, 2)}`
    : "";
  $("#export-xml").href = `/api/sets/${result.id}/export`;
  $("#export-report").href = `/api/sets/${result.id}/report`;

  const timeline = result.timeline;
  const duration = timeline.length ? timeline[timeline.length - 1].end_s : 1;
  const select = (i) => {
    document.querySelectorAll("#set-table tbody tr").forEach((row, j) => row.classList.toggle("selected", i === j));
    showSetDetail(data.positions[i], data);
  };
  renderTimeline($("#set-timeline"), {
    duration,
    items: data.positions.map((p, i) => ({
      id: p.track.id,
      start: timeline[i].start_s,
      end: timeline[i].end_s,
      lane: i % 2,
      color: keyColor(p.track.key),
      label: `${p.position}. ${trackName(p.track)}`,
      sub: `${fmt(p.track.bpm, 1)} BPM · ${p.track.key || "?"} · E${fmt(p.track.energy, 1)}`,
      onclick: () => select(i),
    })),
    overlaps: data.positions.slice(0, -1).map((p, i) => {
      const tr = p.transition_to_next;
      return {
        start: timeline[i + 1].start_s,
        end: timeline[i].end_s,
        tag: tr ? String(Math.round(tr.total)) : "",
        label: tr ? `${p.position} → ${p.position + 1}: ${tr.suggested_type.replace("_", " ")}, ${tr.suggested_length_bars} bars, score ${Math.round(tr.total)}` : "",
      };
    }),
    energy: {
      points: data.positions.filter((p) => p.track.energy !== null).map((p) => ({
        t: (timeline[p.position - 1].start_s + timeline[p.position - 1].end_s) / 2, value: p.track.energy,
      })),
      targets: data.positions.map((p) => ({
        t: (timeline[p.position - 1].start_s + timeline[p.position - 1].end_s) / 2, value: p.target_energy,
      })),
    },
  });

  const head = el("thead", {}, el("tr", {},
    ["#", "Track", "BPM", "Key", "E (target)", "Next", "Transition", "Flags", "Alternates"].map((h, i) =>
      el("th", { class: [0, 2, 4, 5].includes(i) ? "num" : "" }, h))));
  const body = el("tbody", {}, data.positions.map((p, i) => {
    const tr = p.transition_to_next;
    return el("tr", { onclick: () => select(i) },
      el("td", { class: "num" }, p.position),
      el("td", { class: "track" }, trackName(p.track), p.track.id === EXTERNAL_ID ? el("span", { class: "muted" }, " (not in your library)") : null),
      el("td", { class: "num" }, fmt(p.track.bpm, 1)),
      el("td", {}, keyChip(p.track.key)),
      el("td", { class: "num" }, `${fmt(p.track.energy, 1)} (${fmt(p.target_energy, 1)})`),
      scoreCell(tr ? tr.total : null),
      el("td", {}, tr ? `${tr.suggested_type.replace("_", " ")} ${tr.suggested_length_bars}b` : "end"),
      el("td", { class: "flags" }, tr ? tr.flags.join(", ") : ""),
      el("td", { class: "alts" }, p.alternates.map((a) => trackName(a.track)).join(" · ")),
    );
  }));
  setChildren($("#set-table"), head, body);
  setChildren($("#set-detail"), el("p", { class: "muted" }, "Click a track for details."));
}

function initBuild() {
  const form = $("#build-form");
  const curve = form.elements.curve;
  curve.addEventListener("change", () => { $(".custom-curve").hidden = curve.value !== "__custom"; });
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const button = $("button[type=submit]", form);
    button.disabled = true;
    status($("#build-status"), "Building...");
    try {
      const result = await api("/api/build", { method: "POST", body: JSON.stringify(buildBody(form)) });
      renderSet(result);
      const warnings = result.set.warnings || [];
      status($("#build-status"), warnings.length ? `Note: ${warnings.join("; ")}` : "");
    } catch (err) {
      status($("#build-status"), `Could not build a set: ${err.message}`, true);
    } finally {
      button.disabled = false;
    }
  });
  window.addEventListener("resize", () => { if (lastBuild && !$("#tab-build").hidden) renderSet(lastBuild); });
}

// ---------------------------------------------------------------- suggest

function initSuggest() {
  const form = $("#suggest-form");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const seed = seedOf('[data-search="seed"]');
    if (!seed) { status($("#suggest-status"), "Type a song, or pick one from your library.", true); return; }
    const f = new FormData(form);
    const params = new URLSearchParams({ ...seed, top: "15", energy_delta: f.get("energy_delta") || "0" });
    if (f.get("style")) params.set("style", f.get("style"));
    if (f.get("learned") === "on") params.set("learned", "true");
    status($("#suggest-status"), seed.song ? "Looking up the song, then scoring..." : "Scoring...");
    try {
      const data = await api(`/api/suggest?${params}`);
      status($("#suggest-status"), `After ${seedLine(data.seed)}${data.style ? ` Style ${data.style}.` : ""}`);
      const styled = data.suggestions.some((s) => s.style_fit);
      const head = el("thead", {}, el("tr", {},
        el("th", { class: "num" }, "#"), el("th", {}, "Track"), el("th", { class: "num" }, "BPM"), el("th", {}, "Key"),
        el("th", { class: "num" }, "E"), el("th", { class: "num" }, "Score"), styled ? el("th", { class: "num" }, "Style") : null,
        el("th", {}, "Transition"), el("th", {}, "Flags")));
      const rows = data.suggestions.map((s, i) => el("tr", {
        onclick: (ev) => {
          document.querySelectorAll("#suggest-table tbody tr").forEach((r, j) => r.classList.toggle("selected", j === i));
          setChildren($("#suggest-detail"), 
            el("h3", {}, `${s.rank}. ${trackName(s.track)}`),
            el("pre", {}, s.explain.join("\n")),
            s.style_fit ? el("p", { class: "muted" }, `Style fit ${fmt(s.style_fit.score, 2)}`) : null);
        },
      },
        el("td", { class: "num" }, s.rank), el("td", { class: "track" }, trackName(s.track)), el("td", { class: "num" }, fmt(s.track.bpm, 1)),
        el("td", {}, keyChip(s.track.key)), el("td", { class: "num" }, fmt(s.track.energy, 1)), scoreCell(s.score.total),
        styled ? el("td", { class: "num" }, s.style_fit ? Math.round(100 * s.style_fit.score) : "-") : null,
        el("td", {}, `${s.score.suggested_type.replace("_", " ")} ${s.score.suggested_length_bars}b`),
        el("td", { class: "flags" }, s.score.flags.join(", "))));
      setChildren($("#suggest-table"), head, el("tbody", {}, rows));
      $("#suggest-result").hidden = false;
    } catch (err) {
      status($("#suggest-status"), err.message, true);
    }
  });
}

// ---------------------------------------------------------------- discover

function safeLink(url, text) {
  // Links come partly from outside services: only http(s) may become clickable.
  if (!/^https?:\/\//i.test(url || "")) return null;
  return el("a", { href: url, target: "_blank", rel: "noopener noreferrer" }, text);
}

function renderAttribution(target, items) {
  const parts = [];
  items.forEach((a, i) => { if (i) parts.push(" · "); parts.push(safeLink(a.url, a.text) || a.text); });
  setChildren(target, ...parts);
}

// BPM/key estimated from a Deezer preview rather than looked up in a catalog.
function estimated(r) {
  return Boolean(r.bpm_key_source) && r.bpm_key_source !== "GetSongBPM";
}

function initDiscover() {
  const form = $("#discover-form");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const seed = seedOf('[data-search="discover"]');
    if (!seed) { status($("#discover-status"), "Type a song, or pick one from your library.", true); return; }
    const f = new FormData(form);
    const params = new URLSearchParams({ ...seed, top: "20" });
    if (f.get("style")) params.set("style", f.get("style"));
    if (f.get("learned") === "on") params.set("learned", "true");
    const button = $("button[type=submit]", form);
    button.disabled = true;
    status($("#discover-status"), "Asking Last.fm, then getting BPM and key from GetSongBPM and Deezer previews (the first search for a seed can take up to a minute)...");
    try {
      const data = await api(`/api/discover?${params}`);
      const notes = data.warnings.length ? ` Note: ${data.warnings.join("; ")}` : "";
      status($("#discover-status"),
        `After ${seedLine(data.seed)} Last.fm suggested ${data.candidates} tracks (already in your library: ${data.in_library}).${notes}`);
      renderAttribution($("#discover-attribution"), data.attribution);
      const styled = data.results.some((r) => r.style_fit);
      const scened = data.results.some((r) => r.scene_match !== null);
      const head = el("thead", {}, el("tr", {},
        el("th", { class: "num" }, "#"), el("th", {}, "Track"), el("th", { class: "num" }, "BPM"), el("th", {}, "Key"),
        el("th", {}, "Genre"), el("th", { class: "num" }, "Score"), styled ? el("th", { class: "num" }, "Style") : null,
        scened ? el("th", { class: "num", title: "How similar the artist is to the seed's artist on Last.fm" }, "Scene") : null,
        el("th", {}, "Transition"), el("th", {}, "Listen / buy")));
      const rows = data.results.map((r, i) => el("tr", {
        onclick: () => {
          document.querySelectorAll("#discover-table tbody tr").forEach((row, j) => row.classList.toggle("selected", j === i));
          setChildren($("#discover-detail"), 
            el("h3", {}, `${r.artist} - ${r.title}`),
            el("p", { class: "muted" }, `Found as ${r.via} (Last.fm match ${fmt(r.lastfm_match, 2)})`),
            r.scene_match === null ? null : el("p", { class: "muted" }, r.off_scene
              ? "Outside the seed artist's scene on Last.fm (listeners overlap, but the artists aren't similar), so it's listed last."
              : `Scene match ${Math.round(100 * r.scene_match)}%: how similar the artist is to the seed's artist on Last.fm.`),
            r.bpm_key_source ? el("p", { class: "muted" }, `BPM/key ${r.bpm_key_source}${r.bpm_key_source === "GetSongBPM" ? "" : " (30 seconds of audio: a rough estimate; Rekordbox's analysis of the full track is better)"}.`) : null,
            r.bpm_key_known
              ? el("pre", {}, r.explain.join("\n"))
              : el("p", {}, "No BPM or key data yet, so Setsmith can't judge the mix. Listen via the links; once it's in Rekordbox, Suggest and Build will score it properly."),
            r.style_fit ? el("p", { class: "muted" }, `Style fit ${fmt(r.style_fit.score, 2)}`) : null);
        },
      },
        el("td", { class: "num" }, i + 1),
        el("td", { class: "track" }, `${r.artist} - ${r.title}`),
        el("td", { class: "num", title: r.bpm_key_source ? `BPM/key ${r.bpm_key_source}` : "" }, r.bpm && estimated(r) ? `≈${fmt(r.bpm, 1)}` : fmt(r.bpm, 1)),
        el("td", {}, r.key ? [estimated(r) ? "≈" : "", keyChip(r.key)] : "-"),
        el("td", {}, r.genre || "-"),
        r.bpm_key_known
          ? scoreCell(r.score.total)
          : el("td", { class: "num muted", title: "BPM/key unknown: ranked by Last.fm similarity" }, "?"),
        styled ? el("td", { class: "num" }, r.style_fit ? Math.round(100 * r.style_fit.score) : "-") : null,
        scened ? (r.off_scene
          ? el("td", { class: "num muted", title: "Outside the seed artist's scene: listed last" }, "other")
          : el("td", { class: "num" }, r.scene_match === null ? "-" : `${Math.round(100 * r.scene_match)}%`)) : null,
        el("td", {}, r.bpm_key_known ? `${r.score.suggested_type.replace("_", " ")} ${r.score.suggested_length_bars}b` : "-"),
        el("td", { class: "links" },
          [safeLink(r.links.soundcloud, "SoundCloud"), safeLink(r.links.beatport, "Beatport"), safeLink(r.links.lastfm, "Last.fm"), safeLink(r.links.deezer, "Deezer")]
            .filter(Boolean).flatMap((a, j) => (j ? [" · ", a] : [a]))),
      ));
      setChildren($("#discover-table"), head, el("tbody", {}, rows));
      $("#discover-result").hidden = false;
    } catch (err) {
      status($("#discover-status"), err.message, true);
    } finally {
      button.disabled = false;
    }
  });
}

// ---------------------------------------------------------------- live sets

let livesetsLoaded = false;

async function loadLivesets() {
  if (livesetsLoaded) return;
  livesetsLoaded = true;
  const select = $("#liveset-select");
  const sets = await api("/api/livesets");
  sets.forEach((s) => select.append(el("option", { value: s.id }, `#${s.id} ${s.name}`)));
  select.addEventListener("change", () => select.value && showLiveset(select.value));
}

async function showLiveset(id) {
  const data = await api(`/api/livesets/${encodeURIComponent(id)}`);
  const ls = data.liveset, stats = data.stats;
  $("#liveset-result").hidden = false;
  $("#liveset-name").textContent = ls.name;
  $("#liveset-summary").textContent =
    `${stats.matched} matched of ${ls.matches.length} lines` + (ls.duration_s ? ` · recording ${mmss(ls.duration_s)}` : "") +
    (ls.spans.length ? ` · timings: ${ls.spans[0].method === "dtw" ? "aligned" : "approximate"}` : " · no recording");

  const byPos = Object.fromEntries(ls.matches.map((m) => [m.position, m]));
  const spans = ls.spans.length
    ? ls.spans
    : ls.matches.filter((m) => !m.together).map((m, i) => ({ position: m.position, mix_start_s: i * 300, mix_end_s: i * 300 + 300 }));
  const duration = ls.duration_s || Math.max(...spans.map((s) => s.mix_end_s), 1);
  const energy = stats.energy.length ? { points: stats.energy.map(([pos, v]) => ({ t: pos * duration, value: v })), targets: [] } : null;
  renderTimeline($("#liveset-timeline"), {
    duration,
    energy,
    items: spans.map((s, i) => {
      const m = byPos[s.position] || {};
      return {
        id: `p${s.position}`, start: s.mix_start_s, end: s.mix_end_s, lane: i % 2, color: keyColor(m.camelot),
        label: `${s.position}. ${m.track_display || m.raw || ""}`,
        sub: `${fmt(m.bpm, 1)} BPM · ${m.camelot || "?"}${s.orig_start_s !== null && s.orig_start_s !== undefined ? ` · from ${mmss(s.orig_start_s)}` : ""}`,
      };
    }),
    overlaps: ls.transitions.filter((t) => t.overlap_s).map((t) => {
      const next = spans.find((s) => s.position === t.to_position);
      return next ? { start: next.mix_start_s, end: next.mix_start_s + t.overlap_s, tag: `${fmt(t.overlap_bars, 0)}b`, label: `${t.from_position} → ${t.to_position}: ${fmt(t.overlap_bars, 1)} bars overlap` } : null;
    }).filter(Boolean),
  });

  const head = el("thead", {}, el("tr", {}, ["From", "To", "Key move", "Tempo", "Overlap", "Cue out", "Cue in"].map((h) => el("th", {}, h))));
  const name = (pos) => (byPos[pos] && (byPos[pos].track_display || byPos[pos].raw)) || pos;
  const rows = ls.transitions.map((t) => el("tr", {},
    el("td", {}, name(t.from_position)), el("td", {}, name(t.to_position)), el("td", {}, t.key_move || "-"),
    el("td", {}, t.tempo_change_pct === null ? "-" : `${fmt(t.tempo_change_pct, 1)}%`),
    el("td", {}, t.overlap_bars === null ? "-" : `${fmt(t.overlap_bars, 1)} bars`),
    el("td", {}, mmss(t.cue_out_s)), el("td", {}, mmss(t.cue_in_s))));
  setChildren($("#liveset-table"), head, el("tbody", {}, rows));
}

// ---------------------------------------------------------------- start

async function init() {
  initTabs();
  document.querySelectorAll(".search").forEach(initSearch);
  initBuild();
  initSuggest();
  initDiscover();
  try {
    const info = await api("/api/info");
    const analyzed = info.analysis && info.analysis.analyzed ? ` · ${info.analysis.analyzed} analyzed` : "";
    $("#collection-info").textContent = `${info.collection} · ${info.tracks} tracks${analyzed} · v${info.version}`;
    const curve = $("#build-form").elements.curve;
    curve.append(el("option", { value: "" }, "Style's curve, else journey"));
    info.curves.forEach((c) => curve.append(el("option", { value: c }, c.replace("_", " "))));
    curve.append(el("option", { value: "__custom" }, "Custom..."));
    document.querySelectorAll('select[name="style"]').forEach((select) => {
      info.styles.forEach((s) => select.append(el("option", { value: s.key }, `${s.name} (${s.bpm_band[0]}-${s.bpm_band[1]})`)));
    });
    const discovery = info.discovery || {};
    $("#discover-setup").hidden = Boolean(discovery.lastfm);
    $("#discover-form").hidden = !discovery.lastfm;
    document.querySelectorAll('input[name="learned"]').forEach((box) => {
      box.disabled = !info.learned_available;
      if (!info.learned_available) box.parentElement.title = "Run 'setsmith learn' first";
    });
  } catch (err) {
    status($("#build-status"), `Could not reach Setsmith: ${err.message}`, true);
  }
}

init();
