/* SegEvalKit viewer. Data come from the local server (segevalkit view) or are embedded (--export). */
(() => {
  "use strict";
  const NV = window.niivue;
  const MODE = window.SEK_MODE || "server";
  const qs = new URLSearchParams(location.search);
  if (qs.get("delay")) { // screenshot harness only: hold the load event while WebGL renders
    const im = new Image(); im.src = "/__delay=" + qs.get("delay"); im.style.display = "none";
    document.body.appendChild(im);
  }
  window.SEK_STATUS = "loading";
  window.SEK_ERRORS = [];

  const st = { view: "multi", layout: "single", overlay: "labels", source: "ref", sel: null,
               opacity: 0.6, win: "abdomen" };
  let M = null;
  let viewers = [];
  const bufs = new Map();
  const live = new Map();

  // ------------------------------------------------------------------ utilities
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const fmt = (v, k) => (v === null || v === undefined || Number.isNaN(v)) ? "–" :
    (/hd|assd|masd|distance/.test(k) ? v.toFixed(1) : /volume_difference|rvd/.test(k) ? (100 * v).toFixed(0) + "%" :
     /lesions$|count/.test(k) ? String(Math.round(v)) : v.toFixed(3));
  function toast(msg, ms = 2200) {
    const t = $("toast"); t.textContent = msg; t.hidden = false;
    clearTimeout(toast._t); if (ms) toast._t = setTimeout(() => (t.hidden = true), ms);
  }
  function fail(e) { console.error(e); window.SEK_ERRORS.push(String(e && e.message || e)); toast("Error: " + (e.message || e), 6000); }
  window.addEventListener("error", (e) => window.SEK_ERRORS.push(String(e.message)));

  const enc = (p) => p.split("/").map(encodeURIComponent).join("/");
  async function getBuf(path) {
    if (!bufs.has(path)) {
      bufs.set(path, (async () => {
        if (MODE === "static") {
          const b64 = (window.SEK_FILES || {})[path];
          if (!b64) return null;
          const bin = atob(b64); const u = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
          return u.buffer;
        }
        const r = await fetch(enc(path));
        return r.ok ? r.arrayBuffer() : null;
      })());
    }
    const b = await bufs.get(path);
    return b ? b.slice(0) : null;
  }
  async function getManifest() {
    if (MODE === "static") return window.SEK_MANIFEST;
    const r = await fetch("api/manifest");
    if (!r.ok) throw new Error("manifest: " + r.status);
    return r.json();
  }
  async function liveMetrics(model, s) {
    if (MODE === "static") return ((M.live || {})[model] || {})[s] || null;
    const k = model + "\u0000" + s;
    if (!live.has(k)) live.set(k, fetch(`api/metrics?model=${encodeURIComponent(model)}&structure=${encodeURIComponent(s)}`)
      .then((r) => (r.ok ? r.json() : null)));
    return live.get(k);
  }
  const S = (name) => M.structures.find((s) => s.name === name);
  const nameOf = (idx) => (M.structures.find((s) => s.index === idx) || {}).name;

  // ------------------------------------------------------------------ one canvas
  class Viewer {
    constructor(source, grid) {
      this.source = source;
      this.el = document.createElement("div"); this.el.className = "tile";
      this.canvas = document.createElement("canvas");
      this.badge = document.createElement("div"); this.badge.className = "badge";
      this.el.append(this.canvas, this.badge); grid.appendChild(this.el);
      this.errVol = null; this.meshSig = ""; this.lastLoc = null;
    }
    get isRef() { return this.source === "ref"; }
    async init() {
      this.nv = new NV.Niivue({
        backColor: [0.06, 0.04, 0.11, 1], crosshairColor: [0.72, 0.61, 0.96, 0.85], show3Dcrosshair: true,
        dragAndDropEnabled: false, isColorbar: false, textHeight: 0.03, isOrientCube: false,
        multiplanarShowRender: NV.SHOW_RENDER.ALWAYS, logLevel: "error",
      });
      await this.nv.attachToCanvas(this.canvas);
      const [lo, hi] = M.windows[st.win];
      // NVImage.loadFromUrl honours `buffer` (Niivue.loadVolumes does not), so data can be embedded.
      const img = await NV.NVImage.loadFromUrl({ url: "image.nii.gz", name: "image.nii.gz",
        buffer: await getBuf("vol/image.nii.gz"), colormap: "gray", cal_min: lo, cal_max: hi });
      this.nv.addVolume(img);
      const lab = await getBuf(`vol/labels/${this.source}.nii.gz`);
      if (lab) { // volume 1: every structure; volume 2: the selected structure only (label alpha is binary)
        const L = await NV.NVImage.loadFromUrl({ url: "labels.nii.gz", name: "labels.nii.gz", buffer: lab.slice(0), opacity: st.opacity });
        this.nv.addVolume(L);
        const H = await NV.NVImage.loadFromUrl({ url: "highlight.nii.gz", name: "highlight.nii.gz", buffer: lab, opacity: 0 });
        this.nv.addVolume(H);
      }
      this.setWindow();
      this.hasLabels = !!lab;
      this.applyLabels();
      this.nv.onLocationChange = (d) => { this.lastLoc = d; };
      this.canvas.addEventListener("pointerdown", (e) => { this.down = [e.clientX, e.clientY]; });
      this.canvas.addEventListener("pointerup", (e) => {
        if (this.down && Math.hypot(e.clientX - this.down[0], e.clientY - this.down[1]) < 4) setTimeout(() => this.pick(), 30);
      });
      this.applyView();
    }
    setWindow() {
      const [lo, hi] = M.windows[st.win]; const v = this.nv.volumes[0];
      v.cal_min = lo; v.cal_max = hi; this.nv.updateGLVolume();
    }
    pick() {
      if (!this.lastLoc || !this.hasLabels || st.view === "3d") return;
      const vals = this.lastLoc.values || [];
      const v = vals[1] && Math.round(vals[1].value);
      const n = v > 0 ? nameOf(v) : null;
      if (n && n !== st.sel) select(n, { jump: false });
    }
    labelOpacity() {
      if (st.view === "3d") return 0;
      return (st.overlay === "labels" || this.isRef) ? st.opacity : 0;
    }
    applyLabels() {
      if (!this.hasLabels) return;
      const lut = (only) => {
        const R = [0], G = [0], B = [0], A = [0], I = [0], labels = [""];
        for (const s of M.structures) {
          const [r, g, b] = rgb(s.color);
          R.push(r); G.push(g); B.push(b); I.push(s.index); labels.push(s.name);
          A.push(!only || s.name === only ? 255 : 0);
        }
        return { R, G, B, A, I, labels };
      };
      const all = this.nv.volumes[1], hi = this.nv.volumes[2];
      all.setColormapLabel(lut(null));
      hi.setColormapLabel(lut(st.sel || "\u0000"));
      const op = this.labelOpacity();
      all.opacity = st.sel ? op * 0.3 : op;
      hi.opacity = st.sel ? Math.min(1, op * 1.35) : 0;
      this.nv.updateGLVolume();
    }
    async applyErrors() {
      if (this.isRef) return;
      if (this.errVol) { this.nv.removeVolume(this.errVol); this.errVol = null; }
      if (st.overlay === "errors" && st.sel && st.view !== "3d") {
        const buf = await getBuf(`vol/error/${this.source}/${st.sel}.nii.gz`);
        if (buf) {
          const img = await NV.NVImage.loadFromUrl({ url: "errors.nii.gz", name: "errors.nii.gz", buffer: buf });
          const c = ["tp", "fn", "fp"].map((k) => rgb(M.error_colors[k]));
          img.setColormapLabel({ R: [0, c[0][0], c[1][0], c[2][0]], G: [0, c[0][1], c[1][1], c[2][1]],
                                 B: [0, c[0][2], c[1][2], c[2][2]], A: [0, 255, 255, 255], I: [0, 1, 2, 3],
                                 labels: ["", "TP", "FN", "FP"] });
          img.opacity = Math.max(st.opacity, 0.35);
          this.nv.addVolume(img); this.errVol = img;
        }
      }
      this.nv.updateGLVolume();
    }
    meshList() {
      const src = this.source;
      if (st.overlay === "errors" && !this.isRef && st.sel) {
        return ["tp", "fn", "fp"].map((p) => ({ path: `mesh/${src}/${p}/${st.sel}.mz3`, name: `${p}`,
          rgba: [...rgb(M.error_colors[p]), 255], opacity: p === "tp" ? 0.45 : 1 }));
      }
      return M.structures.filter((s) => (this.isRef ? s.in_ref : s.models[src]))
        .map((s) => ({ path: `mesh/${src}/${s.name}.mz3`, name: s.name, rgba: [...rgb(s.color), 255],
                       opacity: st.sel ? (s.name === st.sel ? 1 : 0.16) : 1 }));
    }
    async applyMeshes() {
      const want = st.view === "3d" ? this.meshList() : [];
      const sig = want.map((w) => w.path).join("|");
      if (sig !== this.meshSig) {
        for (const m of [...this.nv.meshes]) this.nv.removeMesh(m);
        this.meshSig = sig;
        if (want.length) toast("building surfaces…", 0);
        for (const w of want) {
          const buf = await getBuf(w.path);
          if (!buf || buf.byteLength < 64) continue; // empty surface
          try {
            // NiiVue takes the mesh format from the name's extension (gzipped MZ3)
            await this.nv.addMeshFromUrl({ url: w.name + ".mz3", buffer: buf, rgba255: w.rgba,
                                           opacity: w.opacity, name: w.name + ".mz3" });
          } catch (e) { console.warn("mesh", w.path, e); }
        }
        document.getElementById("toast").hidden = true;
      } else {
        for (const m of this.nv.meshes) {
          const w = want.find((x) => x.name + ".mz3" === m.name);
          if (w) this.nv.setMeshProperty(m.id, "opacity", w.opacity);
        }
      }
      this.nv.setOpacity(0, st.view === "3d" ? 0 : 1);
      this.nv.drawScene();
    }
    applyView() {
      const T = NV.SLICE_TYPE; const nv = this.nv;
      nv.setSliceMosaicString("");
      nv.opts.show3Dcrosshair = st.view !== "3d";
      if (st.view !== "3d") nv.scene.volScaleMultiplier = 1;
      if (st.view === "multi") nv.setSliceType(T.MULTIPLANAR);
      else if (st.view === "axial") nv.setSliceType(T.AXIAL);
      else if (st.view === "coronal") nv.setSliceType(T.CORONAL);
      else if (st.view === "sagittal") nv.setSliceType(T.SAGITTAL);
      else if (st.view === "3d") { nv.setSliceType(T.RENDER); nv.setRenderAzimuthElevation(120, 15); nv.scene.volScaleMultiplier = 1.3; }
      else if (st.view === "montage") { nv.setSliceType(T.AXIAL); nv.setSliceMosaicString(mosaic()); }
      this.applyLabels();
    }
    setBadge() {
      let t = this.isRef ? "<b>Reference</b>" : `<b>${esc(this.source)}</b>`;
      const s = st.sel && S(st.sel);
      if (s && !this.isRef && s.models[this.source]) t += ` · ${esc(st.sel)} DSC ${s.models[this.source].dice.toFixed(3)}`;
      else if (s && !this.isRef) t += ` · ${esc(st.sel)}: not predicted`;
      this.badge.innerHTML = t;
    }
  }

  function mosaic() {
    let lo = Infinity, hi = -Infinity;
    const ext = st.sel ? [S(st.sel).extent] : M.structures.map((s) => s.extent);
    for (const e of ext) if (e) { lo = Math.min(lo, e.min_mm[2]); hi = Math.max(hi, e.max_mm[2]); }
    if (!Number.isFinite(lo)) return "";
    const n = 12, zs = [];
    for (let i = 0; i < n; i++) zs.push((lo + (hi - lo) * (i + 0.5) / n).toFixed(1));
    return `A ${zs.slice(0, 4).join(" ")} ; A ${zs.slice(4, 8).join(" ")} ; A ${zs.slice(8).join(" ")}`;
  }

  // ------------------------------------------------------------------ layout
  async function build() {
    const grid = $("grid"); grid.innerHTML = ""; viewers = [];
    const sources = st.layout === "compare" ? [...(M.has_ref ? ["ref"] : []), ...M.models] : [st.source];
    const n = sources.length; const cols = n <= 2 ? n : n <= 4 ? 2 : 3;
    grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    grid.style.gridTemplateRows = `repeat(${Math.ceil(n / cols)}, 1fr)`;
    for (const s of sources) viewers.push(new Viewer(s, grid));
    for (const v of viewers) await v.init();
    if (viewers.length > 1) for (const v of viewers) v.nv.broadcastTo(viewers.filter((o) => o !== v).map((o) => o.nv), { "2d": true, "3d": true });
    await refresh();
  }
  async function refresh() {
    for (const v of viewers) { v.applyView(); await v.applyErrors(); await v.applyMeshes(); v.setBadge(); }
    legend(); renderStructures(); renderMetrics();
  }
  function jumpTo(mm) {
    if (!mm) return;
    for (const v of viewers) {
      v.nv.scene.crosshairPos = v.nv.mm2frac([mm[0], mm[1], mm[2]]);
      v.nv.drawScene(); v.nv.createOnLocationChange();
    }
  }
  async function select(name, { jump = true, mm = null } = {}) {
    st.sel = name;
    const s = S(name);
    if (jump) jumpTo(mm || (s.extent && s.extent.centroid_mm));
    await refresh();
  }

  // ------------------------------------------------------------------ side panels
  function renderStructures() {
    const box = $("structures"); box.innerHTML = "";
    for (const s of M.structures) {
      const d = document.createElement("div"); d.className = "row" + (s.name === st.sel ? " sel" : "");
      const chips = M.models.map((m) => s.models[m] ? `<span title="${esc(m)} Dice">${s.models[m].dice.toFixed(2)}</span>` : `<span title="${esc(m)}: not predicted">–</span>`).join("");
      const vol = s.ref_ml !== null ? `${s.ref_ml < 10 ? s.ref_ml.toFixed(2) : s.ref_ml.toFixed(0)} mL` : "pred only";
      d.innerHTML = `<i class="sw" style="background:${s.color}"></i><span class="nm">${esc(s.name)}<small>${vol}</small></span><span class="chips">${chips}</span>`;
      d.title = s.lesion ? "lesion structure" : "";
      d.onclick = () => select(s.name).catch(fail);
      box.appendChild(d);
    }
  }
  function renderLesions() {
    const L = M.lesions || {}; const names = Object.keys(L);
    $("lesion-section").hidden = !names.length;
    const box = $("lesions"); box.innerHTML = "";
    for (const n of names) {
      const t = L[n];
      const h = document.createElement("div"); h.className = "subhead";
      h.textContent = `${n}: ${t.ref.length} in the reference`; box.appendChild(h);
      t.ref.forEach((r, i) => {
        const d = document.createElement("div"); d.className = "lesion";
        const tags = M.models.filter((m) => r.found[m] !== null && r.found[m] !== undefined).map((m) => r.found[m]
          ? `<span class="tag found" title="${esc(m)}: found, lesion Dice ${r.dice[m].toFixed(2)}">${esc(short(m))} ✓ ${r.dice[m].toFixed(2)}</span>`
          : `<span class="tag missed" title="${esc(m)}: missed">${esc(short(m))} ✗</span>`).join(" ");
        d.innerHTML = `<span>Lesion ${i + 1} <span class="who">${r.ml.toFixed(2)} mL</span></span><span class="tags">${tags}</span>`;
        d.onclick = () => { st.overlay = "errors"; setSeg("overlays", "overlay", "errors"); select(n, { mm: r.centroid_mm }).catch(fail); };
        box.appendChild(d);
      });
      for (const m of M.models) for (const r of (t.fp[m] || [])) {
        const d = document.createElement("div"); d.className = "lesion";
        d.innerHTML = `<span>False positive <span class="who">${r.ml.toFixed(2)} mL</span></span><span class="tags"><span class="tag fp">${esc(short(m))}</span></span>`;
        d.onclick = () => { st.overlay = "errors"; setSeg("overlays", "overlay", "errors"); select(n, { mm: r.centroid_mm }).catch(fail); };
        box.appendChild(d);
      }
    }
  }
  const short = (m) => m.replace(/ResEnc-?M?/i, "").replace("TotalSegmentator", "TotalSeg").trim();
  const COLS = [["dice", "Dice", 1], ["nsd", "NSD 2 mm", 1], ["hd95", "HD95", -1], ["assd", "ASSD", -1],
                ["relative_volume_difference", "RVD", 0], ["lesion_f1", "L-F1", 1], ["lesion_recall", "L-sensitivity", 1],
                ["lesion_precision", "L-precision", 1], ["false_positive_lesions", "FP lesions", -1]];
  async function renderMetrics() {
    const box = $("metrics"); const src = $("metric-source");
    if (!st.sel || !M.models.length) { box.innerHTML = `<div class="note">Select a structure to see its metrics.</div>`; src.textContent = ""; return; }
    const s = S(st.sel); const stored = M.models.some((m) => (M.results[m] || {})[st.sel]);
    let rows = {};
    if (stored) { for (const m of M.models) rows[m] = (M.results[m] || {})[st.sel] || null; src.textContent = "from the results folder"; }
    else { src.textContent = "computed live"; box.innerHTML = `<div class="note">computing…</div>`;
           for (const m of M.models) rows[m] = s.models[m] ? await liveMetrics(m, st.sel) : null; }
    const cols = COLS.filter(([k]) => M.models.some((m) => rows[m] && rows[m][k] !== undefined && rows[m][k] !== null));
    const best = {};
    for (const [k, , dir] of cols) {
      const vals = M.models.map((m) => rows[m] && rows[m][k]).filter((v) => v !== null && v !== undefined);
      if (vals.length > 1 && dir) best[k] = dir > 0 ? Math.max(...vals) : Math.min(...vals);
      if (vals.length > 1 && !dir) best[k] = vals.reduce((a, b) => (Math.abs(b) < Math.abs(a) ? b : a));
    }
    // metrics as rows, models as columns (fits the side panel for any number of metrics)
    let h = `<table class="m"><tr><th></th>${M.models.map((m) => `<th>${esc(short(m))}</th>`).join("")}</tr>`;
    for (const [k, l] of cols) {
      h += `<tr><td>${l}</td>${M.models.map((m) => {
        const v = rows[m] ? rows[m][k] : null;
        return `<td class="${v !== null && v !== undefined && v === best[k] ? "best" : ""}">${fmt(v, k)}</td>`;
      }).join("")}</tr>`;
    }
    h += `<tr><td>volume (mL)</td>${M.models.map((m) => `<td>${s.models[m] ? s.models[m].pred_ml.toFixed(1) : "–"}</td>`).join("")}</tr>`;
    h += `</table><div class="note">Reference ${s.ref_ml !== null ? s.ref_ml.toFixed(2) + " mL" : "–"} · best per row in bold · HD95 in mm · RVD = relative volume difference · L- = lesion-wise.</div>`;
    box.innerHTML = h;
  }
  function legend() {
    const L = $("legend"); const c = M.error_colors; const s = st.sel && S(st.sel);
    let h = "";
    if (st.overlay === "errors" && st.layout === "compare" || st.overlay === "errors" && st.source !== "ref") {
      h += `<span class="k"><i style="background:${c.tp}"></i>agreement (TP)</span>`;
      h += `<span class="k"><i style="background:${c.fn}"></i>missed by the model: under-segmentation (FN)</span>`;
      h += `<span class="k"><i style="background:${c.fp}"></i>added by the model: over-segmentation (FP)</span>`;
      if (!st.sel) h += `<span class="k">select a structure to see its errors</span>`;
    } else {
      h += `<span class="k">labels: ${s ? `<i style="background:${s.color}"></i>${esc(s.name)} highlighted, other structures dimmed` : "each structure in its own colour"}</span>`;
    }
    h += `<span class="k"><i style="background:#b89cf5;height:2px"></i>crosshair</span>`;
    h += `<span class="hint2">${st.view === "3d" ? "drag to rotate · scroll to zoom" : "click a structure · scroll to change slice · right-drag to window"}`
      + ` · <a href="https://github.com/niivue/niivue" target="_blank" rel="noopener">NiiVue</a></span>`;
    L.innerHTML = h;
  }

  // ------------------------------------------------------------------ controls
  function setSeg(id, attr, val) {
    for (const b of $(id).querySelectorAll("button")) b.classList.toggle("on", b.dataset[attr] === val);
  }
  function wire() {
    $("views").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; st.view = b.dataset.view; setSeg("views", "view", st.view); refresh().catch(fail); };
    $("layouts").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; st.layout = b.dataset.layout; setSeg("layouts", "layout", st.layout); $("source").disabled = st.layout === "compare"; build().catch(fail); };
    $("overlays").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; st.overlay = b.dataset.overlay; setSeg("overlays", "overlay", st.overlay);
      if (st.overlay === "errors" && st.layout === "single" && st.source === "ref" && M.models.length) { st.source = M.models[0]; $("source").value = st.source; build().catch(fail); return; }
      if (st.overlay === "errors" && !st.sel && M.structures.length) { select(M.structures[0].name).catch(fail); return; }
      refresh().catch(fail); };
    $("source").onchange = (e) => { st.source = e.target.value; build().catch(fail); };
    $("window").onchange = (e) => { st.win = e.target.value; for (const v of viewers) v.setWindow(); };
    $("opacity").oninput = (e) => { st.opacity = e.target.value / 100; for (const v of viewers) { v.applyLabels(); if (v.errVol) { v.errVol.opacity = Math.max(st.opacity, 0.35); v.nv.updateGLVolume(); } } };
    document.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      const k = { a: "axial", c: "coronal", s: "sagittal", m: "montage", r: "3d", p: "multi" }[e.key.toLowerCase()];
      if (k) { st.view = k; setSeg("views", "view", k); refresh().catch(fail); }
      if (e.key.toLowerCase() === "e") $("overlays").querySelector(`[data-overlay=${st.overlay === "errors" ? "labels" : "errors"}]`).click();
    });
  }

  async function main() {
    M = await getManifest();
    $("case").textContent = `${M.case} · ${M.shape.join("×")}${M.step > 1 ? ` (stride ${(M.steps || [M.step]).join("×")})` : ""} · ${M.models.length} model${M.models.length === 1 ? "" : "s"}`;
    const src = $("source");
    src.innerHTML = (M.has_ref ? `<option value="ref">Reference</option>` : "") + M.models.map((m) => `<option value="${esc(m)}">${esc(m)}</option>`).join("");
    st.source = M.has_ref ? "ref" : M.models[0];
    $("window").innerHTML = Object.keys(M.windows).map((w) => `<option${w === st.win ? " selected" : ""}>${w}</option>`).join("");
    if (qs.get("layout")) { st.layout = qs.get("layout"); setSeg("layouts", "layout", st.layout); }
    if (qs.get("view")) { st.view = qs.get("view"); setSeg("views", "view", st.view); }
    if (qs.get("overlay")) { st.overlay = qs.get("overlay"); setSeg("overlays", "overlay", st.overlay); }
    if (qs.get("source")) st.source = qs.get("source");
    if (st.layout === "single" && st.overlay === "errors" && st.source === "ref" && M.models.length) st.source = M.models[0];
    src.value = st.source;
    wire(); renderLesions();
    await build();
    const pre = qs.get("select") || (M.lesions && Object.keys(M.lesions).length ? Object.keys(M.lesions)[0] : null);
    if (pre && S(pre)) {
      const l = (M.lesions[pre] || {}).ref;
      await select(pre, { mm: qs.get("select") ? null : (l && l.length ? l[0].centroid_mm : null) });
    }
    window.SEK_STATUS = "ready";
    report();
  }
  function report() { // screenshot harness only
    if (!qs.get("report")) return;
    const gl = document.createElement("canvas").getContext("webgl2");
    fetch("/__report", { method: "POST", body: JSON.stringify({ status: window.SEK_STATUS, errors: window.SEK_ERRORS,
      webgl2: !!gl, viewers: viewers.length, volumes: viewers.map((v) => v.nv.volumes.length),
      meshes: viewers.map((v) => v.nv.meshes.length), sel: st.sel }) }).catch(() => {});
  }
  main().catch((e) => { fail(e); window.SEK_STATUS = "error"; report(); });
})();
