// Sidebar charts (plain SVG + canvas). Each chart builder returns
// { setTime(localTime) } so the shared time cursor can move without a rebuild.

const NS = "http://www.w3.org/2000/svg";

// Temperature ramp: one hue (orange), dark = cold -> light = hot. Shared with the globe.
export const RAMP = ["#120804", "#3d1708", "#7a2e0f", "#c24e1c", "#eb6834", "#f6a57c", "#fde9dc"];
const RAMP_RGB = RAMP.map(h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)));

export function rampRGB(t) {
  t = Math.min(1, Math.max(0, t)) * (RAMP_RGB.length - 1);
  const i = Math.min(RAMP_RGB.length - 2, Math.floor(t));
  const f = t - i, a = RAMP_RGB[i], b = RAMP_RGB[i + 1];
  return [0, 1, 2].map(c => Math.round(a[c] + (b[c] - a[c]) * f));
}

// Display units. All data stay in kelvin; conversion happens only for display.
export const UNITS = {
  F: { label: "°F", name: "Fahrenheit", conv: k => (k - 273.15) * 9 / 5 + 32 },
  C: { label: "°C", name: "Celsius", conv: k => k - 273.15 },
  K: { label: "K", name: "Kelvin", conv: k => k },
};
export const units = { key: "F" };
export const tconv = k => UNITS[units.key].conv(k);
export const uLabel = () => UNITS[units.key].label;
export const fmtT = (k, d = 0) => `${tconv(k).toFixed(d)} ${uLabel()}`;

// Habitability scale for the depth/time heatmap (kelvin): blue saturates at the
// coldest and red at the hottest temperatures people live through; neutral grey
// is a comfortable room temperature. Beyond either end the colour stays saturated.
export const HABIT = { cold: 233.15, comfort: 293.15, hot: 323.15 };   // -40 C, 20 C, 50 C
const DIV_VARS = ["--div-cold-3", "--div-cold-2", "--div-cold-1", "--div-mid", "--div-hot-1", "--div-hot-2", "--div-hot-3"];

function hexRGB(h) { h = h.trim(); return [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)); }
function lerpStops(stops, t) {
  t = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t)), f = t - i;
  return [0, 1, 2].map(c => Math.round(stops[i][c] + (stops[i + 1][c] - stops[i][c]) * f));
}
// returns T(kelvin) -> [r, g, b], using the current theme's diverging stops
export function habitScale() {
  const cs = getComputedStyle(document.documentElement);
  const st = DIV_VARS.map(v => hexRGB(cs.getPropertyValue(v)));
  const cold = st.slice(0, 4).reverse(), hot = st.slice(3);   // both run neutral -> saturated
  return T => T < HABIT.comfort
    ? lerpStops(cold, (HABIT.comfort - T) / (HABIT.comfort - HABIT.cold))
    : lerpStops(hot, (T - HABIT.comfort) / (HABIT.hot - HABIT.comfort));
}
export function habitWord(T) {
  return T < HABIT.cold ? "colder than people can live in" : T > HABIT.hot ? "hotter than people can live in"
    : "within the range people live in";
}

export function fmtLT(h) {
  h = ((h % 24) + 24) % 24;
  let hh = Math.floor(h), mm = Math.round((h - hh) * 60);
  if (mm === 60) { hh = (hh + 1) % 24; mm = 0; }
  return `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
}

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function niceTicks(lo, hi, n = 5) {
  const span = hi - lo, raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
  return { ticks: out, lo: Math.floor(lo / step) * step, hi: Math.ceil(hi / step) * step };
}

// ---------------------------------------------------------------- data access
export function interpDepth(row, depth, d) {
  if (d <= depth[0]) return row[0];
  for (let k = 1; k < depth.length; k++) {
    if (depth[k] >= d) {
      const w = (d - depth[k - 1]) / (depth[k] - depth[k - 1]);
      return row[k - 1] * (1 - w) + row[k] * w;
    }
  }
  return row[row.length - 1];
}

export function rowAt(data, L) {
  const lt = data.local_time, n = lt.length, dlt = 24 / n;
  const f = (((L % 24) + 24) % 24) / dlt;
  const i0 = Math.floor(f) % n, i1 = (i0 + 1) % n, w = f - Math.floor(f);
  return data.T[i0].map((v, k) => v * (1 - w) + data.T[i1][k] * w);
}

function periodicAt(arr, L) {
  const n = arr.length, f = (((L % 24) + 24) % 24) / (24 / n);
  const i0 = Math.floor(f) % n, w = f - Math.floor(f);
  return arr[i0] * (1 - w) + arr[(i0 + 1) % n] * w;
}

function sunAt(sun, L) {
  // nearest sample: azimuth and hour angle wrap, so do not interpolate them
  const n = sun.elevation_deg.length;
  const i = Math.round((((L % 24) + 24) % 24) / (24 / n)) % n;
  return { el: periodicAt(sun.elevation_deg, L), az: sun.azimuth_deg[i], disk: sun.disk_radius_deg[i] };
}

export function horizonAt(h, az) {
  if (!h) return 0;
  const a = h.azimuth_deg, e = h.elevation_deg, n = a.length;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n, a0 = a[i], a1 = j === 0 ? a[0] + 360 : a[j];
    const x = az < a0 ? az + 360 : az;
    if (x >= a0 && x <= a1) return e[i] + (e[j] - e[i]) * (x - a0) / Math.max(1e-6, a1 - a0);
  }
  return e[0];
}

export function sunState(data, L) {
  const s = sunAt(data.sun, L);
  const h = horizonAt(data.horizon, s.az);
  const top = s.el + s.disk;
  let state = "night";
  if (top > h) state = s.el - s.disk >= h ? "up" : "partial";
  else if (top > 0 || s.el > -s.disk) state = "terrain";
  return { ...s, horizon: h, state };
}

// ------------------------------------------------------------ chart scaffold
function frame(container, { width, height, m }) {
  container.innerHTML = "";
  container.style.height = height + "px";
  const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}` }, container);
  const tip = document.createElement("div");
  tip.className = "tip";
  container.appendChild(tip);
  const pw = width - m.l - m.r, ph = height - m.t - m.b;
  return { svg, tip, pw, ph, m };
}

function showTip(tip, container, x, y, html) {
  tip.innerHTML = html;
  tip.style.display = "block";
  const cw = container.clientWidth, tw = tip.offsetWidth;
  let left = x + 14;
  if (left + tw > cw) left = x - tw - 14;
  tip.style.left = Math.max(0, left) + "px";
  tip.style.top = Math.max(0, y - 10) + "px";
}

function ltAxis(svg, m, pw, ph) {
  for (const h of [0, 6, 12, 18, 24]) {
    const x = m.l + pw * h / 24;
    el("line", { x1: x, x2: x, y1: m.t, y2: m.t + ph, class: "gridline" }, svg);
    const t = el("text", { x, y: m.t + ph + 14, "text-anchor": "middle" }, svg);
    t.textContent = h === 12 ? "noon" : h === 0 || h === 24 ? (h === 0 ? "00:00" : "24:00") : `${String(h).padStart(2, "0")}:00`;
  }
}

function depthTicks(maxD) {
  return [0, 0.1, 0.3, 0.5, 1, 2, 3].filter(d => d <= maxD + 1e-9);
}

function dragTime(target, xToLT, onSetTime) {
  let down = false;
  const set = e => onSetTime(xToLT(e.offsetX));
  target.addEventListener("pointerdown", e => { down = true; target.setPointerCapture(e.pointerId); set(e); });
  target.addEventListener("pointermove", e => { if (down) set(e); });
  target.addEventListener("pointerup", () => { down = false; });
}

// --------------------------------------------------------------- heatmap
export function heatmap(container, data, { onSetTime, width }) {
  const height = 210, m = { l: 44, r: 10, t: 6, b: 22 };
  const { svg, tip, pw, ph } = frame(container, { width, height, m });
  const depth = data.depth, maxD = depth[depth.length - 1];
  const y = d => m.t + ph * Math.sqrt(d / maxD);
  let lo = Infinity, hi = -Infinity;
  for (const r of data.T) for (const v of r) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const color = habitScale();

  const cv = document.createElement("canvas");
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(pw * dpr); cv.height = Math.round(ph * dpr);
  Object.assign(cv.style, { left: m.l + "px", top: m.t + "px", width: pw + "px", height: ph + "px" });
  container.insertBefore(cv, svg);
  const ctx = cv.getContext("2d");
  const nlt = data.local_time.length, dlt = 24 / nlt;
  const bounds = depth.map((d, k) => k === 0 ? 0 : 0.5 * (depth[k - 1] + d)).concat([maxD]);
  for (let i = 0; i < nlt; i++) {
    const x0 = (data.local_time[i] - dlt / 2) / 24 * cv.width, x1 = (data.local_time[i] + dlt / 2) / 24 * cv.width;
    for (let k = 0; k < depth.length; k++) {
      const [r, g, b] = color(data.T[i][k]);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      const y0 = (y(bounds[k]) - m.t) * dpr, y1 = (y(bounds[k + 1]) - m.t) * dpr;
      ctx.fillRect(Math.floor(x0), Math.floor(y0), Math.ceil(x1 - x0) + 1, Math.ceil(y1 - y0) + 1);
      if (x0 < 0) ctx.fillRect(Math.floor(x0 + cv.width), Math.floor(y0), Math.ceil(x1 - x0) + 1, Math.ceil(y1 - y0) + 1);
    }
  }
  for (const h of [0, 6, 12, 18, 24]) {
    const t = el("text", { x: m.l + pw * h / 24, y: m.t + ph + 14, "text-anchor": "middle" }, svg);
    t.textContent = h === 12 ? "noon" : `${String(h).padStart(2, "0")}:00`;
  }
  for (const d of depthTicks(maxD)) {
    el("line", { x1: m.l - 4, x2: m.l, y1: y(d), y2: y(d), class: "axisline" }, svg);
    const t = el("text", { x: m.l - 7, y: y(d) + 3.5, "text-anchor": "end" }, svg);
    t.textContent = d === 0 ? "surface" : `${d} m`;
  }
  // cursor with a light halo so it reads on both saturated blue and red
  const halo = el("line", { y1: m.t, y2: m.t + ph, stroke: "#ffffff", "stroke-width": 4, opacity: 0.85 }, svg);
  const cursor = el("line", { y1: m.t, y2: m.t + ph, stroke: "#0b0b0b", "stroke-width": 1.5 }, svg);
  const hit = el("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "transparent", style: "cursor:pointer" }, svg);
  const xToLT = px => Math.min(24, Math.max(0, (px - m.l) / pw * 24));
  dragTime(hit, xToLT, onSetTime);
  hit.addEventListener("pointermove", e => {
    const L = xToLT(e.offsetX);
    const d = maxD * ((e.offsetY - m.t) / ph) ** 2;
    const v = interpDepth(rowAt(data, L), depth, d);
    showTip(tip, container, e.offsetX, e.offsetY,
      `<b>${fmtT(v)}</b>, ${habitWord(v)}<br>${fmtLT(L)} local time<br>${d < 0.005 ? "surface" : (d < 1 ? (d * 100).toFixed(0) + " cm" : d.toFixed(2) + " m")} deep`);
  });
  hit.addEventListener("pointerleave", () => { tip.style.display = "none"; });

  // colour key, linear in temperature from the cold to the hot limit
  const key = document.createElement("div");
  key.className = "habit-key";
  key.style.marginLeft = m.l + "px";
  key.style.width = pw + "px";
  const kc = document.createElement("canvas");
  kc.width = 300; kc.height = 1;
  const kx = kc.getContext("2d");
  for (let i = 0; i < kc.width; i++) {
    const [r, g, b] = color(HABIT.cold + (HABIT.hot - HABIT.cold) * (i + 0.5) / kc.width);
    kx.fillStyle = `rgb(${r},${g},${b})`;
    kx.fillRect(i, 0, 1, 1);
  }
  const pos = T => `${(T - HABIT.cold) / (HABIT.hot - HABIT.cold) * 100}%`;
  key.appendChild(kc);
  key.insertAdjacentHTML("beforeend", `<div class="ticks">
      <span style="left:0;transform:none">${fmtT(HABIT.cold)} or colder</span>
      <span style="left:${pos(HABIT.comfort)}">${fmtT(HABIT.comfort)}</span>
      <span style="left:100%;transform:translateX(-100%)">${fmtT(HABIT.hot)} or hotter</span></div>`);
  container.style.height = "";
  container.appendChild(key);
  return {
    range: [lo, hi],
    setTime(L) {
      const x = m.l + pw * (((L % 24) + 24) % 24) / 24;
      for (const ln of [halo, cursor]) { ln.setAttribute("x1", x); ln.setAttribute("x2", x); }
    },
  };
}

// ------------------------------------------------------------ depth lines
export const LINE_DEPTHS = [0, 0.1, 0.3, 1.0];
const LINE_VARS = ["--series-1", "--series-2", "--series-3", "--series-4"];
const depthName = d => d === 0 ? "Surface" : d < 1 ? `${Math.round(d * 100)} cm` : `${d} m`;

export function lineLegendHTML() {
  return LINE_DEPTHS.map((d, i) => `<span><i style="background:var(${LINE_VARS[i]})"></i>${depthName(d)}</span>`).join("");
}

export function lineChart(container, data, { onSetTime, width }) {
  const height = 200, m = { l: 40, r: 50, t: 8, b: 22 };
  const { svg, tip, pw, ph } = frame(container, { width, height, m });
  const series = LINE_DEPTHS.map(d => data.T.map(r => tconv(interpDepth(r, data.depth, d))));
  const all = series.flat();
  const nt = niceTicks(Math.min(...all), Math.max(...all), 5);
  const x = L => m.l + pw * L / 24, y = v => m.t + ph * (1 - (v - nt.lo) / (nt.hi - nt.lo));
  ltAxis(svg, m, pw, ph);
  for (const v of nt.ticks) {
    el("line", { x1: m.l, x2: m.l + pw, y1: y(v), y2: y(v), class: "gridline" }, svg);
    const t = el("text", { x: m.l - 6, y: y(v) + 3.5, "text-anchor": "end" }, svg);
    t.textContent = v;
  }
  const unit = el("text", { x: m.l - 6, y: y(nt.ticks[nt.ticks.length - 1]) - 9, "text-anchor": "end" }, svg);
  unit.textContent = uLabel();
  const lt = data.local_time;
  const labelY = [];
  series.forEach((s, i) => {
    const pts = s.map((v, j) => `${x(lt[j]).toFixed(1)},${y(v).toFixed(1)}`);
    pts.push(`${x(24).toFixed(1)},${y(s[0]).toFixed(1)}`);
    el("polyline", { points: pts.join(" "), fill: "none", stroke: `var(${LINE_VARS[i]})`, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    labelY.push([y(s[0]), i]);
  });
  // direct labels at the right end, de-overlapped
  labelY.sort((a, b) => a[0] - b[0]);
  for (let k = 1; k < labelY.length; k++) labelY[k][0] = Math.max(labelY[k][0], labelY[k - 1][0] + 12);
  for (const [yy, i] of labelY) {
    const t = el("text", { x: m.l + pw + 6, y: yy + 3.5 }, svg);
    t.textContent = depthName(LINE_DEPTHS[i]);
  }
  const hover = el("line", { y1: m.t, y2: m.t + ph, class: "hover", visibility: "hidden" }, svg);
  const cursor = el("line", { y1: m.t, y2: m.t + ph, class: "cursor" }, svg);
  const dots = series.map((_, i) => el("circle", { r: 4, fill: `var(${LINE_VARS[i]})`, stroke: "var(--surface-1)", "stroke-width": 2 }, svg));
  const hit = el("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "transparent", style: "cursor:pointer" }, svg);
  const xToLT = px => Math.min(24, Math.max(0, (px - m.l) / pw * 24));
  dragTime(hit, xToLT, onSetTime);
  hit.addEventListener("pointermove", e => {
    const L = xToLT(e.offsetX);
    hover.setAttribute("x1", x(L)); hover.setAttribute("x2", x(L)); hover.setAttribute("visibility", "visible");
    const vals = series.map(s => periodicAt(s, L));
    showTip(tip, container, e.offsetX, e.offsetY, `<b>${fmtLT(L)}</b> local time<br>` +
      vals.map((v, i) => `<span class="sw" style="background:var(${LINE_VARS[i]})"></span>${depthName(LINE_DEPTHS[i])}: <b>${v.toFixed(0)} ${uLabel()}</b>`).join("<br>"));
  });
  hit.addEventListener("pointerleave", () => { tip.style.display = "none"; hover.setAttribute("visibility", "hidden"); });
  return {
    setTime(L) {
      L = ((L % 24) + 24) % 24;
      cursor.setAttribute("x1", x(L)); cursor.setAttribute("x2", x(L));
      series.forEach((s, i) => { dots[i].setAttribute("cx", x(L)); dots[i].setAttribute("cy", y(periodicAt(s, L))); });
    },
  };
}

// ----------------------------------------------------------- depth profile
export function profileChart(container, data, { width }) {
  const height = 230, m = { l: 52, r: 14, t: 8, b: 30 };
  const { svg, tip, pw, ph } = frame(container, { width, height, m });
  const depth = data.depth, maxD = depth[depth.length - 1];
  const nd = depth.length;
  const mn = depth.map((_, k) => tconv(Math.min(...data.T.map(r => r[k]))));
  const mx = depth.map((_, k) => tconv(Math.max(...data.T.map(r => r[k]))));
  const mean = depth.map((_, k) => tconv(data.T.reduce((s, r) => s + r[k], 0) / data.T.length));
  const nt = niceTicks(Math.min(...mn), Math.max(...mx), 5);
  const x = v => m.l + pw * (v - nt.lo) / (nt.hi - nt.lo), y = d => m.t + ph * Math.sqrt(d / maxD);
  for (const v of nt.ticks) {
    el("line", { x1: x(v), x2: x(v), y1: m.t, y2: m.t + ph, class: "gridline" }, svg);
    const t = el("text", { x: x(v), y: m.t + ph + 14, "text-anchor": "middle" }, svg);
    t.textContent = v;
  }
  const xl = el("text", { x: m.l + pw, y: m.t + ph + 27, "text-anchor": "end" }, svg);
  xl.textContent = `temperature (${uLabel()})`;
  for (const d of depthTicks(maxD)) {
    el("line", { x1: m.l, x2: m.l + pw, y1: y(d), y2: y(d), class: "gridline" }, svg);
    const t = el("text", { x: m.l - 6, y: y(d) + 3.5, "text-anchor": "end" }, svg);
    t.textContent = d === 0 ? "surface" : `${d} m`;
  }
  const band = mx.map((v, k) => `${x(v).toFixed(1)},${y(depth[k]).toFixed(1)}`)
    .concat(mn.map((v, k) => [v, k]).reverse().map(([v, k]) => `${x(v).toFixed(1)},${y(depth[k]).toFixed(1)}`));
  el("polygon", { points: band.join(" "), fill: "var(--band)", stroke: "none" }, svg);
  el("polyline", { points: mean.map((v, k) => `${x(v).toFixed(1)},${y(depth[k]).toFixed(1)}`).join(" "),
    fill: "none", stroke: "var(--text-muted)", "stroke-width": 1.5, "stroke-dasharray": "4 3" }, svg);
  const line = el("polyline", { fill: "none", stroke: "var(--series-1)", "stroke-width": 2, "stroke-linejoin": "round" }, svg);
  const hover = el("line", { x1: m.l, x2: m.l + pw, class: "hover", visibility: "hidden" }, svg);
  let cur = mean;
  const hit = el("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", e => {
    const d = Math.min(maxD, Math.max(0, maxD * ((e.offsetY - m.t) / ph) ** 2));
    hover.setAttribute("y1", y(d)); hover.setAttribute("y2", y(d)); hover.setAttribute("visibility", "visible");
    const f = a => interpDepth(a, depth, d).toFixed(0);
    showTip(tip, container, e.offsetX, e.offsetY,
      `<b>${d < 0.005 ? "Surface" : d < 1 ? (d * 100).toFixed(0) + " cm" : d.toFixed(2) + " m"}</b><br>now: <b>${f(cur)} ${uLabel()}</b><br>` +
      `day range: ${f(mn)} to ${f(mx)} ${uLabel()}<br>day mean: ${f(mean)} ${uLabel()}`);
  });
  hit.addEventListener("pointerleave", () => { tip.style.display = "none"; hover.setAttribute("visibility", "hidden"); });
  return {
    setTime(L) {
      cur = rowAt(data, L).map(tconv);
      line.setAttribute("points", cur.map((v, k) => `${x(v).toFixed(1)},${y(depth[k]).toFixed(1)}`).join(" "));
    },
  };
}

// ---------------------------------------------------- sky: horizon + sun path
export function skyChart(container, data, opts) {
  return data.horizon ? horizonChart(container, data, opts) : sunElevationChart(container, data, opts);
}

// flat-horizon sites: Sun elevation against local time (aligned with the other time charts)
function sunElevationChart(container, data, { onSetTime, width }) {
  const height = 170, m = { l: 40, r: 50, t: 8, b: 22 };
  const { svg, tip, pw, ph } = frame(container, { width, height, m });
  const sun = data.sun, lt = data.local_time;
  const top = Math.min(90, Math.ceil((Math.max(...sun.elevation_deg) + 5) / 10) * 10);
  const nt = niceTicks(-10, top, 5);
  const x = L => m.l + pw * L / 24, y = e => m.t + ph * (1 - (Math.max(e, nt.lo) - nt.lo) / (nt.hi - nt.lo));
  ltAxis(svg, m, pw, ph);
  el("rect", { x: m.l, y: y(0), width: pw, height: m.t + ph - y(0), fill: "var(--terrain)", opacity: 0.6 }, svg);
  for (const v of nt.ticks) {
    el("line", { x1: m.l, x2: m.l + pw, y1: y(v), y2: y(v), class: v === 0 ? "axisline" : "gridline" }, svg);
    const t = el("text", { x: m.l - 6, y: y(v) + 3.5, "text-anchor": "end" }, svg);
    t.textContent = `${v}°`;
  }
  const e = sun.elevation_deg;
  const pts = e.map((v, j) => `${x(lt[j]).toFixed(1)},${y(v).toFixed(1)}`);
  pts.push(`${x(24).toFixed(1)},${y(e[0]).toFixed(1)}`);
  el("polyline", { points: pts.join(" "), fill: "none", stroke: "var(--series-4)", "stroke-width": 2, "stroke-linejoin": "round" }, svg);
  const lab = el("text", { x: m.l + pw + 6, y: y(0) + 3.5 }, svg);
  lab.textContent = "horizon";
  const cursor = el("line", { y1: m.t, y2: m.t + ph, class: "cursor" }, svg);
  const dot = el("circle", { r: 5, fill: "var(--series-4)", stroke: "var(--surface-1)", "stroke-width": 2 }, svg);
  const hit = el("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "transparent", style: "cursor:pointer" }, svg);
  const xToLT = px => Math.min(24, Math.max(0, (px - m.l) / pw * 24));
  dragTime(hit, xToLT, onSetTime);
  hit.addEventListener("pointermove", ev => {
    const L = xToLT(ev.offsetX), s = sunAt(sun, L);
    showTip(tip, container, ev.offsetX, ev.offsetY,
      `<b>${fmtLT(L)}</b> local time<br>Sun elevation <b>${s.el.toFixed(1)}°</b><br>azimuth ${s.az.toFixed(0)}°`);
  });
  hit.addEventListener("pointerleave", () => { tip.style.display = "none"; });
  return {
    setTime(L) {
      L = ((L % 24) + 24) % 24;
      cursor.setAttribute("x1", x(L)); cursor.setAttribute("x2", x(L));
      dot.setAttribute("cx", x(L)); dot.setAttribute("cy", y(periodicAt(e, L)));
    },
  };
}

// terrain sites: horizon silhouette and Sun track by compass direction
function horizonChart(container, data, { width }) {
  const height = 190, m = { l: 36, r: 10, t: 8, b: 22 };
  const { svg, tip, pw, ph } = frame(container, { width, height, m });
  const sun = data.sun, h = data.horizon;
  const hz = h ? h.elevation_deg : [0];
  const maxSun = Math.max(...sun.elevation_deg);
  const top = Math.min(90, Math.max(maxSun + 2, Math.max(...hz) + 2, 5));
  const bot = Math.max(-90, Math.min(-2, Math.min(...hz) - 1));
  const nt = niceTicks(bot, top, 5);
  const x = a => m.l + pw * a / 360, y = e => m.t + ph * (1 - (e - nt.lo) / (nt.hi - nt.lo));
  for (const v of nt.ticks) {
    el("line", { x1: m.l, x2: m.l + pw, y1: y(v), y2: y(v), class: v === 0 ? "axisline" : "gridline" }, svg);
    const t = el("text", { x: m.l - 6, y: y(v) + 3.5, "text-anchor": "end" }, svg);
    t.textContent = `${v}°`;
  }
  ["N", "E", "S", "W", "N"].forEach((c, i) => {
    const t = el("text", { x: x(i * 90), y: m.t + ph + 14, "text-anchor": "middle" }, svg);
    t.textContent = c;
  });
  // terrain silhouette
  let pts;
  if (h) {
    const a = h.azimuth_deg, e = h.elevation_deg;
    const e0 = horizonAt(h, 0);
    pts = [[0, e0], ...a.map((v, i) => [v, e[i]]), [360, e0]];
  } else pts = [[0, 0], [360, 0]];
  const poly = pts.map(([a, e]) => `${x(a).toFixed(1)},${y(Math.max(e, nt.lo)).toFixed(1)}`);
  poly.push(`${x(360)},${y(nt.lo)}`, `${x(0)},${y(nt.lo)}`);
  el("polygon", { points: poly.join(" "), fill: "var(--terrain)", stroke: "var(--axis)", "stroke-width": 1 }, svg);
  // sun path (broken where azimuth wraps)
  const segs = [[]];
  const n = sun.azimuth_deg.length;
  for (let i = 0; i <= n; i++) {
    const j = i % n, a = sun.azimuth_deg[j], e = sun.elevation_deg[j];
    const last = segs[segs.length - 1];
    if (e < nt.lo) { if (last.length) segs.push([]); continue; }
    if (last.length && Math.abs(a - last[last.length - 1][0]) > 180) segs.push([]);
    segs[segs.length - 1].push([a, e]);
  }
  for (const s of segs) if (s.length > 1)
    el("polyline", { points: s.map(([a, e]) => `${x(a).toFixed(1)},${y(e).toFixed(1)}`).join(" "),
      fill: "none", stroke: "var(--series-4)", "stroke-width": 2, "stroke-linejoin": "round" }, svg);
  const lab = el("text", { x: m.l + pw - 4, y: m.t + 12, "text-anchor": "end" }, svg);
  lab.textContent = "grey: terrain horizon · yellow: Sun path";
  const dot = el("circle", { r: 5, fill: "var(--series-4)", stroke: "var(--surface-1)", "stroke-width": 2 }, svg);
  const hit = el("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", e => {
    const a = Math.min(360, Math.max(0, (e.offsetX - m.l) / pw * 360));
    showTip(tip, container, e.offsetX, e.offsetY, `azimuth ${a.toFixed(0)}°<br>horizon: <b>${horizonAt(h, a % 360).toFixed(1)}°</b>`);
  });
  hit.addEventListener("pointerleave", () => { tip.style.display = "none"; });
  return {
    setTime(L) {
      const s = sunAt(sun, L);
      dot.setAttribute("cx", x(s.az));
      dot.setAttribute("cy", y(Math.max(s.el, nt.lo)));
      dot.setAttribute("opacity", s.el < nt.lo ? 0.25 : 1);
    },
  };
}
