import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";
import {
  RAMP, rampRGB, fmtLT, HABIT, UNITS, units, tconv, uLabel, fmtT, heatmap, lineChart, lineLegendHTML, profileChart, skyChart, rowAt, interpDepth, sunState,
} from "./charts.js";

const C_MIN = 40, C_MAX = 720;            // globe colour scale (K)
const $ = id => document.getElementById(id);

// temperature unit: Fahrenheit by default; the viewer's choice is remembered in this browser
try { const u = localStorage.getItem("mercury-unit"); if (UNITS[u]) units.key = u; } catch (e) { /* storage unavailable */ }

const state = {
  meta: null, tau: 0, playing: false, depth: 0,
  sel: null,          // {lat, lon}
  data: null,         // point payload
  exact: false,
  charts: [],
};

// ------------------------------------------------------------------- scene
const canvas = $("globe");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07080a);
const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 200);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 1.25;
controls.maxDistance = 8;
controls.rotateSpeed = 0.6;
controls.enablePan = false;

// lat/lon <-> body-fixed direction (+y = north; east longitude counter-clockwise seen from north)
function dirOf(latDeg, lonDeg) {
  const la = THREE.MathUtils.degToRad(latDeg), lo = THREE.MathUtils.degToRad(lonDeg);
  return new THREE.Vector3(Math.cos(la) * Math.cos(lo), Math.sin(la), -Math.cos(la) * Math.sin(lo));
}
function latLonOf(v) {
  const n = v.clone().normalize();
  return { lat: THREE.MathUtils.radToDeg(Math.asin(n.y)), lon: (THREE.MathUtils.radToDeg(Math.atan2(-n.z, n.x)) + 360) % 360 };
}
camera.position.copy(dirOf(20, 0).multiplyScalar(4.4));
controls.addEventListener("end", () => writeHash());

// star field
{
  const n = 1800, pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const v = new THREE.Vector3().randomDirection().multiplyScalar(60 + Math.random() * 40);
    pos.set([v.x, v.y, v.z], i * 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0x9a9a96, size: 1.2, sizeAttenuation: false })));
}

// colour look-up texture
const lutData = new Uint8Array(256 * 4);
for (let i = 0; i < 256; i++) { const [r, g, b] = rampRGB(i / 255); lutData.set([r, g, b, 255], i * 4); }
const lut = new THREE.DataTexture(lutData, 256, 1);
lut.magFilter = lut.minFilter = THREE.LinearFilter;
lut.needsUpdate = true;

const blank = new THREE.DataTexture(new Uint8Array(4), 1, 1);
blank.needsUpdate = true;

const uniforms = {
  atlasA: { value: blank }, atlasB: { value: blank }, mixAB: { value: 0 },
  capNorth: { value: new THREE.Vector4() }, capSouth: { value: new THREE.Vector4() },
  capLat: { value: new THREE.Vector2(80, 80) }, capCol: { value: new THREE.Vector2(0, 0) },
  hasNorth: { value: 0 }, hasSouth: { value: 0 }, ready: { value: 0 },
  slopeTex: { value: blank }, hasSlopes: { value: 0 }, gMax: { value: 0.4 },
  gNodes: { value: new Array(9).fill(0) }, nG: { value: 1 },
  tileW: { value: 180 }, tileH: { value: 42 }, capY0: { value: 42 },
  tLo: { value: 20 }, tStep: { value: 2.835 }, cMin: { value: C_MIN }, cMax: { value: C_MAX }, lut: { value: lut },
};

const material = new THREE.ShaderMaterial({
  uniforms,
  vertexShader: /* glsl */`
    varying vec3 vPos; varying vec3 vNormalView; varying vec2 vUv;
    void main() {
      vPos = position;
      vUv = uv;
      vNormalView = normalize(normalMatrix * normal);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`,
  fragmentShader: /* glsl */`
    uniform sampler2D atlasA, atlasB, lut, slopeTex;
    uniform float mixAB, hasNorth, hasSouth, ready, tLo, tStep, cMin, cMax;
    uniform float hasSlopes, gMax, tileW, tileH, capY0;
    uniform float gNodes[9];
    uniform int nG;
    uniform vec4 capNorth, capSouth;   // per cap: x0, y0, dx (km), n (cells per side)
    uniform vec2 capLat, capCol;        // north, south: product edge latitude (deg), atlas column
    varying vec3 vPos; varying vec3 vNormalView; varying vec2 vUv;
    const float R = 2439.4;

    // bilinear interpolation over valid (non-zero) texels inside an atlas region
    vec2 bil(sampler2D a, vec2 px, ivec4 reg, bool wrapX) {
      vec2 q = px - 0.5; ivec2 i0 = ivec2(floor(q)); vec2 f = q - vec2(i0);
      float s = 0.0, w = 0.0;
      for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
        ivec2 p = i0 + ivec2(dx, dy);
        p.x = wrapX ? ((p.x % reg.z) + reg.z) % reg.z : clamp(p.x, 0, reg.z - 1);
        p.y = clamp(p.y, 0, reg.w - 1);
        float v = texelFetch(a, p + reg.xy, 0).r * 255.0;
        float ww = (dx == 0 ? 1.0 - f.x : f.x) * (dy == 0 ? 1.0 - f.y : f.y);
        if (v > 0.5) { s += ww * (tLo + (v - 1.0) * tStep); w += ww; }
      }
      return vec2(w > 0.001 ? s / w : 0.0, w);
    }

    // lower node and weight of a slope value on the (non-uniform) table axis
    void gIndex(float g, out int i, out float w) {
      i = 0; w = 0.0;
      if (nG < 2) return;
      if (g <= gNodes[0]) return;
      if (g >= gNodes[nG - 1]) { i = nG - 2; w = 1.0; return; }
      for (int k = 0; k < 8; k++) {
        if (k < nG - 1 && g <= gNodes[k + 1]) { i = k; w = (g - gNodes[k]) / (gNodes[k + 1] - gNodes[k]); return; }
      }
    }

    float tileT(sampler2D a, int lo, int la, int iy, int ix) {
      float v = texelFetch(a, ivec2(ix * int(tileW) + lo, iy * int(tileH) + la), 0).r * 255.0;
      return tLo + (v - 1.0) * tStep;
    }

    // slope-aware temperature: 4D interpolation in (lat, lon, gy, gx) of the frame's response tiles
    float slopeT(sampler2D a, float latd, float lond, vec2 g) {
      float fa = min(abs(latd), 2.0 * (tileH - 1.0) - 0.001) * 0.5;
      int ia = int(floor(fa)); float wa = fa - float(ia);
      float fo = mod(lond, 360.0) * 0.5;
      int io = int(floor(fo)); float wo = fo - float(io);
      int nW = int(tileW); io = io % nW; int io1 = (io + 1) % nW;
      if (latd < 0.0) g.y = -g.y;   // T(-lat, gx, gy) = T(lat, gx, -gy)
      int ix, iy; float wx, wy;
      gIndex(g.x, ix, wx); gIndex(g.y, iy, wy);
      int ix1 = min(ix + 1, nG - 1), iy1 = min(iy + 1, nG - 1);
      float s = 0.0;
      for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
        int jy = dy == 0 ? iy : iy1, jx = dx == 0 ? ix : ix1;
        float wg = (dy == 0 ? 1.0 - wy : wy) * (dx == 0 ? 1.0 - wx : wx);
        if (wg == 0.0) continue;
        float t00 = tileT(a, io, ia, jy, jx), t01 = tileT(a, io1, ia, jy, jx);
        float t10 = tileT(a, io, ia + 1, jy, jx), t11 = tileT(a, io1, ia + 1, jy, jx);
        s += wg * mix(mix(t00, t01, wo), mix(t10, t11, wo), wa);
      }
      return s;
    }

    float sampleT(sampler2D a, float lat, float lonS, vec2 g) {
      float alat = abs(lat);
      bool north = lat > 0.0;
      if (alat > radians((north ? capLat.x : capLat.y) - 0.1)) {
        vec4 c = north ? capNorth : capSouth;
        if ((north ? hasNorth : hasSouth) > 0.5) {
          float rho = 2.0 * R * tan((1.5707963 - alat) * 0.5);
          float x = rho * sin(lonS), y = (north ? -1.0 : 1.0) * rho * cos(lonS);
          vec2 px = vec2((x - c.x) / c.z + 0.5, (c.y - y) / c.z + 0.5);
          int n = int(c.w);
          vec2 r = bil(a, px, ivec4(int(north ? capCol.x : capCol.y), int(capY0), n, n), false);
          if (r.y > 0.5) return r.x;
        }
      }
      return slopeT(a, degrees(lat), degrees(lonS), g);
    }

    void main() {
      vec3 p = normalize(vPos);
      float lat = asin(clamp(p.y, -1.0, 1.0));
      float lon = atan(-p.z, p.x);
      vec3 col = vec3(0.16);
      if (ready > 0.5) {
        // local slope: sphere uv runs from 180 deg W, so shift by half a turn; gradients of the
        // continuous uv avoid a mip-level seam at 0 deg E
        vec2 g = vec2(0.0);
        if (hasSlopes > 0.5) {
          vec2 st = vec2(fract(vUv.x + 0.5), 1.0 - vUv.y);
          vec2 e = textureGrad(slopeTex, st, dFdx(vUv), dFdy(vUv)).rg;
          g = (e * 255.0 - 128.0) / 127.0 * gMax;
        }
        float T = mix(sampleT(atlasA, lat, lon, g), sampleT(atlasB, lat, lon, g), mixAB);
        col = texture2D(lut, vec2(clamp((T - cMin) / (cMax - cMin), 0.0, 1.0), 0.5)).rgb;
      }
      // graticule every 30 degrees, plus the boundary of each polar terrain model
      float latd = degrees(lat), lond = degrees(lon);
      float capd = latd > 0.0 ? capLat.x : capLat.y;
      float gl = abs(fract(latd / 30.0 + 0.5) - 0.5) * 30.0, go = abs(fract(lond / 30.0 + 0.5) - 0.5) * 30.0;
      float lineLat = 1.0 - smoothstep(0.0, 1.2 * fwidth(latd), gl);
      float lineLon = (1.0 - smoothstep(0.0, 1.2 * fwidth(lond), go)) * step(abs(latd), capd);
      float ring = 1.0 - smoothstep(0.0, 1.2 * fwidth(latd), abs(abs(latd) - capd));
      col = mix(col, vec3(1.0), 0.22 * max(lineLat, lineLon));
      col = mix(col, vec3(0.55, 0.75, 1.0), 0.35 * ring);
      // gentle limb darkening for depth perception
      float limb = 0.72 + 0.28 * clamp(vNormalView.z, 0.0, 1.0);
      gl_FragColor = vec4(col * limb, 1.0);
    }`,
});
const globe = new THREE.Mesh(new THREE.SphereGeometry(1, 192, 128), material);
scene.add(globe);

// spin axis: segments out of each pole, labelled N / S at a constant screen size
{
  const color = 0xc9d3e0;
  const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.85 });
  const pts = [[1, 1.4], [-1, -1.4]].flatMap(([a, b]) => [new THREE.Vector3(0, a, 0), new THREE.Vector3(0, b, 0)]);
  scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), mat));
  for (const [txt, y] of [["N", 1.47], ["S", -1.47]]) {
    const c = document.createElement("canvas");
    c.width = c.height = 64;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#c9d3e0";
    ctx.font = "600 44px system-ui, sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(txt, 32, 34);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, sizeAttenuation: false, transparent: true }));
    s.scale.setScalar(0.045);
    s.position.set(0, y, 0);
    scene.add(s);
  }
}

// selection marker and subsolar marker
function surfaceMarker(color, r0, r1) {
  const g = new THREE.Group();
  const ring = new THREE.Mesh(new THREE.RingGeometry(r0, r1, 48),
    new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, transparent: true, opacity: 0.95 }));
  g.add(ring);
  g.visible = false;
  scene.add(g);
  return g;
}
function placeMarker(m, lat, lon) {
  const d = dirOf(lat, lon);
  m.userData.dir = d;
  m.position.copy(d.clone().multiplyScalar(1.003));
  m.lookAt(d.clone().multiplyScalar(2));
  m.visible = true;
}
const selMarker = surfaceMarker(0xffffff, 0.014, 0.022);
const selDot = new THREE.Mesh(new THREE.CircleGeometry(0.006, 24), new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide }));
selMarker.add(selDot);
const sunMarker = surfaceMarker(0xffd27a, 0.02, 0.026);
{
  const rays = new THREE.Group();
  for (let i = 0; i < 8; i++) {
    const r = new THREE.Mesh(new THREE.PlaneGeometry(0.004, 0.012), new THREE.MeshBasicMaterial({ color: 0xffd27a, side: THREE.DoubleSide }));
    const a = i * Math.PI / 4;
    r.position.set(Math.cos(a) * 0.034, Math.sin(a) * 0.034, 0);
    r.rotation.z = a - Math.PI / 2;
    rays.add(r);
  }
  sunMarker.add(rays);
}

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe($("stage"));

// ------------------------------------------------------------------ frames
const frames = new Map();   // key -> {tex, bitmap, pixels}
const pending = new Map();
const MAX_FRAMES = 160;

function frameKey(depth, k) { return `${depth}|${k}`; }

function loadFrame(depth, k) {
  const key = frameKey(depth, k);
  if (frames.has(key)) return Promise.resolve(frames.get(key));
  if (pending.has(key)) return pending.get(key);
  const p = fetch(`/api/frame?depth=${depth}&k=${k}&v=${state.meta.version}`)
    .then(r => { if (!r.ok) throw new Error(`frame ${r.status}`); return r.blob(); })
    .then(b => createImageBitmap(b, { premultiplyAlpha: "none", colorSpaceConversion: "none" }))
    .then(bmp => {
      const tex = new THREE.Texture(bmp);
      tex.flipY = false;
      tex.magFilter = tex.minFilter = THREE.NearestFilter;
      tex.generateMipmaps = false;
      tex.needsUpdate = true;
      const f = { tex, bitmap: bmp, pixels: null };
      frames.set(key, f);
      pending.delete(key);
      if (frames.size > MAX_FRAMES) {
        for (const [kk, ff] of frames) {
          if (kk.startsWith(`${state.depth}|`)) continue;
          ff.tex.dispose(); ff.bitmap.close?.(); frames.delete(kk);
          if (frames.size <= MAX_FRAMES) break;
        }
      }
      return f;
    })
    .catch(e => { pending.delete(key); throw e; });
  pending.set(key, p);
  return p;
}

let prefetchToken = 0;
async function prefetchAll(depth, startK) {
  const token = ++prefetchToken, N = state.meta.n_frames;
  const order = Array.from({ length: N }, (_, i) => (startK + i) % N);
  let next = 0;
  const worker = async () => {
    while (next < order.length && token === prefetchToken) {
      const k = order[next++];
      try { await loadFrame(depth, k); } catch (e) { /* retried on demand */ }
    }
  };
  await Promise.all([worker(), worker(), worker()]);
}

function frameFor(tau) {
  const N = state.meta.n_frames, f = (((tau % 1) + 1) % 1) * N;
  const kA = Math.floor(f) % N, kB = (kA + 1) % N;
  return { kA, kB, mix: f - Math.floor(f) };
}

function updateGlobeTexture() {
  if (!state.meta) return;
  const { kA, kB, mix } = frameFor(state.tau);
  const fa = frames.get(frameKey(state.depth, kA)), fb = frames.get(frameKey(state.depth, kB));
  if (fa && fb) {
    uniforms.atlasA.value = fa.tex; uniforms.atlasB.value = fb.tex;
    uniforms.mixAB.value = mix;
    uniforms.ready.value = 1;
    $("loading").style.display = "none";
  } else {
    loadFrame(state.depth, kA).catch(() => {});
    loadFrame(state.depth, kB).catch(() => {});
  }
}

// temperature under the mouse, from the server (one request in flight at a time)
const hover = { busy: false, want: null, key: "", T: null };
// GET JSON from the API. Gateway errors (server restarting) are retried a few
// times; a non-JSON body becomes a readable error instead of a parse failure.
async function fetchJSON(url, retries = 3) {
  for (let i = 0; ; i++) {
    const r = await fetch(url);
    if ([502, 503, 504].includes(r.status) && i < retries) {
      await new Promise(res => setTimeout(res, 1500 * (i + 1)));
      continue;
    }
    let d;
    try { d = await r.json(); } catch (e) {
      throw new Error(`server unavailable (HTTP ${r.status}); please try again`);
    }
    if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
    return d;
  }
}

function requestHoverT(lat, lon) {
  hover.want = { lat, lon, t: state.tau, depth: state.depth };
  if (hover.busy) return;
  const q = hover.want;
  hover.want = null;
  hover.busy = true;
  fetch(`/api/sample?lat=${q.lat}&lon=${q.lon}&t=${q.t}&depth=${q.depth}`)
    .then(r => r.json())
    .then(d => { hover.T = d.T; hover.key = `${q.lat.toFixed(2)},${q.lon.toFixed(2)}`; updateHoverTip(); })
    .catch(() => {})
    .finally(() => { hover.busy = false; if (hover.want) requestHoverT(hover.want.lat, hover.want.lon); });
}

// static ~2 km slope texture (east/north slope per pixel), loaded once
async function loadSlopes() {
  const r = await fetch(`/api/slopes.png?v=${state.meta.version}`);
  if (!r.ok) return;
  const bmp = await createImageBitmap(await r.blob(), { premultiplyAlpha: "none", colorSpaceConversion: "none" });
  const tex = new THREE.Texture(bmp);
  tex.flipY = false;
  tex.wrapS = THREE.RepeatWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  tex.needsUpdate = true;
  uniforms.slopeTex.value = tex;
  uniforms.hasSlopes.value = 1;
  console.log(`slope texture ${bmp.width}x${bmp.height} loaded`);
}

// -------------------------------------------------------------------- time
function orbitAt(tau) {
  const o = state.meta.orbit, t = ((tau % 1) + 1) % 1;
  let i = o.tau.findIndex(v => v > t); if (i < 0) i = o.tau.length; i = Math.max(1, i) - 1;
  return { subsolar: o.subsolar_lon[i], r: o.sun_distance_au[i] };
}
function localTimeAt(lon, tau) { return ((12 + lon / 15 + 24 * tau) % 24 + 24) % 24; }

function setTau(t, fromUser = false) {
  state.tau = ((t % 1) + 1) % 1;
  state.dirty = true;
  if (fromUser) writeHash();
}

function updateTimeUI() {
  const m = state.meta, o = orbitAt(state.tau);
  const days = state.tau * m.solar_day_earth_days;
  $("time").value = state.tau;
  const ss = o.subsolar > 180 ? `${(360 - o.subsolar).toFixed(0)}° W` : `${o.subsolar.toFixed(0)}° E`;
  $("timeinfo").innerHTML = `Day ${days.toFixed(1)} of ${m.solar_day_earth_days.toFixed(1)} <span class="sub">(Earth days)</span><br>` +
    `<span class="sub">Sun overhead at ${ss} · ${o.r.toFixed(3)} AU</span>`;
  placeMarker(sunMarker, 0, o.subsolar);
  if (state.data) {
    const L = localTimeAt(state.sel.lon, state.tau);
    for (const c of state.charts) c.setTime(L);
    updateNow(L);
  }
}

$("time").addEventListener("input", e => { setTau(+e.target.value, true); });
const PLAY = '<svg viewBox="0 0 14 14"><path d="M3 1.5v11l9.5-5.5z" fill="currentColor"/></svg>';
const PAUSE = '<svg viewBox="0 0 14 14"><path d="M3 1.5h3v11H3zM8 1.5h3v11H8z" fill="currentColor"/></svg>';
function setPlaying(p) {
  state.playing = p;
  $("play").innerHTML = p ? PAUSE : PLAY;
  $("play").setAttribute("aria-label", p ? "Pause" : "Play");
  if (!p) writeHash();
}
$("play").addEventListener("click", () => setPlaying(!state.playing));
document.addEventListener("keydown", e => {
  if (e.code === "Space" && !["INPUT", "SELECT", "BUTTON"].includes(document.activeElement.tagName)) { e.preventDefault(); setPlaying(!state.playing); }
});

// -------------------------------------------------------------- depth chips
function buildDepthBar() {
  const bar = $("depthbar");
  for (const d of state.meta.globe_depths) {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = d === 0 ? "Surface" : d < 1 ? `${Math.round(d * 100)} cm` : `${d} m`;
    b.setAttribute("aria-pressed", d === state.depth);
    b.addEventListener("click", () => setDepth(d));
    bar.appendChild(b);
  }
}
function setDepth(d) {
  state.depth = d;
  [...$("depthbar").querySelectorAll(".chip")].forEach((b, i) => b.setAttribute("aria-pressed", state.meta.globe_depths[i] === d));
  updateLegendLabel();
  prefetchAll(d, frameFor(state.tau).kA);
  writeHash();
}

function buildLegend() {
  const cv = $("legend-bar"), ctx = cv.getContext("2d");
  for (let x = 0; x < cv.width; x++) {
    const [r, g, b] = rampRGB(x / (cv.width - 1));
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(x, 0, 1, cv.height);
  }
  updateLegendTicks();
}

const LEGEND_TICKS = { K: [100, 700, 100], C: [-200, 400, 100], F: [-200, 800, 200] };
function updateLegendTicks() {
  const ticks = $("legend-ticks"), [lo, hi, step] = LEGEND_TICKS[units.key];
  ticks.innerHTML = "";
  for (let v = lo; v <= hi; v += step) {
    let k = v;   // kelvin position of this tick
    if (units.key === "C") k = v + 273.15;
    if (units.key === "F") k = (v - 32) * 5 / 9 + 273.15;
    if (k < C_MIN || k > C_MAX) continue;
    const s = document.createElement("span");
    s.style.left = `${(k - C_MIN) / (C_MAX - C_MIN) * 100}%`;
    s.textContent = v;
    ticks.appendChild(s);
  }
}
function updateLegendLabel() {
  const d = state.depth;
  $("legend-lbl").textContent = (d === 0 ? "Surface temperature" :
    `Temperature at ${d < 1 ? Math.round(d * 100) + " cm" : d + " m"} depth`) + ` (${uLabel()})`;
}

function buildUnitBar() {
  const bar = $("unitbar");
  for (const [key, u] of Object.entries(UNITS)) {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = u.label;
    b.title = u.name;
    b.dataset.unit = key;
    b.setAttribute("aria-label", u.name);
    b.addEventListener("click", () => setUnit(key));
    bar.appendChild(b);
  }
  markUnit();
}
function markUnit() {
  $("unitbar").querySelectorAll(".chip").forEach(b => b.setAttribute("aria-pressed", b.dataset.unit === units.key));
}
function setUnit(key) {
  units.key = key;
  try { localStorage.setItem("mercury-unit", key); } catch (e) { /* storage unavailable */ }
  markUnit();
  updateLegendLabel();
  updateLegendTicks();
  if (state.data) renderSite();
}

// --------------------------------------------------------- picking & hover
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
function pick(e) {
  const r = canvas.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const hit = ray.intersectObject(globe)[0];
  return hit ? latLonOf(hit.point) : null;
}
let downAt = null;
canvas.addEventListener("pointerdown", e => { downAt = [e.clientX, e.clientY]; });
canvas.addEventListener("pointerup", e => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
  const p = pick(e);
  if (p) select(p.lat, p.lon);
});
const fmtLat = lat => `${Math.abs(lat).toFixed(2)}° ${lat >= 0 ? "N" : "S"}`;
const fmtLon = lon => `${(((lon % 360) + 360) % 360).toFixed(2)}° E`;
let hoverAt = null;
function updateHoverTip() {
  const tip = $("hovertip");
  if (!hoverAt) return;
  const { p, x, y } = hoverAt;
  const fresh = hover.key === `${p.lat.toFixed(2)},${p.lon.toFixed(2)}`;
  tip.innerHTML = `${fmtLat(p.lat)}, ${fmtLon(p.lon)}${fresh && hover.T !== null ? ` · ${fmtT(hover.T)}` : ""}` +
    `<br><span style="color:var(--hud-muted)">local time ${fmtLT(localTimeAt(p.lon, state.tau))} · click for profile</span>`;
  const r = $("stage").getBoundingClientRect();
  tip.style.display = "block";
  tip.style.left = `${Math.min(x - r.left + 14, r.width - tip.offsetWidth - 6)}px`;
  tip.style.top = `${y - r.top + 14}px`;
}
canvas.addEventListener("pointermove", e => {
  const p = pick(e);
  if (!p) { $("hovertip").style.display = "none"; hoverAt = null; return; }
  hoverAt = { p, x: e.clientX, y: e.clientY };
  updateHoverTip();
  requestHoverT(p.lat, p.lon);
});
canvas.addEventListener("pointerleave", () => { $("hovertip").style.display = "none"; hoverAt = null; });

// ----------------------------------------------------------------- sidebar
const SITES = [
  [0, 0, "Equator, hot pole", "0° N, 0° E"],
  [0, 90, "Equator, warm pole", "0° N, 90° E"],
  [45, 45, "Mid-latitude", "45° N, 45° E"],
  [82, 30, "Northern plains", "82° N, 30° E"],
  [85.85, 63.0, "Prokofiev crater floor", "permanent shadow"],
  [87.9, 78.8, "Kandinsky crater", "87.9° N, 78.8° E"],
  [-85, 100, "South polar terrain", "85° S, 100° E"],
  [-81, 250, "South, 81° S", "81° S, 250° E"],
];

// polar terrain-model extent, from the products the server has
const capLatOf = pole => state.meta?.caps?.[pole]?.lat_min ?? 80;
const isPolar = lat => Math.abs(lat) >= capLatOf(lat > 0 ? "north" : "south");
const capText = () => `${capLatOf("north")}° N and ${capLatOf("south")}° S`;

function gotoHTML() {
  return `<form class="goto" id="goto">
      <input id="in-lat" inputmode="decimal" placeholder="lat (−90…90)" aria-label="Latitude">
      <input id="in-lon" inputmode="decimal" placeholder="lon °E (0…360)" aria-label="Longitude, degrees east">
      <button class="btn" type="submit">Go</button></form>`;
}
function bindGoto() {
  $("goto").addEventListener("submit", e => {
    e.preventDefault();
    const lat = parseFloat($("in-lat").value), lon = parseFloat($("in-lon").value);
    if (Number.isFinite(lat) && Number.isFinite(lon) && Math.abs(lat) <= 90) select(lat, lon, true);
  });
}

function renderIntro() {
  const sb = $("sidebar");
  sb.innerHTML = `
    <h2>Pick a location</h2>
    <p class="note" style="margin-top:6px">Click the globe, enter coordinates, or choose a site below. The sidebar shows how temperature
    changes with depth (down to 3 m) and through one Mercury solar day: 176 Earth days, from noon to noon.</p>
    ${gotoHTML()}
    <div class="sites">${SITES.map((s, i) => `<button class="btn" data-i="${i}">${s[2]}<small>${s[3]}</small></button>`).join("")}</div>
    <h3>About the data</h3>
    <p class="note">This is a 1D heat-conduction model of Mercury's regolith, using the Hayne et al. (2017) properties. Poleward of ${capText()}
    it uses terrain-aware polar products built from MESSENGER topography, which add shadowing, slopes and heat from surrounding terrain.
    Elsewhere it includes the slope of each ~2 km patch of ground, from the MESSENGER global elevation map; "Run exact model here" adds
    the full terrain treatment anywhere. Times are mean local solar time: 12:00 is mean noon.</p>
    <h3>Terrain detail outside the poles</h3>
    <p class="note">Detailed elevation data are used only poleward of ${capText()}. There, laser altimetry (north, 250 m per pixel)
    and stereo imaging (south) give the shadows cast by crater walls and hills, which control temperatures where the Sun stays low.
    Toward the equator, the globe and site tables use only the local slope from the coarser, smoothed global elevation map
    (665 m per pixel). They do not include shadows or heat from nearby terrain. This matters little where the Sun is high, but
    crater floors and slopes facing away from the Sun near the edge of the polar regions can be off by tens of degrees.
    "Run exact model here" adds horizons from the same global elevation map, but features smaller than about 1–2 km are still not resolved.</p>`;
  bindGoto();
  sb.querySelectorAll(".sites .btn").forEach(b => b.addEventListener("click", () => {
    const s = SITES[+b.dataset.i]; select(s[0], s[1], true);
  }));
}

async function select(lat, lon, fly = false) {
  lon = ((lon % 360) + 360) % 360;
  state.sel = { lat, lon };
  state.exact = false;
  // the old site's charts and tiles disappear with the "Loading…" sidebar, so drop
  // its data now; otherwise time updates during the fetch write to missing elements
  state.data = null;
  state.charts = [];
  placeMarker(selMarker, lat, lon);
  if (fly) flyTo(lat, lon);
  writeHash();
  $("sidebar").innerHTML = `<h2>${fmtLat(lat)}, ${fmtLon(lon)}</h2><p class="note">Loading…</p>`;
  try {
    const d = await fetchJSON(`/api/point?lat=${lat}&lon=${lon}`);
    if (state.sel?.lat !== lat || state.sel?.lon !== lon) return;
    state.data = d;
    renderSite();
  } catch (e) {
    $("sidebar").innerHTML += `<p class="error">Could not load data: ${e.message}</p>`;
  }
}

async function runExact() {
  const btn = $("btn-exact");
  const { lat, lon } = state.sel;
  btn.disabled = true;
  btn.textContent = "Running terrain model… (~1–5 s)";
  try {
    const d = await fetchJSON(`/api/ondemand?lat=${lat}&lon=${lon}`, 1);
    if (state.sel?.lat !== lat || state.sel?.lon !== lon) return;
    state.data = d; state.exact = true;
    renderSite();
  } catch (e) {
    btn.disabled = false; btn.textContent = "Run exact model here";
    btn.insertAdjacentHTML("afterend", `<p class="error">Run failed: ${e.message}</p>`);
  }
}

const ICON_OK = '<svg viewBox="0 0 12 12"><circle cx="6" cy="6" r="6" fill="var(--good)"/><path d="M3.2 6.2l1.9 1.9 3.7-4" stroke="#fff" stroke-width="1.5" fill="none"/></svg>';
const ICON_SHADOW = '<svg viewBox="0 0 12 12"><circle cx="6" cy="6" r="5.25" fill="none" stroke="var(--text-secondary)" stroke-width="1.5"/><path d="M6 .75a5.25 5.25 0 010 10.5z" fill="var(--text-secondary)"/></svg>';
const fmt = (v, d = 0, u = "") => v === null || v === undefined ? "–" : `${(+v).toFixed(d)}${u}`;

function renderSite() {
  const d = state.data, info = d.info, sb = $("sidebar");
  const w = Math.max(280, sb.clientWidth - 40);
  const surf = d.T.map(r => r[0]);
  const lat = d.lat, lon = d.lon;
  let facts = "";
  if (info.terrain) {
    facts = `
      <dt>Permanent shadow</dt><dd class="status">${info.permanent_shadow ? ICON_SHADOW + "yes, never directly sunlit" : ICON_OK + "no"}</dd>
      <dt>Sun visible</dt><dd>${fmt(100 * info.sun_visible_fraction, 1, "% of the day")}</dd>
      <dt>Mean direct sunlight</dt><dd>${fmt(info.mean_direct_flux, 0, " W/m²")}</dd>
      <dt>Elevation · slope</dt><dd>${fmt(info.elevation_m, 0, " m")} · ${fmt(info.slope_deg, 1, "°")}</dd>
      <dt>Sky view factor</dt><dd>${fmt(info.sky_view_factor, 3)}</dd>
      ${info.dem_quality === 0 ? '<dt>Topography</dt><dd>interpolated (no stereo data): unreliable</dd>' : ""}`;
  } else if (info.slope_aware) {
    const dirs = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"];
    const facing = info.slope_deg < 0.3 ? "level" : `faces ${dirs[Math.round(info.facing_deg / 45) % 8]}`;
    facts = `
      <dt>Elevation</dt><dd>${fmt(info.elevation_m, 0, " m")}</dd>
      <dt>Local slope (2 km)</dt><dd>${fmt(info.slope_deg, 1, "°")}, ${facing}</dd>`;
  }
  sb.innerHTML = `
    <h2>${fmtLat(lat)}, ${fmtLon(lon)}</h2>
    <span class="source">${info.source}${info.seconds ? ` · ran in ${info.seconds.toFixed(2)} s` : ""}</span>
    <div class="tiles" id="now-tiles"></div>
    <dl class="facts"><dt>Local time now</dt><dd id="now-lt"></dd><dt>Sun now</dt><dd id="now-sun"></dd>
      <dt>Surface range over the day</dt><dd>${tconv(Math.min(...surf)).toFixed(0)} to ${fmtT(Math.max(...surf))}</dd>${facts}</dl>
    <div class="row" style="margin-top:12px">
      ${state.exact ? '<button class="btn" id="btn-table">Back to table values</button>'
                    : `<button class="btn primary" id="btn-exact">Run exact model here</button>
                       <span class="muted" style="font-size:12px">${isPolar(lat) ? "full terrain model at 250–665 m detail" : "full terrain model: horizons and heat from surrounding terrain, 665 m detail"}</span>`}
    </div>
    ${info.script ? `<div class="row" style="margin-top:8px">
      <button class="btn" id="btn-script">Download Python script</button>
      <span class="muted" style="font-size:12px">runs the exact model yourself; needs only numpy</span></div>` : ""}

    <h3>Temperature through the day, by depth, compared with human habitability</h3>
    <p class="note">Grey is comfortable room temperature. Blue is colder and red is hotter, fully saturated beyond the
    range people live through (<span id="hm-range"></span>). The line marks the current time; click or drag the chart to change it.</p>
    <div class="chart" id="c-heat"></div>

    <h3>Temperature over the day at four depths</h3>
    <div class="legend">${lineLegendHTML()}</div>
    <div class="chart" id="c-line"></div>

    <h3>Temperature vs depth</h3>
    <div class="legend"><span><i style="background:var(--series-1)"></i>now</span><span><i class="dash"></i>day mean</span>
      <span><i class="box" style="background:var(--band)"></i>day range</span></div>
    <div class="chart" id="c-prof"></div>

    <h3>${info.terrain ? "Sky: horizon and Sun path" : "Sun elevation over the day"}</h3>
    <p class="note">${info.terrain ? "Terrain horizon by compass direction, with the Sun's track over the solar day. The dot is the Sun now." : "Geometric Sun elevation above a flat horizon. Run the exact model to include the real terrain horizon."}</p>
    <div class="chart" id="c-sky"></div>

    ${sourcesHTML(info.sources)}

    <div class="footer-actions">
      <button class="btn" id="btn-csv">Download CSV</button>
      <button class="btn" id="btn-link">Copy link</button>
      <button class="btn" id="btn-clear">Choose another site</button>
    </div>
    ${gotoHTML()}`;
  const setT = L => { setTau((L - 12 - lon / 15) / 24, true); };
  state.charts = [
    heatmap($("c-heat"), d, { onSetTime: setT, width: w }),
    lineChart($("c-line"), d, { onSetTime: setT, width: w }),
    profileChart($("c-prof"), d, { width: w }),
    skyChart($("c-sky"), d, { onSetTime: setT, width: w }),
  ];
  const [lo, hi] = state.charts[0].range;
  $("hm-range").textContent = `${fmtT(HABIT.cold)} to ${fmtT(HABIT.hot)}; this site spans ${fmtT(lo)} to ${fmtT(hi)}`;
  $("btn-exact")?.addEventListener("click", runExact);
  $("btn-table")?.addEventListener("click", () => select(lat, lon));
  $("btn-csv").addEventListener("click", downloadCSV);
  $("btn-script")?.addEventListener("click", downloadScript);
  $("btn-link").addEventListener("click", e => { navigator.clipboard?.writeText(location.href); e.target.textContent = "Link copied"; });
  $("btn-clear").addEventListener("click", () => { state.sel = null; state.data = null; state.charts = []; selMarker.visible = false; writeHash(); renderIntro(); });
  bindGoto();
  updateTimeUI();
}

function updateNow(L) {
  const d = state.data;
  const row = rowAt(d, L);
  const at = z => interpDepth(row, d.depth, z);
  const tile = (k, v) => `<div class="tile"><div class="k">${k}</div><div class="v">${tconv(v).toFixed(0)}<small> ${uLabel()}</small></div></div>`;
  $("now-tiles").innerHTML = tile("Surface now", at(0)) + tile("50 cm now", at(0.5)) + tile("1 m now", at(1.0));
  $("now-lt").textContent = `${fmtLT(L)} (mean solar time)`;
  const s = sunState(d, L);
  const txt = { up: "above the horizon", partial: "partly behind the horizon", terrain: "hidden by terrain", night: "below the horizon (night)" }[s.state];
  $("now-sun").textContent = `${s.el.toFixed(1)}° elevation, ${txt}`;
}

const esc = t => String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function sourcesHTML(sources) {
  if (!sources?.length) return "";
  const item = s => `<li><span class="src-name">${esc(s.name)}</span>
      <span class="src-role">${esc(s.role)}</span>
      <span class="src-links">${s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.label || "dataset page")}</a>` : esc(s.note || "")}${
        s.download ? ` · <a href="${esc(s.download)}" target="_blank" rel="noopener">download file</a>` : ""}</span></li>`;
  return `<h3>Where this data comes from</h3>
    <ul class="sources">${sources.map(item).join("")}</ul>`;
}

async function downloadScript(e) {
  const btn = e.currentTarget, d = state.data;
  btn.disabled = true; btn.textContent = "Preparing script… (~1–5 s)";
  try {
    const r = await fetch(`/api/script?lat=${d.lat}&lon=${d.lon}`);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `HTTP ${r.status}`);
    const name = /filename="([^"]+)"/.exec(r.headers.get("Content-Disposition") || "")?.[1] || "mercury_site.py";
    const a = document.createElement("a");
    a.href = URL.createObjectURL(await r.blob());
    a.download = name;
    a.click();
    URL.revokeObjectURL(a.href);
    btn.textContent = "Download Python script";
  } catch (err) {
    btn.textContent = `Script failed: ${err.message}`;
  }
  btn.disabled = false;
}

function downloadCSV() {
  const d = state.data;
  const u = { F: "degF", C: "degC", K: "K" }[units.key];
  const lines = [`# Mercury regolith temperature (${u}), lat ${d.lat}, lon ${d.lon} E; ${d.info.source}`,
    ["local_time_h", ...d.depth.map(z => `T_${z}m_${u}`)].join(",")];
  d.local_time.forEach((L, i) => lines.push([L, ...d.T[i].map(k => tconv(k).toFixed(2))].join(",")));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
  a.download = `mercury_T_${d.lat.toFixed(2)}_${d.lon.toFixed(2)}${state.exact ? "_exact" : ""}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (state.data) renderSite(); }, 200);
});

// --------------------------------------------------------------- camera fly
let fly = null;
function flyTo(lat, lon) {
  const dist = camera.position.length();
  fly = { from: camera.position.clone().normalize(), to: dirOf(lat, lon), t0: performance.now(), dist };
}

// ---------------------------------------------------------------- URL hash
function writeHash() {
  const p = new URLSearchParams();
  if (state.sel) { p.set("lat", state.sel.lat.toFixed(3)); p.set("lon", state.sel.lon.toFixed(3)); }
  p.set("t", state.tau.toFixed(4));
  if (state.depth) p.set("depth", state.depth);
  const c = latLonOf(camera.position);
  p.set("cam", `${c.lat.toFixed(1)},${c.lon.toFixed(1)},${camera.position.length().toFixed(2)}`);
  history.replaceState(null, "", "#" + p.toString());
}
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  return { lat: p.get("lat"), lon: p.get("lon"), t: p.get("t"), depth: p.get("depth"), cam: p.get("cam") };
}

// ------------------------------------------------------------------- loop
let last = performance.now();
function loop(now) {
  requestAnimationFrame(loop);   // schedule first, so an error below cannot stop rendering
  try {
    frame(now);
  } catch (e) {
    console.error(e);
  }
}
function frame(now) {
  const dt = (now - last) / 1000; last = now;
  if (state.meta && state.playing) setTau(state.tau + dt / +$("speed").value);
  if (fly) {
    const k = Math.min(1, (now - fly.t0) / 900), e = k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2;
    camera.position.copy(fly.from.clone().lerp(fly.to, e).normalize().multiplyScalar(fly.dist));
    if (k >= 1) fly = null;
  }
  controls.update();
  // keep surface markers a constant size on screen (sized for the default 3.4-unit viewing distance)
  for (const mk of [selMarker, sunMarker]) {
    if (!mk.visible) continue;
    const k = Math.min(1.5, camera.position.distanceTo(mk.position) / 3.4);
    mk.scale.setScalar(k);
    mk.position.copy(mk.userData.dir).multiplyScalar(1 + 0.003 * k);   // lift above the surface scales too
  }
  if (state.meta) {
    updateGlobeTexture();
    if (state.dirty) { state.dirty = false; updateTimeUI(); }
  }
  renderer.render(scene, camera);
}

// ------------------------------------------------------------------- init
async function init() {
  resize();
  requestAnimationFrame(loop);
  const meta = await (await fetch("/api/meta")).json();
  state.meta = meta;
  uniforms.tLo.value = meta.t_lo;
  uniforms.tStep.value = meta.t_step;
  for (const [pole, u, h] of [["north", "capNorth", "hasNorth"], ["south", "capSouth", "hasSouth"]]) {
    const c = meta.caps[pole];
    if (c) {
      uniforms[u].value.set(c.x0, c.y0, c.dx, c.n);
      uniforms[h].value = 1;
      uniforms.capLat.value.setComponent(pole === "north" ? 0 : 1, c.lat_min);
      uniforms.capCol.value.setComponent(pole === "north" ? 0 : 1, c.col);
    }
  }
  uniforms.nG.value = meta.g_nodes.length;
  uniforms.gNodes.value = meta.g_nodes.concat(new Array(9 - meta.g_nodes.length).fill(0));
  uniforms.gMax.value = meta.g_max;
  uniforms.tileW.value = meta.tile_w;
  uniforms.tileH.value = meta.tile_h;
  uniforms.capY0.value = meta.cap_y0;
  if (meta.slopes) loadSlopes().catch(e => console.warn("slope texture failed:", e));
  const h = readHash();
  if (h.t !== null) state.tau = +h.t || 0;
  if (h.depth !== null && meta.globe_depths.includes(+h.depth)) state.depth = +h.depth;
  buildDepthBar();
  buildUnitBar();
  buildLegend();
  setDepth(state.depth);
  setPlaying(false);
  state.dirty = true;
  const cam = (h.cam || "").split(",").map(Number);
  if (cam.length === 3 && cam.every(Number.isFinite)) camera.position.copy(dirOf(cam[0], cam[1]).multiplyScalar(cam[2]));
  if (h.lat !== null && h.lon !== null) select(+h.lat, +h.lon, !h.cam);
  else renderIntro();
}
init().catch(e => { $("loading").textContent = `Failed to start: ${e.message}`; });
