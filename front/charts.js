/* Minimal SVG charts for the QOPO dashboard: line (lin/log), dot-whisker, bars.
   No dependencies. Marks: 2px lines, ≥8px markers with a 2px surface ring,
   hairline solid grid, 4px rounded bar ends, crosshair / per-mark tooltips. */
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const SURFACE = '#ffffff';

  function el(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    for (const k in attrs || {}) if (attrs[k] !== undefined && attrs[k] !== null) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }
  function text(parent, x, y, str, attrs) {
    const t = el('text', Object.assign({ x, y }, attrs || {}), parent);
    t.textContent = str;
    return t;
  }

  /* ---------- tooltip ---------- */
  const tip = () => document.getElementById('tooltip');
  function showTip(html, evt) {
    const t = tip();
    t.innerHTML = html;
    t.classList.add('show');
    const pad = 14;
    const w = t.offsetWidth, h = t.offsetHeight;
    let x = evt.pageX + pad, y = evt.pageY + pad;
    if (x + w > window.scrollX + document.documentElement.clientWidth - 8) x = evt.pageX - w - pad;
    if (y + h > window.scrollY + window.innerHeight - 8) y = evt.pageY - h - pad;
    t.style.left = x + 'px';
    t.style.top = y + 'px';
  }
  function hideTip() { tip().classList.remove('show'); }
  function swatch(color) { return `<span class="tt-sw" style="background:${color}"></span>`; }
  function row(color, label, val) {
    return `<div class="tt-row"><span>${color ? swatch(color) : ''}${label}</span><span class="tt-val">${val}</span></div>`;
  }

  /* ---------- scales ---------- */
  function linScale(d0, d1, r0, r1) {
    const s = (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
    s.domain = [d0, d1];
    return s;
  }
  function logScale(d0, d1, r0, r1) {
    const l0 = Math.log10(d0), l1 = Math.log10(d1);
    const s = (v) => r0 + ((Math.log10(Math.max(v, d0)) - l0) / (l1 - l0 || 1)) * (r1 - r0);
    s.domain = [d0, d1];
    return s;
  }
  function niceLin(min, max, count) {
    if (min === max) { max = min + 1; }
    const span = max - min;
    const step0 = Math.pow(10, Math.floor(Math.log10(span / count)));
    const err = span / count / step0;
    const step = step0 * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
    const lo = Math.floor(min / step) * step, hi = Math.ceil(max / step) * step;
    const ticks = [];
    for (let v = lo; v <= hi + step / 2; v += step) ticks.push(+v.toFixed(10));
    return { lo, hi, ticks };
  }
  function niceLog(min, max) {
    let lo = Math.pow(10, Math.floor(Math.log10(min)));
    let hi = Math.pow(10, Math.ceil(Math.log10(max)));
    if (hi === lo) hi = lo * 10;
    const ticks = [];
    for (let v = lo; v <= hi * 1.0001; v *= 10) ticks.push(v);
    if (ticks.length <= 2) { // under one decade: add 2 and 5 subdivisions
      const extra = [];
      ticks.forEach((t) => { extra.push(t); if (t * 2 <= hi) extra.push(t * 2); if (t * 5 <= hi) extra.push(t * 5); });
      return { lo, hi, ticks: extra };
    }
    return { lo, hi, ticks };
  }

  /* ---------- markers ---------- */
  function marker(g, x, y, s, size) {
    const r = size || 4.5;
    const fill = s.hollow ? SURFACE : s.color;
    const common = { fill, stroke: s.hollow ? s.color : SURFACE, 'stroke-width': 2 };
    let node;
    if (s.shape === 'square') node = el('rect', Object.assign({ x: x - r, y: y - r, width: 2 * r, height: 2 * r, rx: 1.5 }, common), g);
    else if (s.shape === 'diamond') node = el('path', Object.assign({ d: `M${x} ${y - r - 1}L${x + r + 1} ${y}L${x} ${y + r + 1}L${x - r - 1} ${y}Z` }, common), g);
    else node = el('circle', Object.assign({ cx: x, cy: y, r }, common), g);
    if (!s.hollow) { // surface ring outside the fill
      const ring = node.cloneNode();
      ring.setAttribute('fill', 'none');
      ring.setAttribute('stroke', SURFACE);
      ring.setAttribute('stroke-width', 2);
      node.setAttribute('stroke', 'none');
      g.insertBefore(ring, node);
    }
    return node;
  }
  function legendKey(s) {
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('width', 22); svg.setAttribute('height', 12); svg.setAttribute('aria-hidden', 'true');
    if (!s.noLine) el('line', { x1: 1, y1: 6, x2: 21, y2: 6, stroke: s.color, 'stroke-width': s.ref ? 1.5 : 2, 'stroke-linecap': 'round' }, svg);
    marker(svg, 11, 6, s, 3.5);
    return svg;
  }

  function frame(container, height, margin) {
    container.innerHTML = '';
    const width = Math.max(280, container.clientWidth || 560);
    const svg = el('svg', { viewBox: `0 0 ${width} ${height}`, width, height, role: 'img' }, container);
    return { svg, width, height, inner: { x0: margin.l, x1: width - margin.r, y0: margin.t, y1: height - margin.b } };
  }

  function empty(container, html) {
    container.innerHTML = `<div class="empty">${html}</div>`;
  }

  function legend(container, series) {
    const box = document.createElement('div');
    box.className = 'legend';
    series.forEach((s) => {
      const item = document.createElement('span');
      item.appendChild(legendKey(s));
      item.appendChild(document.createTextNode(s.label));
      box.appendChild(item);
    });
    container.appendChild(box);
  }

  /* ---------- line chart ---------- */
  function line(container, opt) {
    const series = (opt.series || []).filter((s) => s.points && s.points.length);
    if (!series.length) return empty(container, opt.empty || 'No data for the current selection.');
    const margin = { t: 12, r: opt.endLabels === false ? 16 : 118, b: 40, l: 54 };
    const f = frame(container, opt.height || 260, margin);
    const { svg, inner } = f;
    if (opt.ariaLabel) svg.setAttribute('aria-label', opt.ariaLabel);

    const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p.x)))].sort((a, b) => a - b);
    const ys = series.flatMap((s) => s.points.map((p) => p.y)).filter((v) => v !== null && isFinite(v) && (!opt.y.log || v > 0));
    let yInfo;
    if (opt.y.log) yInfo = niceLog(Math.min(...ys), Math.max(...ys));
    else yInfo = niceLin(opt.y.min !== undefined ? opt.y.min : Math.min(0, ...ys), opt.y.max !== undefined ? opt.y.max : Math.max(...ys), 5);
    if (opt.y.min !== undefined && !opt.y.log) { yInfo.lo = opt.y.min; yInfo.ticks = yInfo.ticks.filter((t) => t >= opt.y.min); }
    if (opt.y.max !== undefined && !opt.y.log) { yInfo.hi = opt.y.max; yInfo.ticks = yInfo.ticks.filter((t) => t <= opt.y.max); }
    const ysc = (opt.y.log ? logScale : linScale)(yInfo.lo, yInfo.hi, inner.y1, inner.y0);
    const xmin = xs[0], xmax = xs[xs.length - 1];
    const xpad = xmin === xmax ? 1 : 0;
    const xsc = linScale(xmin - xpad, xmax + xpad, inner.x0 + 10, inner.x1 - 10);

    const axis = el('g', { class: 'axis' }, svg);
    // keep tick labels ≥ 22px apart: thin a dense log axis to every 2nd/3rd decade
    const stride = Math.max(1, Math.ceil(22 / ((inner.y1 - inner.y0) / Math.max(1, yInfo.ticks.length - 1))));
    yInfo.ticks.forEach((t, i) => {
      if (i % stride) return;
      const y = ysc(t);
      el('line', { class: 'gridline', x1: inner.x0, x2: inner.x1, y1: y, y2: y }, axis);
      text(axis, inner.x0 - 8, y + 3.5, opt.y.fmt(t), { 'text-anchor': 'end' });
    });
    el('line', { class: 'baseline', x1: inner.x0, x2: inner.x1, y1: inner.y1, y2: inner.y1 }, axis);
    (opt.x.ticks || xs).forEach((t) => text(axis, xsc(t), inner.y1 + 16, opt.x.fmt ? opt.x.fmt(t) : t, { 'text-anchor': 'middle' }));
    if (opt.x.label) text(svg, (inner.x0 + inner.x1) / 2, f.height - 4, opt.x.label, { class: 'axis-title', 'text-anchor': 'middle' });

    const g = el('g', {}, svg);
    // reference series first (recessive), then data series
    const ordered = series.filter((s) => s.ref).concat(series.filter((s) => !s.ref));
    ordered.forEach((s) => {
      const pts = s.points.filter((p) => p.y !== null && isFinite(p.y) && (!opt.y.log || p.y > 0)).sort((a, b) => a.x - b.x);
      if (pts.length > 1) el('path', { class: 'series-line' + (s.ref ? ' ref' : ''), stroke: s.color, d: pts.map((p, i) => `${i ? 'L' : 'M'}${xsc(p.x)} ${ysc(p.y)}`).join('') }, g);
      pts.forEach((p) => marker(g, xsc(p.x), ysc(p.y), s, s.ref ? 3.5 : 4.5));
      s._last = pts[pts.length - 1];
    });

    // direct end labels — only where they don't collide (legend carries the rest)
    if (opt.endLabels !== false) {
      const labels = ordered.filter((s) => s._last).map((s) => ({ s, y: ysc(s._last.y), x: xsc(s._last.x) })).sort((a, b) => a.y - b.y);
      let lastY = -Infinity;
      labels.forEach((l) => {
        if (l.y - lastY < 13) return;
        text(svg, l.x + 9, l.y + 3.5, l.s.short || l.s.label, { class: 'end-label' + (l.s.ref ? ' ref' : '') });
        lastY = l.y;
      });
    }

    // crosshair + tooltip
    const cross = el('line', { class: 'crosshair', y1: inner.y0, y2: inner.y1, visibility: 'hidden' }, svg);
    const hit = el('rect', { class: 'hit', x: inner.x0, y: inner.y0, width: inner.x1 - inner.x0, height: inner.y1 - inner.y0 }, svg);
    const pick = (evt) => {
      const rect = svg.getBoundingClientRect();
      const px = ((evt.clientX - rect.left) / rect.width) * f.width;
      let best = xs[0];
      xs.forEach((x) => { if (Math.abs(xsc(x) - px) < Math.abs(xsc(best) - px)) best = x; });
      return best;
    };
    hit.addEventListener('mousemove', (evt) => {
      const x = pick(evt);
      cross.setAttribute('x1', xsc(x)); cross.setAttribute('x2', xsc(x)); cross.setAttribute('visibility', 'visible');
      const hits = ordered.map((s) => ({ s, p: s.points.find((p) => p.x === x) })).filter((h) => h.p && h.p.y !== null);
      hits.sort((a, b) => b.p.y - a.p.y);
      const head = opt.x.tipLabel ? opt.x.tipLabel(x) : x;
      showTip(`<div class="tt-head">${head}</div>` + hits.map((h) => row(h.s.color, h.s.label, opt.y.tipFmt ? opt.y.tipFmt(h.p.y, h.p) : opt.y.fmt(h.p.y))).join(''), evt);
    });
    hit.addEventListener('mouseleave', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });

    if (series.length > 1 || opt.legend) legend(container, series);
  }

  /* ---------- dot / whisker chart (rows = bands, x = value) ---------- */
  function dots(container, opt) {
    const series = (opt.series || []).filter((s) => Object.keys(s.values || {}).length);
    if (!series.length) return empty(container, opt.empty || 'No data for the current selection.');
    const bands = opt.bands.filter((b) => series.some((s) => s.values[b]));
    const rowH = Math.max(30, 14 * series.length + 12);
    const margin = { t: 8, r: 20, b: 36, l: 136 };
    const f = frame(container, margin.t + margin.b + rowH * bands.length, margin);
    const { svg, inner } = f;
    const xsc = linScale(opt.x.min, opt.x.max, inner.x0, inner.x1);
    const axis = el('g', { class: 'axis' }, svg);
    opt.x.ticks.forEach((t) => {
      el('line', { class: 'gridline', x1: xsc(t), x2: xsc(t), y1: inner.y0, y2: inner.y1 }, axis);
      text(axis, xsc(t), inner.y1 + 16, opt.x.fmt(t), { 'text-anchor': 'middle' });
    });
    if (opt.x.label) text(svg, (inner.x0 + inner.x1) / 2, f.height - 3, opt.x.label, { class: 'axis-title', 'text-anchor': 'middle' });
    bands.forEach((b, bi) => {
      const y0 = inner.y0 + bi * rowH;
      if (bi) el('line', { class: 'gridline', x1: 0, x2: inner.x1, y1: y0, y2: y0 }, axis);
      text(svg, 0, y0 + rowH / 2 + 4, opt.bandLabel ? opt.bandLabel(b) : b, { class: 'band-label' });
      const present = series.filter((s) => s.values[b]);
      present.forEach((s, si) => {
        const v = s.values[b];
        const y = y0 + (rowH / (present.length + 1)) * (si + 1);
        const g = el('g', {}, svg);
        if (v.lo !== undefined && v.hi !== undefined && v.hi > v.lo) el('line', { class: 'whisker', stroke: s.color, x1: xsc(Math.max(opt.x.min, v.lo)), x2: xsc(Math.min(opt.x.max, v.hi)), y1: y, y2: y }, g);
        marker(g, xsc(v.y), y, s, 4.5);
        const hitbox = el('rect', { class: 'hit', x: xsc(v.y) - 10, y: y - 8, width: 20, height: 16 }, g);
        hitbox.style.cursor = 'default';
        hitbox.addEventListener('mousemove', (evt) => showTip(`<div class="tt-head">${opt.bandLabel ? opt.bandLabel(b) : b}</div>` + row(s.color, s.label, '') + (opt.tip ? opt.tip(v, s, b) : ''), evt));
        hitbox.addEventListener('mouseleave', hideTip);
      });
    });
    legend(container, series.map((s) => Object.assign({ noLine: true }, s)));
  }

  /* ---------- horizontal bars (one bar per row, colored by series identity) ---------- */
  function bars(container, opt) {
    const rows = opt.rows || [];
    if (!rows.length) return empty(container, opt.empty || 'No data for the current selection.');
    const rowH = 30, barH = 14;
    const longest = Math.max(...rows.map((r) => r.label.length));
    const margin = { t: 6, r: 78, b: 30, l: Math.min(300, 30 + longest * 6.6) };
    const f = frame(container, margin.t + margin.b + rowH * rows.length, margin);
    const { svg, inner } = f;
    const max = Math.max(...rows.map((r) => r.value));
    const info = niceLin(0, max, 4);
    const xsc = linScale(0, info.hi, inner.x0, inner.x1);
    const axis = el('g', { class: 'axis' }, svg);
    info.ticks.forEach((t) => {
      el('line', { class: 'gridline', x1: xsc(t), x2: xsc(t), y1: inner.y0, y2: inner.y1 }, axis);
      text(axis, xsc(t), inner.y1 + 16, opt.fmt(t), { 'text-anchor': 'middle' });
    });
    rows.forEach((r, i) => {
      const yc = inner.y0 + i * rowH + rowH / 2;
      const g = el('g', {}, svg);
      const keyS = Object.assign({ noLine: true }, r.series);
      const k = el('g', { transform: `translate(0 ${yc - 6})` }, g);
      marker(k, 6, 6, keyS, 4);
      text(g, 18, yc + 4, r.label, { class: 'band-label' });
      const w = Math.max(2, xsc(r.value) - inner.x0), rad = Math.min(4, w / 2);
      // square at the baseline, 4px rounded data-end
      el('path', { fill: r.series.color, opacity: r.series.hollow ? 0.45 : 1, d: `M${inner.x0} ${yc - barH / 2}H${inner.x0 + w - rad}Q${inner.x0 + w} ${yc - barH / 2} ${inner.x0 + w} ${yc - barH / 2 + rad}V${yc + barH / 2 - rad}Q${inner.x0 + w} ${yc + barH / 2} ${inner.x0 + w - rad} ${yc + barH / 2}H${inner.x0}Z` }, g);
      text(g, inner.x0 + w + 6, yc + 4, opt.fmt(r.value, true), { class: 'end-label' });
      const hitbox = el('rect', { class: 'hit', x: 0, y: yc - rowH / 2, width: f.width, height: rowH }, g);
      hitbox.style.cursor = 'default';
      hitbox.addEventListener('mousemove', (evt) => showTip(`<div class="tt-head">${r.label}</div>` + (r.tip || row(r.series.color, 'value', opt.fmt(r.value, true))), evt));
      hitbox.addEventListener('mouseleave', hideTip);
    });
    el('line', { class: 'baseline', x1: inner.x0, x2: inner.x0, y1: inner.y0, y2: inner.y1 }, svg);
  }

  window.Charts = { line, dots, bars, legendKey, row, empty };
})();
