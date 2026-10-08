/* QOPO dashboard: normalises window.QOPO_DATA into series and renders each section. */
(function () {
  'use strict';
  const D = window.QOPO_DATA;
  const $ = (sel) => document.querySelector(sel);

  /* ---------- identity: color follows host·backend, never rank ---------- */
  const CONFIGS = [
    { key: 'spark-gb10|lightning.gpu', color: 'var(--s1)', hex: '#00917f' },
    { key: 'rtx3080|lightning.gpu', color: 'var(--s2)', hex: '#d9731a' },
    { key: 'spark-gb10|lightning.qubit', color: 'var(--s3)', hex: '#3a62b0' },
    { key: 'rtx3080|lightning.qubit', color: 'var(--s4)', hex: '#5d9130' },
    { key: 'rtx3080|default.qubit', color: 'var(--s5)', hex: '#c94f86' },
    { key: 'spark-gb10|default.qubit', color: 'var(--s6)', hex: '#7b5cc4' },
  ];
  const REF_HEX = '#9aa1a8';
  const SOLVERS = ['brute_force', 'simulated_annealing', 'markowitz', 'qaoa', 'random'];
  const SOLVER_LABEL = { brute_force: 'Brute force', simulated_annealing: 'Simulated annealing', markowitz: 'Markowitz (top-k)', qaoa: 'QAOA', random: 'Random' };
  const BASELINE = { key: 'rtx3080|default.qubit', campaign: '2026-06' };

  const hostLabel = (h) => (D && D.hosts[h] ? D.hosts[h].label : h);
  const campaignLabel = (c) => {
    const [y, m] = c.split('-');
    const d = new Date(Date.UTC(+y, +m - 1, 1));
    return isNaN(d) ? c : d.toLocaleString('en', { month: 'short', year: 'numeric', timeZone: 'UTC' });
  };
  const configOf = (a) => `${a.host}|${a.backend}`;
  const configMeta = (key) => CONFIGS.find((c) => c.key === key) || { key, color: REF_HEX, hex: REF_HEX };

  /* ---------- stats & formatting ---------- */
  const mean = (xs) => xs.reduce((s, v) => s + v, 0) / xs.length;
  const std = (xs) => { if (xs.length < 2) return 0; const m = mean(xs); return Math.sqrt(xs.reduce((s, v) => s + (v - m) ** 2, 0) / xs.length); };
  const median = (xs) => { const s = [...xs].sort((a, b) => a - b); const k = s.length >> 1; return s.length % 2 ? s[k] : (s[k - 1] + s[k]) / 2; };
  const groupBy = (xs, f) => xs.reduce((m, x) => { const k = f(x); (m.get(k) || m.set(k, []).get(k)).push(x); return m; }, new Map());
  function fmtDur(sec, precise) {
    if (sec === null || sec === undefined || !isFinite(sec)) return '—';
    if (sec === 0) return '0';
    if (sec < 1e-3) return (sec * 1e6).toFixed(0) + ' µs';
    if (sec < 1) return +(sec * 1e3).toPrecision(precise ? 3 : 2) + ' ms';
    if (sec < 120) return +sec.toPrecision(precise ? 3 : 2) + ' s';
    if (sec < 7200) return +(sec / 60).toPrecision(precise ? 3 : 2) + ' min';
    return +(sec / 3600).toPrecision(3) + ' h';
  }
  function fmtBytes(kb) {
    if (kb === null || kb === undefined || !isFinite(kb)) return '—';
    // decimal units so log-axis decades land on round labels (1 MB, 10 MB, ...)
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let i = 0, v = kb * 1024;
    while (v >= 1000 && i < units.length - 1) { v /= 1000; i++; }
    return `${+v.toPrecision(v >= 100 ? 3 : 2)} ${units[i]}`;
  }
  const fmtRatio = (v) => (v === null || v === undefined ? '—' : v.toFixed(3));
  const pct = (v) => (v === null || v === undefined ? '—' : `${v >= 0 ? '+' : '−'}${Math.abs(v * 100).toFixed(1)} %`);
  const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  /* ---------- empty dataset ---------- */
  if (!D || !D.artifacts || !D.artifacts.length) {
    $('#kpis').outerHTML = '<div class="empty" style="margin-top:2rem">No dataset bundled yet. Run <code>python front/build_data.py</code> from the repository root, then reload.</div>';
    return;
  }

  /* ---------- normalise ---------- */
  const A = D.artifacts;
  const campaigns = [...new Set(A.map((a) => a.campaign))].sort();
  const latestCampaign = campaigns[campaigns.length - 1];
  const configKeys = [...new Set(A.map(configOf))].sort((a, b) => {
    const ia = CONFIGS.findIndex((c) => c.key === a), ib = CONFIGS.findIndex((c) => c.key === b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
  });
  const campaignsPerConfig = groupBy(A, configOf);
  const multiCampaign = (key) => new Set((campaignsPerConfig.get(key) || []).map((a) => a.campaign)).size > 1;

  // A series = host·backend × campaign. Hollow markers mark archived campaigns.
  const seriesCache = new Map();
  function seriesFor(key, campaign) {
    const id = `${key}@${campaign}`;
    if (seriesCache.has(id)) return seriesCache.get(id);
    const [host, backend] = key.split('|');
    const meta = configMeta(key);
    const archived = campaign !== latestCampaign;
    const base = `${hostLabel(host)} · ${backend}`;
    const s = {
      id, key, campaign, host, backend,
      color: meta.hex,
      shape: host === 'spark-gb10' ? 'square' : 'circle',
      hollow: archived,
      label: multiCampaign(key) || archived ? `${base} (${campaignLabel(campaign)})` : base,
      short: `${host === 'spark-gb10' ? 'GB10' : 'RTX'} ${backend.replace('lightning.', 'l.').replace('default.qubit', 'default')}${archived ? ` (${campaignLabel(campaign).split(' ')[0]})` : ''}`,
    };
    seriesCache.set(id, s);
    return s;
  }
  const seriesOfArtifact = (a) => seriesFor(configOf(a), a.campaign);
  const seriesRank = (s) => configKeys.indexOf(s.key) * 100 + campaigns.indexOf(s.campaign);

  // One record per (series, suite, solver, n, seed, QAOA settings): short probes and
  // repeated runs re-solve the same seeded instance; keep the most recent artifact's solve.
  const R = (() => {
    const all = D.records.map((r) => Object.assign({ art: A[r.a] }, r));
    const best = new Map();
    all.forEach((r) => {
      if (r.sd === null || r.sd === undefined) { best.set(Symbol(), r); return; }
      // within one suite only: other suites re-solve the same seeds for different views
      const k = [configOf(r.art), r.art.campaign, r.art.suite, r.s, r.n, r.sd, r.L, r.it, r.rs, r.op].join('|');
      const prev = best.get(k);
      if (!prev || String(r.art.created) > String(prev.art.created)) best.set(k, r);
    });
    return [...best.values()];
  })();

  /* ---------- filter state ---------- */
  const state = {
    campaigns: new Set(campaigns),
    configs: new Set(configKeys),
    qualityN: null, qualityOp: 'adam', scalingPreset: null, depthOp: 'adam', budgetN: null, depthN: null,
  };
  const visibleArt = (a) => state.campaigns.has(a.campaign) && state.configs.has(configOf(a)) && (a.status || 'ok') === 'ok';
  const recs = (pred) => R.filter((r) => visibleArt(r.art) && pred(r));

  function chip(label, pressed, onClick, opts) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip' + (opts && opts.seg ? ' seg' : '');
    b.setAttribute('aria-pressed', pressed ? 'true' : 'false');
    if (opts && opts.series) {
      const k = document.createElement('span');
      k.className = 'key';
      k.appendChild(Charts.legendKey(Object.assign({ noLine: true }, opts.series)));
      b.appendChild(k);
    }
    b.appendChild(document.createTextNode(label));
    b.addEventListener('click', onClick);
    return b;
  }
  function segmented(container, options, current, onPick) {
    container.innerHTML = '';
    options.forEach((o) => container.appendChild(chip(o.label, o.value === current, () => onPick(o.value), { seg: true })));
  }

  function renderFilters() {
    const fc = $('#filter-campaign'), fk = $('#filter-config');
    fc.querySelectorAll('.chip').forEach((n) => n.remove());
    fk.querySelectorAll('.chip').forEach((n) => n.remove());
    campaigns.forEach((c) => fc.appendChild(chip(campaignLabel(c), state.campaigns.has(c), () => toggle(state.campaigns, c))));
    configKeys.forEach((k) => {
      const [h, b] = k.split('|');
      fk.appendChild(chip(`${hostLabel(h)} · ${b}`, state.configs.has(k), () => toggle(state.configs, k), { series: seriesFor(k, latestCampaign) }));
    });
  }
  function toggle(set, v) {
    if (set.has(v)) { if (set.size > 1) set.delete(v); } else set.add(v);
    renderAll();
  }

  /* ---------- KPIs (whole dataset, filter-independent) ---------- */
  const isAdam = (r) => !r.op || r.op === 'adam';
  const OPT_LABEL = { adam: 'Adam', cobyla: 'COBYLA' };
  const optLabel = (o) => OPT_LABEL[o || 'adam'] || o;
  // Benchmark preset: p = 1, 60 iterations, 2 restarts; Adam until October 2026, COBYLA after.
  const isPreset = (r, op = 'adam') => r.L === 1 && r.it === 60 && r.rs === 2 && (r.op || 'adam') === op;
  function renderKpis() {
    const okR = R.filter((r) => (r.art.status || 'ok') === 'ok');
    const q8 = okR.filter((r) => r.s === 'qaoa' && r.n === 8 && r.art.suite === 'quality' && isPreset(r));
    const bySeries = [...groupBy(q8, (r) => seriesOfArtifact(r.art).id).values()].map((rs) => ({ s: seriesOfArtifact(rs[0].art), t: median(rs.map((r) => r.t)) / 1e3, ratio: mean(rs.map((r) => r.r)), hits: rs.filter((r) => r.r >= 1 - 1e-9).length, runs: rs.length }));
    const fastest = bySeries.sort((a, b) => a.t - b.t)[0];
    const base = bySeries.find((x) => x.s.key === BASELINE.key && x.s.campaign === BASELINE.campaign);
    const bestQ = [...bySeries].sort((a, b) => b.ratio - a.ratio)[0];
    const qaoaAll = okR.filter((r) => r.s === 'qaoa');
    const maxN = qaoaAll.reduce((m, r) => Math.max(m, r.n), 0);
    const maxNrec = qaoaAll.filter((r) => r.n === maxN);
    const maxNSeries = [...new Set(maxNrec.map((r) => seriesOfArtifact(r.art).label))];
    const maxNProbe = maxNrec.every((r) => r.it !== null && r.it <= 1) ? ' · 1-iteration memory probe' : '';
    const hosts = new Set(A.map((a) => a.host));
    const tiles = [
      fastest && { label: 'Fastest QAOA solve · 8 assets', value: fmtDur(fastest.t, true), note: `${fastest.s.label}${base && base.s.id !== fastest.s.id ? ` · ${(base.t / fastest.t).toFixed(1)}× faster than the June baseline` : ''}` },
      { label: 'Largest portfolio simulated', value: `${maxN}<small>assets</small>`, note: `2<sup>${maxN}</sup> = ${Math.pow(2, maxN).toLocaleString('en')} amplitudes · ${maxNSeries.slice(0, 2).join(', ')}${maxNSeries.length > 2 ? '…' : ''}${maxNProbe}` },
      bestQ && { label: 'Best QAOA quality · 8 assets', value: fmtRatio(bestQ.ratio), note: `${bestQ.hits}/${bestQ.runs} instances solved to optimum · ${bestQ.s.label}` },
      { label: 'QAOA solves recorded', value: qaoaAll.length.toLocaleString('en'), note: `${A.length} artifacts · ${configKeys.length} host·backend configs · ${hosts.size} host${hosts.size > 1 ? 's' : ''}` },
    ].filter(Boolean);
    $('#kpis').innerHTML = tiles.map((t) => `<div class="kpi"><div class="kpi-label">${t.label}</div><div class="kpi-value">${t.value}</div><div class="kpi-note">${t.note}</div></div>`).join('');
    const gen = D.generated_utc ? D.generated_utc.replace('T', ' ').replace('Z', ' UTC') : '';
    $('#generated').textContent = `Dataset built ${gen} · ${campaigns.map(campaignLabel).join(' + ')} campaigns`;
  }

  /* ---------- 01 quality ---------- */
  function presetArtifacts(suite, op) {
    // artifacts whose QAOA records all use the benchmark preset (classical-only artifacts count too)
    const qa = groupBy(R.filter((r) => r.s === 'qaoa'), (r) => r.a);
    return new Set(A.filter((a) => a.suite === suite && (!qa.has(a.id) || qa.get(a.id).every((r) => isPreset(r, op)))).map((a) => a.id));
  }
  function renderQuality() {
    const ops = ['adam', 'cobyla'].filter((op) => R.some((r) => r.s === 'qaoa' && r.art.suite === 'quality' && isPreset(r, op)));
    if (!ops.includes(state.qualityOp)) state.qualityOp = ops[0] || 'adam';
    segmented($('#quality-op'), ops.map((op) => ({ label: `${optLabel(op)} preset`, value: op })), state.qualityOp, (v) => { state.qualityOp = v; renderQuality(); });
    const ids = presetArtifacts('quality', state.qualityOp);
    const all = recs((r) => ids.has(r.a));
    const ns = [...new Set(all.map((r) => r.n))].sort((a, b) => a - b);
    if (!ns.includes(state.qualityN)) state.qualityN = ns.includes(8) ? 8 : ns[0];
    segmented($('#quality-n'), ns.map((n) => ({ label: `${n} assets`, value: n })), state.qualityN, (v) => { state.qualityN = v; renderQuality(); });
    const rs = all.filter((r) => r.n === state.qualityN);
    const groups = groupBy(rs, (r) => seriesOfArtifact(r.art).id);
    const series = [...groups.values()].map((g) => {
      const s = seriesOfArtifact(g[0].art);
      const values = {};
      groupBy(g, (r) => r.s).forEach((list, solver) => {
        const ratios = list.map((r) => r.r), m = mean(ratios), sd = std(ratios);
        values[solver] = { y: m, lo: m - sd, hi: m + sd, runs: list.length, hits: list.filter((r) => r.r >= 1 - 1e-9).length, t: median(list.map((r) => r.t)) / 1e3 };
      });
      return Object.assign({}, s, { values });
    }).sort((a, b) => seriesRank(a) - seriesRank(b));
    const target = rs.length ? rs[0].art.target : null;
    $('#quality-sub').textContent = rs.length ? `${state.qualityN} assets, select ${target ?? state.qualityN / 2} · paired seeded instances (seed 42)` : '';
    Charts.dots($('#chart-quality'), {
      bands: SOLVERS, series, bandLabel: (b) => SOLVER_LABEL[b] || b,
      x: { min: 0, max: 1.1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: (t) => t.toFixed(2), label: 'approximation ratio (1 = optimum)' },
      tip: (v) => Charts.row(null, 'mean ratio', fmtRatio(v.y)) + Charts.row(null, 'std', fmtRatio(v.hi - v.y)) + Charts.row(null, 'optimal', `${v.hits}/${v.runs}`) + Charts.row(null, 'median time', fmtDur(v.t, true)),
    });
    const rows = series.filter((s) => s.values.qaoa).map((s) => ({ label: s.label, series: s, value: s.values.qaoa.t, tip: Charts.row(s.color, 'median', fmtDur(s.values.qaoa.t, true)) + Charts.row(null, 'mean ratio', fmtRatio(s.values.qaoa.y)) + Charts.row(null, 'solves', s.values.qaoa.runs) }));
    Charts.bars($('#chart-quality-time'), { rows, fmt: (v, precise) => fmtDur(v, precise) });
    appendTable($('#chart-quality'), ['Configuration', 'Solver', 'Mean ratio', 'Std', 'Optimal', 'Median time'],
      series.flatMap((s) => SOLVERS.filter((k) => s.values[k]).map((k) => [s.label, SOLVER_LABEL[k], num(fmtRatio(s.values[k].y)), num(fmtRatio(s.values[k].hi - s.values[k].y)), num(`${s.values[k].hits}/${s.values[k].runs}`), num(fmtDur(s.values[k].t, true))])));
  }

  /* ---------- 02 scaling ---------- */
  const presetKey = (r) => `${r.L}|${r.it}|${r.rs}|${r.op || 'adam'}`;
  const presetLabel = (k) => { const [L, it, rs, op] = k.split('|'); return `p${L} · ${it} it · ${rs} restart${rs > 1 ? 's' : ''}${op && op !== 'adam' ? ` · ${optLabel(op)}` : ''}`; };
  function renderScaling() {
    // offer every QAOA setting that spans at least two sizes; default to the June baseline's
    const pool = recs((r) => r.s === 'qaoa' && (r.art.suite === 'scaling' || r.art.suite === 'quality'));
    const combos = [...groupBy(pool, presetKey).entries()].map(([k, l]) => ({ k, sizes: new Set(l.map((r) => r.n)).size, base: l.some((r) => configOf(r.art) === BASELINE.key && r.art.campaign === BASELINE.campaign && r.art.suite === 'scaling') }))
      .filter((c) => c.sizes > 1).sort((a, b) => b.base - a.base || b.sizes - a.sizes);
    if (!combos.some((c) => c.k === state.scalingPreset)) state.scalingPreset = combos.length ? combos[0].k : null;
    segmented($('#scaling-preset'), combos.map((c) => ({ label: presetLabel(c.k), value: c.k })), state.scalingPreset, (v) => { state.scalingPreset = v; renderScaling(); });
    const q = pool.filter((r) => presetKey(r) === state.scalingPreset);
    const p = { label: state.scalingPreset ? presetLabel(state.scalingPreset) : 'multi-size' };
    const artIds = new Set(q.map((r) => r.a));
    const bySeries = groupBy(q, (r) => seriesOfArtifact(r.art).id);
    const agg = [...bySeries.values()].map((g) => {
      const s = seriesOfArtifact(g[0].art);
      const pts = [...groupBy(g, (r) => r.n).entries()].map(([n, l]) => ({ x: n, t: median(l.map((r) => r.t)) / 1e3, ratio: mean(l.map((r) => r.r)), mem: Math.max(...l.map((r) => r.m || 0)) || null, runs: l.length, conc: l.some((r) => r.art.concurrent), probe: l.some((r) => r.art.probe) }));
      return { s, pts: pts.sort((a, b) => a.x - b.x) };
    }).sort((a, b) => seriesRank(a.s) - seriesRank(b.s));

    // classical references from the same artifacts, per host
    const refs = [];
    const classical = recs((r) => artIds.has(r.a) && (r.s === 'brute_force' || r.s === 'simulated_annealing'));
    groupBy(classical, (r) => `${r.s}|${r.art.host}`).forEach((l, k) => {
      const [solver, host] = k.split('|');
      refs.push({ ref: true, color: REF_HEX, shape: host === 'spark-gb10' ? 'square' : 'circle', label: `${SOLVER_LABEL[solver]} · ${hostLabel(host)}`, short: `${solver === 'brute_force' ? 'brute force' : 'SA'} ${host === 'spark-gb10' ? 'GB10' : 'RTX'}`, points: [...groupBy(l, (r) => r.n).entries()].map(([n, ll]) => ({ x: n, y: median(ll.map((r) => r.t)) / 1e3 })) });
    });
    const xFmt = { label: 'portfolio size n (assets = qubits)', tipLabel: (x) => `${x} assets · 2^${x} amplitudes` };
    const mk = (f) => agg.map(({ s, pts }) => Object.assign({}, s, { points: pts.map((pt) => ({ x: pt.x, y: f(pt), pt })) }));
    const emptyMsg = `No ${p.label} runs spanning several sizes for the selected configurations.`;
    Charts.line($('#chart-scaling-time'), { series: mk((pt) => pt.t).concat(refs), x: xFmt, y: { log: true, fmt: (v) => (v >= 120 ? `${v.toLocaleString('en')} s` : fmtDur(v)), tipFmt: (v, p) => fmtDur(v, true) + (p.pt && p.pt.conc ? ' ·&nbsp;shared&nbsp;host' : '') }, empty: emptyMsg });
    Charts.line($('#chart-scaling-mem'), { series: mk((pt) => (pt.mem ? pt.mem * 1024 : null)), x: xFmt, y: { log: true, fmt: (v) => fmtBytes(v / 1024), tipFmt: (v) => fmtBytes(v / 1024) }, empty: emptyMsg });
    Charts.line($('#chart-scaling-ratio'), { series: mk((pt) => pt.ratio), x: xFmt, y: { min: 0, max: 1, fmt: (v) => v.toFixed(2), tipFmt: (v, pt) => `${fmtRatio(v)} (${pt.pt.runs} solves)` }, empty: emptyMsg });

    const base = agg.find((a) => a.s.key === BASELINE.key && a.s.campaign === BASELINE.campaign);
    const speed = base ? agg.filter((a) => a !== base).map(({ s, pts }) => Object.assign({}, s, { points: pts.map((pt) => { const b = base.pts.find((x) => x.x === pt.x); return b ? { x: pt.x, y: b.t / pt.t } : null; }).filter(Boolean) })) : [];
    Charts.line($('#chart-scaling-speedup'), { series: speed, x: xFmt, y: { log: true, fmt: (v) => `${+v.toPrecision(2)}×`, tipFmt: (v) => `${v.toFixed(1)}×` }, empty: base ? 'No other configuration shares sizes with the June baseline yet.' : 'The June RTX 3080 baseline is filtered out or has no runs at this preset.' });

    appendTable($('#chart-scaling-time'), ['Configuration', 'n', 'Solves', 'Median time', 'Mean ratio', 'Peak Python mem'],
      agg.flatMap(({ s, pts }) => pts.map((pt) => [s.label, num(pt.x), num(pt.runs), num(fmtDur(pt.t, true)), num(fmtRatio(pt.ratio)), num(fmtBytes(pt.mem))])));
  }

  /* ---------- 03 optimizer budget ---------- */
  function renderBudget() {
    const q = recs((r) => r.s === 'qaoa' && r.L === 1 && r.art.suite === 'quality' && !r.art.probe);
    const combos = groupBy(q, (r) => r.n);
    const ns = [...combos.keys()].filter((n) => new Set(combos.get(n).map((r) => `${r.it}|${r.rs}|${r.op}`)).size > 1).sort((a, b) => a - b);
    const box = $('#budget-panels');
    if (!ns.length) {
      $('#budget-n').innerHTML = '';
      box.innerHTML = '<div class="card" style="grid-column:1/-1"><div class="empty">No iteration/restart sweep in the current selection yet. The campaign runs it at 12 assets: <code>--qaoa-iterations 60/150/300 × --qaoa-restarts 2/5</code>.</div></div>';
      return;
    }
    if (!ns.includes(state.budgetN)) state.budgetN = ns.includes(12) ? 12 : ns[0];
    segmented($('#budget-n'), ns.map((n) => ({ label: `${n} assets`, value: n })), state.budgetN, (v) => { state.budgetN = v; renderBudget(); });
    const rs = q.filter((r) => r.n === state.budgetN);
    const panels = [...new Set(rs.map((r) => `${r.op || 'adam'}|${r.rs}`))].sort();
    const allIts = [...new Set(rs.map((r) => r.it))].sort((a, b) => a - b);
    box.innerHTML = '';
    panels.forEach((pk) => {
      const [op, kStr] = pk.split('|');
      const k = +kStr;
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `<h3 class="card-title">${optLabel(op)} · ${k} restart${k > 1 ? 's' : ''}</h3><p class="card-sub">Mean QAOA approximation ratio · ${state.budgetN} assets</p><div class="chart"></div>`;
      box.appendChild(card);
      const sub = rs.filter((r) => r.rs === k && (r.op || 'adam') === op);
      const series = [...groupBy(sub, (r) => seriesOfArtifact(r.art).id).values()].map((g) => Object.assign({}, seriesOfArtifact(g[0].art), {
        points: [...groupBy(g, (r) => r.it).entries()].map(([it, l]) => ({ x: it, y: mean(l.map((r) => r.r)), t: median(l.map((r) => r.t)) / 1e3, runs: l.length })),
      })).sort((a, b) => seriesRank(a) - seriesRank(b));
      Charts.line(card.querySelector('.chart'), { series, legend: true, height: 230, x: { ticks: allIts, label: op === 'cobyla' ? 'maxiter per restart' : 'max iterations per restart', tipLabel: (x) => `${x} iterations · ${k} restart${k > 1 ? 's' : ''}` }, y: { min: 0, max: 1, fmt: (v) => v.toFixed(2), tipFmt: (v, p) => `${fmtRatio(v)} · ${fmtDur(p.t)}` } });
    });
  }

  /* ---------- 04 depth ---------- */
  function renderDepth() {
    const all = recs((r) => r.s === 'qaoa' && r.art.suite === 'layers');
    const ops = [...new Set(all.map((r) => r.op || 'adam'))].sort();
    if (!ops.includes(state.depthOp)) state.depthOp = ops[0];
    const q = all.filter((r) => (r.op || 'adam') === state.depthOp);
    const ns = [...new Set(q.map((r) => r.n))].sort((a, b) => a - b);
    if (!ns.includes(state.depthN)) state.depthN = ns.includes(10) ? 10 : ns[ns.length - 1];
    const ctl = $('#depth-n');
    segmented(ctl, ns.map((n) => ({ label: `${n} assets`, value: n })), state.depthN, (v) => { state.depthN = v; renderDepth(); });
    if (ops.length > 1) {
      const sep = document.createElement('span'); sep.style.width = '.6rem'; ctl.appendChild(sep);
      ops.forEach((o) => ctl.appendChild(chip(optLabel(o), o === state.depthOp, () => { state.depthOp = o; renderDepth(); }, { seg: true })));
    }
    const rs = q.filter((r) => r.n === state.depthN);
    const agg = [...groupBy(rs, (r) => seriesOfArtifact(r.art).id).values()].map((g) => ({
      s: seriesOfArtifact(g[0].art),
      pts: [...groupBy(g, (r) => r.L).entries()].map(([L, l]) => ({ x: L, ratio: mean(l.map((r) => r.r)), t: median(l.map((r) => r.t)) / 1e3, runs: l.length, hits: l.filter((r) => r.r >= 1 - 1e-9).length })),
    })).sort((a, b) => seriesRank(a.s) - seriesRank(b.s));
    const x = { label: 'QAOA layers p', tipLabel: (v) => `p = ${v}` };
    Charts.line($('#chart-depth-ratio'), { series: agg.map(({ s, pts }) => Object.assign({}, s, { points: pts.map((p) => ({ x: p.x, y: p.ratio, p })) })), x, y: { min: 0, max: 1, fmt: (v) => v.toFixed(2), tipFmt: (v, pt) => `${fmtRatio(v)} · ${pt.p.hits}/${pt.p.runs} optimal` } });
    Charts.line($('#chart-depth-time'), { series: agg.map(({ s, pts }) => Object.assign({}, s, { points: pts.map((p) => ({ x: p.x, y: p.t })) })), x, y: { min: 0, fmt: (v) => fmtDur(v), tipFmt: (v) => fmtDur(v, true) } });
  }

  /* ---------- 05 market ---------- */
  function studyName(symbols) {
    const crypto = symbols.filter((s) => s.endsWith('-USD')).length;
    if (crypto === symbols.length) return 'Crypto';
    if (crypto) return 'Mixed stocks + crypto';
    return 'S&P 500 subset';
  }
  function renderMarket() {
    const box = $('#market-studies');
    const studies = (D.studies || []).filter((st) => visibleArt(A[st.a]));
    if (!studies.length) { box.innerHTML = '<div class="empty">No market studies in the current selection.</div>'; return; }
    const groups = groupBy(studies, (st) => st.symbols.join(','));
    box.innerHTML = '';
    groups.forEach((list) => {
      const first = list[0];
      const card = document.createElement('div');
      card.className = 'card';
      const target = A[first.a].target;
      card.innerHTML = `<h3 class="card-title">${studyName(first.symbols)} · select ${target} of ${first.symbols.length}</h3>
        <div class="study-meta"><span><b>Symbols</b> ${first.symbols.map((s) => s.replace('-USD', '')).join(', ')}</span><span><b>Window</b> ${first.start} → ${first.end}</span><span><b>Split</b> ${first.in_rows} in-sample / ${first.out_rows} out-of-sample days</span></div>`;
      const rows = [];
      const latestPerSeries = [...groupBy(list, (st) => seriesOfArtifact(A[st.a]).id).values()].map((l) => l[l.length - 1]).sort((a, b) => seriesRank(seriesOfArtifact(A[a.a])) - seriesRank(seriesOfArtifact(A[b.a])));
      const classicalFrom = latestPerSeries[0];
      SOLVERS.filter((k) => k !== 'qaoa' && classicalFrom.solvers[k]).forEach((k) => rows.push({ who: SOLVER_LABEL[k], s: null, v: classicalFrom.solvers[k] }));
      latestPerSeries.forEach((st) => { if (st.solvers.qaoa) rows.push({ who: 'QAOA', s: seriesOfArtifact(A[st.a]), v: st.solvers.qaoa }); });
      const bestSharpe = Math.max(...rows.map((r) => r.v.sharpe_ratio ?? -Infinity));
      const wrap = document.createElement('div');
      wrap.className = 'table-wrap';
      wrap.innerHTML = `<table><thead><tr><th>Solver</th><th>Selection</th><th class="num">QUBO ratio</th><th class="num">OOS return</th><th class="num">OOS vol</th><th class="num">Sharpe</th><th class="num">Max DD</th><th class="num">Time</th></tr></thead><tbody>${rows.map((r) => `<tr><td><span class="tag">${r.s ? keyHtml(r.s) : ''}${esc(r.who)}${r.s ? ` <span style="color:var(--muted)">· ${esc(r.s.label)}</span>` : ''}</span></td><td>${esc((r.v.assets || []).map((x) => x.replace('-USD', '')).join(', '))}</td><td class="num${r.v.ratio >= 1 - 1e-9 ? ' best' : ''}">${fmtRatio(r.v.ratio)}</td><td class="num">${pct(r.v.annualized_return)}</td><td class="num">${pct(r.v.annualized_volatility).replace('+', '')}</td><td class="num${r.v.sharpe_ratio === bestSharpe ? ' best' : ''}">${r.v.sharpe_ratio?.toFixed(2) ?? '—'}</td><td class="num">${pct(r.v.max_drawdown)}</td><td class="num">${fmtDur(r.v.ms / 1e3, true)}</td></tr>`).join('')}</tbody></table>`;
      card.appendChild(wrap);
      box.appendChild(card);
    });
  }

  /* ---------- 06 runs ---------- */
  function keyHtml(s) {
    const holder = document.createElement('span');
    holder.appendChild(Charts.legendKey(Object.assign({ noLine: true }, s)));
    return holder.innerHTML;
  }
  function renderRuns() {
    const qaoaN = groupBy(R.filter((r) => r.s === 'qaoa'), (r) => r.a);
    const rows = A.filter((a) => state.campaigns.has(a.campaign) && state.configs.has(configOf(a))).map((a) => ({
      created: a.created, s: seriesOfArtifact(a), suite: a.probe ? 'quality · n>20 probe' : a.suite, conc: a.concurrent,
      n: (() => { const ns = [...new Set((qaoaN.get(a.id) || []).map((r) => r.n))].sort((x, y) => x - y); return ns.length > 1 ? `${ns[0]}–${ns[ns.length - 1]}` : ns[0] ?? a.n; })(),
      preset: (() => { const qs = qaoaN.get(a.id) || []; if (!qs.length) return '—'; const Ls = [...new Set(qs.map((r) => r.L))].join(','); const q = qs[0]; return `p${Ls} · ${q.it} it · ${q.rs} rs${q.op && q.op !== 'adam' ? ` · ${optLabel(q.op)}` : ''}`; })(),
      repeats: a.repeats, wall: a.wall_s, mem: a.peak_pool_mb ? a.peak_pool_mb * 1024 : a.max_rss_kb, status: a.status || 'ok', file: a.file.split('/').pop(), notes: a.notes,
    }));
    (D.runs_without_artifact || []).forEach((m) => {
      const s = seriesFor(`${m.host}|${m.backend}`, (m.started_utc || '').replace(/-/g, '').slice(0, 6).replace(/^(\d{4})(\d{2})$/, '$1-$2') || latestCampaign);
      if (!state.configs.has(s.key)) return;
      rows.push({ created: (m.started_utc || '').replace(/[-:]/g, ''), s, suite: m.suite, n: (m.command || '').match(/--assets\s+(\d+)/)?.[1] ?? '—', preset: '—', repeats: (m.command || '').match(/--repeats\s+(\d+)/)?.[1] ?? '—', wall: m.wall_s, mem: m.peak_pool_used_mb ? m.peak_pool_used_mb * 1024 : m.max_rss_kb, status: m.status || 'failed', file: '—', notes: m.notes });
    });
    rows.sort((a, b) => String(b.created).localeCompare(String(a.created)));
    const pill = (st) => `<span class="pill ${st === 'ok' ? 'ok' : st === 'aborted' ? 'warn' : 'bad'}">${esc(st)}</span>`;
    const date = (c) => (c && c.length >= 13 ? `${c.slice(0, 4)}-${c.slice(4, 6)}-${c.slice(6, 8)} ${c.slice(9, 11)}:${c.slice(11, 13)}` : esc(c));
    $('#runs-table').innerHTML = `<table><thead><tr><th>Created (UTC)</th><th>Host · backend</th><th>Suite</th><th class="num">n</th><th>QAOA settings</th><th class="num">Repeats</th><th class="num">Wall</th><th class="num" title="RTX 3080: process max RSS. DGX Spark: whole unified pool in use (includes ~4 GB of other processes).">Peak mem (RSS · pool)</th><th>Status</th><th>Artifact / notes</th></tr></thead><tbody>${rows.map((r) => `<tr><td class="mono">${date(r.created)}</td><td><span class="tag">${keyHtml(r.s)}${esc(r.s.label)}</span></td><td>${esc(r.suite)}</td><td class="num">${esc(r.n)}</td><td class="mono">${esc(r.preset)}</td><td class="num">${esc(r.repeats)}</td><td class="num">${r.wall ? fmtDur(r.wall, true) : '—'}</td><td class="num">${r.mem ? fmtBytes(r.mem) : '—'}</td><td>${pill(r.status)}${r.conc ? ' <span class="pill warn" title="Ran while another benchmark used the host; timings are not isolated">shared</span>' : ''}</td><td class="mono" title="${esc(r.notes)}">${esc(r.file)}${r.notes ? `<br><span style="font-family:var(--sans)">${esc(r.notes)}</span>` : ''}</td></tr>`).join('')}</tbody></table>`;
  }

  /* ---------- table views (accessibility: every chart has its numbers) ---------- */
  const num = (v) => ({ num: true, v });
  function appendTable(chartEl, head, rows) {
    if (!rows.length) return;
    const d = document.createElement('details');
    d.className = 'data';
    d.innerHTML = `<summary>Show data table</summary><div class="table-wrap" style="margin-top:.6rem"><table><thead><tr>${head.map((h, i) => `<th${i > 1 ? ' class="num"' : ''}>${h}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => (c && c.num ? `<td class="num">${esc(c.v)}</td>` : `<td>${esc(c)}</td>`)).join('')}</tr>`).join('')}</tbody></table></div>`;
    chartEl.appendChild(d);
  }

  /* ---------- wiring ---------- */
  function renderAll() {
    renderFilters();
    renderQuality();
    renderScaling();
    renderBudget();
    renderDepth();
    renderMarket();
    renderRuns();
  }
  renderKpis();
  renderAll();

  let resizeTimer;
  let lastWidth = window.innerWidth;
  window.addEventListener('resize', () => {
    if (window.innerWidth === lastWidth) return;
    lastWidth = window.innerWidth;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(renderAll, 150);
  });

  const links = [...document.querySelectorAll('.site-nav a')];
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) links.forEach((l) => l.classList.toggle('active', l.getAttribute('href') === `#${e.target.id}`)); });
  }, { rootMargin: '-40% 0px -55% 0px' });
  document.querySelectorAll('main .section').forEach((s) => obs.observe(s));
})();
