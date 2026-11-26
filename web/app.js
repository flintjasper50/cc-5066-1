/* 样本驱动纹理扩展工作室前端：查看 run、图层、接缝交互、候选追溯、在线生成。 */
'use strict';

const IS_FILE = location.protocol === 'file:';
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function asset(p) {
  if (!p) return '';
  return IS_FILE ? '../' + p : '/' + p;
}
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function f3(x) { return (Math.round(x * 1000) / 1000).toFixed(3); }

const GROUP_LABELS = {
  grain: '随机颗粒 · 常规',
  stripes: '定向条纹 · 常规',
  markers: '独特标记 · 常规',
  'grain-periodic': '随机颗粒 · 周期闭环',
  'stripes-periodic': '定向条纹 · 周期闭环',
  'markers-periodic': '独特标记 · 周期闭环',
  conflict: '互斥锚点 A/B 双样片',
  stress: '压力测试 / 可见失败',
  custom: '自定义生成',
};
const GROUP_ORDER = ['grain', 'stripes', 'markers', 'grain-periodic',
  'stripes-periodic', 'markers-periodic', 'conflict', 'stress', 'custom'];

const state = {
  manifest: null,
  run: null,
  layer: 'result',
  showSeams: true,
  showBoxes: false,
  selected: null,
  hover: null,
  sampleImages: {},
  uploaded: [],
};

/* ---------------- 数据加载 ---------------- */

async function loadManifest() {
  if (IS_FILE) {
    state.manifest = window.QS_MANIFEST || { runs: [] };
  } else {
    const r = await fetch('/api/manifest', { cache: 'no-store' });
    state.manifest = await r.json();
  }
  renderRunList();
  const first = state.manifest.runs[0];
  if (first) selectRun(first.run_id);
}

function loadRunDetail(id) {
  if (IS_FILE) {
    return new Promise((resolve, reject) => {
      const old = $('#run-data-script');
      if (old) old.remove();
      const sc = document.createElement('script');
      sc.id = 'run-data-script';
      sc.src = `data/runs/${encodeURIComponent(id)}.js`;
      sc.onload = () => {
        const d = window.QS_RUN_DATA;
        window.QS_RUN_DATA = null;
        d ? resolve(d) : reject(new Error('run 数据为空'));
      };
      sc.onerror = () => reject(new Error('找不到 ' + sc.src));
      document.head.appendChild(sc);
    });
  }
  return fetch('/api/run?id=' + encodeURIComponent(id), { cache: 'no-store' })
    .then(r => { if (!r.ok) throw new Error('run not found'); return r.json(); });
}

/* ---------------- run 列表 ---------------- */

function renderRunList() {
  const runs = state.manifest.runs || [];
  $('#run-count').textContent = `(${runs.length})`;
  const groups = {};
  for (const r of runs) (groups[r.group] ||= []).push(r);
  const root = $('#run-groups');
  root.innerHTML = '';
  const keys = Object.keys(groups).sort((a, b) => {
    const ia = GROUP_ORDER.indexOf(a), ib = GROUP_ORDER.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  for (const g of keys) {
    const h = document.createElement('div');
    h.className = 'run-group-title';
    h.textContent = GROUP_LABELS[g] || g;
    root.appendChild(h);
    for (const r of groups[g]) {
      const item = document.createElement('div');
      item.className = 'run-item' + (state.run && state.run.run_id === r.run_id ? ' active' : '');
      const seam = r.metrics ? r.metrics.seam.seam_rmse : 0;
      item.innerHTML = `
        <img loading="lazy" src="${asset(r.files.result)}" alt="">
        <div>
          <div class="t">${esc(r.title)}</div>
          <div class="s">seed ${r.config.seed} · 缝 ${f3(seam)} ·
            ${r.config.periodic ? '周期' : '常规'}</div>
        </div>
        <span class="dot ${r.status}" title="${esc(r.status)}"></span>`;
      item.onclick = () => selectRun(r.run_id);
      root.appendChild(item);
    }
  }
}

async function selectRun(id) {
  try {
    state.run = await loadRunDetail(id);
  } catch (e) {
    $('#inspect-empty').textContent = '加载 run 失败：' + e.message;
    return;
  }
  state.selected = null;
  state.sampleImages = {};
  renderRunList();
  const r = state.run;
  $('#view-img').src = asset(r.files[state.layer] || r.files.result);
  $('#view-img').onload = () => fitOverlay();
  renderSampleStrip();
  renderMetrics();
  renderInspectEmpty();
  preloadSampleImages();
  requestAnimationFrame(fitOverlay);
}

/* ---------------- 样本条 ---------------- */

function renderSampleStrip() {
  const r = state.run;
  const strip = $('#sample-strip');
  strip.innerHTML = '';
  (r.samples || []).forEach((s, i) => {
    const key = 'sample_' + i;
    if (!r.files[key]) return;
    const chip = document.createElement('div');
    chip.className = 'sample-chip';
    chip.innerHTML = `<img src="${asset(r.files[key])}" alt=""><div class="cap" title="${esc(s.name)}">原始：${esc(s.name)}</div>`;
    strip.appendChild(chip);
  });
  const res = document.createElement('div');
  res.className = 'sample-chip result';
  res.innerHTML = `<img src="${asset(r.files.result)}"><div class="cap">扩展结果（非放大）</div>`;
  strip.appendChild(res);
}

function preloadSampleImages() {
  const r = state.run;
  (r.samples || []).forEach((s, i) => {
    const im = new Image();
    im.src = asset(r.files['sample_' + i]);
    state.sampleImages[i] = im;
  });
}

/* ---------------- 图层 / overlay ---------------- */

$$('#layer-seg button').forEach(b => b.onclick = () => {
  $$('#layer-seg button').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  state.layer = b.dataset.layer;
  $('#view-img').src = asset(state.run.files[state.layer] || state.run.files.result);
  $('#view-img').onload = fitOverlay;
});
$('#t-seams').onchange = e => { state.showSeams = e.target.checked; drawOverlay(); };
$('#t-boxes').onchange = e => { state.showBoxes = e.target.checked; drawOverlay(); };
window.addEventListener('resize', fitOverlay);

function fitOverlay() {
  const r = state.run;
  const img = $('#view-img'), cv = $('#overlay'), stage = $('#stage');
  if (!r || !img.naturalWidth) return;
  const ir = img.getBoundingClientRect(), sr = stage.getBoundingClientRect();
  cv.style.left = (ir.left - sr.left) + 'px';
  cv.style.top = (ir.top - sr.top) + 'px';
  cv.style.width = ir.width + 'px';
  cv.style.height = ir.height + 'px';
  cv.width = r.config.out_w;
  cv.height = r.config.out_h;
  drawOverlay();
}

function drawOverlay() {
  const r = state.run, cv = $('#overlay');
  if (!r || !cv.getContext) return;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);

  if (state.showBoxes) {
    ctx.lineWidth = 1;
    for (const p of r.placements) {
      ctx.strokeStyle = p.flags.length ? 'rgba(248,113,113,.9)' : 'rgba(255,255,255,.45)';
      for (const rc of p.rects) ctx.strokeRect(rc.x + .5, rc.y + .5, rc.w - 1, rc.h - 1);
    }
  }
  if (state.showSeams && (state.layer === 'result'
      || state.layer === 'provenance_id' || state.layer === 'provenance_xy')) {
    ctx.lineWidth = 1.6;
    ctx.lineJoin = 'round';
    for (const s of r.seam_segments) {
      const hot = state.selected === s.placement;
      ctx.strokeStyle = hot ? '#ffffff' : s.color;
      ctx.globalAlpha = state.selected == null || hot ? .95 : .25;
      ctx.lineWidth = hot ? 2.6 : 1.4;
      ctx.beginPath();
      s.points.forEach(([x, y], i) => i ? ctx.lineTo(x + .5, y + .5) : ctx.moveTo(x + .5, y + .5));
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
  if (state.selected != null) {
    const p = r.placements.find(x => x.id === state.selected);
    if (p) {
      ctx.strokeStyle = '#fde047';
      ctx.lineWidth = 2;
      for (const rc of p.rects) ctx.strokeRect(rc.x + 1.5, rc.y + 1.5, rc.w - 3, rc.h - 3);
    }
  }
}

function eventToXY(e) {
  const img = $('#view-img'), ir = img.getBoundingClientRect();
  const r = state.run;
  return {
    x: Math.floor((e.clientX - ir.left) / ir.width * r.config.out_w),
    y: Math.floor((e.clientY - ir.top) / ir.height * r.config.out_h),
  };
}
function pickPlacement(x, y) {
  const r = state.run;
  let hit = null;
  for (const p of r.placements) {
    for (const rc of p.rects) {
      if (x >= rc.x && x < rc.x + rc.w && y >= rc.y && y < rc.y + rc.h) { hit = p; }
    }
  }
  return hit;
}
$('#overlay').addEventListener('mousemove', e => {
  if (!state.run) return;
  const { x, y } = eventToXY(e);
  const p = pickPlacement(x, y);
  const tip = $('#hover-tip');
  if (p) {
    state.hover = p.id;
    tip.classList.remove('hidden');
    tip.style.left = (e.offsetX + 14) + 'px';
    tip.style.top = (e.offsetY + 12) + 'px';
    tip.innerHTML = `#${p.id} ${esc(p.decision)} · 缝 ${f3(p.seam_worst_rmse)}`;
  } else {
    tip.classList.add('hidden');
  }
});
$('#overlay').addEventListener('mouseleave', () => $('#hover-tip').classList.add('hidden'));
$('#overlay').addEventListener('click', e => {
  const { x, y } = eventToXY(e);
  const p = pickPlacement(x, y);
  state.selected = p ? p.id : null;
  if (p) renderInspect(p); else renderInspectEmpty();
  drawOverlay();
});

/* ---------------- 图块检查器 ---------------- */

function renderInspectEmpty() {
  $('#inspect-empty').classList.remove('hidden');
  $('#inspect-body').classList.add('hidden');
}

function patchCanvas(sampleIdx, sy, sx, size, big) {
  const r = state.run, B = r.config.patch;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = '#222a40';
  ctx.fillRect(0, 0, size, size);
  const draw = im => {
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(im, sx, sy, B, B, 0, 0, size, size);
  };
  const cached = state.sampleImages[sampleIdx];
  if (cached && cached.complete && cached.naturalWidth) {
    draw(cached);
  } else if (r.files['sample_' + sampleIdx]) {
    const im = new Image();
    im.onload = () => draw(im);
    im.src = asset(r.files['sample_' + sampleIdx]);
  }
  if (big) c.style.width = c.style.height = '108px';
  return c;
}

function renderInspect(p) {
  $('#inspect-empty').classList.add('hidden');
  const body = $('#inspect-body');
  body.classList.remove('hidden');
  const c = p.chosen;
  const decBadge = `<span class="badge ${p.flags.length ? 'failed-likely' : 'ok'}">${esc(p.decision)}</span>`;
  const flagHtml = p.flags.map(f => `<span class="badge failed-likely">${esc(f)}</span>`).join(' ');

  body.innerHTML = `
    <div class="section-title">图块 #${p.id}（网格 ${p.ix},${p.iy}） ${decBadge} ${flagHtml}</div>
    <div style="display:flex;gap:12px;align-items:flex-start">
      <div id="chosen-patch"></div>
      <div class="kv" style="flex:1">
        <span class="k">来源样片</span><span>#${c.sample_id} ${esc(c.sample_name)}</span>
        <span class="k">来源左上坐标</span><span>y=${c.source_y}, x=${c.source_x}</span>
        <span class="k">候选匹配 RMSE</span><span>${f3(c.match_rmse)}</span>
        <span class="k">切缝 worst/mean</span><span>${f3(p.seam_worst_rmse)} / ${f3(p.seam_mean_rmse)}</span>
        <span class="k">合格候选池</span><span>${p.eligible_total} 个 · 阈值 MSE ${p.threshold_mse.toExponential(2)}</span>
        <span class="k">子种子</span><code>${p.child_seed}</code>
      </div>
    </div>
    <div class="infobox">${esc(p.reason)}</div>

    <div class="section-title">随机性追踪（每次抽取的均匀随机数）</div>
    <div id="draws"></div>

    <div class="section-title">重叠代价最低候选 Top ${p.top_candidates.length}（绿框=最终选中）</div>
    <div id="candidates"></div>

    <details open>
      <summary>切缝区域误差（mincut vs 同位置硬直切）</summary>
      <div id="regions"></div>
    </details>
    <details>
      <summary>全部尝试 ${p.attempts.length}（重排/重采样可追踪）</summary>
      <div id="attempts"></div>
    </details>`;

  const host = $('#chosen-patch', body);
  const cv = patchCanvas(c.sample_id, c.source_y, c.source_x, 108, true);
  cv.title = '所选 B×B 图块';
  cv.style.border = '1px solid #2f9e6e';
  cv.style.borderRadius = '8px';
  host.appendChild(cv);

  $('#draws', body).innerHTML = (p.draws || []).map(d => `
    <div class="attempt"><span>${esc(d.purpose)}</span>
      <span>u=${f3(d.u)} pool=${d.pool} → #${d.pick_index}
      [s${d.picked.sample_id} @${d.picked.source_y},${d.picked.source_x}]</span></div>`).join('')
    || '<div class="hint">无额外随机抽取</div>';

  const cands = $('#candidates', body);
  p.top_candidates.forEach(t => {
    const picked = t.sample_id === c.sample_id && t.source_y === c.source_y
      && t.source_x === c.source_x;
    const row = document.createElement('div');
    row.className = 'cand-row' + (picked ? ' picked' : '');
    const thumb = patchCanvas(t.sample_id, t.source_y, t.source_x, 46);
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.innerHTML = `s${t.sample_id} @${t.source_y},${t.source_x}
      ${picked ? '· <b style="color:#34d399">选中</b>' : ''}`;
    const cost = document.createElement('div');
    cost.className = 'cost';
    cost.innerHTML = `RMSE ${f3(Math.sqrt(Math.max(0, t.mse)))}`;
    row.append(thumb, meta, cost);
    cands.appendChild(row);
  });

  $('#regions', body).innerHTML = p.regions.map(rg => `
    <div class="attempt"><span>${esc(rg.kind)} (${rg.axis}, ${rg.orient})</span>
      <span>mincut ${f3(rg.rmse)} · 直切 ${f3(rg.straight_rmse)}</span></div>`).join('');

  $('#attempts', body).innerHTML = p.attempts.map(a => `
    <div class="attempt">
      <span class="st">${esc(a.stage)}</span>
      <span>s${a.key.sample_id} @${a.key.source_y},${a.key.source_x}
        · 匹配 ${f3(a.match_rmse)} · 缝 ${f3(a.seam_worst_rmse)}</span>
    </div>`).join('');
}

/* ---------------- 指标页 ---------------- */

function metricCard(label, value, cls) {
  return `<div class="metric"><div class="v ${cls || ''}">${value}</div>
          <div class="l">${label}</div></div>`;
}
function rmseClass(v, ref) { return v <= ref * 1.15 ? 'good' : v <= ref * 1.5 ? 'mid' : 'bad'; }

function orientationCards(m) {
  const o = m.orientation || {};
  return metricCard('输出方向相干性', f3(o.orientation_coherence || 0))
    + metricCard('输出主方向', (o.orientation_deg || 0).toFixed(1) + '°');
}

function anchorCards(m) {
  const fc = m.feature_cuts || {};
  if (!fc.available) {
    return `<div class="metric" style="grid-column:1/-1">
      <div class="v" style="font-size:12.5px;color:var(--muted)">不适用</div>
      <div class="l">语义锚点切割仅对自带真值锚点掩膜的准备样本（独特标记 / 互斥箭头）评估；用户上传样本无掩膜。</div>
    </div>`;
  }
  // 兼容旧版 run.json 的 HSV 字段
  if (fc.anchor_cut_events == null) {
    const frac = fc.feature_hit_fraction || 0;
    return metricCard('接缝切过显著结构比例(旧)', f3(frac),
        frac > .3 ? 'bad' : frac > .2 ? 'mid' : 'good')
      + metricCard('特征切割像素对',
        (fc.feature_inside_hits || 0) + '/' + (fc.feature_edge_hits || 0), '');
  }
  const fcs = m.feature_cuts_straight || {};
  const evCls = fc.anchor_cut_events >= 22 ? 'bad'
    : fc.anchor_cut_events >= 5 ? 'mid' : 'good';
  const sevCls = fc.anchor_severe_events >= 8 ? 'bad'
    : fc.anchor_severe_events >= 2 ? 'mid' : 'good';
  let cmp = '';
  if (fcs.available && fcs.anchor_cut_events != null) {
    const d = fcs.anchor_cut_events - fc.anchor_cut_events;
    cmp = d > 0 ? `硬切基线 ${fcs.anchor_cut_events}，最小割少切 ${d} 次`
      : d < 0 ? `硬切基线仅 ${fcs.anchor_cut_events}，本次反增`
      : `与硬切基线相同（${fc.anchor_cut_events}）`;
  }
  return metricCard('锚点切割事件', fc.anchor_cut_events, evCls)
    + metricCard('其中严重长切', fc.anchor_severe_events, sevCls)
    + `<div class="metric" style="grid-column:1/-1">
        <div class="l">连续接缝穿过同一真值锚点笔画内部计为一次切割；
        内部像素对 ${fc.anchor_inside_hits}，切边 ${fc.anchor_edge_hits}。
        ${cmp ? '<br>' + esc(cmp) : ''}</div>
      </div>`;
}

function renderMetrics() {
  const r = state.run, m = r.metrics, cfg = r.config;
  const ref = m.samples.reduce((a, s) => Math.min(a, s.natural_adj_rmse), 1);
  const imp = m.seam_improvement_vs_straight_pct;
  const reproduce = buildCommand(r);
  const links = Object.entries(r.files).filter(([k]) =>
    ['comparison', 'patch_wall', 'tile2x2', 'seams', 'provenance_id',
     'provenance_xy', 'tile_grid', 'tileid', 'source_map_npz'].includes(k))
    .map(([k, v]) => `<a class="link" target="_blank" href="${asset(v)}">${k}</a>`).join(' · ');

  $('#metrics-body').innerHTML = `
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
      <span class="badge ${r.status}">${statusText(r.status)}</span>
      <strong>${esc(r.title)}</strong>
    </div>
    <div class="hint">run_id <code>${r.run_id}</code> · ${r.n_placements} 图块 · ${m.elapsed_sec}s</div>

    ${r.warnings.map(w => `<div class="warnbox">⚠ ${esc(w)}</div>`).join('')}
    ${(r.infos || []).map(w => `<div class="infobox">ℹ ${esc(w)}</div>`).join('')}

    <div class="section-title">接缝误差（合成后沿真实 tile 边界测量）</div>
    <div class="metric-grid">
      ${metricCard('mincut 接缝 RMSE', f3(m.seam.seam_rmse), rmseClass(m.seam.seam_rmse, ref))}
      ${metricCard('硬直切 RMSE（同图块）', f3(m.straight_seam.seam_rmse), '')}
      ${metricCard('相对直切改善', imp.toFixed(1) + '%', imp >= 5 ? 'good' : 'mid')}
      ${metricCard('原样本自然相邻差', f3(ref), '')}
      <div class="metric"><div class="v">${f3(m.seam_to_natural_ratio)}×</div>
        <div class="l">接缝 / 自然地板</div>
        <div class="bar"><i style="width:${Math.min(100, m.seam_to_natural_ratio / 1.6 * 100)}%"></i></div>
      </div>
      ${metricCard('朴素平铺边界 RMSE', f3(m.naive_tile.naive_tile_seam_rmse), 'bad')}
      ${cfg.periodic ? metricCard('周期首尾边 RMSE', f3(m.periodic.edge_rmse),
        rmseClass(m.periodic.edge_rmse, ref)) : ''}
      ${cfg.periodic && m.seam.closure_seam_rmse != null
        ? metricCard('闭环切缝 RMSE', f3(m.seam.closure_seam_rmse),
                     rmseClass(m.seam.closure_seam_rmse, ref)) : ''}
    </div>

    <div class="section-title">重复痕迹</div>
    <div class="metric-grid">
      ${metricCard('不同来源图块', `${m.repetition.distinct_source_tiles}/${m.repetition.placements}`)}
      ${metricCard('单块最大复用', m.repetition.max_exact_reuse,
        m.repetition.max_exact_reuse >= 3 ? 'bad' : 'good')}
      ${metricCard('高重叠来源对', m.repetition.high_source_overlap_pairs)}
      ${metricCard('自相关峰值', f3(m.autocorrelation.autocorr_peak),
        m.autocorrelation.autocorr_peak > .55 ? 'bad' : '')}
      ${metricCard('自相关 lag', `(${m.autocorrelation.autocorr_lag.join(',')})`)}
      ${metricCard('随机图块重复率', f3(m.duplicates.dup_pair_rate))}
    </div>

    <div class="section-title">方向结构与语义切割</div>
    <div class="metric-grid">
      ${orientationCards(m)}
      ${anchorCards(m)}
    </div>
    <div class="kv" style="margin-top:6px">
      ${m.samples.map(s => `<span class="k">样本</span>
        <span>${esc(s.name)}：相干 ${f3(s.orientation_coherence)} / ${s.orientation_deg.toFixed(1)}°
        · 自然差 ${f3(s.natural_adj_rmse)}</span>`).join('')}
    </div>

    <div class="section-title">干预统计</div>
    <div class="kv">
      <span class="k">切缝重排</span><span>${m.interventions.reroute} 次</span>
      <span class="k">来源重采样</span><span>${m.interventions.resample} 次</span>
      <span class="k">仍无解</span><span>${m.interventions.unresolved} 块</span>
    </div>

    <div class="section-title">复现（真实种子 ${cfg.seed}）</div>
    <pre class="cmd" id="cmd-text">${esc(reproduce)}</pre>
    <button class="btn mini" id="btn-copy-cmd">复制命令</button>

    <div class="section-title">产物文件</div>
    <div style="font-size:11.5px;line-height:2">${links}</div>`;

  $('#btn-copy-cmd').onclick = () => {
    navigator.clipboard?.writeText(reproduce);
    $('#btn-copy-cmd').textContent = '已复制 ✓';
  };
}

function statusText(s) {
  return { ok: '通过 / 无明显失败', warning: '有局部问题（可见告警）',
           'failed-likely': '很可能失败 / 结构超出能力' }[s] || s;
}

function buildCommand(r) {
  const c = r.config;
  let sampleKey = (r.group || 'grain').replace('-periodic', '');
  if (sampleKey === 'stress') sampleKey = 'markers';
  if (sampleKey === 'custom') sampleKey = 'grain';
  const sampleArg = (r.samples.length === 1 && String(r.samples[0].id).startsWith('upload'))
    ? '--images <你的图片路径>' : `--sample ${sampleKey}`;
  return `/usr/bin/python3 cli.py run ${sampleArg} \\\n` +
    `  --seed ${c.seed} --width ${c.out_w} --height ${c.out_h} \\\n` +
    `  --patch ${c.patch} --overlap ${c.overlap} --policy ${c.policy}` +
    (c.periodic ? ' --periodic' : '');
}

/* ---------------- tabs ---------------- */
$$('.right .tabs button').forEach(b => b.onclick = () => {
  $$('.right .tabs button').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  $$('.tab-panel').forEach(p => p.classList.add('hidden'));
  $('#tab-' + b.dataset.tab).classList.remove('hidden');
  requestAnimationFrame(fitOverlay);
});

/* ---------------- 生成表单 ---------------- */

$('#f-sample').onchange = e => {
  $('#upload-row').classList.toggle('hidden', e.target.value !== 'upload');
};
$('#f-patch').oninput = e => $('#v-patch').textContent = e.target.value;
$('#f-overlap').oninput = e => $('#v-overlap').textContent = e.target.value;
$('#btn-dice').onclick = () => {
  $('#f-seed').value = Math.floor(Math.random() * 2_000_000_000);
};
$('#btn-refresh').onclick = loadManifest;

if (IS_FILE) {
  $('#mode-pill').textContent = 'file:// 只读模式';
  $('#server-notice').classList.remove('hidden');
  $('#btn-generate').disabled = true;
} else {
  $('#mode-pill').textContent = '本地服务模式 · 可在线生成';
}

async function uploadFiles(files) {
  state.uploaded = [];
  for (const file of files) {
    const buf = await file.arrayBuffer();
    const r = await fetch('/api/upload?name=' + encodeURIComponent(file.name), {
      method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: buf,
    });
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error || '上传失败');
    state.uploaded.push(j.name);
  }
}

$('#btn-generate').onclick = async () => {
  const status = $('#gen-status');
  const sample = $('#f-sample').value;
  try {
    if (sample === 'upload') {
      const files = $('#f-files').files;
      if (!files.length) throw new Error('请先选择 1–2 张图片');
      status.textContent = '上传中…';
      await uploadFiles(files);
    }
    const [w, h] = $('#f-size').value.split('x').map(Number);
    const params = {
      sample,
      uploads: state.uploaded,
      seed: Number($('#f-seed').value) || 1,
      out_w: w, out_h: h,
      patch: Number($('#f-patch').value),
      overlap: Number($('#f-overlap').value),
      periodic: $('#f-periodic').checked,
      policy: $('#f-policy').value,
    };
    status.textContent = '合成中（含硬切基线，约 1–5 秒）…';
    const r = await fetch('/api/generate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error || '生成失败');
    status.textContent = `完成：${j.run_id}（${j.status}）`;
    await loadManifest();
    selectRun(j.run_id);
  } catch (e) {
    status.textContent = '错误：' + e.message;
  }
};

/* ---------------- 启动 ---------------- */
loadManifest().catch(e => {
  $('#run-groups').innerHTML =
    `<div class="warnbox">manifest 加载失败：${esc(e.message)}。<br>
     先运行 <code>/usr/bin/python3 run_all.py</code> 生成实验数据。</div>`;
});
