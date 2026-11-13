// Pipeline viewer. Two modes share every line below the data layer:
//   recorded: replays data/demo.json, captured from a real run of the pipeline (static hosting, no backend)
//   live:     talks to the local FastAPI service (/health, /index/map, /query/stream)
const $ = (s) => document.querySelector(s);
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const STEPS = [['Ingest', 'load'], ['Chunk', 'split'], ['Embed', 'vectors'], ['Retrieve', 'top-k'], ['Generate', 'answer']];
const KIND = { direct: 'answerable', paraphrase: 'answerable, reworded', near_miss: 'not in the documents', off_topic: 'off topic' };
const STOP = new Set('the and for with that this from have has had are was were been not but you your our their they them then than what which when where who how can could should would will into about over under there here does did its'.split(' '));

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const ease = (t) => 1 - Math.pow(1 - t, 3);
const lerp = (a, b, t) => a + (b - a) * t;
const fmt = (n, d = 0) => Number(n).toLocaleString('en-GB', { minimumFractionDigits: d, maximumFractionDigits: d });
const sleep = (ms) => new Promise((r) => setTimeout(r, REDUCED ? 0 : ms));

// ---------------------------------------------------------------- stage (canvas) -------------
class Stage {
  constructor(canvas) {
    this.c = canvas; this.ctx = canvas.getContext('2d');
    this.chunks = []; this.docs = [];
    this.v = { split: 0, morph: 0, q: 0, line: 0, hit: 0 };
    this.tweens = []; this.raf = 0; this.visible = true; this.hits = new Map(); this.qxy = null; this.t0 = performance.now();
    new ResizeObserver(() => this.resize()).observe(canvas);
    new IntersectionObserver((e) => { this.visible = e[0].isIntersecting; if (this.visible) this.kick(); }).observe(canvas);
  }
  setData(chunks) {
    this.chunks = chunks;
    const docs = new Map();
    for (const c of chunks) { const d = docs.get(c.source) || { name: c.source, len: 0 }; d.len = Math.max(d.len, c.end); docs.set(c.source, d); }
    this.docs = [...docs.values()];
    this.docIdx = new Map(this.docs.map((d, i) => [d.name, i]));
    this.layout(); this.kick();
  }
  resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = this.c.clientWidth, h = this.c.clientHeight;
    if (!w || !h) return;
    this.c.width = Math.round(w * dpr); this.c.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = w; this.h = h; this.layout(); this.kick();
  }
  layout() {
    if (!this.w || !this.chunks.length) return;
    const w = this.w, h = this.h, padX = 14, top = 10, bottom = 10;
    this.small = w < 560;
    const nd = this.docs.length, rh = (h - top - bottom) / nd;
    this.barH = Math.min(this.small ? 9 : 12, rh * 0.42);
    this.barX0 = padX; this.barW = w - padX * 2;
    const xs = this.chunks.map((c) => c.x), ys = this.chunks.map((c) => c.y);
    this.bx = [Math.min(...xs), Math.max(...xs)]; this.by = [Math.min(...ys), Math.max(...ys)];
    const mp = 26;
    this.mapX = (x) => mp + ((x - this.bx[0]) / (this.bx[1] - this.bx[0] || 1)) * (w - 2 * mp);
    this.mapY = (y) => mp + 18 + ((y - this.by[0]) / (this.by[1] - this.by[0] || 1)) * (h - 2 * mp - 18);
    this.pos = this.chunks.map((c) => {
      const di = this.docIdx.get(c.source), d = this.docs[di], rowY = top + di * rh;
      const x0 = this.barX0 + (c.start / d.len) * this.barW, x1 = this.barX0 + (c.end / d.len) * this.barW;
      return { bx: (x0 + x1) / 2, bw: x1 - x0, by: rowY + rh - this.barH / 2 - 3, mx: this.mapX(c.x), my: this.mapY(c.y), row: rowY };
    });
  }
  tween(key, to, dur) {
    if (REDUCED || dur <= 0) { this.v[key] = to; this.kick(); return Promise.resolve(); }
    return new Promise((res) => {
      this.tweens = this.tweens.filter((t) => t.key !== key);
      this.tweens.push({ key, from: this.v[key], to, dur, t0: performance.now(), res });
      this.kick();
    });
  }
  setHits(ids, xy) { this.hits = new Map(ids.map((id, i) => [id, i + 1])); this.qxy = xy; this.kick(); }
  kick() { if (!this.raf && this.visible) this.raf = requestAnimationFrame((t) => this.frame(t)); }
  frame(now) {
    this.raf = 0;
    for (const t of this.tweens) {
      const p = Math.min(1, (now - t.t0) / t.dur);
      this.v[t.key] = lerp(t.from, t.to, ease(p));
      if (p >= 1) t.done = true;
    }
    this.tweens.filter((t) => t.done).forEach((t) => t.res());
    this.tweens = this.tweens.filter((t) => !t.done);
    this.draw(now);
    if (this.tweens.length || (this.v.q > 0.99 && this.hits.size)) this.kick();
  }
  draw(now) {
    const { ctx, w, h, v } = this;
    if (!w || !this.pos) return;
    ctx.clearRect(0, 0, w, h);
    ctx.font = '400 10px "JetBrains Mono", ui-monospace, monospace';
    ctx.textBaseline = 'alphabetic';
    const m = ease(v.morph), s = ease(v.split);
    // document labels fade out as the bars turn into points
    if (m < 0.98) {
      ctx.fillStyle = `rgba(237,236,230,${0.45 * (1 - m)})`;
      this.docs.forEach((d, i) => {
        const rowY = this.pos.find((p, j) => this.docIdx.get(this.chunks[j].source) === i)?.row ?? 0;
        ctx.fillText(d.name.replace('.txt', '').replace(/_/g, ' '), this.barX0, rowY + 10);
      });
    }
    // query -> hit lines
    const hitIdx = [...this.hits.keys()].map((id) => this.chunks.findIndex((c) => c.chunk_id === id));
    const qx = this.qxy ? this.mapX(this.qxy[0]) : 0, qy = this.qxy ? this.mapY(this.qxy[1]) : 0;
    const qStart = { x: w / 2, y: 14 };
    const qpx = lerp(qStart.x, qx, ease(v.q)), qpy = lerp(qStart.y, qy, ease(v.q));
    if (this.qxy && v.q > 0 && m > 0.9) {
      for (const i of hitIdx) {
        if (i < 0) continue;
        const p = this.pos[i], lt = ease(v.line);
        ctx.strokeStyle = `rgba(215,255,63,${0.55 * lt})`; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(qpx, qpy);
        ctx.lineTo(lerp(qpx, p.mx, lt), lerp(qpy, p.my, lt)); ctx.stroke();
      }
    }
    // chunks
    const hot = ease(v.hit), r0 = this.small ? 3.2 : 4;
    this.chunks.forEach((c, i) => {
      const p = this.pos[i], rank = this.hits.get(c.chunk_id);
      const isHit = rank && hot > 0;
      const gap = 2 * s;
      const bw = Math.max(1, p.bw - gap);
      const rad = isHit ? r0 * (1 + 0.9 * hot) : r0;
      const cx = lerp(p.bx, p.mx, m), cy = lerp(p.by, p.my, m);
      const width = lerp(bw, rad * 2, m), height = lerp(this.barH, rad * 2, m);
      const dim = this.hits.size && hot > 0 ? lerp(1, 0.32, hot) : 1;
      let col;
      if (isHit) col = `rgba(215,255,63,${lerp(0.5, 1, hot)})`;
      else col = `rgba(237,236,230,${(c.chunk % 2 ? 0.34 : 0.55) * dim * lerp(1, 0.9, m)})`;
      if (isHit && hot > 0.2) { ctx.shadowColor = 'rgba(215,255,63,.8)'; ctx.shadowBlur = 12 * hot; }
      ctx.fillStyle = col;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(cx - width / 2, cy - height / 2, width, height, Math.min(height / 2, lerp(1.5, height / 2, m)));
      else ctx.rect(cx - width / 2, cy - height / 2, width, height);
      ctx.fill(); ctx.shadowBlur = 0;
      if (isHit && hot > 0.6 && m > 0.9) {
        ctx.fillStyle = `rgba(237,236,230,${hot})`;
        ctx.fillText(String(rank), cx + rad + 3, cy - rad - 1);
      }
    });
    // the query point
    if (this.qxy && v.q > 0) {
      const a = Math.min(1, v.q * 2), pulse = REDUCED ? 0 : (Math.sin((now - this.t0) / 380) + 1) / 2;
      ctx.strokeStyle = `rgba(237,236,230,${a})`; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.arc(qpx, qpy, 6 + pulse * 2, 0, 7); ctx.stroke();
      ctx.fillStyle = `rgba(237,236,230,${a})`;
      ctx.beginPath(); ctx.arc(qpx, qpy, 2.2, 0, 7); ctx.fill();
    }
  }
}

// ---------------------------------------------------------------- app ------------------------
const state = { mode: 'recorded', demo: null, step: 0, run: 0, current: null };
let stage;

function setCaption(t) { $('#caption').textContent = t; }

function buildRail() {
  $('#rail').innerHTML = STEPS.map(([n, s], i) =>
    `<li><button type="button" data-i="${i}">${n}<b>${s}</b></button></li>`).join('');
  $('#rail').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    goStep(+b.dataset.i);
  });
}

function markRail(i) {
  state.step = i;
  document.querySelectorAll('#rail button').forEach((b) => {
    const k = +b.dataset.i;
    b.toggleAttribute('aria-current', k === i);
    if (k === i) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current');
    b.classList.toggle('done', k < i);
  });
}

function captionFor(i) {
  const d = state.demo, m = d.meta, r = m.retrieval, q = state.current;
  if (i === 0) return `${m.corpus.documents} documents, ${fmt(m.corpus.words)} words, one bar each. Bar length is the document length.`;
  if (i === 1) return `Split into ${m.corpus.chunks} chunks of at most ${r.chunk_size} characters (${r.chunk_overlap} overlap) with RecursiveCharacterTextSplitter.`;
  if (i === 2) return `Each chunk becomes a ${m.embedder.dim}-number vector (${m.embedder.short}). Drawn here by PCA in 2-D, so distances are only indicative.`;
  if (i === 3) return q ? `${m.retrieval.retriever} retrieval returns the top ${r.k} of ${m.corpus.chunks} chunks for the question.` : 'Choose a question to retrieve for.';
  return q ? (q.done.abstained ? 'The model is asked to answer from the retrieved text only, and it declines.' : `The ${m.llm.short} model answers from the retrieved text only, streamed token by token.`) : 'Choose a question to generate for.';
}

async function goStep(i, { animate = true } = {}) {
  const run = ++state.run;
  markRail(i);
  setCaption(captionFor(i));
  const dur = animate ? 900 : 0;
  await Promise.all([
    stage.tween('split', i >= 1 ? 1 : 0, dur),
    stage.tween('morph', i >= 2 ? 1 : 0, dur * 1.3),
  ]);
  if (run !== state.run) return;
  if (i >= 3 && state.current) await showQuery(state.current, i === 4);
  else if (i < 3) { stage.tween('q', 0, 300); stage.tween('line', 0, 300); stage.tween('hit', 0, 300); stage.setHits([], null); }
}

async function playIntro() {
  const run = ++state.run;
  stage.v.split = 0; stage.v.morph = 0; stage.setHits([], null); stage.v.q = stage.v.line = stage.v.hit = 0;
  for (let i = 0; i <= 2; i++) {
    markRail(i); setCaption(captionFor(i));
    await Promise.all([stage.tween('split', i >= 1 ? 1 : 0, 800), stage.tween('morph', i >= 2 ? 1 : 0, 1100)]);
    if (run !== state.run) return;
    await sleep(i === 2 ? 200 : 900);
    if (run !== state.run) return;
  }
}

// ---- answer area
function resetAnswer(question) {
  $('#qline').textContent = question;
  $('#ans').className = 'ans'; $('#ans').innerHTML = '';
  $('#cites').innerHTML = ''; $('#timings').innerHTML = ''; $('#why').textContent = ''; $('#why').className = 'why mono';
  const chip = $('#stateChip'); chip.textContent = 'retrieving'; chip.className = 'chip mono';
  $('#chunks').innerHTML = '';
}

function contentWords(text) {
  return new Set((text.toLowerCase().match(/[a-z0-9]+(?:[.:'][a-z0-9]+)*/g) || []).filter((w) => /\d/.test(w) || (w.length > 3 && !STOP.has(w))));
}

function highlight(text, words, cls) {
  return esc(text).replace(/[A-Za-z0-9]+(?:[.:'][A-Za-z0-9]+)*/g, (w) => {
    const l = w.toLowerCase();
    return words.has(l) || words.has(l.replace(/s$/, '')) ? `<${cls}>${w}</${cls}>` : w;
  });
}

function renderChunks(sources, question, answer, abstained) {
  const aw = abstained ? contentWords(question) : contentWords(answer || '');
  const cls = abstained ? 'u' : 'mark';
  const maxBm = Math.max(1e-6, ...sources.map((s) => s.bm25));
  const host = $('#chunks'); host.innerHTML = '';
  sources.forEach((s, i) => {
    const el = document.createElement('article'); el.className = 'chunk'; el.id = `chunk-${i + 1}`;
    const pct = (x) => Math.max(0, Math.min(1, x));
    el.style.setProperty('--d', `${i * 90}ms`);
    el.innerHTML = `<header class="mono"><b>${i + 1}</b><span class="id">${esc(s.chunk_id)}</span><span>chars ${s.start}-${s.end}</span></header>
      <p>${highlight(s.text, aw, cls)}</p>
      <div class="scores"><div>dense cosine <b>${s.dense.toFixed(3)}</b><div class="bar"><i style="--w:${pct(s.dense)}"></i></div></div>
      <div>BM25 <b>${s.bm25.toFixed(2)}</b><div class="bar"><i style="--w:${pct(s.bm25 / maxBm)}"></i></div></div>
      <div>rank score <b>${s.score.toFixed(3)}</b><div class="bar"><i style="--w:${pct(s.score / Math.max(...sources.map((x) => x.score), 1e-6))}"></i></div></div></div>`;
    host.appendChild(el);
    setTimeout(() => el.classList.add('on'), REDUCED ? 0 : 120 + i * 140);
  });
}

function finishAnswer(done, q) {
  const chip = $('#stateChip'), ans = $('#ans');
  const t = done.timings || {};
  if (done.abstained) {
    ans.className = 'ans declined'; ans.textContent = "I don't know";
    chip.textContent = 'declined'; chip.className = 'chip mono no';
    const why = { model_declined: 'The model read the retrieved chunks and found no answer in them. Underlined words are the question terms that did match, which is why those chunks came back.',
      low_retrieval_score: 'The best chunk scored below the similarity gate, so the model was never called.', empty_index: 'No documents are indexed yet.' };
    $('#why').textContent = why[done.abstain_reason] || 'No answer.'; $('#why').className = 'why mono no';
  } else {
    ans.className = 'ans'; ans.textContent = done.answer;
    chip.textContent = 'answered'; chip.className = 'chip mono ok';
    const best = Math.max(...done.sources.map((s) => s.support ?? 0));
    $('#cites').innerHTML = done.sources.map((s, i) =>
      `<button type="button" class="cite${(s.support ?? 0) >= 0.5 ? '' : ' weak'}" data-i="${i + 1}">[${i + 1}] ${esc(s.chunk_id)} <span>${Math.round((s.support ?? 0) * 100)}%</span></button>`).join('');
    $('#why').textContent = `Percentages are the share of the answer's content words found in that chunk (best ${Math.round(best * 100)}%).`;
    renderChunks(done.sources, state.current?.question || '', done.answer, false);
  }
  $('#timings').innerHTML = `<dt>retrieve</dt><dd>${fmt(t.retrieve_ms, 0)} ms</dd><dt>generate</dt><dd>${t.generate_ms ? fmt(t.generate_ms, 0) + ' ms' : 'not called'}</dd><dt>retriever</dt><dd>${esc(done.retriever)}</dd><dt>model</dt><dd>${esc(done.model.llm.replace('local:', ''))}</dd>`;
}

document.addEventListener('click', (e) => {
  const c = e.target.closest('.cite'); if (!c) return;
  const el = document.getElementById(`chunk-${c.dataset.i}`); if (!el) return;
  el.scrollIntoView({ behavior: REDUCED ? 'auto' : 'smooth', block: 'center' });
  el.classList.add('flash'); setTimeout(() => el.classList.remove('flash'), 1400);
});

// ---- a single query, replayed or live. Both drive the same three callbacks.
async function showQuery(q, withAnswer) {
  const run = state.run;
  resetAnswer(q.question);
  const ev = q.retrieval;
  const ids = ev.sources.map((s) => s.chunk_id);
  stage.setHits(ids, ev.query_xy);
  await stage.tween('q', 1, 900);
  if (run !== state.run) return;
  stage.tween('line', 1, 700); stage.tween('hit', 1, 800);
  renderChunks(ev.sources, q.question, '', true);
  $('#stateChip').textContent = 'retrieved';
  if (!withAnswer) return;
  setCaption(captionFor(4)); markRail(4);
  await sleep(700);
  if (run !== state.run) return;
  const ans = $('#ans');
  if (q.done.abstained && !q.tokens.length) { await sleep(400); finishAnswer(q.done, q); return; }
  $('#stateChip').textContent = 'generating';
  const total = q.tokens.length ? q.tokens[q.tokens.length - 1].t : 0;
  const speed = total > 6500 ? total / 6500 : 1;
  let text = '', t0 = performance.now();
  for (const tok of q.tokens) {
    const wait = tok.t / speed - (performance.now() - t0);
    if (wait > 0) await sleep(wait);
    if (run !== state.run) return;
    text += tok.text;
    ans.innerHTML = esc(text) + '<span class="caret"></span>';
  }
  finishAnswer(q.done, q);
  if (!q.done.abstained) renderChunks(q.done.sources, q.question, q.done.answer, false);
  else renderChunks(q.done.sources, q.question, '', true);
  document.querySelectorAll('.chunk').forEach((c) => c.classList.add('on'));
}

function buildChips() {
  const host = $('#qchips');
  host.innerHTML = state.demo.queries.map((q, i) =>
    `<button type="button" class="qchip${q.done.abstained ? ' decline' : ''}" data-i="${i}" aria-pressed="false">${esc(q.question)}<small>${KIND[q.kind]}</small></button>`).join('');
  host.addEventListener('click', (e) => {
    const b = e.target.closest('.qchip'); if (!b) return;
    pickQuery(+b.dataset.i);
  });
}

async function pickQuery(i) {
  const q = state.demo.queries[i];
  state.current = q;
  document.querySelectorAll('.qchip').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.i === i)));
  const run = ++state.run;
  stage.setHits([], null); stage.v.q = 0; stage.v.line = 0; stage.v.hit = 0;
  await Promise.all([stage.tween('split', 1, 300), stage.tween('morph', 1, 500)]);
  if (run !== state.run) return;
  markRail(3); setCaption(captionFor(3));
  await showQuery(q, true);
}

// ---- results tables, generated from results/*.json by scripts/check_numbers.py
async function renderResults() {
  try {
    const r = await (await fetch('data/results.json')).json();
    $('#resNote').textContent = r.note;
    $('#resBody').innerHTML = r.tables.map((t) => `<div class="table-wrap"><table class="results"><caption>${esc(t.title)}</caption>
      <thead><tr>${t.head.map((h) => `<th>${esc(h)}</th>`).join('')}</tr></thead>
      <tbody>${t.rows.map((row, i) => `<tr${t.hl === i ? ' class="hl"' : ''}>${row.map((c) => `<td>${esc(c)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`).join('');
    $('#notesList').innerHTML = r.notes.map((n) => `<li>${esc(n)}</li>`).join('');
  } catch (e) {
    $('#resBody').innerHTML = '<p class="muted">Results are not available in this build.</p>';
  }
}

// ---- live mode
async function detectLive() {
  try {
    const res = await fetch('health', { headers: { Accept: 'application/json' } });
    if (!res.ok || !(res.headers.get('content-type') || '').includes('json')) return null;
    return await res.json();
  } catch { return null; }
}

async function liveChunks() {
  const m = await (await fetch('index/map')).json();
  return m.chunks;
}

function parseSSE(buf) {
  const out = []; const parts = buf.split('\n\n'); const rest = parts.pop();
  for (const p of parts) {
    const ev = /^event: (.+)$/m.exec(p)?.[1], data = /^data: (.+)$/m.exec(p)?.[1];
    if (ev && data) out.push([ev, JSON.parse(data)]);
  }
  return [out, rest];
}

async function askLive(question) {
  const run = ++state.run;
  const q = { question, tokens: [], retrieval: null, done: null, kind: 'direct' };
  state.current = q;
  document.querySelectorAll('.qchip').forEach((b) => b.setAttribute('aria-pressed', 'false'));
  resetAnswer(question); markRail(3); setCaption('Retrieving...');
  stage.setHits([], null); stage.v.q = 0; stage.v.line = 0; stage.v.hit = 0;
  await Promise.all([stage.tween('split', 1, 200), stage.tween('morph', 1, 300)]);
  const res = await fetch('query/stream', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question }) });
  if (!res.ok) { const d = await res.json().catch(() => ({})); $('#why').textContent = typeof d.detail === 'string' ? d.detail : 'The request failed.'; $('#why').className = 'why mono no'; $('#stateChip').textContent = 'error'; return; }
  const reader = res.body.getReader(), dec = new TextDecoder(); let buf = '', text = '';
  for (;;) {
    const { value, done } = await reader.read(); if (done) break;
    buf += dec.decode(value, { stream: true });
    const [events, rest] = parseSSE(buf); buf = rest;
    for (const [name, data] of events) {
      if (run !== state.run) return;
      if (name === 'retrieval') {
        q.retrieval = data; stage.setHits(data.sources.map((s) => s.chunk_id), data.query_xy);
        stage.tween('q', 1, 700); stage.tween('line', 1, 600); stage.tween('hit', 1, 700);
        renderChunks(data.sources, question, '', true); markRail(4); setCaption(`Retrieved ${data.sources.length} chunks in ${fmt(data.retrieve_ms)} ms. Generating...`);
        $('#stateChip').textContent = 'generating';
      } else if (name === 'token') { text += data.text; $('#ans').innerHTML = esc(text) + '<span class="caret"></span>'; }
      else if (name === 'done') { q.done = data; finishAnswer(data, q); renderChunks(data.sources, question, data.abstained ? '' : data.answer, data.abstained); document.querySelectorAll('.chunk').forEach((c) => c.classList.add('on')); setCaption(data.abstained ? 'Declined.' : 'Answered from the retrieved chunks.'); }
      else if (name === 'error') { $('#why').textContent = data.detail; $('#why').className = 'why mono no'; $('#stateChip').textContent = 'error'; }
    }
  }
}

async function main() {
  buildRail();
  stage = new Stage($('#stage'));
  const demo = await (await fetch('data/demo.json')).json();
  state.demo = demo;
  const live = await detectLive();
  $('#corpusNote').textContent = `${demo.meta.corpus.documents} documents, ${demo.meta.corpus.chunks} chunks`;
  if (live) {
    state.mode = 'live';
    $('#modePill').textContent = `Live · ${live.llm.replace('local:', '')}`; $('#modePill').classList.add('live');
    $('#liveForm').hidden = false;
    $('#askNote').textContent = 'Live: your question goes to the local service. Recorded questions below replay the saved run.';
    if (live.chunks > 0) { stage.setData(await liveChunks()); }
    else { stage.setData(demo.chunks); $('#loadSample').hidden = false; }
    $('#loadSample').addEventListener('click', async () => {
      const b = $('#loadSample'); b.disabled = true; b.textContent = 'Loading...';
      const r = await fetch('documents/sample', { method: 'POST' });
      if (r.ok) { stage.setData(await liveChunks()); b.hidden = true; } else { b.textContent = 'No sample corpus found'; }
    });
    $('#liveForm').addEventListener('submit', (e) => { e.preventDefault(); const v = $('#liveQ').value.trim(); if (v) askLive(v); });
  } else {
    $('#askNote').textContent = 'Recorded run: saved from a real execution of this code, replayed here with no backend.';
    stage.setData(demo.chunks);
  }
  const m = demo.meta;
  $('#modePill').title = `Recorded ${m.date} on ${m.device}; ${m.llm.model}; ${m.embedder.model}`;
  buildChips(); renderResults();
  $('#replay').addEventListener('click', () => { state.current && 0; playIntro(); });
  await playIntro();
  if (!live) await pickQuery(0);
}

main().catch((e) => { console.error(e); setCaption('Could not load the demo data.'); });
