"""控制台单页 UI：原生 fetch + 少量 JS，无构建步骤。

布局：左侧 tabs；右侧内容。"人类可读层"（画像/叙事/工作记忆）和"机器索引层"（记忆碎片）分明。
"""

INDEX_HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>个人智能体 · 控制台</title>
<style>
  :root {
    --bg:#0e0f13; --panel:#161821; --panel2:#1d2030; --line:#262a3a;
    --text:#e6e8ef; --dim:#8a90a3; --accent:#7cc4ff; --warn:#ffb454; --bad:#ff6b6b; --good:#7ce0b0;
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  * { box-sizing:border-box }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.5 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif }
  header { padding:14px 20px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:18px; flex-wrap:wrap }
  header h1 { font-size:16px; margin:0; font-weight:600; letter-spacing:.5px }
  header h1 span { color:var(--accent) }
  .chips { display:flex; gap:8px; flex-wrap:wrap; margin-left:auto }
  .chip { background:var(--panel2); border:1px solid var(--line); padding:4px 10px; border-radius:12px; font-size:12px; color:var(--dim) }
  .chip b { color:var(--text); margin-right:4px }
  .chip.alert { color:var(--warn); border-color:#4a3a26 }
  nav { display:flex; gap:2px; padding:0 12px; background:var(--panel); border-bottom:1px solid var(--line); overflow-x:auto }
  nav .group { color:#5b6177; font-size:11px; padding:10px 6px; align-self:end; letter-spacing:.5px }
  nav button { background:transparent; color:var(--dim); border:0; padding:10px 14px; cursor:pointer; font-size:13px; border-bottom:2px solid transparent; white-space:nowrap }
  nav button:hover { color:var(--text) }
  nav button.active { color:var(--accent); border-bottom-color:var(--accent) }
  main { padding:18px 20px }
  .empty { color:var(--dim); padding:24px; text-align:center; border:1px dashed var(--line); border-radius:8px }
  .toolbar { margin-bottom:12px; display:flex; gap:8px; align-items:center; flex-wrap:wrap }
  table { width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden }
  th,td { padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; font-size:13px }
  th { background:var(--panel2); font-weight:500; color:var(--dim); font-size:12px; letter-spacing:.3px }
  tr:last-child td { border-bottom:0 }
  .tag { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px; background:var(--panel2); color:var(--dim); border:1px solid var(--line) }
  .tag.good { color:var(--good); border-color:#264a3a }
  .tag.warn { color:var(--warn); border-color:#4a3a26 }
  .tag.bad  { color:var(--bad);  border-color:#4a2626 }
  button.act { background:transparent; border:1px solid var(--line); color:var(--dim); padding:3px 10px; border-radius:6px; cursor:pointer; font-size:12px; margin-right:6px }
  button.act:hover { color:var(--text); border-color:var(--dim) }
  button.danger:hover { color:var(--bad); border-color:var(--bad) }
  button.primary { color:var(--accent); border-color:#2a4a66 }
  button.primary:hover { background:#13283c }
  button.big { padding:8px 18px; font-size:14px }
  input.inline { background:var(--panel2); border:1px solid var(--line); color:var(--text); padding:4px 8px; border-radius:5px; font-size:13px; width:280px }
  .mono { font-family:var(--mono); font-size:12px; color:var(--dim) }
  .col-id { font-family:var(--mono); font-size:11px; color:#5b6177 }
  .diff { display:grid; grid-template-columns:1fr auto 1fr; gap:12px; align-items:center }
  .diff .side { background:var(--panel2); padding:8px 12px; border-radius:6px; border:1px solid var(--line) }
  .diff .arrow { color:var(--dim) }
  .row-actions { white-space:nowrap }
  pre.md { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; white-space:pre-wrap; font-family:var(--mono); font-size:13px; line-height:1.6; color:var(--text); max-height:70vh; overflow:auto }
  details.trace { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px 14px; margin-bottom:8px }
  details.trace summary { cursor:pointer; outline:none; color:var(--text); list-style:none }
  details.trace summary::marker { content:'' }
  details.trace[open] { border-color:var(--accent) }
  details.trace .body { margin-top:10px; padding-top:10px; border-top:1px dashed var(--line); display:grid; gap:8px }
  details.trace .body h4 { margin:6px 0 4px; font-size:12px; color:var(--dim); font-weight:500; letter-spacing:.3px }
  details.trace .body pre { background:var(--panel2); padding:8px 10px; border-radius:5px; white-space:pre-wrap; font-family:var(--mono); font-size:12px; margin:0 }
  .bar { display:inline-block; height:6px; background:#2a4a66; border-radius:3px; vertical-align:middle; margin:0 4px }
</style>
</head>
<body>
<header>
  <h1>个人智能体 · <span>控制台</span></h1>
  <div class="chips" id="chips"></div>
</header>
<nav id="nav"></nav>
<main id="main"></main>

<script>
const TABS = [
  // 人类可读层（直接展示给人）
  ["__g1", "可读层"],
  ["overview",     "概览"],
  ["facts",        "画像"],
  ["narratives",   "叙事笔记"],
  ["working",      "MEMORY.md"],
  // 机器索引层
  ["__g2", "索引层"],
  ["chunks",       "记忆碎片"],
  // 运行时
  ["__g3", "运行时"],
  ["commitments",  "承诺"],
  ["proactive",    "主动消息"],
  ["mood",         "情绪"],
  ["pending",      "待巩固"],
  // 能力层
  ["__g5", "能力"],
  ["tools",        "工具/技能"],
  // 信任层
  ["__g4", "信任"],
  ["pendingact",   "待确认"],
  ["contradictions","矛盾"],
  ["audit",        "审计"],
  ["history",      "记忆历史"],
  ["trace",        "Trace"],
];

const nav = document.getElementById('nav');
const main = document.getElementById('main');
let current = "overview";

function setTab(t){
  current = t;
  for (const b of nav.children) if (b.dataset && b.dataset.t) b.classList.toggle('active', b.dataset.t === t);
  render();
}

for (const [id, label] of TABS) {
  if (id.startsWith('__')) {
    const span = document.createElement('span');
    span.className = 'group';
    span.textContent = label;
    nav.appendChild(span);
    continue;
  }
  const b = document.createElement('button');
  b.textContent = label; b.dataset.t = id;
  b.onclick = () => setTab(id);
  nav.appendChild(b);
}

async function api(method, path, body){
  const opts = { method, headers: {"Content-Type":"application/json"} };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.status === 204 ? null : r.json();
}

function fmt(ts){
  if (!ts) return "";
  const d = new Date(ts);
  return isNaN(d) ? ts : d.toLocaleString('zh-CN', { hour12:false });
}

function esc(s){ return (s ?? "").toString().replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]); }

async function refreshChips(){
  const s = await api('GET', '/api/stats');
  const pendingChip = s.pending_intake > 0
    ? `<span class="chip alert"><b>${s.pending_intake}</b>待巩固</span>` : '';
  document.getElementById('chips').innerHTML = `
    <span class="chip"><b>${s.facts}</b>画像</span>
    <span class="chip"><b>${s.narratives}</b>叙事</span>
    <span class="chip"><b>${s.memories}</b>碎片</span>
    <span class="chip"><b>${s.open_commitments}</b>待跟进</span>
    ${pendingChip}
    <span class="chip"><b>${s.pending_actions}</b>待确认</span>
    <span class="chip"><b>${s.contradictions}</b>矛盾</span>
    <span class="chip"><b>${s.proactive_used_today}/${s.proactive_quota_today || '∞'}</b>今日主动</span>
  `;
}

const VIEWS = {
  async overview(){
    const s = await api('GET', '/api/stats');
    return `
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px">
        ${[
          ['画像事实', s.facts, 'facts'],
          ['叙事笔记', s.narratives, 'narratives'],
          ['记忆碎片', s.memories, 'chunks'],
          ['待巩固 pending', s.pending_intake, 'pending'],
          ['未完成承诺', s.open_commitments, 'commitments'],
          ['待确认动作', s.pending_actions, 'pendingact'],
          ['矛盾待裁决', s.contradictions, 'contradictions'],
          ['今日主动', `${s.proactive_used_today} / ${s.proactive_quota_today || '∞'}`, 'proactive'],
        ].map(([k,v,t]) => `
          <div onclick="setTab('${t}')" style="cursor:pointer;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px">
            <div style="color:var(--dim);font-size:12px">${k}</div>
            <div style="font-size:20px;margin-top:4px">${v}</div>
          </div>
        `).join('')}
      </div>
      <p class="mono" style="margin-top:20px">人类可读层（画像/叙事/MEMORY.md）直接编辑生效；索引层（记忆碎片）自动同步。所有写操作都进「记忆历史」可回溯。</p>
    `;
  },

  async facts(){
    const items = await api('GET', '/api/facts');
    if (!items.length) return `<div class="empty">还没记下任何画像事实。先去聊一会儿吧～</div>`;
    return `<table>
      <thead><tr><th>分类</th><th>key</th><th>value</th><th>置信</th><th>更新于</th><th></th></tr></thead>
      <tbody>${items.map(f => `
        <tr data-id="${f.id}">
          <td><span class="tag">${esc(f.category)}</span></td>
          <td class="mono">${esc(f.key)}</td>
          <td><span class="val">${esc(f.value)}</span></td>
          <td class="mono">${f.confidence.toFixed(2)}</td>
          <td class="mono">${fmt(f.updated_at)}</td>
          <td class="row-actions">
            <button class="act primary" onclick="editFact('${f.id}', this)">编辑</button>
            <button class="act danger" onclick="forgetFact('${f.id}')">遗忘</button>
          </td>
        </tr>`).join('')}
      </tbody></table>`;
  },

  async narratives(){
    const items = await api('GET', '/api/narratives');
    if (!items.length) return `<div class="empty">还没有叙事笔记。聊几轮后 Consolidator 会从对话里抽出来。</div>`;
    return `<table>
      <thead><tr><th>类型</th><th>内容</th><th>事件时间</th><th>合并键</th><th>重要度</th><th>来源</th><th>创建于</th><th></th></tr></thead>
      <tbody>${items.map(n => `
        <tr><td><span class="tag">${esc(n.kind)}</span></td>
            <td>${esc(n.content)}</td>
            <td class="mono">${esc(n.event_at || '—')}</td>
            <td class="mono col-id">${esc(n.canonical_key || '—')}</td>
            <td class="mono">${n.importance.toFixed(2)}</td>
            <td class="mono">${esc(n.source)}</td>
            <td class="mono">${fmt(n.created_at)}</td>
            <td class="row-actions"><button class="act danger" onclick="forgetNarr('${n.id}')">遗忘</button></td>
        </tr>`).join('')}
      </tbody></table>
      <p class="mono" style="margin-top:10px">遗忘任一条 → 对应的向量碎片自动级联失效；合并/过期记录见「记忆历史」标签页。</p>`;
  },

  async working(){
    const wm = await api('GET', '/api/working-memory');
    const body = wm.content
      ? `<pre class="md">${esc(wm.content)}</pre>`
      : `<div class="empty">MEMORY.md 还没生成。先聊几句或在「待巩固」里点「立即巩固」。</div>`;
    return `<div class="toolbar">
        <span class="mono">${esc(wm.path)}</span>
        <button class="act primary" onclick="refreshMD()">重新生成</button>
      </div>` + body;
  },

  async chunks(){
    const items = await api('GET', '/api/memories');
    if (!items.length) return `<div class="empty">索引层还没有任何碎片。</div>`;
    return `<p class="mono" style="margin-bottom:10px">这些是给检索用的向量化片段，每条都指回它的人类可读源（叙事笔记/画像）。</p>
      <table>
      <thead><tr><th>内容</th><th>关键词</th><th>来源</th><th>embedder</th><th>记于</th><th></th></tr></thead>
      <tbody>${items.map(m => `
        <tr><td>${esc(m.content)}</td>
            <td class="mono">${(m.keywords || []).map(esc).join('、')}</td>
            <td class="mono">${esc(m.source_table || '—')}<br/><span class="col-id">${esc(m.source_id || '')}</span></td>
            <td class="mono">${esc(m.embedder_version || '—')}</td>
            <td class="mono">${fmt(m.created_at)}</td>
            <td class="row-actions"><button class="act danger" onclick="forgetMem('${m.id}')">遗忘</button></td>
        </tr>`).join('')}
      </tbody></table>`;
  },

  async pending(){
    const data = await api('GET', '/api/pending');
    const bar = `<div class="toolbar">
        <span>pending: <b>${data.count}</b></span>
        <button class="act primary big" onclick="consolidate()">立即巩固（Deep Dream）</button>
        <span class="mono">把缓冲的对话全跑成画像/叙事/承诺/情绪 + 重写 MEMORY.md</span>
      </div>`;
    if (!data.items.length) return bar + `<div class="empty">没有待巩固的对话。</div>`;
    return bar + `<table>
      <thead><tr><th>状态</th><th>用户文本</th><th>创建于</th><th>处理于</th></tr></thead>
      <tbody>${data.items.map(p => `
        <tr><td><span class="tag ${p.status==='processed'?'good':p.status==='processing'?'warn':''}">${esc(p.status)}</span></td>
            <td>${esc(p.user_text)}</td>
            <td class="mono">${fmt(p.created_at)}</td>
            <td class="mono">${fmt(p.processed_at)}</td>
        </tr>`).join('')}
      </tbody></table>`;
  },

  async mood(){
    const items = await api('GET', '/api/mood');
    if (!items.length) return `<div class="empty">还没有情绪记录。</div>`;
    return `<table>
      <thead><tr><th>时间</th><th>valence</th><th>arousal</th><th>signals</th><th>note</th></tr></thead>
      <tbody>${items.map(m => {
        const v = m.valence ?? 0, a = m.arousal ?? 0;
        const vColor = v >= 0 ? 'var(--good)' : 'var(--bad)';
        return `<tr>
          <td class="mono">${fmt(m.ts)}</td>
          <td><span class="bar" style="width:${Math.min(60, Math.abs(v)*60)}px;background:${vColor}"></span><span class="mono">${v.toFixed(2)}</span></td>
          <td><span class="bar" style="width:${a*60}px"></span><span class="mono">${a.toFixed(2)}</span></td>
          <td class="mono">${(m.signals||[]).map(esc).join('、')}</td>
          <td>${esc(m.note || '')}</td>
        </tr>`;
      }).join('')}
      </tbody></table>`;
  },


  async commitments(){
    const items = await api('GET', '/api/commitments');
    if (!items.length) return `<div class="empty">还没有承诺/开放回路。</div>`;
    const k = { open:'good', done:'', dropped:'bad' };
    return `<table>
      <thead><tr><th>状态</th><th>类型</th><th>内容</th><th>到期</th><th>创建于</th></tr></thead>
      <tbody>${items.map(c => `
        <tr><td><span class="tag ${k[c.status]||''}">${esc(c.status)}</span></td>
            <td class="mono">${esc(c.kind)}</td>
            <td>${esc(c.content)}</td>
            <td class="mono">${fmt(c.due_at)}</td>
            <td class="mono">${fmt(c.created_at)}</td>
        </tr>`).join('')}
      </tbody></table>`;
  },

  async proactive(){
    const items = await api('GET', '/api/proactive');
    if (!items.length) return `<div class="empty">还没发过主动消息。</div>`;
    return `<table>
      <thead><tr><th>状态</th><th>主动消息</th><th>reason（为什么发）</th><th>分</th><th>时间</th></tr></thead>
      <tbody>${items.map(p => `
        <tr><td><span class="tag ${p.status==='sent'?'good':'warn'}">${esc(p.status)}</span></td>
            <td>${esc(p.content)}</td>
            <td class="mono">${esc(p.reason)}</td>
            <td class="mono">${(p.score ?? 0).toFixed(2)}</td>
            <td class="mono">${fmt(p.created_at)}</td>
        </tr>`).join('')}
      </tbody></table>`;
  },

  async pendingact(){
    const items = await api('GET', '/api/pending-actions?status=pending');
    if (!items.length) return `<div class="empty">没有待确认的外部动作。</div>`;
    return `<table>
      <thead><tr><th>类型</th><th>摘要</th><th>payload</th><th>创建于</th><th></th></tr></thead>
      <tbody>${items.map(a => `
        <tr><td class="mono">${esc(a.action_type)}</td>
            <td>${esc(a.summary)}</td>
            <td class="mono"><pre style="margin:0;white-space:pre-wrap">${esc(JSON.stringify(a.payload, null, 2))}</pre></td>
            <td class="mono">${fmt(a.created_at)}</td>
            <td class="row-actions">
              <button class="act danger" onclick="cancelAct('${a.id}')">取消</button>
            </td>
        </tr>`).join('')}
      </tbody></table>
      <p class="mono" style="margin-top:12px">注：「确认通过」请在对话里回「确认」走主流程；控制台只提供「取消」与可视化（避免绕过审计）。</p>`;
  },

  async contradictions(){
    const items = await api('GET', '/api/contradictions');
    if (!items.length) return `<div class="empty">没有矛盾待裁决。</div>`;
    return items.map(c => `
      <div style="background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:12px">
        <div style="margin-bottom:8px"><span class="tag warn">CONFLICT</span> <span class="mono">key = ${esc(c.key)}</span> · <span class="mono">${fmt(c.created_at)}</span></div>
        <div class="diff">
          <div class="side"><div class="mono" style="color:var(--dim);margin-bottom:4px">旧值（仍生效）</div>${esc(c.old?.value || '—')}<div class="mono" style="margin-top:4px">置信 ${(c.old?.confidence ?? 0).toFixed(2)}</div></div>
          <div class="arrow">⇄</div>
          <div class="side"><div class="mono" style="color:var(--dim);margin-bottom:4px">新值（挂起）</div>${esc(c.new?.value || '—')}<div class="mono" style="margin-top:4px">置信 ${(c.new?.confidence ?? 0).toFixed(2)}</div></div>
        </div>
        <div style="margin-top:10px;text-align:right">
          <button class="act" onclick="resolveC('${c.id}','old')">保留旧值</button>
          <button class="act primary" onclick="resolveC('${c.id}','new')">改用新值</button>
        </div>
      </div>`).join('');
  },

  async audit(){
    const items = await api('GET', '/api/audit');
    if (!items.length) return `<div class="empty">还没有审计记录。</div>`;
    return `<table>
      <thead><tr><th>时间</th><th>操作类型</th><th>摘要</th><th>操作者</th></tr></thead>
      <tbody>${items.map(a => `
        <tr><td class="mono">${fmt(a.ts)}</td>
            <td class="mono">${esc(a.kind)}</td>
            <td>${esc(a.summary)}</td>
            <td class="mono">${esc(a.actor)}</td>
        </tr>`).join('')}
      </tbody></table>`;
  },

  async history(){
    const items = await api('GET', '/api/history');
    if (!items.length) return `<div class="empty">还没有记忆变更历史。</div>`;
    // 合并/生命周期类原因高亮，便于一眼看出"哪些记忆被合并/更新/过期"（方案 §10）
    const LIFECYCLE = {MERGE:'#7c5cff', UPDATE:'#2d8cff', EXPIRE:'#b06a00', SAME:'#5a5a5a', CONTRADICTION:'#c0392b'};
    return `<p class="mono" style="margin-bottom:10px">每条写操作可回溯。<b>MERGE/UPDATE/EXPIRE/SAME</b> = 叙事 resolve 的合并历史。</p>
      <table>
      <thead><tr><th>时间</th><th>对象</th><th>原因</th><th>操作者</th><th>原值 → 新值</th></tr></thead>
      <tbody>${items.map(h => `
        <tr><td class="mono">${fmt(h.ts)}</td>
            <td class="mono">${esc(h.target_table)}<br/><span class="col-id">${esc(h.target_id)}</span></td>
            <td><span class="tag"${LIFECYCLE[h.reason] ? ` style="background:${LIFECYCLE[h.reason]};color:#fff"` : ''}>${esc(h.reason)}</span></td>
            <td class="mono">${esc(h.actor)}</td>
            <td><span class="mono" style="color:var(--dim)">${esc(h.prev_value ?? '—')}</span> → ${esc(h.new_value ?? '—')}</td>
        </tr>`).join('')}
      </tbody></table>`;
  },

  async tools(){
    const [tools, traces] = await Promise.all([
      api('GET', '/api/tools'),
      api('GET', '/api/tool-trace?n=100'),
    ]);
    const srcTag = s => s && s.startsWith('mcp') ? 'warn' : s === 'skill' ? 'good' : '';
    const toolTable = tools.length ? `<table>
      <thead><tr><th>来源</th><th>名称</th><th>说明</th><th>危险</th></tr></thead>
      <tbody>${tools.map(t => `
        <tr><td><span class="tag ${srcTag(t.source)}">${esc(t.source)}</span></td>
            <td class="mono">${esc(t.name)}</td>
            <td>${esc(t.description)}</td>
            <td>${t.dangerous ? '<span class="tag bad">需确认</span>' : ''}</td>
        </tr>`).join('')}
      </tbody></table>` : `<div class="empty">没有已注册的工具。检查 config 的 [tools].enabled、agent/skills/ 与 [mcp].servers。</div>`;
    const traceView = traces.length ? `<h3 style="margin:24px 0 12px">工具调用记录 (tool_trace)</h3>
      <table>
      <thead><tr><th>时间</th><th>步</th><th>工具</th><th>参数</th><th>结果</th><th>耗时</th></tr></thead>
      <tbody>${traces.map(t => `
        <tr><td class="mono">${fmt(t.ts)}</td>
            <td class="mono">${t.step}</td>
            <td class="mono">${esc(t.tool_name)}<br/><span class="col-id">${esc(t.source||'')}</span></td>
            <td class="mono"><pre style="margin:0;white-space:pre-wrap;max-width:240px">${esc(t.arguments||'')}</pre></td>
            <td><span class="tag ${t.ok?'good':'bad'}">${t.ok?'ok':'err'}</span> ${esc((t.result||'').slice(0,160))}</td>
            <td class="mono">${t.ms ?? ''}ms</td>
        </tr>`).join('')}
      </tbody></table>` : '<p class="mono" style="margin-top:16px">还没有工具调用记录。聊一句需要查时间/查记忆的话试试。</p>';
    return `<p class="mono" style="margin-bottom:10px">三个来源统一注册：<b>builtin</b> 原生 · <b>mcp:*</b> 外部服务器 · <b>skill</b> 技能（渐进披露，只暴露描述）。危险工具调用前走确认门。</p>${toolTable}${traceView}`;
  },

  async trace(){
    const [turns, ticks] = await Promise.all([
      api('GET', '/api/turn-trace?n=30'),
      api('GET', '/api/tick-trace?n=30'),
    ]);
    const turnView = turns.length ? turns.map(t => `
      <details class="trace">
        <summary><span class="mono">${fmt(t.ts)}</span> · ${t.latency_ms}ms · 你 > ${esc((t.user_text||'').slice(0,80))}</summary>
        <div class="body">
          <h4>稳定 prefix（命中 cache 的那段，SOUL + MEMORY.md）</h4>
          <pre>${esc(t.stable_prefix)}</pre>
          <h4>动态 suffix（按本轮 query 召回的碎片）</h4>
          <pre>${esc(t.dynamic_suffix || '（无召回）')}</pre>
          <h4>召回明细（${t.retrieved.length} 条 · rel 相关 / rec 新近 / imp 重要 · gate 注入门控）</h4>
          <pre>${t.retrieved.map(h => {
            const c = h.components || {};
            const br = (c.relevance != null) ? ` rel ${c.relevance} rec ${c.recency} imp ${c.importance} →` : '';
            const g = h.gate ? ` 〔${h.gate === 'skip' ? '✗skip' : '✓inject'}:${h.gate_reason || ''}〕` : '';
            return `[${h.score.toFixed(2)}]${br} ${h.content}${g}`;
          }).join('\n') || '—'}</pre>
          <h4>回复</h4>
          <pre>${esc(t.reply)}</pre>
        </div>
      </details>`).join('') : '<div class="empty">还没有 turn trace。</div>';
    const tickView = ticks.length ? `<table style="margin-top:18px">
      <thead><tr><th>时间</th><th>状态</th><th>reason / 消息</th></tr></thead>
      <tbody>${ticks.map(t => `
        <tr><td class="mono">${fmt(t.ts)}</td>
            <td><span class="tag ${t.sent?'good':'warn'}">${t.sent?'sent':'skip'}</span></td>
            <td>${esc(t.message || t.reason)}<div class="mono" style="color:var(--dim)">${esc(t.reason)}</div></td>
        </tr>`).join('')}
      </tbody></table>` : '';
    return `<h3 style="margin:8px 0 12px">最近对话 (turn_trace)</h3>${turnView}<h3 style="margin:24px 0 12px">最近心跳 (tick_trace)</h3>${tickView}`;
  },
};

async function editFact(id, btn){
  const tr = btn.closest('tr');
  const valTd = tr.querySelector('.val');
  const old = valTd.textContent;
  valTd.innerHTML = `<input class="inline" value="${esc(old)}">`;
  const input = valTd.querySelector('input');
  input.focus(); input.select();
  btn.textContent = '保存';
  btn.onclick = async () => {
    const v = input.value.trim();
    if (!v || v === old) { render(); return; }
    await api('PATCH', `/api/facts/${id}`, { value: v });
    await refreshChips(); render();
  };
  input.onkeydown = e => { if (e.key === 'Enter') btn.click(); if (e.key === 'Escape') render(); };
}

async function forgetFact(id){
  if (!confirm('确认遗忘这条画像？软删后不再注入对话；记忆历史仍可见。')) return;
  await api('DELETE', `/api/facts/${id}`);
  await refreshChips(); render();
}

async function forgetNarr(id){
  if (!confirm('确认遗忘这条叙事？对应的向量碎片也会级联失效。')) return;
  await api('DELETE', `/api/narratives/${id}`);
  await refreshChips(); render();
}

async function forgetMem(id){
  if (!confirm('确认遗忘这条记忆碎片？')) return;
  await api('DELETE', `/api/memories/${id}`);
  await refreshChips(); render();
}

async function cancelAct(id){
  await api('POST', `/api/pending-actions/${id}/cancel`);
  await refreshChips(); render();
}

async function resolveC(id, keep){
  if (!confirm(keep === 'new' ? '改用新值，并把旧值标为过期？' : '保留旧值，丢弃新值？')) return;
  await api('POST', `/api/contradictions/${id}/resolve`, { keep });
  await refreshChips(); render();
}

async function refreshMD(){
  await api('POST', '/api/working-memory/refresh');
  await refreshChips(); render();
}

async function consolidate(){
  const r = await api('POST', '/api/pending/consolidate');
  alert(`巩固完成：画像 +${r.added} 更新 ${r.updated} · 叙事 +${r.memories} · 承诺 +${r.commitments} · 矛盾 ${r.contradictions}`);
  await refreshChips(); render();
}

async function render(){
  main.innerHTML = '<div class="empty mono">loading…</div>';
  try {
    main.innerHTML = await VIEWS[current]();
  } catch (e) {
    main.innerHTML = `<div class="empty" style="color:var(--bad)">加载失败：${esc(e.message)}</div>`;
  }
}

(async () => { await refreshChips(); setTab('overview'); setInterval(refreshChips, 15000); })();
</script>
</body>
</html>
"""
