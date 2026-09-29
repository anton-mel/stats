from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

from pie_evals.schema import CellStatus, Tier

from . import flops
from .matrix import Matrix
from .store import Store

DEFAULT_MODEL = "gemma-4-26b-a4b-ollama"
OLLAMA_ICON = "data:image/png;base64,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"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dashboard</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2067.89%2067.89'%3E%3Cpath%20d='M52.96,11.53l-43.52,6.4c-3.85.57-5.64,5.08-3.22,8.13l27.3,34.49c2.41,3.05,7.22,2.34,8.65-1.27l16.21-40.89c1.43-3.61-1.58-7.42-5.43-6.86Z'%20fill='none'%20stroke='%23000'%20stroke-miterlimit='10'%20stroke-width='8'/%3E%3C/svg%3E">
<style>
  * { box-sizing: border-box; }
  body { font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 0; color: #1f2328; background: #f6f8fa; }
  header { background: #fff; border-bottom: 1px solid #d8dee4; }
  .bar { max-width: 1000px; margin: 0 auto; padding: 12px 16px; display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
  .brand { font-weight: 700; font-size: 17px; display: inline-flex; align-items: center; gap: 3px; }
  .logo { width: 22px; height: 22px; }
  nav { display: flex; gap: 8px; flex-wrap: wrap; }
  nav button, .pill, .signin, select, button.act {
    box-sizing: border-box; height: 32px; display: inline-flex; align-items: center; gap: 6px;
    border: 1px solid #d0d7de; border-radius: 999px; padding: 0 14px; background: #fff; color: #424a53;
    font: inherit; font-size: 14px; line-height: 1; cursor: pointer; }
  nav button:hover, .pill:hover, select:hover { background: #f6f8fa; }
  nav button.on { background: #1f2328; border-color: #1f2328; color: #fff; }
  .grow { flex: 1; }
  main { max-width: 1000px; margin: 0 auto; padding: 20px 16px 48px; }
  .controls { max-width: 1000px; margin: 0 auto; padding: 16px 16px 0; display: flex; gap: 8px; flex-wrap: nowrap; align-items: center; color: #424a53; font-size: 14px; }
  .controls[hidden] { display: none; }
  .up { color: #1a7f37; } .down { color: #b3261e; }
  .controls #commit { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .controls .seg { flex: none; }
  .controls [hidden] { display: none; }
  .controls .grow { flex: 1; }
  tr.push.flat { cursor: default; }
  input.rowpick { width: 16px; height: 16px; margin: 0; cursor: pointer; vertical-align: middle; }
  table.compact td.pickcell { vertical-align: middle; }
  td.pickcell label { display: flex; align-items: center; justify-content: center; height: calc(14px * 1.4); margin: -3px -8px; padding: 3px 8px; cursor: pointer; }
  td.pickcell:hover { background: #eef1f4; }
  .dot.wait { background: #d4a72c; }
  .dot.bad { background: #cf222e; }
  .dot.part { background: #bc4c00; }
  .seg { display: inline-flex; height: 32px; border: 1px solid #d0d7de; border-radius: 999px; background: #fff; padding: 2px; box-sizing: border-box; }
  .seg button { border: 0; background: none; font: inherit; font-size: 13px; color: #424a53; padding: 0 12px; border-radius: 999px; cursor: pointer; }
  .seg button.on { background: #1f2328; color: #fff; }
  .commit-head { margin: 0 0 16px; }
  .backlink { display: inline-block; font-size: 13px; color: #656d76; margin-bottom: 8px; }
  .backlink:hover { color: #1f2328; }
  .commit-head .title { font-size: 18px; font-weight: 600; color: #1f2328; line-height: 1.35; }
  .commit-head .meta { display: flex; align-items: center; gap: 14px; margin-top: 6px; color: #656d76; font-size: 13px; flex-wrap: wrap; }
  .commit-head .meta span { display: inline-flex; align-items: center; }
  .commit-head .avatar { width: 18px; height: 18px; }
  .commit-head .sha code { background: #fff; border: 1px solid #d0d7de; border-radius: 999px; padding: 2px 8px; color: #0969da; }
  .card { background: #fff; border: 1px solid #d8dee4; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  h2 { font-size: 16px; margin: 0 0 12px; }
  .muted { color: #656d76; font-size: 13px; }
  .up { color: #1a7f37; } .down { color: #cf222e; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
  @media (max-width: 800px) { .tiles { grid-template-columns: 1fr 1fr; } }
  @media (max-width: 520px) { .tiles { grid-template-columns: 1fr; } }
  .tile { background: #fff; border: 1px solid #d8dee4; border-radius: 8px; padding: 10px 12px; min-width: 0; }
  .tile .name { font-weight: 600; font-size: 14px; }
  .tile .now { font-size: 13px; color: #424a53; margin: 2px 0 6px; }
  .chart { position: relative; height: 120px; }
  .phase { font-size: 13px; font-weight: 600; color: #656d76; text-transform: uppercase; letter-spacing: .04em; margin: 0 0 8px; }
  .tiles + .phase { margin-top: 20px; }
  @media (max-width: 700px) { .row { grid-template-columns: 1fr; } }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #eaeef2; vertical-align: top; }
  th { font-size: 13px; color: #424a53; font-weight: 600; }
  td.num, th.num { text-align: right; }
  table.compact { font-size: 13px; }
  table.fixed { table-layout: fixed; }
  table.compact .avatar { width: 18px; height: 18px; }
  table.compact .tag { margin: 0 4px 0 0; }
  table.compact th, table.compact td { padding: 3px 8px; line-height: 1.4; }
  td.clip { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
  .dot.on { background: #1a7f37; margin: 0 0 1px 6px; }
  .dot.run { background: #1a7f37; }
  .idle { background: #1a7f37; } .busy { background: #bf8700; } .offline { background: #cf222e; }
  .tag { display: inline-block; background: #eaeef2; border-radius: 10px; padding: 0 8px; margin: 0 4px 4px 0; font-size: 13px; }
  .avatar { width: 22px; height: 22px; border-radius: 50%; vertical-align: middle; margin-right: 6px; }
  select { appearance: none; -webkit-appearance: none; padding-right: 30px;
    background: #fff url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23656d76'/%3E%3C/svg%3E") no-repeat right 12px center; }
  input { font: inherit; font-size: 14px; height: 32px; box-sizing: border-box; padding: 0 14px; border: 1px solid #d0d7de; border-radius: 999px; }
  input[type=checkbox] { height: auto; }
  button.act { background: #1f2328; border-color: #1f2328; color: #fff; font-weight: 600; }
  button.act:hover { background: #32383f; }
  label.check { display: block; padding: 4px 0; }
  table.compact input.switch { vertical-align: middle; }
  label.toggle { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 7px 0; border-bottom: 1px solid #eaeef2; cursor: pointer; font-size: 14px; }
  label.toggle:last-child { border-bottom: 0; }
  input.switch { appearance: none; -webkit-appearance: none; flex: none; width: 34px; height: 20px; border-radius: 999px; background: #d0d7de;
    position: relative; cursor: pointer; transition: background .15s; margin: 0; border: 0; padding: 0; }
  input.switch::after { content: ""; position: absolute; top: 2px; left: 2px; width: 16px; height: 16px; border-radius: 50%; background: #fff;
    box-shadow: 0 1px 2px rgba(0,0,0,.2); transition: transform .15s; }
  input.switch:checked { background: #1f883d; }
  input.switch:checked::after { transform: translateX(14px); }
  code { background: #eaeef2; border-radius: 4px; padding: 1px 5px; font-size: 13px; }
  a { color: #0969da; text-decoration: none; }
  .signin { background: #1f2328; border-color: #1f2328; color: #fff; font-weight: 600; }
  .signin:hover { background: #32383f; }
  .signin svg { fill: currentColor; }
  #who { display: flex; align-items: center; gap: 8px; }
  .pill { padding: 0 12px 0 4px; color: #1f2328; }
  .pill .avatar { margin: 0; }
  .modal { position: fixed; inset: 0; background: rgba(31, 35, 40, .45); display: flex; align-items: flex-start; justify-content: center; padding: 8vh 16px; z-index: 10; }
  .modal[hidden] { display: none; }
  .sheet { position: relative; background: #fff; border-radius: 10px; width: min(640px, 100%); max-height: 84vh; overflow: auto; box-shadow: 0 8px 24px rgba(0,0,0,.2); }
  .sheet .card { border: 0; margin: 0; }
  .sheet-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin: 0 0 14px; }
  .sheet-head h2 { margin: 0; font-size: 17px; letter-spacing: -0.01em; }
  .chips { display: flex; gap: 6px; flex-wrap: wrap; }
  .chips code { background: #f0f3f6; border-radius: 999px; padding: 2px 10px; font-size: 12px; color: #424a53; }
  .group { border: 1px solid #eaeef2; border-radius: 12px; padding: 4px 14px 6px; margin-bottom: 12px; }
  .group .label { font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: #7a838d; margin: 10px 0 2px; }
  .sheet-foot { position: sticky; bottom: 0; background: #fff; display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 0 2px; border-top: 1px solid #eaeef2; }
  .tag.new { background: #fff8c5; }
  .signin-wrap { min-height: calc(100vh - 57px - 68px); display: flex; align-items: center; justify-content: center; padding-bottom: 12vh; box-sizing: border-box; }
  .signin-page { width: 100%; max-width: 380px; margin: 0; padding: 32px 28px; text-align: center; display: flex; flex-direction: column; align-items: stretch; gap: 12px; }
  .signin-page .gh-mark { align-self: center; fill: #1f2328; }
  .signin-page h2 { margin: 4px 0 0; font-size: 18px; }
  .signin-page p { margin: 0 0 4px; }
  .signin-page input { width: 100%; text-align: center; }
  .signin-page button.act { justify-content: center; width: 100%; }
  .signin-page .err { margin: 0; min-height: 0; }
  .signin-page .err:empty { display: none; }
  .signrow { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .err { font-size: 12px; color: #cf222e; margin: 6px 0 0; min-height: 16px; }
  .x { position: absolute; top: 8px; right: 10px; border: 0; background: none; font-size: 22px; line-height: 1; cursor: pointer; color: #656d76; }
  tr.push { cursor: pointer; } tr.push:hover { background: #f6f8fa; }
  .auto { font-size: 14px; color: #424a53; display: inline-flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .auto .tag, .pill.small { box-sizing: border-box; height: 28px; display: inline-flex; align-items: center; margin: 0;
    font-size: 13px; line-height: 1; border: 1px solid #d0d7de; border-radius: 999px; background-color: #fff; color: #1f2328; }
  .auto .tag { padding: 0 12px; }
  .auto .tag.off { background: #fff8c5; border-color: #eac54f; }
  .on-word { color: #656d76; }
  .pill.small { padding: 0 12px; gap: 6px; margin-left: 4px; }
  .savebar { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  button.act:disabled { opacity: .45; cursor: default; }
  .pager { display: flex; justify-content: center; align-items: center; gap: 6px; padding: 14px 0 0; flex-wrap: wrap; }
  .pager .pill { padding: 0 12px; min-width: 32px; justify-content: center; }
  .pager .pill.on { background: #1f2328; border-color: #1f2328; color: #fff; }
  .pager .pill:disabled { opacity: .4; cursor: default; }
  .pager .gap { color: #656d76; padding: 0 2px; }
  button.link { border: 0; background: none; color: #0969da; font: inherit; cursor: pointer; padding: 0 0 0 6px; }
  #who { position: relative; }
  .menu { position: absolute; right: 0; top: 40px; background: #fff; border: 1px solid #d0d7de; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,.12); padding: 6px; z-index: 20; min-width: 160px; display: flex; flex-direction: column; }
  .menu button { display: flex; align-items: center; gap: 8px; border: 0; background: none; font: inherit; font-size: 14px; text-align: left; padding: 8px 12px; border-radius: 8px; cursor: pointer; color: #1f2328; }
  .menu button:hover { background: #f6f8fa; }
  .menu button svg { color: #656d76; flex: none; width: 13px; height: 13px; }
  .gridwrap { overflow-x: auto; margin: 12px 0; }
  table.grid td, table.grid th { padding: 5px 8px; font-size: 13px; }
  table.grid .c { text-align: center; }
  .ok { color: #1a7f37; font-weight: 700; }
  .label { font-size: 13px; font-weight: 600; color: #424a53; margin: 8px 0 4px; }
  img.ol { height: 14px; vertical-align: -2px; margin-right: 6px; }
  .model-head { display: flex; align-items: baseline; gap: 10px; margin-bottom: 6px; }
  .grip { margin-left: auto; cursor: grab; color: #8c959f; font-size: 14px; letter-spacing: -2px; user-select: none; }
  .model-card.dragging { opacity: .45; }
  .model-card.drop-before { box-shadow: 0 -3px 0 #0969da; }
  .model-card.drop-after { box-shadow: 0 3px 0 #0969da; }
  .model-head h2 { margin: 0; font-size: 16px; }
  .model-head h2 img.ol { height: 16px; }
  table.ov td { vertical-align: middle; }
  .gapbar { position: relative; height: 10px; background: #f0f3f6; border-radius: 5px; }
  .gapbar.empty { background: repeating-linear-gradient(90deg, #f0f3f6 0 6px, #fff 6px 10px); }
  .gapbar .mid { position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: #8c959f; }
  .gapbar .fill { position: absolute; top: 0; bottom: 0; }
  .gapbar .fill.ahead { left: 50%; background: #2da44e; border-radius: 0 5px 5px 0; }
  .gapbar .fill.behind { right: 50%; background: #cf222e; border-radius: 5px 0 0 5px; }
  .gapbar.noisy .fill { opacity: .4; }
  details.quality { margin-top: 12px; }
  table.ov td { vertical-align: middle; }
  table.ov th.vs-h, td.vs-c { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
  table.ov th { vertical-align: bottom; }
  .model-head .who { font-size: 13px; color: #7a838d; }
  .barlab { display: flex; justify-content: space-between; gap: 8px; font-size: 12px; color: #424a53; margin-bottom: 3px; white-space: nowrap; font-variant-numeric: tabular-nums; }
  .barlab .unit { color: #8c959f; }
  table.ov td.num, table.ov th.num { white-space: nowrap; }
  details.quality > summary { list-style: none; display: inline-flex; align-items: center; gap: 6px; height: 28px; padding: 0 12px; border: 1px solid #d0d7de; border-radius: 999px; background: #fff; color: #424a53; font-size: 13px; cursor: pointer; user-select: none; }
  details.quality > summary::-webkit-details-marker { display: none; }
  details.quality > summary:hover { background: #f6f8fa; }
  details.quality > summary::after { content: ""; width: 6px; height: 6px; border-right: 1.5px solid currentColor; border-bottom: 1.5px solid currentColor; transform: translateY(-2px) rotate(45deg); transition: transform .15s; }
  details.quality[open] > summary { background: #1f2328; border-color: #1f2328; color: #fff; }
  details.quality[open] > summary::after { transform: translateY(1px) rotate(-135deg); }
  table.quality { margin-top: 4px; }
  .running { color: #9a6700; background: #fff8c5; border-radius: 999px; padding: 0 8px; font-size: 12px; }
  .legend { display: flex; justify-content: space-between; font-weight: 400; font-size: 11px; color: #7a838d; }
  table.rmlist { width: 100%; }
  table.rmlist td { vertical-align: middle; padding: 8px 6px; }
  .cmd { display: flex; gap: 8px; align-items: stretch; margin: 8px 0 4px; }
  .cmd code { flex: 1; background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; padding: 10px 12px; font-size: 12px; word-break: break-all; }
  .steps { list-style: none; padding: 0; margin: 14px 0 4px; display: flex; flex-direction: column; gap: 10px; }
  .steps li { display: flex; align-items: center; gap: 10px; font-size: 14px; color: #7a838d; }
  .steps li .st { width: 18px; height: 18px; border-radius: 50%; border: 2px solid #d0d7de; flex: none; box-sizing: border-box; }
  .steps li.now { color: #1f2328; }
  .steps li.now .st { border-color: #0969da; border-top-color: transparent; animation: spin 0.9s linear infinite; }
  .steps li.done { color: #1f2328; }
  .steps li.done .st { border-color: #1a7f37; background: #1a7f37; }
  .steps li.bad { color: #cf222e; }
  .steps li.bad .st { border-color: #cf222e; background: #cf222e; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<header><div class="bar">
  <span class="brand"><svg class="logo" viewBox="0 0 67.89 67.89" aria-hidden="true"><path d="M52.96,11.53l-43.52,6.4c-3.85.57-5.64,5.08-3.22,8.13l27.3,34.49c2.41,3.05,7.22,2.34,8.65-1.27l16.21-40.89c1.43-3.61-1.58-7.42-5.43-6.86Z" fill="none" stroke="currentColor" stroke-miterlimit="10" stroke-width="8"/></svg>pie</span>
  <nav id="tabs"></nav>
  <span class="grow"></span>
  <span id="who"></span>
</div></header>
<div class="controls" id="controls">
  <select id="fmac" aria-label="machine"></select><select id="fmodel" aria-label="model"></select>
  <span class="grow"></span>
  <button class="act" id="bench" disabled>Bench</button>
</div>
<main id="main"></main>
<div id="modal" class="modal" hidden><div class="sheet"><button class="x" id="close" aria-label="close">×</button><div id="sheet"></div></div></div>
<script>
const DATA = __DATA__;
const TABS = ["Overview", "Pipeline", "History", "Machines", "People"];
const NL = String.fromCharCode(10);
const REPO_NAME = DATA.repo.split("/").pop();
const esc = x => String(x ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
let RUNNABLE = [...new Map(DATA.pool.filter(m => m.os === "macos").map(m => [m.id, m])).values()];
async function refreshPool() {
  const runners = (await gh(`repos/${DATA.repo}/actions/runners?per_page=100`).catch(() => null))?.runners;
  if (!runners) return;
  DATA.pool = runners.flatMap(r => {
    const id = r.labels.map(l => l.name).find(l => DATA.specs[l]);
    if (!id) return [];
    const spec = DATA.specs[id];
    return [{ name: spec.accelerator, id, kind: "self-hosted", os: spec.os, memory_gib: spec.memory_gib,
      status: r.status !== "online" ? "offline" : r.busy ? "busy" : "idle", last: DATA.last[id] || {} }];
  });
  RUNNABLE = [...new Map(DATA.pool.filter(m => m.os === "macos").map(m => [m.id, m])).values()];
  fillFilters();
}
setInterval(async () => {
  if (!me || document.hidden || !document.getElementById("modal").hidden) return;
  const was = JSON.stringify(DATA.pool);
  await refreshPool();
  if (JSON.stringify(DATA.pool) !== was) draw();
}, 15000);
const modelOf = id => DATA.models.find(m => m.id === id) || DATA.models.find(m => m.pie === id) || { name: id, quant: "" };
const modelName = id => modelOf(id).name;
const pieOf = id => modelOf(id).pie || id;
const modelTag = m => `<img class="ol" src="${DATA.ollama_icon}" alt="">${esc(m.name)}`;
const macName = id => (DATA.pool.find(m => m.id === id) || DATA.results[id] || { name: id }).name;
const PER_PAGE = 20;
let tab = "Overview", back = "Overview", me = null, page = 0, denied = "";
const filt = { mac: "", model: "" };
const chosen = new Set();
const testName = id => DATA.benchmarks.find(b => b.id === id)?.name || id;
function fillFilters() {
  const macs = [...new Set([...Object.keys(DATA.results), ...RUNNABLE.map(m => m.id)])];
  const models = DATA.models.map(m => m.id)
    .sort((a, b) => (a === DATA.default_model ? -1 : b === DATA.default_model ? 1 : 0));
  const opts = (all, ids, name, on) => ids.map(id => `<option value="${esc(id)}"${id === on ? " selected" : ""}>${esc(name(id))}</option>`).join("");
  if (!macs.includes(filt.mac)) filt.mac = macs[0] || "";
  if (!models.includes(filt.model)) filt.model = models.includes(DATA.default_model) ? DATA.default_model : models[0] || "";
  const fm = document.getElementById("fmac"), fo = document.getElementById("fmodel");
  fm.innerHTML = opts("all machines", macs, id => id, filt.mac);
  fo.innerHTML = opts("all models", models, modelName, filt.model);
  fm.onchange = () => { filt.mac = fm.value; draw(); };
  fo.onchange = () => { filt.model = fo.value; draw(); };
}
const token = () => { try { return localStorage.getItem("pie-evals-token"); } catch { return null; } };

const when = d => d ? new Date(d) : null;
const fmtDate = d => { const t = when(d); return t && !isNaN(t) ? `${t.getFullYear()}-${String(t.getMonth() + 1).padStart(2, "0")}-${String(t.getDate()).padStart(2, "0")}` : ""; };
const fmtTime = d => { const t = when(d); return t && !isNaN(t) && d.length > 10 ? `${String(t.getHours()).padStart(2, "0")}:${String(t.getMinutes()).padStart(2, "0")}` : ""; };
function relTime(d) {
  const secs = (new Date(d) - Date.now()) / 1000, rtf = new Intl.RelativeTimeFormat("en", { numeric: "auto" });
  for (const [unit, n] of [["year", 31536000], ["month", 2592000], ["week", 604800], ["day", 86400], ["hour", 3600], ["minute", 60]])
    if (Math.abs(secs) >= n) return rtf.format(Math.round(secs / n), unit);
  return "just now";
}
const ALL = [...DATA.commits, ...DATA.history].sort((a, b) => (b.date || "").localeCompare(a.date || ""));
const commitOf = sha => ALL.find(c => c.sha === sha) || { sha, message: "", author: "", date: "" };
const parentOf = sha => { const i = ALL.findIndex(c => c.sha === sha); return i >= 0 ? ALL[i + 1] : null; };
const measured = [...DATA.commits].sort((a, b) => (b.date || "").localeCompare(a.date || ""));
const inFlight = new Set(DATA.running || []);
let sel = [...measured, ...ALL.filter(c => inFlight.has(c.sha))].sort((a, b) => (b.date || "").localeCompare(a.date || ""))[0]?.sha || "";
const home = sel;
const openQuality = new Set();

function valueAt(mac, model, wl, sha) { return DATA.results[mac]?.models[model]?.[wl]?.[sha]; }
function ms(v) { return v == null ? "–" : v >= 1000 ? (v / 1000).toFixed(1) + "s" : Math.round(v) + "ms"; }
const vs = (a, b) => `${a} / ${b}`;
function pair(p, o, higher, fmt, tip) {
  const win = p == null || o == null ? "" : (higher ? p >= o : p <= o) ? "up" : "down";
  return `<td class="vs-c ${win || "muted"}" title="${esc(tip)}">${vs(fmt(p), fmt(o))}</td>`;
}
function orderedModels() {
  let saved = [];
  try { saved = JSON.parse(localStorage.getItem("stats-model-order") || "[]"); } catch {}
  const rank = id => { const i = saved.indexOf(id); return i < 0 ? saved.length + DATA.models.findIndex(m => m.id === id) : i; };
  return [...DATA.models].sort((a, b) => rank(a.id) - rank(b.id));
}
function ollamaAt(mac, pie, wl, label, at) {
  const runs = DATA.baselines?.[mac]?.[pie]?.[wl]?.[label] || [];
  const upto = at ? runs.filter(r => r.at <= at) : runs;
  return (upto.length ? upto : runs).reduce((a, b) => (!a || b.at > a.at ? b : a), null);
}
const pct = x => `${x > 0 ? "+" : ""}${(x * 100).toFixed(x > -0.1 && x < 0.1 ? 1 : 0)}%`;
const tok = x => x == null ? "–" : Math.round(x).toLocaleString();
function gapBar(gap, noisy) {
  if (gap == null) return `<div class="gapbar empty"><span class="mid"></span></div>`;
  const w = Math.min(Math.abs(gap), 1) * 50;
  return `<div class="gapbar${noisy ? " noisy" : ""}"><span class="mid"></span><span class="fill ${gap >= 0 ? "ahead" : "behind"}" style="width:${w}%"></span></div>`;
}
function metricBar(pv, ov, noisy, unit, m, running, now, note) {
  const gap = pv != null && ov ? pv / ov - 1 : null;
  const cls = gap == null ? "muted" : gap >= 0 ? "up" : "down";
  const tip = note || (`pie ${tok(pv)} · ${m.label} ${tok(ov)} ${unit}` + (gap != null ? ` · ${pct(gap)}` : "") + (noisy ? ` · not steady, ${noisy}` : ""));
  const text = note ? "–" : pv == null && ov == null ? (running && !m.unsupported ? "running" : "–") :
    pv == null && running && !m.unsupported ? "running" : vs(tok(pv), tok(ov));
  return `<td>${gapBar(gap)}</td><td class="vs-c ${cls}" title="${esc(tip)}">${text}</td>`;
}
function overview() {
  const main = document.getElementById("main");
  const mac = filt.mac, done = measured.find(c => !inFlight.has(c.sha))?.sha || "";
  const c = sel ? commitOf(sel) : null;
  let html = `<div class="commit-head"><div class="title">${c ? esc(c.message) || sel.slice(0, 7) : "No pie commit measured yet"}</div><div class="meta">` +
    (c ? `<a class="sha" href="https://github.com/${DATA.pie_repo}/commit/${sel}" target="_blank"><code>${sel.slice(0, 7)}</code></a>` : "") +
    (c?.author ? `<span><img class="avatar" src="https://github.com/${esc(c.author)}.png?size=40">${esc(c.author)}</span>` : "") +
    (c?.date ? `<span title="${fmtDate(c.date)} ${fmtTime(c.date)}">${relTime(c.date)}</span>` : "") +
    `<span class="muted">${esc(macName(mac))}</span>` +
    (inFlight.has(sel) ? `<span class="running">measuring now</span>` : "") +
    (sel && sel !== home ? `<a href="#" id="latest">back to latest</a>` : "") +
    (inFlight.has(sel) && done && done !== sel ? `<a href="#" id="done">last complete: ${done.slice(0, 7)}</a>` : "") + `</div></div>`;
  const rows = [];
  for (const m of orderedModels()) {
    const byTest = DATA.results[mac]?.models[m.pie] || {};
    let body = "";
    const running = inFlight.has(sel);
    for (const b of DATA.benchmarks) {
      const now = sel ? byTest[b.id]?.[sel] : null;
      const ol = ollamaAt(mac, m.pie, b.id, m.label, now?.at);
      const noisy = [now?.noisy && `pie: ${now.noisy}`, ol?.noisy && `${m.label}: ${ol.noisy}`].filter(Boolean).join("; ");
      const multi = b.concurrency > 1;
      body += `<tr title="${esc(b.description)}"><td class="clip">${esc(b.name)}</td>` +
        `${metricBar(now?.decode, ol?.decode, noisy, multi ? "tok/s total" : "tok/s", m, running, now)}` +
        `${metricBar(now?.prefill, ol?.prefill, noisy, multi ? "tok/s per request" : "tok/s", m, running, now)}` +
        pair(now?.ttft, ol?.ttft, false, ms, `time to first token (median), pie / ${m.label}`) + `</tr>`;
    }
    if (m.unsupported) body = `<tr><td colspan="6" class="muted">${esc(m.unsupported)}; only ${esc(m.label)} is measured on this model.</td></tr>` +
      DATA.benchmarks.map(b => {
        const ol = ollamaAt(mac, m.pie, b.id, m.label, null);
        const multi = b.concurrency > 1;
        return `<tr title="${esc(b.description)}"><td class="clip">${esc(b.name)}</td>` +
          `${metricBar(null, ol?.decode, ol?.noisy, multi ? "tok/s total" : "tok/s", m, false, null)}` +
          `${metricBar(null, ol?.prefill, ol?.noisy, multi ? "tok/s per request" : "tok/s", m, false, null)}` +
          pair(null, ol?.ttft, false, ms, `time to first token, ${m.label}`) + `</tr>`;
      }).join("");
    const qtasks = [
      ["gsm8k", "GSM8K", "grade school math, exact final answer"],
      ["math500", "MATH-500", "competition math, exact boxed answer"],
      ["mmlu", "MMLU", "general knowledge across 57 subjects, multiple choice"],
      ["arc", "ARC-Challenge", "grade school science reasoning, multiple choice"],
      ["ifeval", "IFEval", "follows verifiable format instructions"],
    ];
    const qcell = q => q?.score == null ? `<td class="num muted">not run</td>` :
      `<td class="num" title="95% interval ${(q.ci95[0] * 100).toFixed(0)}–${(q.ci95[1] * 100).toFixed(0)}%, n=${q.n}${q.version ? ", " + esc(q.version) : ""}">${(q.score * 100).toFixed(0)}%</td>`;
    const qrows = qtasks;
    const qhtml = `<details class="quality" data-model="${esc(m.id)}"${openQuality.has(m.id) ? " open" : ""}>` +
      `<summary>Quality scores</summary>` +
      `<table class="compact fixed quality"><colgroup><col><col style="width:120px"></colgroup>` +
      qrows.map(([t, name, desc]) => `<tr title="${esc(desc)}"><td>${name}</td>${qcell(DATA.quality?.[m.id]?.[t])}</tr>`).join("") + `</table></details>`;
    rows.push(`<div class="card model-card" data-model="${esc(m.id)}"><div class="model-head"><h2>${modelTag(m)}</h2><span class="who">${m.unsupported ? esc(m.label) : "pie / " + esc(m.label)}</span><span class="grip" title="drag to reorder">⋮⋮</span></div>` +
      `<table class="compact fixed ov"><colgroup><col style="width:17%"><col><col style="width:110px"><col><col style="width:120px"><col style="width:112px"></colgroup>` +
      `<tr><th>benchmark</th><th title="decode tok/s (total across requests when concurrent)">decode</th><th class="vs-h" title="pie / ${esc(m.label)}">tok/s</th><th title="prefill tok/s">prefill</th><th class="vs-h" title="pie / ${esc(m.label)}">tok/s</th><th class="vs-h" title="median time to first token, pie / ${esc(m.label)}">TTFT</th></tr>${body}</table>${qhtml}</div>`);
  }
  main.innerHTML = html + rows.join("");
  let dragged = null;
  main.querySelectorAll(".model-card").forEach(card => {
    card.addEventListener("dragstart", e => { dragged = card.dataset.model; card.classList.add("dragging"); e.dataTransfer.effectAllowed = "move"; });
    const grip = card.querySelector(".grip");
    grip.addEventListener("pointerdown", () => { card.draggable = true; });
    grip.addEventListener("pointerup", () => { card.draggable = false; });
    card.addEventListener("dragend", () => { card.draggable = false; card.classList.remove("dragging"); main.querySelectorAll(".drop-before,.drop-after").forEach(c => c.classList.remove("drop-before", "drop-after")); });
    card.addEventListener("dragover", e => {
      if (!dragged || dragged === card.dataset.model) return;
      e.preventDefault();
      const r = card.getBoundingClientRect(), after = e.clientY > r.top + r.height / 2;
      card.classList.toggle("drop-after", after);
      card.classList.toggle("drop-before", !after);
    });
    card.addEventListener("dragleave", () => card.classList.remove("drop-before", "drop-after"));
    card.addEventListener("drop", e => {
      e.preventDefault();
      if (!dragged || dragged === card.dataset.model) return;
      const ids = orderedModels().map(m => m.id).filter(id => id !== dragged);
      const at = ids.indexOf(card.dataset.model) + (card.classList.contains("drop-after") ? 1 : 0);
      ids.splice(at, 0, dragged);
      try { localStorage.setItem("stats-model-order", JSON.stringify(ids)); } catch {}
      dragged = null;
      draw();
    });
  });
  main.querySelectorAll("details.quality").forEach(d => d.addEventListener("toggle", () => {
    if (d.open) openQuality.add(d.dataset.model); else openQuality.delete(d.dataset.model);
  }));
  document.getElementById("latest")?.addEventListener("click", e => { e.preventDefault(); sel = home; draw(); });
  document.getElementById("done")?.addEventListener("click", e => { e.preventDefault(); sel = done; draw(); });
}
function openRuns(shas) {
  if (!me) return openSignIn();
  const list = Array.isArray(shas) ? shas : [shas];
  const sha = list[0], c = commitOf(sha), parent = list.length > 1 ? null : parentOf(sha);
  const row = (kind, x, on) => `<label class="toggle"><span>${kind === "models" ? modelTag(x) : esc(x.name)}</span><input type="checkbox" class="switch" data-kind="${kind}" value="${x.id}" ${on ? "checked" : ""}></label>`;
  sheet(`<div class="sheet-head"><h2>Bench</h2>` +
    `<div class="chips">${list.map(x => `<code>${x.slice(0, 7)}</code>`).join("")}</div></div>` +
    `<div class="group"><div class="label">Models</div>${DATA.models.map(m => row("models", m, false)).join("")}</div>` +
    `<div class="group"><div class="label">Machines</div>${RUNNABLE.map(m => row("machines", m, true)).join("") || `<div class="muted">No machine is connected.</div>`}</div>` +
    `<div class="group"><div class="label">Benchmarks</div>${DATA.benchmarks.map(b => row("benchmarks", b, true)).join("")}` +
    (parent ? `<label class="toggle"><span>Also measure <code>${parent.sha.slice(0, 7)}</code>, the commit before</span><input type="checkbox" class="switch" id="parent" checked></label>` : "") +
    `</div><div class="sheet-foot"><span id="msg" class="muted"></span><button class="act" id="run">Run</button></div>`);
  document.getElementById("run").onclick = async () => {
    const pick = k => [...document.querySelectorAll(`#sheet input[data-kind=${k}]:checked`)].map(x => x.value);
    const inputs = { models: pick("models").join(","), macs: pick("machines").join(","), benchmarks: pick("benchmarks").join(",") };
    const msg = document.getElementById("msg");
    if (!inputs.models || !inputs.macs || !inputs.benchmarks) { msg.textContent = "Pick at least one model, machine and benchmark."; return; }
    const shas = [...list, ...(document.getElementById("parent")?.checked ? [parent.sha] : [])];
    msg.textContent = "Starting…";
    try {
      for (const s of shas)
        await gh(`repos/${DATA.repo}/actions/workflows/pie-eval.yml/dispatches`, { method: "POST", body: JSON.stringify({ ref: "main", inputs: { pie_commit: s, ...inputs } }) });
      msg.innerHTML = `Started ${shas.length} run${shas.length > 1 ? "s" : ""}. <a href="https://github.com/${DATA.repo}/actions/workflows/pie-eval.yml" target="_blank">Follow</a>`;
    } catch (e) { msg.textContent = "Could not start: " + e.message; }
  };
}

async function configure() {
  const main = document.getElementById("main");
  main.innerHTML = `<div class="card muted">Loading…</div>`;
  const path = `repos/${DATA.repo}/contents/config.json`;
  let setup = { models: [], machines: [], benchmarks: [] }, last = null;
  try {
    const file = await gh(path);
    if (file) setup = { ...setup, ...JSON.parse(atob(file.content)) };
    last = (await gh(`repos/${DATA.repo}/commits?path=config.json&per_page=1`) || [])[0];
  } catch (e) { main.innerHTML = `<div class="card down">${esc(e.message)}</div>`; return; }
  const on = (kind, id) => (setup[kind] || []).includes(id);
  const gib = g => g == null ? "–" : `${g} GB`;
  const models = DATA.models.filter(m => on("models", m.id));
  const machines = RUNNABLE.filter(m => on("machines", m.id));
  const tests = DATA.benchmarks.filter(b => on("benchmarks", b.id));

  let html = `<div class="card"><table class="compact"><tr><th>model</th><th>precision</th><th class="num">size</th></tr>`;
  for (const m of models) html += `<tr><td>${modelTag(m)}</td><td class="muted">${esc(m.quant || "")}</td><td class="num">${gib(m.gib)}</td></tr>`;
  if (!models.length) html += `<tr><td colspan="3" class="muted">No model runs.</td></tr>`;
  html += `</table></div><div class="card"><table class="compact"><tr><th>machine</th><th>id</th><th>memory</th><th>status</th></tr>`;
  for (const m of machines) html += `<tr><td>${esc(m.name)}</td><td class="muted">${m.id}</td><td>${m.memory_gib ? m.memory_gib + " GB" : "–"}</td>` +
    `<td><span class="dot ${m.status}"></span>${m.status}</td></tr>`;
  if (!machines.length) html += `<tr><td colspan="4" class="muted">No machine runs.</td></tr>`;
  html += `</table></div><div class="card"><table class="compact"><tr><th>benchmark</th><th>description</th></tr>`;
  for (const b of tests) html += `<tr><td>${esc(b.name)}</td><td class="muted">${esc(b.description)}</td></tr>`;
  if (!tests.length) html += `<tr><td class="muted">No benchmark runs.</td></tr>`;
  const by = last ? `Set <span title="${fmtDate(last.commit.committer.date)} ${fmtTime(last.commit.committer.date)}">${relTime(last.commit.committer.date)}</span> in config.json` : "Not set up yet: pushes run nothing";
  main.innerHTML = html + `</table></div><div class="muted">${by}</div>`;
}

function pushes() {
  let html = `<div class="card"><table class="compact fixed"><colgroup><col><col style="width:120px"><col style="width:96px"><col style="width:60px"><col style="width:48px"></colgroup>` +
             `<tr><th>commit</th><th>author</th><th>date</th><th>time</th><th></th></tr>`;
  const pages = Math.max(1, Math.ceil(ALL.length / PER_PAGE));
  page = Math.min(page, pages - 1);
  const isMeasured = new Set(DATA.commits.map(c => c.sha));
  const running = new Set(DATA.running || []);
  const fits = x => (!filt.mac || x.mac === filt.mac) && (!filt.model || x.model === pieOf(filt.model));
  const failedAt = sha => (DATA.failed || []).some(f => f.sha === sha && fits(f));
  const measuredAt = sha => Object.entries(DATA.results).some(([mac, r]) =>
    (!filt.mac || mac === filt.mac) && Object.entries(r.models).some(([model, byTest]) =>
      (!filt.model || model === pieOf(filt.model)) && Object.values(byTest).some(byCommit => byCommit[sha])));
  const state = sha => running.has(sha) ? ["wait", "waiting to run"]
    : failedAt(sha) ? [measuredAt(sha) ? "part" : "bad", measuredAt(sha) ? "some benchmarks failed" : "failed"]
    : measuredAt(sha) ? ["run", "benchmarked"] : ["", "not benchmarked"];
  for (const c of ALL.slice(page * PER_PAGE, (page + 1) * PER_PAGE)) {
    html += `<tr class="push${isMeasured.has(c.sha) ? "" : " flat"}" data-sha="${c.sha}"><td class="clip" title="${esc(c.message)}"><span class="dot ${state(c.sha)[0]}" title="${state(c.sha)[1]}"></span><code>${c.sha.slice(0, 7)}</code> ${esc(c.message)}</td>` +
            `<td class="clip">${esc(c.author)}</td><td>${fmtDate(c.date)}</td><td class="muted">${fmtTime(c.date)}</td>` +
            `<td class="pickcell"><label><input type="checkbox" class="rowpick" value="${c.sha}"${chosen.has(c.sha) ? " checked" : ""}></label></td></tr>`;
  }
  document.getElementById("main").innerHTML = html + `</table>${pager(pages)}</div>`;
  document.querySelectorAll("tr.push").forEach(tr => tr.onclick = e => {
    if (e.target.classList.contains("pickcell")) return e.target.querySelector("input.rowpick").click();
    if (e.target.closest(".pickcell")) return;
    if (!isMeasured.has(tr.dataset.sha)) return;
    sel = tr.dataset.sha; tab = "Overview"; draw();
  });
  document.querySelectorAll("input.rowpick").forEach(x => x.onchange = () => {
    x.checked ? chosen.add(x.value) : chosen.delete(x.value);
    document.getElementById("bench").disabled = !chosen.size;
  });
  document.getElementById("bench").disabled = !chosen.size;
  document.querySelectorAll(".pager button[data-page]").forEach(b => b.onclick = () => { page = +b.dataset.page; draw(); window.scrollTo(0, 0); });
}
function pager(pages) {
  if (pages < 2) return "";
  const want = [...new Set([0, pages - 1, page - 1, page, page + 1].filter(i => i >= 0 && i < pages))].sort((a, b) => a - b);
  let out = "", prev = -1;
  for (const i of want) {
    if (i - prev > 1) out += `<span class="gap">…</span>`;
    out += `<button class="pill ${i === page ? "on" : ""}" data-page="${i}">${i + 1}</button>`;
    prev = i;
  }
  return `<div class="pager"><button class="pill" data-page="${Math.max(0, page - 1)}" ${page === 0 ? "disabled" : ""}>‹ Prev</button>${out}` +
         `<button class="pill" data-page="${Math.min(pages - 1, page + 1)}" ${page === pages - 1 ? "disabled" : ""}>Next ›</button></div>`;
}

function pool() {
  let html = "";
  for (const kind of [...new Set(["self-hosted", ...DATA.pool.map(m => m.kind)])]) {
    const list = DATA.pool.filter(m => m.kind === kind);
    html += `<div class="card"><table class="compact fixed">` +
            `<colgroup><col><col style="width:90px"><col style="width:100px"><col style="width:150px"><col style="width:90px"><col style="width:150px"></colgroup>` +
            `<tr><th>machine</th><th>memory</th><th>status</th><th>last run</th><th>commit</th><th>author</th></tr>`;
    for (const m of list) {
      const c = m.last.commit ? commitOf(m.last.commit) : null;
      html += `<tr><td class="clip">${esc(m.name)} <span class="muted">${m.id}</span></td>` +
        `<td>${m.memory_gib ? m.memory_gib + " GB" : ""}</td><td><span class="dot ${m.status}"></span>${m.status}</td>` +
        `<td>${m.last.at ? `${fmtDate(m.last.at)} <span class="muted">${fmtTime(m.last.at)}</span>` : "–"}</td>` +
        `<td>${m.last.commit ? `<code>${m.last.commit.slice(0, 7)}</code>` : "–"}</td><td class="clip">${c?.author ? esc(c.author) : "–"}</td></tr>`;
    }
    if (!list.length) html += `<tr><td colspan="6" class="muted">None connected.</td></tr>`;
    html += `</table></div>`;
  }
  document.getElementById("main").innerHTML = html;
}
function people() {
  const ago = d => { if (!d) return "–"; const h = (Date.now() - new Date(d)) / 36e5; return h < 1 ? "just now" : h < 24 ? `${Math.round(h)}h ago` : `${Math.round(h / 24)}d ago`; };
  let html = `<div class="card"><table class="compact"><tr><th>who</th><th>access</th><th class="num">last active</th></tr>`;
  for (const p of DATA.people)
    html += `<tr><td><img class="avatar" src="https://github.com/${p.login}.png?size=44">${esc(p.login)}</td><td class="muted">${esc(p.role || "–")}</td><td class="num muted">${ago(p.last)}</td></tr>`;
  if (!DATA.people.length) html += `<tr><td colspan="3" class="muted">Nobody yet.</td></tr>`;
  document.getElementById("main").innerHTML = html + `</table></div>`;
}

async function gh(path, opts = {}) {
  const r = await fetch(`https://api.github.com/${path}`, { cache: "no-store", ...opts, headers: { Authorization: `Bearer ${token()}`, Accept: "application/vnd.github+json", ...(opts.headers || {}) } });
  if (!r.ok && r.status !== 404) throw new Error(`${r.status} ${await r.text()}`);
  if (r.status === 404 || r.status === 204) return null;
  return r.json();
}
function sheet(html) { document.getElementById("sheet").innerHTML = `<div class="card">${html}</div>`; document.getElementById("modal").hidden = false; }
function closeSheet() { document.getElementById("modal").hidden = true; stopMacPoll(); }
document.getElementById("close").onclick = closeSheet;
document.getElementById("modal").onclick = e => { if (e.target.id === "modal") closeSheet(); };
document.addEventListener("keydown", e => { if (e.key === "Escape") { closeSheet(); closeMenu(); } });

let macPoll = null;
function stopMacPoll() { if (macPoll) { clearInterval(macPoll); macPoll = null; } }
async function addMac() {
  stopMacPoll();
  sheet(`<div class="sheet-head"><h2>Add this Mac</h2></div><p class="muted">Getting a registration token for ${esc(DATA.repo)}…</p>`);
  let reg = null, err = "";
  try { reg = await gh(`repos/${DATA.repo}/actions/runners/registration-token`, { method: "POST" }); } catch (e) { err = String(e.message || e); }
  if (!reg?.token) {
    sheet(`<div class="sheet-head"><h2>Add this Mac</h2></div><p>GitHub refused a runner registration token. Adding runners needs admin rights on ${esc(DATA.repo)} and a token with the <code>repo</code> scope (or Administration: write).</p><p class="muted">${esc(err.slice(0, 200))}</p>`);
    return;
  }
  const before = new Set(((await gh(`repos/${DATA.repo}/actions/runners?per_page=100`).catch(() => null))?.runners || []).map(r => r.id));
  const url = `https://raw.githubusercontent.com/${DATA.repo}/main/infra/mac/setup-runner.sh`;
  const cmd = `curl -fsSL ${url} | bash -s ${reg.token}`;
  const expires = reg.expires_at ? `${fmtTime(reg.expires_at)}` : "in an hour";
  sheet(`<div class="sheet-head"><h2>Add this Mac</h2></div>` +
    `<p>Open Terminal on the Mac you want to add and run:</p>` +
    `<div class="cmd"><code id="cmd">${esc(cmd)}</code><button class="act" id="copy">Copy</button></div>` +
    `<p class="muted">The token works once and expires at ${esc(expires)}. The script finds the chip and memory, installs the GitHub runner as a service and registers it here.</p>` +
    `<ul class="steps" id="steps"><li class="now" data-s="mac"><span class="st"></span><span>Waiting for the Mac to connect</span></li></ul>`);
  document.getElementById("copy").onclick = async e => {
    try { await navigator.clipboard.writeText(cmd); e.target.textContent = "Copied"; } catch { e.target.textContent = "Select and copy"; }
  };
  const set = (key, state, text) => {
    const li = document.querySelector(`#steps li[data-s="${key}"]`);
    if (!li) return;
    li.className = state;
    if (text) li.lastElementChild.innerHTML = text;
  };
  const tick = async () => {
    if (document.getElementById("modal").hidden) return stopMacPoll();
    const runners = (await gh(`repos/${DATA.repo}/actions/runners?per_page=100`).catch(() => null))?.runners || [];
    const fresh = runners.find(r => !before.has(r.id));
    if (!fresh) return;
    const labels = fresh.labels.map(l => l.name);
    const plat = labels.find(l => DATA.platforms.includes(l));
    const where = plat ? ` on <code>${esc(plat)}</code>` : ` <span class="muted">(platform not in matrix/platforms.yaml yet)</span>`;
    if (fresh.status === "online") {
      set("mac", "done", `<code>${esc(fresh.name)}</code> connected${where}`);
      stopMacPoll();
      await refreshPool();
      draw();
    } else set("mac", "now", `<code>${esc(fresh.name)}</code> registered, starting${where}`);
  };
  macPoll = setInterval(tick, 3000);
}
async function removeMac() {
  stopMacPoll();
  const runners = (await gh(`repos/${DATA.repo}/actions/runners?per_page=100`).catch(() => null))?.runners || [];
  if (!runners.length) { sheet(`<div class="sheet-head"><h2>Remove a Mac</h2></div><p class="muted">No Macs are registered to ${esc(DATA.repo)}.</p>`); return; }
  const state = r => r.status !== "online" ? "offline" : r.busy ? "running a job" : "idle";
  sheet(`<div class="sheet-head"><h2>Remove a Mac</h2></div><table class="compact rmlist">` +
    runners.map(r => `<tr><td><code>${esc(r.name)}</code></td><td class="muted">${state(r)}</td><td class="num"><button class="act" data-rm="${r.id}">Remove</button></td></tr>`).join("") +
    `</table>`);
  document.querySelectorAll("[data-rm]").forEach(b => b.onclick = () => removeOne(runners.find(r => String(r.id) === b.dataset.rm)));
}
async function removeOne(r) {
  let tok = null, err = "";
  try { tok = await gh(`repos/${DATA.repo}/actions/runners/remove-token`, { method: "POST" }); } catch (e) { err = String(e.message || e); }
  if (!tok?.token) {
    sheet(`<div class="sheet-head"><h2>Remove ${esc(r.name)}</h2></div><p>GitHub refused a removal token. Removing runners needs admin rights on ${esc(DATA.repo)}.</p><p class="muted">${esc(err.slice(0, 200))}</p>`);
    return;
  }
  const cmd = `curl -fsSL https://raw.githubusercontent.com/${DATA.repo}/main/infra/mac/remove-runner.sh | bash -s ${tok.token}`;
  sheet(`<div class="sheet-head"><h2>Remove ${esc(r.name)}</h2></div>` +
    `<p>On that Mac, run this in Terminal to stop the service and unregister it:</p>` +
    `<div class="cmd"><code>${esc(cmd)}</code><button class="act" id="copy">Copy</button></div>` +
    `<p class="muted">Mac gone or unreachable? <a href="#" id="force">Remove it from GitHub only</a>; its runner files stay on that machine.</p>` +
    `<ul class="steps" id="steps"><li class="now" data-s="gone"><span class="st"></span><span>Waiting for ${esc(r.name)} to unregister</span></li></ul>`);
  document.getElementById("copy").onclick = async e => {
    try { await navigator.clipboard.writeText(cmd); e.target.textContent = "Copied"; } catch { e.target.textContent = "Select and copy"; }
  };
  document.getElementById("force").onclick = async e => {
    e.preventDefault();
    try { await gh(`repos/${DATA.repo}/actions/runners/${r.id}`, { method: "DELETE" }); } catch (x) { e.target.textContent = `failed: ${String(x.message || x).slice(0, 80)}`; }
  };
  macPoll = setInterval(async () => {
    if (document.getElementById("modal").hidden) return stopMacPoll();
    const left = (await gh(`repos/${DATA.repo}/actions/runners?per_page=100`).catch(() => null))?.runners;
    if (left && !left.some(x => x.id === r.id)) {
      const li = document.querySelector('#steps li[data-s="gone"]');
      li.className = "done";
      li.lastElementChild.textContent = `${r.name} removed`;
      stopMacPoll();
      await refreshPool();
      draw();
    }
  }, 3000);
}
function openSignIn() { if (tab !== "Sign in") back = tab; tab = "Sign in"; closeSheet(); draw(); }
function signInPage() {
  document.getElementById("main").innerHTML = `<div class="signin-wrap"><div class="card signin-page"><svg width="40" height="40" class="gh-mark" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg><h2>Sign in with GitHub</h2><p class="muted">Paste a GitHub token with access to ${REPO_NAME}.</p>` +
    `<input id="tok" type="password" placeholder="Paste GitHub access token"><button class="act" id="go">Sign in</button><div id="err" class="err"></div></div></div>`;
  const tok = document.getElementById("tok"), go = document.getElementById("go");
  tok.focus();
  tok.onkeydown = e => { if (e.key === "Enter") go.click(); };
  go.onclick = async () => {
    try { localStorage.setItem("pie-evals-token", tok.value.trim()); } catch {}
    await signIn();
    if (me) { tab = back; draw(); } else document.getElementById("err").textContent = denied || "That token did not work.";
  };
}
function closeMenu() { document.querySelector(".menu")?.remove(); }
document.addEventListener("click", e => { if (!e.target.closest("#who")) closeMenu(); });
async function signIn() {
  me = null; denied = "";
  if (!token()) return renderWho();
  try { me = await gh("user"); } catch { me = null; denied = "That token did not work."; }
  if (me) {
    let repo = null;
    try { repo = await gh(`repos/${DATA.repo}`); } catch { repo = null; }
    if (!repo?.permissions?.push) {
      denied = `${me.login} does not have write access to ${REPO_NAME}. Ask an admin to add you.`;
      me = null;
      try { localStorage.removeItem("pie-evals-token"); } catch {}
    }
  }
  if (me) await refreshPool();
  renderWho();
}
function renderWho() {
  const who = document.getElementById("who");
  who.innerHTML = me
    ? `<button class="pill" id="me"><img class="avatar" src="${me.avatar_url}">${esc(me.login)}</button>`
    : `<button class="signin" id="in"><svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>Sign in with GitHub</button>`;
  const i = document.getElementById("in"), m = document.getElementById("me");
  if (i) i.onclick = openSignIn;
  if (m) m.onclick = () => {
    if (document.querySelector(".menu")) return closeMenu();
    who.insertAdjacentHTML("beforeend", `<div class="menu"><button id="addmac"><svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M7.75 2a.75.75 0 0 1 .75.75V7h4.25a.75.75 0 0 1 0 1.5H8.5v4.25a.75.75 0 0 1-1.5 0V8.5H2.75a.75.75 0 0 1 0-1.5H7V2.75A.75.75 0 0 1 7.75 2Z"/></svg>Add Mac</button><button id="rmmac"><svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M2.75 7.25h10.5a.75.75 0 0 1 0 1.5H2.75a.75.75 0 0 1 0-1.5Z"/></svg>Remove Mac</button><button id="out"><svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M2 2.75C2 1.784 2.784 1 3.75 1h2.5a.75.75 0 0 1 0 1.5h-2.5a.25.25 0 0 0-.25.25v10.5c0 .138.112.25.25.25h2.5a.75.75 0 0 1 0 1.5h-2.5A1.75 1.75 0 0 1 2 13.25Zm10.44 4.5-1.97-1.97a.749.749 0 0 1 .326-1.275.749.749 0 0 1 .734.215l3.25 3.25a.75.75 0 0 1 0 1.06l-3.25 3.25a.749.749 0 0 1-1.275-.326.749.749 0 0 1 .215-.734l1.97-1.97H6.75a.75.75 0 0 1 0-1.5Z"/></svg>Sign out</button></div>`);
    document.getElementById("addmac").onclick = () => { closeMenu(); addMac(); };
    document.getElementById("rmmac").onclick = () => { closeMenu(); removeMac(); };
    document.getElementById("out").onclick = () => { closeMenu(); try { localStorage.removeItem("pie-evals-token"); } catch {} me = null; renderWho(); draw(); };
  };
}

function draw() {
  if (!me && tab !== "Overview" && tab !== "Sign in") tab = "Overview";
  else if (me && tab === "Sign in") tab = back;
  document.getElementById("tabs").innerHTML = (me ? TABS : ["Overview"]).map(t => `<button class="${t === tab ? "on" : ""}">${t}</button>`).join("");
  document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => { tab = b.textContent; draw(); });
  document.getElementById("controls").hidden = tab !== "Overview" && tab !== "History";
  document.getElementById("fmac").hidden = tab !== "History" && tab !== "Overview";
  document.getElementById("fmodel").hidden = tab !== "History";
  const bench = document.getElementById("bench");
  bench.hidden = tab !== "History";
  bench.disabled = !chosen.size;
  document.getElementById("in")?.classList.toggle("on", tab === "Sign in");
  ({ "Overview": overview, "Pipeline": configure, "History": pushes, "Machines": pool, "People": people, "Sign in": signInPage })[tab]();
}
fillFilters();
document.getElementById("bench").onclick = () => { if (chosen.size) openRuns([...chosen]); };
signIn().then(draw);
</script>
</body>
</html>
"""


def _gh(path: str) -> dict | list | None:
    try:
        out = subprocess.run(["gh", "api", path], capture_output=True, text=True, check=True).stdout
        return json.loads(out)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def history(pie_repo: str) -> list[dict]:
    try:
        out = subprocess.run(["gh", "api", "--paginate", "--slurp", f"repos/{pie_repo}/commits?sha=main&per_page=100"],
                             capture_output=True, text=True, check=True).stdout
        pages = json.loads(out)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return []
    return [{"sha": c["sha"], "message": (c["commit"]["message"] or "").splitlines()[0][:120] if c["commit"]["message"] else "",
             "author": (c.get("author") or {}).get("login") or c["commit"]["author"]["name"],
             "date": c["commit"]["committer"]["date"]} for page in pages for c in page]


def runners(repo: str) -> list[dict] | None:
    got = _gh(f"repos/{repo}/actions/runners?per_page=100")
    return got.get("runners", []) if isinstance(got, dict) else None


def _pool(live: list[dict] | None, matrix: Matrix, last: dict[str, dict]) -> list[dict]:
    pool = []
    for r in live or []:
        labels = [lab["name"] for lab in r.get("labels", [])]
        pid = next((lab for lab in labels if lab in matrix.platforms), None)
        if not pid:
            continue
        spec = matrix.platforms[pid]
        pool.append({
            "name": spec.accelerator,
            "id": pid,
            "kind": "self-hosted",
            "os": spec.os,
            "memory_gib": int(spec.memory_gib),
            "status": "busy" if r["status"] == "online" and r.get("busy") else "idle" if r["status"] == "online" else "offline",
            "last": last.get(pid, {}),
        })
    return sorted(pool, key=lambda m: (m["kind"] != "self-hosted", m["name"]))


def _tokens(n: int) -> str:
    return f"{n // 1024}k" if n >= 1024 and n % 1024 == 0 else str(n)


def benchmarks(matrix: Matrix) -> list[dict]:
    workloads = {c.workload.id: c.workload for c in matrix.cells_for(Tier.TARGETED)
                 if str(c.workload.kind) in ("single_stream", "long_context", "concurrency", "prefix_shared")}
    order = sorted(workloads.values(), key=lambda w: (int(w.params.get("concurrency") or 1), int(w.params.get("prefill") or 0)))
    tests = []
    for w in order:
        n, prompt, out = int(w.params.get("concurrency") or 1), int(w.params.get("prefill") or 0), int(w.params.get("decode") or 0)
        head = "story prompt" if w.params.get("prompt") else f"{_tokens(prompt)}-token prompt"
        label = w.params.get("label") or f"{head}, {out} tokens out" + (f", {n} requests at once" if n > 1 else "")
        metric = "prefill" if w.params.get("primary") == "prefill_tok_s" else "decode"
        tests.append({"id": w.id, "name": label, "description": w.params.get("description", ""), "concurrency": n, "metric": metric})
    return tests


def verdicts(results: dict, commits: list[dict]) -> dict[str, list[dict]]:
    order = [c["sha"] for c in sorted(commits, key=lambda c: c["date"] or "")]
    out: dict[str, list[dict]] = {}
    for mac, r in results.items():
        for model, by_test in r["models"].items():
            for wl, by_commit in by_test.items():
                for phase in ("prefill", "decode"):
                    prev = None
                    for sha in order:
                        v = (by_commit.get(sha) or {}).get(phase)
                        if v is None:
                            continue
                        if prev:
                            out.setdefault(sha, []).append(
                                {"mac": mac, "model": model, "test": wl, "phase": phase, "pct": round(100 * (v / prev - 1), 1)})
                        prev = v
    return out


def quant_label(scheme: str) -> str:
    low = scheme.lower()
    if "u4" in low or "q4" in low:
        return "4-bit"
    if "mxfp4" in low:
        return "mxfp4"
    if "bf16" in low:
        return "bf16"
    if "nvfp4" in low:
        return "nvfp4"
    return scheme


def mac_models(matrix: Matrix) -> list[dict]:
    macs = [p for p in matrix.platforms.values() if p.os == "macos"]
    why = {}
    for c in matrix.expand():
        if str(c.engine) == "pie":
            why.setdefault(c.artifact.id, set()).add(c.declared_unsupported_reason)
    out = []
    for a in matrix.artifacts.values():
        pie = matrix.artifacts.get(a.baseline_of or "")
        if a.source_format.value != "ollama" or pie is None or not macs:
            continue
        out.append({"id": a.id, "name": a.ollama_tag, "pie": pie.id, "label": a.baseline_label or "Ollama", "family": a.family,
                    "scheme": str(a.scheme), "format": str(a.source_format), "gib": a.expected_gib, "context": pie.max_context,
                    "quant": quant_label(str(a.scheme)),
                    "unsupported": next(iter(why[pie.id])) if pie.id in why and None not in why[pie.id] else None})
    return sorted(out, key=lambda m: m["name"])


def _paginate(path: str) -> list:
    try:
        out = subprocess.run(["gh", "api", "--paginate", "--slurp", path], capture_output=True, text=True, check=True).stdout
        return json.loads(out)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return []


def in_flight(repo: str) -> list[str]:
    runs = _gh(f"repos/{repo}/actions/workflows/pie-eval.yml/runs?per_page=50&status=in_progress") or {}
    queued = _gh(f"repos/{repo}/actions/workflows/pie-eval.yml/runs?per_page=50&status=queued") or {}
    out = []
    for run in [*(runs.get("workflow_runs") or []), *(queued.get("workflow_runs") or [])]:
        title = run.get("display_title") or ""
        for word in title.replace(",", " ").split():
            if len(word) >= 7 and all(ch in "0123456789abcdef" for ch in word):
                out.append(word)
    return sorted(set(out))


def usage(repo: str, authors: dict[str, str]) -> dict[str, dict]:
    pages = _paginate(f"repos/{repo}/actions/workflows/pie-eval.yml/runs?per_page=100")
    out: dict[str, dict] = {}
    for run in (r for page in pages for r in page.get("workflow_runs", [])):
        sha = (run.get("display_title") or "").split()[-1] if run.get("display_title") else ""
        who = (run.get("triggering_actor") or {}).get("login") if run.get("event") == "workflow_dispatch" else authors.get(sha)
        if not who:
            continue
        u = out.setdefault(who, {"last": ""})
        u["last"] = max(u["last"], run.get("created_at") or "")
    return out


def people(repo: str, authors: dict[str, str]) -> list[dict]:
    roles = {c["login"]: c.get("role_name", "") for page in _paginate(f"repos/{repo}/collaborators?affiliation=all&per_page=100") for c in page}
    used = usage(repo, authors)
    none = {"last": ""}
    rows = [{"login": who, "role": roles.get(who, ""), **used.get(who, none)} for who in set(roles) | set(used)]
    rows.sort(key=lambda p: p["login"].lower())
    return sorted(rows, key=lambda p: p["last"], reverse=True)


def quality(store: Store) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in sorted((store.root / "quality").glob("*.json")) if (store.root / "quality").exists() else []:
        q = json.loads(f.read_text())
        out.setdefault(q["artifact"], {})[q["task"]] = {k: q.get(k) for k in ("score", "ci95", "n", "version", "date", "engine", "empty")}
    return out


def build(store: Store, matrix: Matrix, live: list[dict] | None, *, repo: str, pie_repo: str,
          lookup_commits: bool = True) -> dict:
    tests = benchmarks(matrix)
    concurrency = {b["id"]: b["concurrency"] for b in tests}
    t = store.table(Tier.TARGETED)
    every = [r for r in t.to_pylist() if r["pie_commit"]] if t.num_rows else []
    latest = {}
    for r in every:
        latest[(r["run_id"], r["cell_id"] or r["cell_key"])] = r
    every = list(latest.values())
    measured_ok = (str(CellStatus.PASS), str(CellStatus.NOISY))
    rows = [r for r in every if r["status"] in measured_ok and (r["decode_tok_s"] is not None or r["output_tok_s"] is not None or r["prefill_tok_s"] is not None)]
    rows.sort(key=lambda r: r["started_at"])
    baseline_of = {a.id: a.baseline_of for a in matrix.artifacts.values() if a.baseline_of}
    label_of = {a.id: a.baseline_label or a.id for a in matrix.artifacts.values() if a.baseline_of}
    shown = {m["pie"] for m in mac_models(matrix)}
    baselines: dict[str, dict] = {}
    for r in rows:
        if r["engine"] != "pie" and r["workload"] in concurrency:
            single = concurrency[r["workload"]] == 1
            baselines.setdefault(r["platform"], {}).setdefault(baseline_of.get(r["artifact"], r["artifact"]), {}).setdefault(r["workload"], {}).setdefault(label_of.get(r["artifact"], r["engine"]), []).append({
                "at": r["started_at"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "prefill": r["prefill_tok_s"],
                "decode": r["decode_tok_s"] if single else r["output_tok_s"],
                "version": r["engine_version"] or "", "ttft": r["ttft_ms_p50"],
                "noisy": r["invalid_reason"] if r["status"] == str(CellStatus.NOISY) else None})
    rows = [r for r in rows if r["engine"] == "pie"]
    failed = sorted({(r["pie_commit"], r["platform"], r["artifact"]) for r in every
                     if r["status"] == str(CellStatus.FAIL)
                     and r["workload"] in concurrency and r["artifact"] in shown})

    results: dict[str, dict] = {}
    order: list[str] = []
    last: dict[str, dict] = {}
    have: set[str] = set()
    for r in rows:
        if r["pie_commit"] not in order:
            order.append(r["pie_commit"])
        last[r["platform"]] = {"at": r["started_at"].strftime("%Y-%m-%dT%H:%M:%SZ"), "commit": r["pie_commit"]}
        if r["workload"] not in concurrency:
            continue
        mac = results.setdefault(r["platform"], {"name": r["accelerator"], "models": defaultdict(lambda: defaultdict(dict))})
        cell = json.loads(r["record_json"])["cell"]
        tf = flops.tflops(r, cell["workload"]["params"], flops.model_config(cell["artifact"]["base_model"]))
        single = concurrency[r["workload"]] == 1
        mac["models"][r["artifact"]][r["workload"]][r["pie_commit"]] = {
            "prefill": r["prefill_tok_s"], "prefill_tflops": tf["prefill_tflops"],
            "decode": r["decode_tok_s"] if single else r["output_tok_s"], "decode_tflops": tf["decode_tflops"], "output": r["output_tok_s"],
            "at": r["started_at"].strftime("%Y-%m-%dT%H:%M:%SZ"), "ttft": r["ttft_ms_p50"],
            "noisy": r["invalid_reason"] if r["status"] == str(CellStatus.NOISY) else None}
        have.add(r["artifact"])

    known = {c["sha"]: c for c in (history(pie_repo) if lookup_commits else [])}
    commits = []
    for sha in order:
        if sha in known:
            commits.append(dict(known[sha]))
            continue
        info = _gh(f"repos/{pie_repo}/commits/{sha}") if lookup_commits else None
        commit = (info or {}).get("commit", {})
        commits.append({
            "sha": sha,
            "message": (commit.get("message") or "").splitlines()[0][:90] if commit else "",
            "author": ((info or {}).get("author") or {}).get("login") or (commit.get("author") or {}).get("name", ""),
            "date": (commit.get("committer") or {}).get("date", ""),
        })
    commits.sort(key=lambda c: c["date"] or "~")
    measured = {c["sha"] for c in commits}
    all_commits = [c for c in known.values() if c["sha"] not in measured]

    models = mac_models(matrix)
    for m in models:
        m["has_results"] = m["pie"] in have
    return {
        "repo": repo, "pie_repo": pie_repo, "default_model": DEFAULT_MODEL,
        "benchmarks": tests,
        "verdicts": verdicts(results, commits), "baselines": baselines,
        "baseline_labels": sorted(set(label_of.values())),
        "quality": quality(store),
        "ollama_icon": OLLAMA_ICON,
        "commits": commits, "history": all_commits, "results": results, "models": models,
        "failed": [{"sha": sha, "mac": mac, "model": model} for sha, mac, model in failed],
        "running": in_flight(repo) if lookup_commits else [],
        "pool": _pool(live, matrix, last),
        "platforms": sorted(matrix.platforms),
        "specs": {p.id: {"accelerator": p.accelerator, "os": p.os, "memory_gib": int(p.memory_gib)} for p in matrix.platforms.values()},
        "last": last,
        "people": people(repo, {c["sha"]: c["author"] for c in [*known.values(), *commits]}) if lookup_commits else [],
    }


def render(store: Store, matrix: Matrix, out: Path, live: list[dict] | None = None, *, repo: str = "pie-project/pie-evals",
           pie_repo: str = "pie-project/pie", lookup_commits: bool = True) -> int:
    data = build(store, matrix, live, repo=repo, pie_repo=pie_repo, lookup_commits=lookup_commits)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(PAGE.replace("__DATA__", json.dumps(data)))
    return len(data["commits"])
