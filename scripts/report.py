"""Build a self-contained HTML report from experiment run directories.

Each run directory should contain run.json and a summary CSV. Run metadata and
summary rows are embedded in the output, so it opens directly without a server.
"""

import argparse
import csv
from html import escape
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_csv(path):
    if not path or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        return list(csv.DictReader(input_file))


def _find_csv(run_dir, requested, fallbacks):
    candidates = ([requested] if requested else []) + list(fallbacks)
    for candidate in candidates:
        if candidate:
            path = run_dir / candidate
            if path.is_file():
                return path
    return None


def collect_runs(experiments_dir):
    runs = []
    for metadata_path in sorted(experiments_dir.rglob("run.json")):
        run_dir = metadata_path.parent
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"Skipping {metadata_path}: cannot read run metadata ({error})")
            continue

        summary_path = _find_csv(
            run_dir,
            metadata.get("summary_csv"),
            ("summary.csv", "metrics.csv", "results.csv"),
        )
        if summary_path is None:
            print(f"Skipping {metadata_path}: no summary CSV found")
            continue
        try:
            display_path = run_dir.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            display_path = run_dir.as_posix()
        runs.append(
            {
                "id": run_dir.relative_to(experiments_dir).as_posix(),
                "name": metadata.get("name") or run_dir.name,
                "path": display_path,
                "metadata": metadata,
                "summary": _read_csv(summary_path),
                "summary_file": summary_path.name,
            }
        )
    runs.sort(
        key=lambda run: (
            str(run["metadata"].get("experiment", "")),
            str(run["metadata"].get("split", "")),
            int(run["metadata"].get("run_number", 0) or 0),
            run["id"],
        )
    )
    return runs


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment results</title>
<style>
:root{color-scheme:light;--ink:#18232d;--muted:#687681;--line:#dfe5e8;--paper:#f5f7f6;--card:#fff;--green:#276b58;--green-soft:#e7f1ed;--blue:#476f91;--shadow:0 10px 32px #1d34320b}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}header{background:#fff;border-bottom:1px solid var(--line);padding:25px max(28px,calc((100vw - 1440px)/2));display:flex;align-items:center;justify-content:space-between;gap:20px}.eyebrow{margin:0 0 4px;color:var(--green);font-size:11px;font-weight:750;letter-spacing:.13em;text-transform:uppercase}h1{font-size:25px;line-height:1.2;margin:0;letter-spacing:-.025em}header p.sub{margin:6px 0 0;color:var(--muted)}main{max-width:1440px;margin:0 auto;padding:25px 28px 48px}.toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:18px}.toolbar label,.toolbar select,.search{color:var(--muted);font-size:12px}.toolbar select,.search{border:1px solid var(--line);background:white;border-radius:8px;padding:9px 11px;color:var(--ink)}.search{margin-left:auto;min-width:210px}.pill{border-radius:99px;padding:5px 10px;background:var(--green-soft);color:var(--green);font-size:11px;font-weight:700}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px;margin-bottom:17px}.card,.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow)}.card{padding:16px 18px}.card .label{color:var(--muted);font-size:11px;font-weight:650;letter-spacing:.04em;text-transform:uppercase}.card .value{font-size:24px;font-weight:720;letter-spacing:-.03em;margin-top:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.card .hint{color:var(--muted);font-size:11px;margin-top:2px}.panel{padding:18px 20px;margin-bottom:16px}.panel-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px}.panel h2{font-size:15px;margin:0;letter-spacing:-.01em}.panel .note{font-size:11px;color:var(--muted)}.chart{display:flex;flex-direction:column;gap:9px;max-height:330px;overflow:auto}.bar-row{display:grid;grid-template-columns:minmax(130px,220px) 1fr 90px;align-items:center;gap:12px;font-size:12px}.bar-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#40505a}.bar-track{height:10px;background:#edf1ef;border-radius:99px;overflow:hidden}.bar{height:100%;background:var(--green);border-radius:99px;min-width:2px}.bar-value{text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}.table-wrap{overflow:auto;max-height:55vh;border:1px solid var(--line);border-radius:8px}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th,td{padding:9px 11px;text-align:left;border-bottom:1px solid #edf0f1}th{position:sticky;top:0;background:#f8faf9;color:#5d6b74;font-size:10px;text-transform:uppercase;letter-spacing:.055em;cursor:pointer;z-index:1}tbody tr:hover{background:#f6faf8}td.num{font-variant-numeric:tabular-nums;text-align:right}.empty{padding:28px;text-align:center;color:var(--muted)}.meta{display:flex;flex-wrap:wrap;gap:7px;margin:0 0 12px}.meta span{background:#f1f4f3;padding:4px 8px;border-radius:5px;font-size:11px;color:#52616a}.footer{color:var(--muted);font-size:11px;text-align:center;padding:12px}.error{color:#8b4d32}@media(max-width:850px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}header{padding:20px 22px}main{padding:20px 16px 35px}.search{margin-left:0}}@media(max-width:500px){.cards{grid-template-columns:1fr 1fr;gap:8px}.card{padding:12px}.card .value{font-size:19px}.bar-row{grid-template-columns:110px 1fr 65px;gap:8px}}
</style>
</head>
<body>
<header><div><p class="eyebrow">Research workspace · experiment browser</p><h1>Experiment results</h1><p class="sub">Compare runs, configurations, and summary metrics.</p></div><span class="pill" id="run-count">Loading</span></header>
<main id="app">
<div class="toolbar"><label for="run-select">Experiment</label><select id="run-select"></select><label for="metric-select">Pivot metric</label><select id="metric-select"></select><input class="search" id="search" type="search" placeholder="Filter configurations…" aria-label="Filter configurations"></div>
<section class="cards" id="cards"></section>
<section class="panel"><div class="panel-head"><h2>Configuration summary</h2><span class="note" id="summary-note">Chunk metrics count relevant selected chunks; abstract hit rate counts a question when any chunk from its relevant abstract is selected.</span></div><div class="table-wrap" id="summary-table">__INITIAL_SUMMARY__</div></section>
<section class="panel"><div class="panel-head"><h2 id="pivot-title">Chunking × retrieval</h2><span class="note" id="pivot-note">Cell values show the selected metric for each configuration.</span></div><div class="table-wrap" id="pivot-table"></div></section>
<div class="footer" id="footer"></div>
</main>
<script>
const RUNS=__RUNS__;
const COLUMN_LABELS={method:'retrieval',mean_retrieval_latency_ms:'latency (ms)',mean_tokens_small:'avg tokens @small',mean_tokens_medium:'avg tokens @medium',mean_tokens_large:'avg tokens @large',recall_small:'recall@small',recall_medium:'recall@medium',recall_large:'recall@large'};
const COLUMN_HELP={recall_small:'Macro-averaged recall of relevant source documents among the chunks returned at the small size cutoff.',recall_medium:'Macro-averaged recall of relevant source documents among the chunks returned at the medium size cutoff.',recall_large:'Macro-averaged recall of relevant source documents among the chunks returned at the large size cutoff.'};
const byId=id=>document.getElementById(id);
const parseNumber=v=>v!==null&&v!==undefined&&String(v).trim()!==''&&Number.isFinite(Number(v))?Number(v):null;
const fmt=(v)=>{const n=parseNumber(v);if(n===null)return v==null?'':String(v);return new Intl.NumberFormat(undefined,{maximumFractionDigits:5}).format(n)};
const labelColumn=k=>COLUMN_LABELS[k]||k.replaceAll('_',' ');
function columns(rows){const seen=new Set();for(const row of rows)Object.keys(row).forEach(k=>seen.add(k));return [...seen]}
function numericColumns(rows){return columns(rows).filter(k=>rows.some(r=>parseNumber(r[k])!==null))}
function runNumber(run){const m=run.metadata||{};if(m.run_number!==undefined&&m.run_number!==null)return m.run_number;const leaf=String(run.id||'').split('/').pop();return /^\d+$/.test(leaf)?Number(leaf):'legacy'}
function labelRun(run){const m=run.metadata||{};const raw=String(run.name||'Experiment');const kind=String(m.experiment||raw.split(/[ _]/)[0]||'Experiment').replaceAll('_',' ');const parts=[kind,m.split&&String(m.split),`run ${runNumber(run)}`];if(!m.split)parts.splice(1,1,raw);return parts.filter(Boolean).join(' · ')}
function renderTable(target,rows,emptyText){if(!rows.length){target.innerHTML=`<div class="empty">${emptyText}</div>`;return}const cols=columns(rows);const nums=new Set(numericColumns(rows));target.innerHTML=`<table><thead><tr>${cols.map(c=>`<th data-column="${esc(c)}" title="${esc(COLUMN_HELP[c]||'Click to sort')}">${esc(labelColumn(c))} ↕</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${cols.map(c=>`<td class="${nums.has(c)?'num':''}" title="${esc(r[c]??'')}">${esc(fmt(r[c]))}</td>`).join('')}</tr>`).join('')}</tbody></table>`;target.querySelectorAll('th').forEach(th=>th.addEventListener('click',()=>{const c=th.dataset.column;const sorted=[...rows].sort((a,b)=>{const av=parseNumber(a[c]),bv=parseNumber(b[c]);return av!==null&&bv!==null?av-bv:String(a[c]??'').localeCompare(String(b[c]??''))});renderTable(target,sorted,emptyText)}))}
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function currentRun(){return RUNS[Number(byId('run-select').value)]}
function init(){byId('run-count').textContent=`${RUNS.length} run${RUNS.length===1?'':'s'} loaded`;byId('run-select').innerHTML=RUNS.map((r,i)=>`<option value="${i}">${esc(labelRun(r))}</option>`).join('');byId('run-select').addEventListener('change',render);byId('metric-select').addEventListener('change',render);byId('search').addEventListener('input',render);render()}
function render(){const run=currentRun();if(!run)return;let rows=run.summary||[];const query=byId('search').value.toLowerCase();if(query)rows=rows.filter(r=>Object.values(r).some(v=>String(v).toLowerCase().includes(query)));const kind=String((run.metadata||{}).experiment||'').toLowerCase();const metrics=numericColumns(run.summary||[]);const metricOptions=kind==='diagnostic'?metrics.filter(m=>['mean_retrieval_latency_ms','recall_small','recall_medium','recall_large'].includes(m)):metrics;const metricSelect=byId('metric-select');const previous=metricSelect.value;metricSelect.innerHTML=metricOptions.map(m=>`<option value="${esc(m)}">${esc(labelColumn(m))}</option>`).join('');if(metricOptions.includes(previous))metricSelect.value=previous;const metric=metricSelect.value;
const meta=run.metadata||{};const experiment=meta.experiment||String(run.name||'Experiment').split(/[ _]/)[0];const cards=[['Experiment',experiment,'Experiment type'],['Split',meta.split||'—','Dataset split'],['Run',runNumber(run),'Number within this experiment and split']];if(meta.target_context_tokens)cards.push(['Target context tokens',meta.target_context_tokens,'Estimated context budget']);byId('cards').innerHTML=cards.map(([label,value,hint])=>`<div class="card"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div><div class="hint">${esc(hint)}</div></div>`).join('');
const dimension=rows.some(r=>r.chunking)?'chunking':rows.some(r=>r.chunking_method)?'chunking_method':null;const variantFor=r=>r.method||[r.retrieval_method||'no context',r.scoring_method].filter(Boolean).join(' · ')||r.configuration||r.name||'configuration';const variants=[...new Set(rows.map(variantFor))];const groups=dimension?[...new Set(rows.map(r=>r[dimension]).filter(Boolean))]:[...new Set(rows.map(r=>r.configuration||r.name).filter(Boolean))];const pivotRows=groups.map(group=>{const row={[dimension||'configuration']:group};for(const variant of variants){const match=rows.find(r=>(dimension?r[dimension]===group:(r.configuration||r.name)===group)&&variantFor(r)===variant);row[variant]=match?match[metric]:''}return row});byId('pivot-title').textContent=metric?`${dimension?labelColumn(dimension):'Configuration'} × setup · ${labelColumn(metric)}`:'Configuration comparison';renderTable(byId('pivot-table'),pivotRows,'No configurations match this filter.');
let summaryRows=rows;if(kind==='diagnostic'){const expected=['chunking','method','mean_retrieval_latency_ms','mean_tokens_small','mean_tokens_medium','mean_tokens_large','recall_small','recall_medium','recall_large'];summaryRows=rows.map(row=>Object.fromEntries(expected.map(key=>[key,row[key]??''])));const hasRecall=rows.every(row=>row.recall_small!==undefined&&row.recall_medium!==undefined&&row.recall_large!==undefined);byId('summary-note').textContent=hasRecall?`${run.summary_file} · ${rows.length} configurations. Recall is the fraction of relevant source documents retrieved, averaged over questions; chunk cutoffs are scaled using average chunk lengths.`:`${run.summary_file} · This legacy run has no recall-by-size data. Rerun the diagnostic to populate the requested metrics.`}else{byId('summary-note').textContent=`${run.summary_file} · ${rows.length} configurations. Summary metrics and token usage are shown for this ${kind||'experiment'} run.`}renderTable(byId('summary-table'),summaryRows,'No summary rows match this filter.');byId('footer').textContent=`Source: ${run.path} · Self-contained report generated with embedded data.`}
init();
</script>
</body></html>'''


def build_report(runs):
    embedded_data = json.dumps(runs, ensure_ascii=False, separators=(",", ":"))
    # Prevent data strings from escaping the script element in the HTML document.
    embedded_data = embedded_data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    first_run = runs[0] if runs else {"summary": []}
    initial_summary = _render_initial_summary(first_run.get("summary", []))
    return HTML_TEMPLATE.replace("__RUNS__", embedded_data).replace(
        "__INITIAL_SUMMARY__", initial_summary
    )


def _render_initial_summary(rows):
    if not rows:
        return '<div class="empty">This experiment has no summary rows.</div>'
    headers = list(dict.fromkeys(key for row in rows for key in row))
    head = "".join(
        f"<th>{escape('retrieval' if key == 'method' else key.replace('_', ' '))}</th>"
        for key in headers
    )
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(row.get(key, '') or ''))}</td>" for key in headers) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-dir",
        default=str(PROJECT_ROOT / "data" / "experiments"),
        help="Root directory to scan recursively for run.json files",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "experiments" / "report.html"),
        help="Path for the standalone HTML report",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    experiments_dir = Path(args.experiments_dir).resolve()
    output_path = Path(args.output).resolve()
    if not experiments_dir.is_dir():
        raise SystemExit(f"Experiments directory does not exist: {experiments_dir}")
    runs = collect_runs(experiments_dir)
    if not runs:
        raise SystemExit(f"No experiment run directories found under {experiments_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(runs), encoding="utf-8")
    print(f"Wrote report for {len(runs)} run(s) to {output_path}")


if __name__ == "__main__":
    main()
