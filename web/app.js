'use strict';
const $ = id => document.getElementById(id);
const state = {status:null,busy:false,answer:null,history:[]};
const node = (tag, text, cls) => {const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
const pct = v => v==null?'—':`${(v*100).toFixed(1)}%`;
let toastTimer;
function toast(message){$('toast').textContent=message;$('toast').classList.remove('hidden');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.add('hidden'),4500);}
function showError(error){$('global-alert').textContent=error.message||String(error);$('global-alert').classList.remove('hidden');}
function busy(value){state.busy=value;document.querySelectorAll('.button, #restore-samples, #file-input').forEach(b=>b.disabled=value);}
async function api(path,options={}){
  options={...options};
  if(options.method && options.method!=='GET' && !(options.body instanceof FormData)){
    options.headers={'Content-Type':'application/json',...(options.headers||{})};options.body=JSON.stringify(options.body||{});
  }
  const r=await fetch('/api'+path,options);
  let result;try{result=await r.json();}catch{throw new Error('服务返回异常，请确认终端中的服务仍在运行。');}
  if(!r.ok){let msg=result.detail;if(Array.isArray(msg))msg=msg.map(x=>`${x.loc?.slice(-1)[0]}: ${x.msg}`).join('；');throw new Error(msg||`请求失败（${r.status}）`);}
  return result;
}
async function job(path,body,onProgress){
  const started=await api(path,{method:'POST',body});
  for(;;){await new Promise(r=>setTimeout(r,450));const j=await api('/jobs/'+started.job_id);if(onProgress)onProgress(j.message);if(j.state==='done')return j.result;if(j.state==='error')throw new Error(j.message);}
}
function page(name){
  document.querySelectorAll('.page').forEach(n=>n.classList.toggle('active',n.id==='page-'+name));
  document.querySelectorAll('.nav-item').forEach(n=>n.classList.toggle('active',n.dataset.page===name));
  $('crumb').textContent={chat:'资料问答',library:'资料库',experiments:'实验评测',settings:'接口设置'}[name];
}
document.querySelectorAll('[data-page]').forEach(b=>b.addEventListener('click',()=>page(b.dataset.page)));
async function refresh(){
  const s=await api('/status');state.status=s;
  $('stat-docs').textContent=s.documents.length;$('nav-docs').textContent=String(s.documents.length).padStart(2,'0');
  $('stat-chunks').textContent=s.index.count||0;$('stat-index').textContent=s.needs_rebuild?'待建立 / 更新':'已就绪';
  $('stat-answer').textContent=s.config.answer_mode==='llm'?'大模型 · 引用校验':'原文摘录';
  $('mode-pill').textContent=s.config.answer_mode==='llm'?'模型已配置':'无 Key 演示模式';
  $('index-pill').textContent=s.needs_rebuild?'需重建':'已就绪';
  $('doc-summary').textContent=`${s.documents.length} 份 · 共 ${s.documents.reduce((a,d)=>a+d.chars,0).toLocaleString()} 字符`;
  const prev=$('doc-filter').value;$('doc-filter').replaceChildren(node('option','全部资料'));$('doc-filter').firstChild.value='';
  s.documents.forEach(d=>{const o=node('option',d.name);o.value=d.id;$('doc-filter').append(o);});$('doc-filter').value=prev;
  if($('doc-filter').selectedIndex<0)$('doc-filter').value='';
  if(s.index.size){$('chunk-size').value=s.index.size;$('chunk-overlap').value=s.index.overlap;}
  $('build-status').textContent=s.index.count?`${s.index.count} 个片段 · 隔离 ${s.index.blocked_count} 个可疑片段 · ${s.index.dimension} 维`:'尚未建立索引';
  if(s.needs_rebuild)$('build-status').textContent+=' · 资料或模型已变化，请重建后提问。';
  renderDocuments(s.documents);
  $('connection-info').replaceChildren();
  [['Embedding',s.config.embedding_model],['回答模型',s.config.llm_model],['Qdrant',s.config.qdrant_mode],['外部数据传输',s.config.external_requests?'已配置外部接口':'当前全部在本机运行']].forEach(([k,v])=>{const row=node('div');row.append(node('dt',k),node('dd',v));$('connection-info').append(row);});
}
function renderDocuments(docs){
  $('document-list').replaceChildren();
  if(!docs.length){$('document-list').append(node('p','还没有资料。导入文件，或载入内置演示资料。','empty-table'));return;}
  for(const d of docs){const row=node('div',undefined,'document-row');const icon=node('div',d.name.split('.').pop().toUpperCase(),'file-icon');const desc=node('div',undefined,'file-description');desc.append(node('strong',d.name),node('small',`${d.chars.toLocaleString()} 字符 · ${d.created.slice(0,10)}`));const actions=node('div',undefined,'document-actions');const view=node('button','查看原文','text-button');view.onclick=()=>openSource(d.id);const remove=node('button','移除','text-button danger');remove.onclick=async()=>{if(state.busy)return;if(!confirm(`从当前资料库移除“${d.name}”？原始文件不会被删除。`))return;busy(true);try{await api('/documents/'+d.id,{method:'DELETE'});await refresh();toast('已移除资料，请重建索引。');}catch(e){showError(e);}finally{busy(false);}};actions.append(view,remove);row.append(icon,desc,actions);$('document-list').append(row);}
}
async function openSource(id,evidence){
  try{const d=await api('/documents/'+id);$('source-title').textContent=d.name;$('source-content').replaceChildren();
    d.units.forEach((u,i)=>{const section=node('section',undefined,'source-unit');section.append(node('h3',u.location));const text=node('p');if(evidence && evidence.unit===i){section.classList.add('active');text.append(document.createTextNode(u.text.slice(0,evidence.start)),node('mark',u.text.slice(evidence.start,evidence.end)),document.createTextNode(u.text.slice(evidence.end)));}else text.textContent=u.text;section.append(text);$('source-content').append(section);});
    $('source-dialog').showModal();if(evidence)setTimeout(()=>$('source-content').querySelector('.active')?.scrollIntoView({block:'center'}),30);
  }catch(e){showError(e);}
}
$('close-source').onclick=()=>$('source-dialog').close();
$('source-dialog').addEventListener('click',e=>{if(e.target===$('source-dialog')){const r=$('source-dialog').getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)$('source-dialog').close();}});
function highlight(id){document.querySelectorAll('.evidence-item').forEach(e=>e.classList.remove('highlight'));const target=$('evidence-'+id);if(target){target.classList.add('highlight');target.scrollIntoView({behavior:'smooth',block:'center'});}}
function renderEvidence(evidence){
  $('evidence-count').textContent=evidence.length;$('evidence-list').replaceChildren();
  if(!evidence.length){const empty=node('div',undefined,'empty-evidence');empty.append(node('h3','没有可用证据'),node('p','可以补充资料、改写问题，或检查检索阈值。'));$('evidence-list').append(empty);return;}
  for(const e of evidence){const card=node('article',undefined,'evidence-item');card.id='evidence-'+e.id;const top=node('div',undefined,'evidence-top');top.append(node('span',e.id,'evidence-id'),node('span',e.name,'evidence-name'));card.append(top,node('div',`${e.location} · 字符 ${e.start+1}–${e.end}`,'evidence-location'),node('div',e.text,'evidence-text'));const scores=node('div',undefined,'evidence-scores');scores.append(node('span',`余弦 ${e.cosine.toFixed(3)}`),node('span',`BM25 ${e.bm25.toFixed(2)}`),node('span',`RRF ${e.rrf.toFixed(4)}`));const source=node('button','查看原文 ↗','text-button');source.onclick=()=>openSource(e.doc_id,e);scores.append(source);card.append(scores);$('evidence-list').append(card);}
}
function answerMarkdown(a){let txt=`# ${a.question}\n\n模式：${a.mode==='llm'?'大模型回答':'原文摘录'}\n\n`;if(a.status==='insufficient')txt+='资料不足：'+a.missing.join('；')+'\n\n';for(const c of a.claims)txt+=`${c.text} ${c.evidence.map(e=>`[${e.id}]`).join(' ')}\n\n`;txt+=`${a.notice}\n\n## 证据\n\n`;for(const e of a.evidence)txt+=`### [${e.id}] ${e.name} / ${e.location} / 字符 ${e.start+1}–${e.end}\n\n${e.text}\n\n`;return txt;}
function download(text,name,type){const a=node('a');const url=URL.createObjectURL(new Blob([text],{type}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function renderAnswer(a){
  state.answer=a;$('suggestions').classList.add('hidden');$('answer-area').classList.remove('hidden');$('answer-area').replaceChildren();
  const card=node('article',undefined,'card answer-card');const heading=node('div',undefined,'answer-header');heading.append(node('h2',a.status==='insufficient'?'当前资料不足':a.mode==='llm'?'基于资料的回答':'检索到的原文摘录'),node('span',`${a.timing.total_ms} ms · ${a.verified_quotes} 条引文已匹配`,'answer-meta'));card.append(heading,node('p',a.question,'answer-question'));
  if(a.status==='insufficient'){for(const missing of a.missing)card.append(node('p',missing,'answer-claim'));}
  for(const c of a.claims){const p=node('p',c.text,'answer-claim');const ids=[...new Set(c.evidence.map(r=>r.id))];for(const id of ids){const cite=node('button',id,'citation');cite.title='定位证据 '+id;cite.onclick=()=>highlight(id);p.append(cite);}card.append(p);}
  card.append(node('p',a.notice,'answer-notice'));const actions=node('div',undefined,'answer-actions');const exp=node('button','导出回答 .md','text-button');exp.onclick=()=>download(answerMarkdown(a),'RAG问答记录.md','text/markdown;charset=utf-8');const copy=node('button','复制回答','text-button');copy.onclick=async()=>{try{await navigator.clipboard.writeText(answerMarkdown(a));toast('已复制回答与证据。');}catch{toast('无法访问剪贴板，请使用导出。');}};actions.append(exp,copy);card.append(actions);
  const details=node('details',undefined,'query-details');details.append(node('summary',`检索查询 · ${a.queries.length} 条`));a.queries.forEach(q=>details.append(node('p',q)));card.append(details);$('answer-area').append(card);renderEvidence(a.evidence);
}
function renderHistory(){
  $('history').replaceChildren();if(state.history.length<2)return;$('history').append(node('div','本次会话中的其他问题','small-label'));
  [...state.history].reverse().filter(a=>a!==state.answer).slice(0,5).forEach(a=>{const b=node('button',a.question+' ↗');b.onclick=()=>{if(!state.busy){renderAnswer(a);renderHistory();}};$('history').append(b);});
}
async function ask(event){
  event?.preventDefault();if(state.busy)return;const question=$('question').value.trim();if(!question){$('question').focus();return;}
  if(state.status?.needs_rebuild){page('library');toast('请先重建索引，再进行问答。');return;}
  $('global-alert').classList.add('hidden');busy(true);$('answer-area').classList.remove('hidden');const loading=node('div',undefined,'card answer-loading');loading.append(node('span',undefined,'spinner'),node('span','正在检索资料并核对引用…'));$('answer-area').replaceChildren(loading);
  try{const filter=$('doc-filter').value;const result=await job('/ask',{question,top_k:Number($('top-k').value),strategy:$('strategy').value,threshold:Number($('threshold').value),expand:$('expand').checked,doc_ids:filter?[filter]:[]});state.history.push(result);state.history=state.history.slice(-12);renderAnswer(result);renderHistory();}
  catch(e){$('answer-area').replaceChildren();showError(e);}finally{busy(false);}
}
$('ask-form').addEventListener('submit',ask);$('question').addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){e.preventDefault();$('ask-form').requestSubmit();}});
document.querySelectorAll('[data-question]').forEach(b=>b.onclick=()=>{if(state.busy)return;$('question').value=b.dataset.question;$('ask-form').requestSubmit();});
$('clear-chat').onclick=()=>{if(state.busy)return;$('question').value='';$('answer-area').classList.add('hidden');$('suggestions').classList.remove('hidden');$('question').focus();};
function updateParams(){$('param-summary').textContent=($('strategy').value==='hybrid'?'混合检索':'向量检索')+' · top-k '+$('top-k').value;}
$('top-k').oninput=updateParams;$('strategy').onchange=updateParams;
async function uploadFiles(files){if(state.busy||!files.length)return;busy(true);$('global-alert').classList.add('hidden');let added=0;try{for(const file of files){if(file.size>10*1024*1024)throw new Error(file.name+' 超过 10 MB。');$('upload-status').textContent='正在导入：'+file.name;const data=new FormData();data.append('file',file);const result=await api('/documents',{method:'POST',body:data});if(!result.duplicate)added++;}toast(`已导入 ${added} 份新资料，请重建索引。`);$('upload-status').textContent=`导入完成，新增 ${added} 份；重复文件已跳过。`;}catch(e){showError(e);$('upload-status').textContent='部分文件未导入，请查看提示。';}finally{await refresh().catch(showError);busy(false);$('file-input').value='';}}
$('file-input').onchange=e=>uploadFiles([...e.target.files]);
for(const type of ['dragenter','dragover'])$('dropzone').addEventListener(type,e=>{e.preventDefault();$('dropzone').classList.add('dragging');});
for(const type of ['dragleave','drop'])$('dropzone').addEventListener(type,e=>{e.preventDefault();$('dropzone').classList.remove('dragging');});
$('dropzone').addEventListener('drop',e=>uploadFiles([...e.dataTransfer.files]));
$('restore-samples').onclick=async()=>{if(state.busy)return;busy(true);try{await api('/samples',{method:'POST'});await refresh();toast('演示资料已载入，请重建索引。');}catch(e){showError(e);}finally{busy(false);}};
$('rebuild').onclick=async()=>{if(state.busy)return;busy(true);$('global-alert').classList.add('hidden');try{await job('/build',{size:Number($('chunk-size').value),overlap:Number($('chunk-overlap').value)},s=>$('build-status').textContent=s);await refresh();toast('索引已更新，可以开始提问。');}catch(e){showError(e);$('build-status').textContent='重建未完成，原索引未被替换。';}finally{busy(false);}};
$('run-eval').onclick=async()=>{if(state.busy)return;busy(true);$('global-alert').classList.add('hidden');$('eval-status').textContent='开始评测…';try{const result=await job('/evaluate',{},s=>$('eval-status').textContent=s);renderEvaluation(result);toast('评测完成，可下载逐题结果。');}catch(e){showError(e);$('eval-status').textContent='评测未完成';}finally{busy(false);}};
function renderEvaluation(result){
  $('eval-status').textContent=`${result.runs.length} 组 · ${result.question_count} 题 / 组`;$('eval-results').className='eval-table-wrap';$('eval-results').replaceChildren();const table=node('table',undefined,'eval-table');const head=node('thead');const hr=node('tr');['chunk','top-k','检索','证据召回','完整证据','多证据完整','无答案拒答'].forEach(t=>hr.append(node('th',t)));head.append(hr);table.append(head);const body=node('tbody');const best=Math.max(...result.runs.map(r=>r.metrics.all_evidence_at_k));for(const r of result.runs){const row=node('tr');if(r.metrics.all_evidence_at_k===best)row.className='best';[r.size,r.top_k,r.strategy==='hybrid'?'混合':'向量',pct(r.metrics.evidence_recall),pct(r.metrics.all_evidence_at_k),pct(r.metrics.multi_all_evidence),pct(r.metrics.refusal_rate_on_unanswerable)].forEach((v,i)=>row.append(node('td',String(v),i===4?'bar-cell':undefined)));body.append(row);}table.append(body);$('eval-results').append(table,node('p','浅色行表示本次完整证据率最高的配置。小样本结果不能代表真实课程总体效果。','help'));
  const quoteRates=result.runs.map(r=>r.metrics.quote_match_rate).filter(x=>x!=null);$('eval-results').append(node('p',`已展示引文的原文匹配率：${quoteRates.length?pct(Math.min(...quoteRates))+'–'+pct(Math.max(...quoteRates)):'无引用'}。回答先经引用校验；该数值不代表语义正确率。`,'help'));
  $('eval-downloads').replaceChildren();for(const [ext,file] of Object.entries(result.files)){const a=node('a','下载 '+ext.toUpperCase());a.href='/api/reports/'+encodeURIComponent(file);a.download=file;$('eval-downloads').append(a);}
}
refresh().catch(e=>{showError(e);$('mode-pill').textContent='连接失败';});
