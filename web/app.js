'use strict';
const $ = id => document.getElementById(id);
const state = {status:null,busy:false,answer:null,history:[],sessionId:null,settingsBusy:false,evaluations:new Map(),evaluationTotal:0,evaluationLoading:false};
const node = (tag, text, cls) => {const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
const pct = v => v==null?'—':`${(v*100).toFixed(1)}%`;
let toastTimer;
function toast(message){$('toast').textContent=message;$('toast').classList.remove('hidden');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.add('hidden'),4500);}
function showError(error){displayFailure(error);}
function updateDeleteSession(){const button=$('delete-session');button.disabled=state.busy||!state.sessionId;button.title=state.busy?'正在处理任务，请完成后再删除会话。':!state.sessionId?'请先选择一条已保存的会话。':'删除当前会话及其问答记录';}
function busy(value,activeButton){state.busy=value;document.querySelectorAll('.button, #restore-samples, #file-input, #saved-session').forEach(b=>{if(!b.closest('#help-dialog, #welcome-dialog, #field-help-dialog, #error-dialog, #global-alert')&&b.id!=='open-help')b.disabled=value;});document.querySelectorAll('[aria-busy="true"]').forEach(b=>b.removeAttribute('aria-busy'));if(value&&activeButton)activeButton.setAttribute('aria-busy','true');updateDeleteSession();updateEvaluationControls();}
async function api(path,options={}){
  options={...options};
  if(options.method && options.method!=='GET' && !(options.body instanceof FormData)){
    options.headers={'Content-Type':'application/json',...(options.headers||{})};options.body=JSON.stringify(options.body||{});
  }
  const r=await fetch('/api'+path,options);
  let result;try{result=await r.json();}catch{throw new Error('服务返回异常，请确认终端中的服务仍在运行。');}
  if(!r.ok){let msg=result.detail;if(Array.isArray(msg))msg=msg.map(x=>`${x.loc?.slice(-1)[0]}: ${x.msg}`).join('；');const e=new Error(msg||`请求失败（${r.status}）`);e.diagnostic=result.error;throw e;}
  return result;
}
async function job(path,body,onProgress,onStarted){
  const started=await api(path,{method:'POST',body});
  onStarted?.(started);
  return waitForJob(started.job_id,onProgress);
}
async function waitForJob(id,onProgress){
  const poll=async()=>{for(;;){await new Promise(r=>setTimeout(r,450));const j=await api('/jobs/'+id);onProgress?.(j.message);if(j.state==='done')return j.result;if(j.state==='error'){const e=new Error(j.message);e.diagnostic=j.error;throw e;}}};
  if(!window.EventSource)return poll();
  return new Promise((resolve,reject)=>{
    const source=new EventSource('/api/jobs/'+id+'/events');
    source.addEventListener('progress',event=>onProgress?.(JSON.parse(event.data).message));
    source.addEventListener('done',event=>{source.close();resolve(JSON.parse(event.data));});
    source.addEventListener('failed',event=>{source.close();const data=JSON.parse(event.data);const e=new Error(data.message);e.diagnostic=data.error;reject(e);});
    source.onerror=()=>{source.close();poll().then(resolve,reject);};
  });
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
  [['向量检索',s.config.embedding_model],['回答模型',s.config.llm_model],['资料索引（Qdrant）',s.config.qdrant_mode],['Agent 规划',s.config.agent_mode],['重排模型',s.config.rerank_model],['外部数据传输',s.config.external_requests?'已配置外部接口':'当前全部在本机运行']].forEach(([k,v])=>{const row=node('div');const term=node('dt',k);term.append(helpButton(k.includes('Qdrant')?'database':k==='向量检索'?'embedding':k==='重排模型'?'rerank':'current'));row.append(term,node('dd',v));$('connection-info').append(row);});
  $('rerank').disabled=s.config.rerank_mode!=='api';if($('rerank').disabled)$('rerank').checked=false;
  updateWorkflow();
}
function updateWorkflow(){
  $('workflow-note').textContent=$('workflow').value==='agent'?(state.status?.config.answer_mode==='llm'?'模型决定检索与原文查阅步骤，最多 4 轮、6 次只读工具调用。':'当前演示使用规则拆分与工具流程；配置 LLM 后可启用模型工具决策。'):'一次检索后生成并核验回答，可在同一会话继续追问。';
}
$('workflow').onchange=updateWorkflow;
async function refreshSessions(){
  const sessions=await api('/sessions');$('saved-session').replaceChildren();
  const empty=node('option','新会话 · 首次提问后保存');empty.value='';$('saved-session').append(empty);
  sessions.forEach(s=>{const option=node('option',s.title+' · '+s.turn_count+' 问');option.value=s.id;$('saved-session').append(option);});
  $('saved-session').value=state.sessionId||'';updateDeleteSession();
}
async function loadSession(id){
  if(state.busy)return;
  if(!id){clearConversation();return;}
  try{const session=await api('/sessions/'+id);state.sessionId=id;state.history=session.turns.map(t=>t.response);rememberSession();
    if(state.history.length)renderAnswer(state.history[state.history.length-1]);else resetAnswer();renderHistory();await refreshSessions();
  }catch(error){showError(error);}
}
function rememberSession(){try{if(state.sessionId)localStorage.setItem('evidence-session',state.sessionId);else localStorage.removeItem('evidence-session');}catch{}}
function resetAnswer(){state.answer=null;$('question').value='';$('answer-area').replaceChildren();$('answer-area').classList.add('hidden');$('suggestions').classList.remove('hidden');renderEvidence([]);}
function clearConversation(){state.sessionId=null;state.history=[];rememberSession();resetAnswer();renderHistory();refreshSessions().catch(showError);$('question').focus();}
$('saved-session').onchange=event=>loadSession(event.target.value);
$('delete-session').onclick=async()=>{if(state.busy||!state.sessionId)return;if(!confirm('删除这个会话和它保存的问答记录？'))return;try{await api('/sessions/'+state.sessionId,{method:'DELETE'});clearConversation();}catch(error){showError(error);}};
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
  for(const e of evidence){const card=node('article',undefined,'evidence-item');card.id='evidence-'+e.id;const top=node('div',undefined,'evidence-top');top.append(node('span',e.id,'evidence-id'),node('span',e.name,'evidence-name'));card.append(top,node('div',`${e.location} · 字符 ${e.start+1}–${e.end}`,'evidence-location'),node('div',e.text,'evidence-text'));const scores=node('div',undefined,'evidence-scores');scores.append(node('span',`余弦 ${e.cosine.toFixed(3)}`),node('span',`BM25 ${e.bm25.toFixed(2)}`),node('span',`RRF ${e.rrf.toFixed(4)}`));if(e.lexical_rescue)scores.append(node('span','明确术语补召回'));if(typeof e.rerank_score==='number')scores.append(node('span',`重排 ${e.rerank_score.toFixed(3)}`));const source=node('button','查看原文 ↗','text-button');source.onclick=()=>openSource(e.doc_id,e);scores.append(source);card.append(scores);$('evidence-list').append(card);}
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
  if(a.retrieval?.warning)card.append(node('p',a.retrieval.warning,'answer-notice'));
  for(const warning of a.retrieval?.warnings||[])card.append(node('p',warning,'answer-notice'));
  if(a.termination==='budget_reached')card.append(node('p','已达到查阅步骤上限，回答仅使用已取得的证据。','answer-notice'));
  if(a.trace)card.append(renderTrace(a));
}
function renderTrace(a){
  const detail=node('details',undefined,'trace-details');detail.append(node('summary','运行详情 · '+(a.workflow==='agent'?'Agent 分步核查':'RAG 检索问答')));
  const metrics=node('div',undefined,'trace-metrics');const usage=a.trace.usage;const requests=a.trace.http_calls.filter(c=>c.service==='llm').length;
  [['证据上下文',`${a.context?.used_chars||0} / ${a.context?.budget_chars||0} 字符`],['向量缓存',`${a.trace.embedding_cache.embedding_hits} 命中 / ${a.trace.embedding_cache.embedding_misses} 新计算`],['模型请求',String(requests)],['服务商返回 token',usage.reported_calls?`${usage.total_tokens}${usage.unreported_calls?'（部分请求未报告）':''}`:'未报告']].forEach(([key,value])=>{const entry=node('div');entry.append(node('small',key),node('strong',value));metrics.append(entry);});detail.append(metrics);
  if(a.agent)detail.append(node('p',`${a.agent.planner==='llm'?'模型工具调用':'规则流程演示'} · ${a.agent.decision_rounds} 轮决策 · ${a.agent.tool_calls} 次工具调用`,'help'));
  const list=node('ol',undefined,'trace-list');const labels={context:'读取会话',decide:'选择查阅步骤',tool:'查阅资料',retrieve:'检索证据',pack:'整理证据',answer:'回答与引用校验'};
  a.trace.steps.forEach(step=>{const item=node('li');item.append(node('span',(labels[step.name]||step.name)+(step.tool?' · '+step.tool:'')),node('span',`${step.duration_ms} ms${step.outcome==='cached'?' · 已复用':step.outcome==='rejected'?' · 参数被拒绝':''}`));list.append(item);});detail.append(list);
  const exportButton=node('button','导出运行记录 .json','text-button');exportButton.onclick=()=>download(JSON.stringify(a.trace,null,2),'知据运行记录_'+a.trace.id.slice(0,8)+'.json','application/json');detail.append(exportButton);return detail;
}
function renderHistory(){
  $('history').replaceChildren();if(state.history.length<2)return;$('history').append(node('div','本次会话中的其他问题','small-label'));
  [...state.history].reverse().filter(a=>a!==state.answer).slice(0,5).forEach(a=>{const b=node('button',a.question+' ↗');b.onclick=()=>{if(!state.busy){renderAnswer(a);renderHistory();}};$('history').append(b);});
}
async function ask(event){
  event?.preventDefault();if(state.busy)return;const question=$('question').value.trim();if(!question){$('question').focus();return;}
  if(state.status?.needs_rebuild){page('library');toast('请先重建索引，再进行问答。');return;}
  busy(true,$('ask-button'));$('answer-area').classList.remove('hidden');const loading=node('div',undefined,'card answer-loading');loading.append(node('span',undefined,'spinner'),node('span','正在检索资料并核对引用…'));$('answer-area').replaceChildren(loading);
  try{if(!state.sessionId){const session=await api('/sessions',{method:'POST'});state.sessionId=session.id;rememberSession();}
    const filter=$('doc-filter').value;const result=await job('/ask',{question,top_k:Number($('top-k').value),strategy:$('strategy').value,threshold:Number($('threshold').value),expand:$('expand').checked,doc_ids:filter?[filter]:[],workflow:$('workflow').value,session_id:state.sessionId,rerank:$('rerank').checked,context_chars:Number($('context-chars').value),lexical_rescue:$('lexical-rescue').checked,answer_guard:$('answer-guard').checked},message=>loading.lastChild.textContent=message);state.history.push(result);state.history=state.history.slice(-100);renderAnswer(result);renderHistory();await refreshSessions();}
  catch(e){$('answer-area').replaceChildren();showError(e);}finally{busy(false);}
}
$('ask-form').addEventListener('submit',ask);$('question').addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){e.preventDefault();$('ask-form').requestSubmit();}});
document.querySelectorAll('[data-question]').forEach(b=>b.onclick=()=>{if(state.busy)return;$('question').value=b.dataset.question;$('ask-form').requestSubmit();});
$('clear-chat').onclick=()=>{if(!state.busy)clearConversation();};
function updateParams(){$('param-summary').textContent=($('strategy').value==='hybrid'?'混合检索':'向量检索')+' · top-k '+$('top-k').value;}
$('top-k').oninput=updateParams;$('strategy').onchange=updateParams;
async function uploadFiles(files){if(state.busy||!files.length)return;busy(true);let added=0;try{for(const file of files){if(file.size>10*1024*1024)throw new Error(file.name+' 超过 10 MB。');$('upload-status').textContent='正在导入：'+file.name;const data=new FormData();data.append('file',file);const result=await api('/documents',{method:'POST',body:data});if(!result.duplicate)added++;}if($('auto-index').checked&&added){await job('/build',{size:Number($('chunk-size').value),overlap:Number($('chunk-overlap').value)},s=>$('upload-status').textContent=s);toast(`已导入 ${added} 份新资料并建立索引，可以直接提问。`);}else toast(`已导入 ${added} 份新资料${added?'，请重建索引':'，重复文件已跳过'}。`);$('upload-status').textContent=`导入完成，新增 ${added} 份；重复文件已跳过。`;}catch(e){showError(e);$('upload-status').textContent='导入或索引未全部完成，请查看提示。已成功导入的资料会保留。';}finally{await refresh().catch(showError);busy(false);$('file-input').value='';}}
$('file-input').onchange=e=>uploadFiles([...e.target.files]);
for(const type of ['dragenter','dragover'])$('dropzone').addEventListener(type,e=>{e.preventDefault();$('dropzone').classList.add('dragging');});
for(const type of ['dragleave','drop'])$('dropzone').addEventListener(type,e=>{e.preventDefault();$('dropzone').classList.remove('dragging');});
$('dropzone').addEventListener('drop',e=>uploadFiles([...e.dataTransfer.files]));
$('restore-samples').onclick=async()=>{if(state.busy)return;busy(true);try{await api('/samples',{method:'POST'});await refresh();toast('演示资料已载入，请重建索引。');}catch(e){showError(e);}finally{busy(false);}};
$('rebuild').onclick=async()=>{if(state.busy)return;busy(true,$('rebuild'));try{await job('/build',{size:Number($('chunk-size').value),overlap:Number($('chunk-overlap').value)},s=>$('build-status').textContent=s);await refresh();toast('索引已更新，可以开始提问。');}catch(e){showError(e);$('build-status').textContent='重建未完成，原索引未被替换。';}finally{busy(false);}};
const evaluationLabels={parameters:'参数对照评测',ablation:'优化前后对比',live:'真实模型评测'};
function updateEvaluationControls(){if(!$('eval-load-more'))return;$('eval-load-more').classList.toggle('hidden',state.evaluations.size>=state.evaluationTotal);$('eval-load-more').disabled=state.busy||state.evaluationLoading;$('eval-empty').classList.toggle('hidden',state.evaluations.size>0);}
function evaluationProgress(id,message){const record=state.evaluations.get(id);if(record)record.message=message;const card=$('evaluation-'+id);if(card)card.querySelector('.evaluation-progress').textContent=message;}
function evaluationFailureButton(diagnostic){const button=node('button','查看原因与解决方法','text-button');button.type='button';button.onclick=()=>{currentFailure=diagnostic;openFailureGuide();};return button;}
function renderEvaluationCard(record,prepend=false,expand=false){
  const old=$('evaluation-'+record.id),card=node('details',undefined,'card eval-card evaluation-run');card.id='evaluation-'+record.id;card.dataset.state=record.state;card.dataset.evaluationId=record.id;card.open=old?old.open:expand;
  state.evaluations.set(record.id,record);
  const summary=node('summary',undefined,'evaluation-heading'),title=node('div',undefined,'evaluation-title');title.append(node('h2',`${evaluationLabels[record.kind]||'评测'} · 第 ${record.number} 次`),node('span',new Date(record.created_at).toLocaleString('zh-CN',{hour12:false})+' · '+(record.request.live?'真实模型':'离线演示'),'muted small'));
  const status=node('span',{running:'正在评测',done:'已完成',error:'评测未完成',interrupted:'评测已中断'}[record.state]||'状态待确认','evaluation-state');summary.append(title,status,node('span','展开 / 收起','muted small evaluation-toggle'));card.append(summary);
  const body=node('div',undefined,'evaluation-body'),progress=node('p',record.state==='done'?`${record.result.runs.length} 组 · ${record.result.question_count} 题 / 组`:record.message,'evaluation-progress help');progress.setAttribute('role','status');body.append(progress);
  const p=record.request.parameters||{},models=record.request.models||{};
  body.append(node('p',`切分 ${p.sizes?.join(' / ')||'见报表'} 字符 · top-k ${p.top_k?.join(' / ')||'见报表'} · ${(p.strategies||[]).map(s=>s==='hybrid'?'混合':'向量').join(' / ')} · 阈值 ${p.thresholds?.join(' / ')??'见报表'} · 重叠 ${p.overlap??'见报表'} 字符 · 查询扩展 ${p.expand?'开启':'关闭'}`,'evaluation-snapshot help'));
  body.append(node('p',`对话：${models.llm_model||'见报表'}；向量：${models.embedding_model||'见报表'}。${record.request.imported?'此记录从旧版完整报表恢复。':'这是运行时保存的设置。'}`,'evaluation-snapshot help'));
  if(record.state==='done')renderEvaluation(record.result,body);
  else if(record.state==='running')body.append(node('p','本轮正在独立运行，完成后会在这里显示结果。历史结果仍可展开和下载。','help'));
  else if(record.state==='interrupted'){
    const guide=node('button','查看中断处理引导','text-button');guide.type='button';guide.onclick=()=>openFieldHelp('eval_history');body.append(node('p','服务重启时，本轮尚未完成，不能作为完整结果。历史报告保留；确认后使用上方对应按钮重新评测，真实模型会产生新的 API 调用。','help'),guide);
  }else{body.append(node('p',record.error?.stage?'失败步骤：'+record.error.stage:'本轮未生成完整报告。','help'));if(record.error)body.append(evaluationFailureButton(record.error));}
  card.append(body);if(old)old.replaceWith(card);else if(prepend)$('evaluation-history').prepend(card);else $('evaluation-history').append(card);updateEvaluationControls();
}
async function loadEvaluations(more=false){
  if(state.evaluationLoading)return;state.evaluationLoading=true;updateEvaluationControls();
  try{let data=await api('/evaluations?offset='+(more?state.evaluations.size:0)+'&limit=20');
    if(more&&(data.total!==state.evaluationTotal||data.items.some(record=>state.evaluations.has(record.id)))){
      // Another tab can insert a new run ahead of this offset. Start a fresh
      // page instead of looping over duplicates and skipping unseen records.
      data=await api('/evaluations?offset=0&limit=20');more=false;toast('评测记录已有更新，已载入最新记录。请继续加载更早的评测。');
    }
    if(!more){state.evaluations.clear();$('evaluation-history').replaceChildren();}state.evaluationTotal=data.total;data.items.forEach((record,i)=>renderEvaluationCard(record,false,!more&&i===0));for(const record of data.items.filter(r=>r.state==='running'))resumeEvaluation(record);}
  finally{state.evaluationLoading=false;updateEvaluationControls();}
}
async function resumeEvaluation(record){
  busy(true,$({parameters:'run-eval',ablation:'run-ablation',live:'run-live-eval'}[record.kind]));
  try{const result=await waitForJob(record.id,message=>evaluationProgress(record.id,message));renderEvaluationCard({...record,state:'done',result,finished_at:new Date().toISOString()});}
  catch(error){try{const saved=await api('/evaluations/'+record.id);renderEvaluationCard(saved);if(saved.state==='running')showError(error);}catch{showError(error);}}
  finally{busy(false);}
}
$('eval-load-more').onclick=()=>loadEvaluations(true).catch(showError);
async function evaluate(live=false,ablation=false){
  if(state.busy||state.evaluationLoading)return;if(live&&!$('live-eval-consent').checked){toast('请先勾选 API 调用和计费确认。');return;}
  let record;busy(true,$(live?'run-live-eval':ablation?'run-ablation':'run-eval'));
  try{const result=await job('/evaluate',{live,ablation},message=>{if(record)evaluationProgress(record.id,message);},started=>{record=started.evaluation;document.querySelectorAll('.evaluation-run').forEach(card=>card.open=false);state.evaluationTotal++;renderEvaluationCard(record,true,true);});renderEvaluationCard({...record,state:'done',result,finished_at:new Date().toISOString()});toast('评测完成，已独立保存，可下载本轮逐题结果。');}
  catch(error){if(record){const diagnostic=error.diagnostic||fallbackDiagnostic(error);let saved;try{saved=await api('/evaluations/'+record.id);}catch{}renderEvaluationCard(saved||{...record,state:'error',message:diagnostic.message,error:diagnostic});}showError(error);}
  finally{busy(false);}
}
$('run-eval').onclick=()=>evaluate();$('run-live-eval').onclick=()=>evaluate(true);$('run-ablation').onclick=()=>evaluate(false,true);
function renderEvaluation(result,target){
  const overview=node('div',undefined,'metric-overview'),results=node('div',undefined,'eval-table-wrap'),downloads=node('div',undefined,'download-links');target.append(overview,results,downloads);
  const table=node('table',undefined,'eval-table');const head=node('thead');const hr=node('tr');['策略','chunk','top-k','检索','证据召回','完整证据','多证据完整','无答案拒答','有答案误拒'].forEach(t=>hr.append(node('th',t)));head.append(hr);table.append(head);const body=node('tbody');const best=Math.max(...result.runs.map(r=>r.metrics.all_evidence_at_k));for(const r of result.runs){const row=node('tr');if(r.metrics.all_evidence_at_k===best)row.className='best';[r.variant||'当前',r.size,r.top_k,r.strategy==='hybrid'?'混合':'向量',pct(r.metrics.evidence_recall),pct(r.metrics.all_evidence_at_k),pct(r.metrics.multi_all_evidence),pct(r.metrics.refusal_rate_on_unanswerable),String(r.metrics.refused_answerable_count??'—')].forEach((v,i)=>row.append(node('td',String(v),i===5?'bar-cell':undefined)));body.append(row);}table.append(body);results.append(table,node('p','浅色行表示本次完整证据率最高的配置。小样本结果不能代表真实课程总体效果。','help'));
  const quoteRates=result.runs.map(r=>r.metrics.quote_match_rate).filter(x=>x!=null);results.append(node('p',`已展示引文的原文匹配率：${quoteRates.length?pct(Math.min(...quoteRates))+'–'+pct(Math.max(...quoteRates)):'无引用'}。回答先经引用校验；该数值不代表语义正确率。`,'help'));
  const quality=r=>[r.metrics.all_evidence_at_k??-1,r.metrics.refusal_rate_on_unanswerable??-1,-(r.metrics.refused_answerable_count??0)];const bestRun=result.runs.reduce((a,b)=>{const x=quality(a),y=quality(b);for(let i=0;i<x.length;i++){if(y[i]>x[i])return b;if(y[i]<x[i])return a;}return a;});
  [['本次模式',result.settings.live?'真实模型':'离线演示'],['最高完整证据率',pct(bestRun.metrics.all_evidence_at_k)],['对应边界题拒答率',pct(bestRun.metrics.refusal_rate_on_unanswerable)],['对应平均耗时',bestRun.metrics.mean_latency_ms+' ms']].forEach(([label,value])=>{const tile=node('div');tile.append(node('small',label),node('strong',value));overview.append(tile);});
  const bars=node('div',undefined,'eval-bars');result.runs.forEach(r=>{const row=node('div');const meter=node('meter');meter.min=0;meter.max=1;meter.value=r.metrics.all_evidence_at_k;meter.setAttribute('aria-label',`chunk ${r.size} top-k ${r.top_k} ${r.strategy} 完整证据率`);row.append(node('span',r.variant||`${r.size} / ${r.top_k} / ${r.strategy==='hybrid'?'混合':'向量'}`),meter,node('b',pct(r.metrics.all_evidence_at_k)));bars.append(row);});results.append(bars);
  for(const [ext,file] of Object.entries(result.files)){const a=node('a','下载 '+ext.toUpperCase());a.href='/api/reports/'+encodeURIComponent(file);a.download=file;downloads.append(a);}
}
function openHelp(){if(!$('help-dialog').open)$('help-dialog').showModal();}
$('open-help').onclick=openHelp;$('close-help').onclick=()=>$('help-dialog').close();$('help-done').onclick=()=>$('help-dialog').close();
$('welcome-start').onclick=()=>{$('welcome-dialog').close();$('question').focus();};$('welcome-help').onclick=()=>{$('welcome-dialog').close();openHelp();};
const connectionLabels={llm:['对话模型','写答案：根据问题与找到的原文组织回答。真实 Agent 还需支持工具调用。'],embedding:['向量模型（可选）','找资料：按文字的意思找相关段落。未配时使用本机词面检索；更换后要重建索引。'],rerank:['重排模型（可选）','挑证据：把更可能回答问题的段落排前面。未配时保留原检索排序。']};
function renderModelFields(config){
  $('model-fields').replaceChildren();$('model-timeout').value=config.timeout;$('model-json').checked=config.json_mode;
  Object.entries(connectionLabels).forEach(([kind,[title,explanation]])=>{
    const c=config[kind],section=node('details',undefined,'connection-section');section.open=kind==='llm'||c.enabled;
    const summary=node('summary');summary.append(document.createTextNode(title),helpButton(kind));section.append(summary,node('p',explanation,'help'));
    const enabledLabel=node('label',undefined,'checkbox-label'),enabled=node('input');enabled.type='checkbox';enabled.id=`model-${kind}-enabled`;enabled.checked=c.enabled;enabledLabel.append(enabled,document.createTextNode('启用这个接口'),helpButton(kind+'_enabled'));section.append(enabledLabel);
    if(kind==='llm'){const preset=node('button','填入 DeepSeek 官方地址示例','text-button');preset.type='button';preset.id='model-deepseek-preset';preset.onclick=()=>{$('model-llm-url').value='https://api.deepseek.com';$('model-llm-model').value='deepseek-flash';$('model-llm-enabled').checked=true;$('model-feedback').textContent='已填入官方地址和模型示例。填写自己账户的调用密钥，先测试再保存。模型名以平台当前列表为准。';};const row=node('div',undefined,'connection-actions');row.append(preset,helpButton('deepseek'));section.append(row);}
    for(const [field,label,hint] of [['url','服务地址（API Base URL）','从当前平台接入文档复制基础地址，不填聊天网页或完整请求路径。'],['key','API Key（调用密钥）',c.has_key?'已保存；留空保留原 Key，输入新 Key 会替换。':'从当前平台 API 控制台创建并复制；不是登录密码。'],['model','模型名／模型 ID','复制符合本卡片用途的模型 ID，或从下方获取列表后选择。']]){
      const wrapper=node('div',undefined,'model-field'),input=node('input');input.id=`model-${kind}-${field}`;input.type='text';input.value=field==='key'?'':c[field];input.maxLength=field==='key'?4096:field==='url'?1500:150;input.autocomplete='off';input.spellcheck=false;
      wrapper.append(fieldHeading(label,kind+'_'+field,input.id),field==='key'?makeApiKeyInput(input):input,node('small',hint));section.append(wrapper);
      if(field==='model'){input.setAttribute('list',`models-${kind}`);const list=node('datalist');list.id=`models-${kind}`;section.append(list);}
    }
    const actions=node('div',undefined,'connection-actions'),show=node('button','显示本次输入','text-button');show.type='button';show.id=`model-${kind}-show`;show.onclick=()=>show.textContent=toggleApiKey($(`model-${kind}-key`))?'显示本次输入':'隐藏本次输入';actions.append(show,helpButton(kind+'_show'));
    const clearLabel=node('label',undefined,'checkbox-label'),clear=node('input');clear.type='checkbox';clear.id=`model-${kind}-clear`;clearLabel.append(clear,document.createTextNode('清除已保存 Key'),helpButton(kind+'_clear'));actions.append(clearLabel);section.append(actions);
    const buttons=node('div',undefined,'connection-actions');for(const [label,route] of [['获取可用模型','models'],['测试这个连接','test']]){const button=node('button',label,'button secondary small');button.type='button';button.id=`model-${kind}-${route}`;button.onclick=()=>probeModel(kind,route);buttons.append(button,helpButton(kind+'_'+route));}section.append(buttons);$('model-fields').append(section);
  });
}
function modelPayload(kind){const payload={timeout:Number($('model-timeout').value),json_mode:$('model-json').checked};for(const k of Object.keys(connectionLabels)){payload[k]={enabled:$(`model-${k}-enabled`).checked,url:$(`model-${k}-url`).value.trim(),model:$(`model-${k}-model`).value.trim(),key:$(`model-${k}-key`).value.trim()||null,clear_key:$(`model-${k}-clear`).checked};}if(kind)payload.kind=kind;return payload;}
function settingsBusy(value){state.settingsBusy=value;document.querySelectorAll('#model-form button, #model-form input').forEach(b=>{if(!b.classList.contains('help-icon'))b.disabled=value;});}
async function openModelSettings(){if(state.busy)return;try{const config=await api('/settings');renderModelFields(config);$('model-feedback').textContent='建议先测试连接，再保存。测试不会改变当前设置。';page('settings');$('model-panel').scrollIntoView({behavior:'smooth',block:'start'});}catch(error){showError(error);}}
async function probeModel(kind,route){if(state.settingsBusy)return;const payload=modelPayload(kind);settingsBusy(true);$('model-feedback').textContent=route==='test'?'正在测试；只发送固定短文本…':'正在获取模型列表…';try{const result=await api('/settings/'+route,{method:'POST',body:payload});if(route==='models'){const list=$('models-'+kind);list.replaceChildren();result.models.forEach(name=>{const option=node('option');option.value=name;list.append(option);});$('model-feedback').textContent=`已获取 ${result.models.length} 个模型。点击模型输入框查看建议，按对话 / 向量 / 重排用途选择。`;}else $('model-feedback').textContent=[...result.checks,...result.warnings,`${result.duration_ms} ms`].join('；');}catch(error){$('model-feedback').textContent=error.message;showError(error);}finally{settingsBusy(false);}}
async function waitForRestart(previous){for(let i=0;i<100;i++){await new Promise(r=>setTimeout(r,400));try{const health=await api('/health');if(health.instance_id!==previous)return;}catch{}}throw new Error('设置已保存，但服务尚未恢复。请查看启动窗口提示。');}
async function saveModelSettings(event,demo=false){event?.preventDefault();if(state.settingsBusy)return;const payload=modelPayload();if(demo)for(const k of Object.keys(connectionLabels))payload[k].enabled=false;settingsBusy(true);$('model-feedback').textContent='正在保存并应用…';try{const result=await api('/settings',{method:'POST',body:payload});renderModelFields(result.settings);$('model-feedback').textContent=result.message;if(result.restart_requested){await waitForRestart(result.previous_instance);$('model-feedback').textContent='服务已自动重启，配置已生效。'+(result.needs_rebuild?'请在资料库重建索引。':'可以继续提问。');}await refresh();toast($('model-feedback').textContent);}catch(error){$('model-feedback').textContent=error.message;showError(error);}finally{settingsBusy(false);}}
$('open-model-settings').onclick=openModelSettings;$('settings-edit').onclick=openModelSettings;$('model-form').onsubmit=saveModelSettings;$('model-demo').onclick=()=>saveModelSettings(null,true);$('help-config').onclick=()=>{$('help-dialog').close();openModelSettings();};
async function initialize(){const integration=await api('/integrations');$('mcp-url').value=integration.mcp.url;renderModelFields(await api('/settings'));await refresh();await refreshSessions();let previous,seen=false;try{previous=localStorage.getItem('evidence-session');seen=localStorage.getItem('zhiju-welcome-1.2')==='seen';localStorage.setItem('zhiju-welcome-1.2','seen');}catch{}if(previous&&[...$('saved-session').options].some(o=>o.value===previous))await loadSession(previous);await loadEvaluations();if(new URLSearchParams(location.search).has('help'))openHelp();else if(!seen)$('welcome-dialog').showModal();}
initialize().catch(e=>{showError(e);$('mode-pill').textContent='连接失败';});

$('copy-mcp').onclick=async()=>{try{await navigator.clipboard.writeText($('mcp-url').value);toast('已复制 MCP 地址。');}catch{toast('请选中地址后复制。');}};
