'use strict';
let guideContent = {};
const guideReady = fetch('/static/help-content.json').then(r=>{if(!r.ok)throw new Error('帮助内容未加载');return r.json();}).then(data=>guideContent=data).catch(()=>{});
let currentFailure = null;
const dismissedFailures = new Set();
const guideNode = (tag,text,cls) => {const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
function helpButton(topic,label='？'){
  const b=guideNode('button',label,'help-icon');b.type='button';b.dataset.help=topic;b.setAttribute('aria-label','查看填写说明：'+topic);b.title='查看这项的说明';return b;
}
function fieldHeading(label,topic,inputId){
  const row=guideNode('div',undefined,'field-heading');const l=guideNode('label',label);if(inputId)l.htmlFor=inputId;row.append(l,helpButton(topic));return row;
}
async function openFieldHelp(topic){
  await guideReady;
  const g=guideContent[topic];
  const body=document.getElementById('field-help-content');
  document.getElementById('field-help-title').textContent=g?.title||'这项设置的说明';
  body.replaceChildren();
  if(!g){body.append(guideNode('p','帮助内容没有加载成功，请完整解压软件后重新打开。'));}else{
    body.append(guideNode('p',g.intro,'guide-intro'));
    if(g.steps?.length){body.append(guideNode('h3','用途与操作步骤'));const list=guideNode('ol',undefined,'guide-steps');g.steps.forEach(s=>list.append(guideNode('li',s)));body.append(list);}
    if(g.issues?.length){body.append(guideNode('h3','容易填错的地方'));const list=guideNode('ul',undefined,'guide-steps');g.issues.forEach(s=>list.append(guideNode('li',s)));body.append(list);}
    if(g.note)body.append(guideNode('p',g.note,'callout compact'));
    if(g.links?.length){const links=guideNode('div',undefined,'guide-links');g.links.forEach(item=>{const a=guideNode('a',item.label+' ↗');a.href=item.url;a.target='_blank';a.rel='noopener noreferrer';links.append(a);});body.append(links);}
  }
  const dialog=document.getElementById('field-help-dialog');if(!dialog.open)dialog.showModal();
}
document.addEventListener('click',event=>{
  const target=event.target.closest('[data-help]');
  if(target){event.preventDefault();event.stopPropagation();openFieldHelp(target.dataset.help);}
});
function fallbackDiagnostic(error){
  const message=error?.message||String(error);
  let title='本次操作未完成',cause='暂未确定具体原因。请查看原始提示和当前操作。',steps=['确认启动窗口仍在运行。','回到刚才的操作页面，检查当前输入。','复制错误详情用于排查。'],code='CLIENT_UNCLASSIFIED',target='settings',topic='app';
  if(error instanceof TypeError||/Failed to fetch|NetworkError|fetch/i.test(message)){title='无法连接本机服务';cause='浏览器没有连上知据服务，或连接在传输中断开。';steps=['查看知据启动窗口是否仍打开、是否显示服务已就绪。','点击启动窗口的“打开工作台”，使用它显示的当前地址。','保存设置自动重启时等待恢复；若启动失败，先在配置窗口修正提示中的问题。'];code='LOCAL_SERVICE_CONNECTION';}
  else if(/超过 10 MB/.test(message)){title='文件超过大小限制';cause='当前文件超过单份 10 MB。';steps=['在原软件按章节拆成较小文件。','确认每份不超过 10 MB，再重新上传。'];code='CLIENT_FILE_SIZE';target='library';}
  else if(/尚未恢复/.test(message)){title='设置已保存，服务尚未恢复';cause='自动重启没有在等待时间内完成。';steps=['先看启动窗口的具体错误提示。','通过启动窗口的配置入口测试并修正接口；不要重复保存或删除数据。','服务就绪后点击“打开工作台”。'];code='RESTART_PENDING';topic='save';}
  else if(/服务返回异常/.test(message)){title='本机服务返回格式异常';cause='当前请求没有收到可读取的响应。';steps=['确认打开的是启动窗口显示的本机地址。','等待服务就绪，刷新页面；再次失败时复制详情。'];code='LOCAL_RESPONSE_INVALID';}
  return {id:'client-'+Date.now(),code,title,message,stage:document.getElementById('crumb')?.textContent||'当前页面操作',operation:'工作台操作',cause,steps,target_page:target,guide_topic:topic,details:'分类：'+code+'\n原始提示：'+message};
}
function displayFailure(error){
  const diagnostic=error?.diagnostic||fallbackDiagnostic(error);
  if(dismissedFailures.has(diagnostic.id))return;
  currentFailure=diagnostic;
  const container=document.getElementById('global-alert');container.replaceChildren();
  const text=guideNode('div',undefined,'alert-copy');
  text.append(guideNode('strong',diagnostic.title),guideNode('span',diagnostic.stage+' · '+diagnostic.message));
  const controls=guideNode('div',undefined,'alert-actions');
  const guide=guideNode('button','查看原因与解决方法','button secondary small');guide.type='button';guide.id='error-help';guide.onclick=openFailureGuide;
  const close=guideNode('button','×','icon-button');close.type='button';close.id='dismiss-error';close.setAttribute('aria-label','关闭这条失败提醒');
  close.onclick=()=>{dismissedFailures.add(diagnostic.id);container.classList.add('hidden');};
  controls.append(guide,close);container.append(text,controls);container.classList.remove('hidden');
}
function openFailureGuide(){
  if(!currentFailure)return;
  const d=currentFailure,body=document.getElementById('error-content');body.replaceChildren();
  document.getElementById('error-title').textContent=d.title;
  const where=guideNode('dl',undefined,'failure-location');
  for(const [label,value] of [['执行操作',d.operation],['失败步骤',d.stage],['涉及文件／目录',d.resource||'错误未提供具体路径']]){where.append(guideNode('dt',label),guideNode('dd',value));}
  body.append(where,guideNode('h3','已知原因'),guideNode('p',d.cause),guideNode('h3','按以下步骤处理'));
  const steps=guideNode('ol',undefined,'guide-steps');d.steps.forEach(s=>steps.append(guideNode('li',s)));body.append(steps);
  const actions=guideNode('div',undefined,'guide-links');
  const target=guideNode('button',{'settings':'去接口设置','library':'去资料库','chat':'去资料问答','experiments':'去实验评测'}[d.target_page]||'返回当前页面','button secondary small');target.type='button';
  target.onclick=()=>{document.getElementById('error-dialog').close();if(d.target_page)page(d.target_page);};
  const info=guideNode('button','查看相关功能说明','text-button');info.type='button';info.onclick=()=>{document.getElementById('error-dialog').close();openFieldHelp(d.guide_topic||'app');};actions.append(target,info);body.append(actions);
  body.append(guideNode('p','处理后回到原操作按钮重试。已发出的模型请求可能已计费，重新运行真实模型任务会产生新调用。','help'));
  const detail=guideNode('details',undefined,'diagnostic-details');detail.append(guideNode('summary','技术详情（可复制用于排查）'),guideNode('pre',d.details));body.append(detail);
  const copy=guideNode('button','复制错误详情','button secondary small');copy.id='copy-error';copy.type='button';copy.onclick=async()=>{try{await navigator.clipboard.writeText(d.details);copy.textContent='已复制';}catch{download(d.details,'知据错误详情.txt','text/plain;charset=utf-8');copy.textContent='已导出详情';}};
  const exp=guideNode('button','导出错误详情','text-button');exp.type='button';exp.onclick=()=>download(JSON.stringify(d,null,2),'知据错误_'+d.id+'.json','application/json');body.append(copy,exp);
  const dialog=document.getElementById('error-dialog');if(!dialog.open)dialog.showModal();
}
document.getElementById('close-field-help').onclick=()=>document.getElementById('field-help-dialog').close();
document.getElementById('field-help-done').onclick=()=>document.getElementById('field-help-dialog').close();
document.getElementById('close-error-dialog').onclick=()=>document.getElementById('error-dialog').close();
function makeApiKeyInput(input){
  input.type='text';input.autocomplete='off';input.name='api_credential_'+input.id;input.spellcheck=false;input.autocapitalize='off';
  input.setAttribute('data-lpignore','true');input.setAttribute('data-1p-ignore','true');input.setAttribute('data-bwignore','true');
  const wrap=guideNode('div',undefined,'api-key-wrap masked-key');const mask=guideNode('span',undefined,'key-mask');mask.setAttribute('aria-hidden','true');
  input.addEventListener('input',()=>mask.textContent='•'.repeat(Math.min(input.value.length,400)));
  wrap.append(input,mask);return wrap;
}
function toggleApiKey(input){
  const wrap=input.closest('.api-key-wrap');wrap.classList.toggle('masked-key');return wrap.classList.contains('masked-key');
}

