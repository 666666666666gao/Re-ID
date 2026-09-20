"""Retain audit tool requests/results only; exclude private workspace initialization."""
from pathlib import Path
import json,datetime
root=Path(__file__).parent
source=Path('C:/Users/gb/.codex/sessions/2026/09/21/rollout-2026-09-21T06-14-00-01a0c0e2-1f03-7e23-b627-9bb872213eec.jsonl')
events=[json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
requests={};pairs=[]
for n,event in enumerate(events,1):
    if event.get('type')!='response_item':continue
    p=event['payload'];kind=p.get('type')
    if kind in ('function_call','custom_tool_call'):
        value=p.get('input',p.get('arguments',''))
        if not isinstance(value,str):value=json.dumps(value,ensure_ascii=False)
        if any(x in value for x in ('SOUL.md','USER.md','.codex/memories','.codex\\\\memories')):continue
        if not any(x in value for x in ('trifusion_supported_balance_m0_independent_audit_20260921','.trifusion_github_publish_22c3bee','audit_local.py')):continue
        requests[p.get('call_id')]=(n,event)
    elif kind in ('function_call_output','custom_tool_call_output') and p.get('call_id') in requests:
        first,request=requests[p['call_id']]
        pairs.append(dict(request_source_line=first,result_source_line=n,request=request,result=event))
result=dict(scope='Exact reviewer audit tool request/result pairs through capture time; no reasoning text or private workspace initialization. Dedicated remote .request.json/.stderr/.transport and read_outputs files preserve execution receipts separately.',
    source_session_id='01a0c0e2-1f03-7e23-b627-9bb872213eec',
    captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_line_count_at_capture=len(events),
    pair_count=len(pairs),pairs=pairs)
(root/'audit_tool_trace.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(tool_request_result_pairs=len(pairs),source_lines=len(events))))
