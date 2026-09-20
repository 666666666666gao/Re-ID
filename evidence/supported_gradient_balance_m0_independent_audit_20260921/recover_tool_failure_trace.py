"""Recover exact missing failed tool calls from this reviewer's own transcript only."""
from pathlib import Path
import json, hashlib
root=Path(__file__).parent
source=Path('C:/Users/gb/.codex/sessions/2026/09/21/rollout-2026-09-21T06-14-00-01a0c0e2-1f03-7e23-b627-9bb872213eec.jsonl')
events=[json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
calls={}
selected=[]
for lineno,event in enumerate(events,1):
    if lineno>310: break  # End of the original failed-attempt interval; excludes later recovery chatter.
    if event.get('type')!='response_item': continue
    p=event['payload'];kind=p.get('type')
    if kind in ('function_call','custom_tool_call'):
        calls[p.get('call_id')]=(lineno,event)
    if kind not in ('function_call_output','custom_tool_call_output'): continue
    output=p.get('output','')
    text=output if isinstance(output,str) else json.dumps(output,ensure_ascii=False)
    if not any(k in text for k in ("KeyError: 'local_relative_path'","SyntaxError: '(' was never closed",'FileNotFoundError','IO error for operation')): continue
    call=calls.get(p.get('call_id'))
    if call is None: continue
    # Do not collect unrelated file-not-found output; these are the two documented misses.
    if ('FileNotFoundError' in text or 'IO error for operation' in text) and not any(x in text for x in ('signal_preserving_v8.py','build_msvr310_signal_oof_protocol.py')): continue
    selected.append(dict(request_source_line=call[0],output_source_line=lineno,request=call[1],output=event))
dest=root/'recovered_failure_tool_trace.json'
result=dict(source_transcript_basename=source.name,source_agent_session_id='01a0c0e2-1f03-7e23-b627-9bb872213eec',
    selection='Exact request/output pairs for AF3, AF4 and AF5; unmodified transcript event objects. No unrelated conversation, system instructions, secret session helper content or personal context retained.',
    pairs=selected)
dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(selected_pairs=len(selected),output=dest.name,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
    source_lines=[[x['request_source_line'],x['output_source_line']] for x in selected]),indent=2))
