"""Publish the concrete accepted closeout only after the original campaign exits."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import urllib.request

import paramiko

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
proof=Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
publication=json.loads((proof/'publication857_local.json').read_bytes())
previous=json.loads((proof/'four_copy856_2025_pending.json').read_bytes())
intake=json.loads((base/'semantic_capacity_complete_intake856/stdout.json').read_bytes())
analysis=json.loads((base/'semantic_capacity_text_analysis857/SUMMARY.json').read_bytes())
assert intake['status']=='CAPACITY_COMPLETE_WITH_FIRST_EVAL_CONTINUATION_VERIFIED'
assert intake['formal_completed']==3 and intake['report_invocations']==1 and intake['completion_launch_exit']['exit_code']==0
assert intake['original_launch_exit']['exit_code']==1 and intake['new_training_invocations']==0
assert analysis['status']=='ACCEPTED_REPORT_TEXT_VISUALIZATION_COMPLETE' and analysis['curve_rows']==600
assert publication['previous_head']==previous['head']=='60e1c4dec1f21a772cde7c703cb29ad2b5ee61bd'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==publication['previous_head']
assert subprocess.check_output(['git','diff','--cached','--name-only'],cwd=repo)==b''
assert not (proof/'commit857.json').exists() and not (proof/'target857.bundle').exists()
assert all(hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest for name,digest in publication['protected_files'].items())

def script(client,code):
    stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
    stdin.write(code);stdin.channel.shutdown_write();stdout.channel.settimeout(60)
    data,error=stdout.read(),stderr.read()
    assert stdout.channel.recv_exit_status()==0,error.decode()
    return data

def run(client,args):
    _,stdout,stderr=client.exec_command(shlex.join(args))
    data,error=stdout.read(),stderr.read()
    assert stdout.channel.recv_exit_status()==0,error.decode()
    return data

root='/data/gaob/Re-ID/Trifusion'
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
gate='''from pathlib import Path
import hashlib,json
root=Path('/data/gaob/Re-ID/Trifusion')
completion=root/'logs/semantic_capacity_first_eval_completion_20261005_857'
state=json.loads((completion/'campaign.json').read_text())
exit_record=json.loads((root/'logs/semantic_capacity_first_eval_launch_20261005_857/EXIT.json').read_text())
assert state['status']=='COMPLETE' and state['report_invocations']==1 and state['report_exit_code']==0
assert exit_record['exit_code']==0 and state['new_training_invocations']==0
plan=json.loads((completion/'PLAN.json').read_text())
assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==d for n,d in plan['original_failure_sha256'].items())
assert json.loads((root/'logs/semantic_capacity_launch_20261005_856/EXIT.json').read_text())['exit_code']==1
assert not Path('/proc/642951').exists() and not Path('/proc/642952').exists() and not Path('/proc/1171133').exists()
print('ORIGINAL_TRAININGS_AND_FIRST_EVAL_CONTINUATION_TERMINAL_FAILURE_UNCHANGED')
'''
script(client,gate)
assert run(client,['git','-C',root,'rev-parse','HEAD']).decode().strip()==publication['previous_head']
changed=publication['files']
subprocess.run(['git','-c','core.autocrlf=false','add','-f','--pathspec-from-file=-','--pathspec-file-nul'],
               input=b''.join(n.encode()+b'\0' for n in changed),cwd=repo,check=True)
subprocess.run(['git','-c','core.autocrlf=false','add','--renormalize','--pathspec-from-file=-','--pathspec-file-nul'],
               input=b''.join(n.encode()+b'\0' for n in changed),cwd=repo,check=True)
staged=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=repo).decode().rstrip('\0').split('\0')
assert set(staged)==set(changed)

def blobs_exact(ref,names):
    data=subprocess.check_output(['git','cat-file','--batch'],cwd=repo,input=''.join(ref+n+'\n' for n in names).encode())
    offset=0
    for name in names:
        end=data.index(b'\n',offset);size=int(data[offset:end].split()[-1]);offset=end+1
        assert data[offset:offset+size]==(repo/name).read_bytes(),name
        offset+=size+1

blobs_exact(':',changed)
subprocess.run(['git','-c','core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof','diff','--cached','--check','--','.',
                *[':(exclude)'+n+'/**' for n in publication['immutable_archive_roots']]],cwd=repo,check=True)
subprocess.run(['git','commit','--quiet','-m','Close semantic near-capacity controls with all-query evidence and primary-source notes'],cwd=repo,check=True)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
subprocess.run(['git','bundle','create',str(proof/'target857.bundle'),publication['previous_head']+'..main'],cwd=repo,check=True)
subprocess.run(['git','push','origin','main'],cwd=repo,check=True)
(proof/'commit857.json').write_bytes((json.dumps(dict(head=head,files=changed,staged_owned_bytes_exact=True),indent=2)+'\n').encode())
files=set()
for number in range(739,858):
    files.update(json.loads((proof/f'publication{number}_local.json').read_bytes())['files'])
files=sorted(files);blobs_exact(head+':',files)
expected={n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in files}
doc='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
assert hashlib.sha256(urllib.request.urlopen('https://raw.githubusercontent.com/666666666666gao/Re-ID/'+head+'/'+doc,timeout=30).read()).hexdigest()==expected[doc]
run(client,['git','-C',root,'sparse-checkout','add',*publication['remote_sparse_roots']])
bundle=proof/'target857.bundle';destination='/tmp/trifusion_capacity_complete857_20261005.bundle'
script(client,f'from pathlib import Path;assert not Path({destination!r}).exists()')
sftp=client.open_sftp();sftp.put(str(bundle),destination);sftp.close()
script(client,f'from pathlib import Path;import hashlib;assert hashlib.sha256(Path({destination!r}).read_bytes()).hexdigest()=={hashlib.sha256(bundle.read_bytes()).hexdigest()!r}')
run(client,['git','-C',root,'fetch',destination,'main'])
run(client,['git','-C',root,'merge','--ff-only','FETCH_HEAD'])
assert run(client,['git','-C',root,'rev-parse','HEAD']).decode().strip()==head
before=json.loads(script(client,f'from pathlib import Path;import hashlib,json;root=Path({root!r});print(json.dumps({{n:hashlib.sha256((root/n).read_bytes()).hexdigest() if (root/n).is_file() else None for n in {files!r}}}))'))
prior=next(r['verified_files'] for r in previous['servers'] if r['port']==2026)
assert all(before[n] in (None,prior.get(n),digest) for n,digest in expected.items())
changed_here=[n for n,digest in expected.items() if before[n]!=digest]
script(client,f'from pathlib import Path;root=Path({root!r});[(root/n).parent.mkdir(parents=True,exist_ok=True) for n in {changed_here!r}]')
sftp=client.open_sftp()
for name in changed_here:sftp.put(str(repo/name),root+'/'+name)
sftp.close()
verified=json.loads(script(client,f'from pathlib import Path;import hashlib,json;root=Path({root!r});expected={expected!r};actual={{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in expected}};assert actual==expected;print(json.dumps({{"port":2026,"verified_files":actual,"text_and_plot_sync_only":True}}))'))
script(client,gate);client.close()
assert hashlib.sha256((Path('C:/Users/gb/Desktop/document')/Path(doc).name).read_bytes()).hexdigest()==expected[doc]
assert all(hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest for name,digest in publication['protected_files'].items())
record=dict(status='FOUR_COPIES_VERIFIED_2025_MIRROR_IO_PENDING',head=head,section='41.857',
    completed_at=datetime.now().astimezone().isoformat(),doc_sha256=expected[doc],doc_bytes=(repo/doc).stat().st_size,
    servers=[verified],owned_count=len(files),pending_mirror=previous['pending_mirror'],
    boundary='Three original capacity trainings, first strict100/once-only report through explicit completion, and text/plot/source-note closeout. Original parent EXIT1/state/counters unchanged; new completion EXIT0. No NN training restart, weight reselection, historical failure reclassification or SOTA assertion. 2025 original I/O remains pending without probe; protected unrelated files unchanged; Goal ACTIVE/UNMET.')
(proof/'four_copy857_2025_pending.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in record.items() if k!='servers'},indent=2))
