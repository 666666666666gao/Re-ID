from pathlib import Path
from datetime import datetime
import hashlib,json,shlex,subprocess,urllib.request,paramiko

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
proof=Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
publication=json.loads((proof/'publication873_local.json').read_bytes())
previous=json.loads((proof/'four_copy872_2025_pending.json').read_bytes())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==publication['previous_head']==previous['head']
assert subprocess.check_output(['git','diff','--cached','--name-only'],cwd=repo)==b''
assert not (proof/'commit873.json').exists() and not (proof/'target873.bundle').exists()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(repo/n)==d for n,d in publication['protected_files'].items())
def script(c,code):
    i,o,e=c.exec_command('/usr/bin/python3 -B -');i.write(code);i.channel.shutdown_write();o.channel.settimeout(300)
    data,error=o.read(),e.read(); assert o.channel.recv_exit_status()==0,error.decode();return data
def run(c,args):
    _,o,e=c.exec_command(shlex.join(args));o.channel.settimeout(300)
    data,error=o.read(),e.read();assert o.channel.recv_exit_status()==0,error.decode();return data
root='/data/gaob/Re-ID/Trifusion'
c=paramiko.SSHClient();c.load_host_keys('C:/Users/gb/.ssh/known_hosts')
c.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
gate=r'''from pathlib import Path
import hashlib,json,sys
root=Path('/data/gaob/Re-ID/Trifusion');sys.path.insert(0,str(root))
from tools import queue_independent_role_heads as panel
for p in Path('/proc').iterdir():
    if p.name.isdigit() and (p/'cmdline').is_file():
        cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
        assert '/data/gaob/Re-ID/Trifusion/tools/' not in cmd,cmd
panel.source_map();panel.previous.require_controls()
for name in ('independent_role_head_storage_retirement_20261006_873','independent_role_head_unused_f2_retirement_20261006_873'):
    j=root/'logs'/name
    assert (j/'COMPLETE.json').is_file()
print('OWN_NN_TERMINAL_385_RAW187_STORAGE_RETIREMENTS_VERIFIED')
'''
script(c,gate)
assert run(c,['git','-C',root,'rev-parse','HEAD']).decode().strip()==previous['head']
changed=publication['files']
assert all((repo/n).is_file() for n in changed)
subprocess.run(['git','-c','core.autocrlf=false','add','-f','--pathspec-from-file=-','--pathspec-file-nul'],
               input=b''.join(n.encode()+b'\0' for n in changed),cwd=repo,check=True)
subprocess.run(['git','-c','core.autocrlf=false','add','--renormalize','--pathspec-from-file=-','--pathspec-file-nul'],
               input=b''.join(n.encode()+b'\0' for n in changed),cwd=repo,check=True)
staged=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=repo).decode().rstrip('\0').split('\0')
assert set(staged)==set(changed)
def blobs_exact(ref,names):
    for start in range(0,len(names),80):
        batch=names[start:start+80]
        data=subprocess.check_output(['git','cat-file','--batch'],cwd=repo,input=''.join(ref+n+'\n' for n in batch).encode());offset=0
        for n in batch:
            end=data.index(b'\n',offset);size=int(data[offset:end].split()[-1]);offset=end+1
            assert data[offset:offset+size]==(repo/n).read_bytes(),n;offset+=size+1
blobs_exact(':',changed)
subprocess.run(['git','-c','core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof','diff','--cached','--check','--','.',
                *[':(exclude)'+n+'/**' for n in publication['immutable_archive_roots']]],cwd=repo,check=True)
subprocess.run(['git','commit','--quiet','-m','Preserve failed storage gate and retire unused owned binaries before head-control launch'],cwd=repo,check=True)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
subprocess.run(['git','bundle','create',str(proof/'target873.bundle'),previous['head']+'..main'],cwd=repo,check=True)
subprocess.run(['git','push','origin','main'],cwd=repo,check=True)
(proof/'commit873.json').write_text(json.dumps(dict(head=head,files=changed,owned_staging_exact=True),indent=2)+'\n')
prior=next(r['verified_files'] for r in previous['servers'] if r['port']==2026)
files=sorted(set(prior)|set(changed));expected={n:sha(repo/n) for n in files}
assert all(expected[n]==d for n,d in prior.items() if n not in changed)
blobs_exact(head+':',changed)
doc='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
assert hashlib.sha256(urllib.request.urlopen('https://raw.githubusercontent.com/666666666666gao/Re-ID/'+head+'/'+doc,timeout=30).read()).hexdigest()==expected[doc]
run(c,['git','-C',root,'sparse-checkout','add',*publication['remote_sparse_roots']])
destination='/tmp/trifusion_independent_role_heads873_20261006.bundle'
script(c,f'from pathlib import Path;assert not Path({destination!r}).exists()')
s=c.open_sftp();s.put(str(proof/'target873.bundle'),destination);s.close()
script(c,f'from pathlib import Path;import hashlib;assert hashlib.sha256(Path({destination!r}).read_bytes()).hexdigest()=={sha(proof/"target873.bundle")!r}')
run(c,['git','-C',root,'fetch',destination,'main'])
run(c,['git','-C',root,'merge','--ff-only','FETCH_HEAD'])
assert run(c,['git','-C',root,'rev-parse','HEAD']).decode().strip()==head
before=json.loads(script(c,f'from pathlib import Path;import hashlib,json;root=Path({root!r});names={files!r};print(json.dumps({{n:hashlib.sha256((root/n).read_bytes()).hexdigest() if (root/n).is_file() else None for n in names}}))'))
assert all(before[n] in (None,prior.get(n),expected[n],publication['original_manifest_sha256'] if n=='MANIFEST.md' else None) for n in files)
missing=[n for n,d in expected.items() if before[n]!=d]
script(c,f'from pathlib import Path;root=Path({root!r});[(root/n).parent.mkdir(parents=True,exist_ok=True) for n in {missing!r}]')
s=c.open_sftp()
for n in missing:s.put(str(repo/n),root+'/'+n)
s.close()
verified=json.loads(script(c,f'from pathlib import Path;import hashlib,json;root=Path({root!r});expected={expected!r};actual={{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in expected}};assert actual==expected;print(json.dumps({{"port":2026,"verified_files":actual,"text_source_sync_only":True}}))'))
script(c,gate)
script(c,f'''from pathlib import Path
import hashlib,json
root=Path({root!r}); p=root/'refine-logs/independent_role_heads_v1/SOURCE_SCOPE.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()=={publication['source_scope_sha256']!r}
sources=json.loads(p.read_text())['source_sha256'];assert len(sources)=={publication['source_count']}
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==d for n,d in sources.items())
print('385_STUDY_SOURCES_EXACT')
''')
c.close()
assert sha(Path('C:/Users/gb/Desktop/document')/Path(doc).name)==expected[doc]
assert all(sha(repo/n)==d for n,d in publication['protected_files'].items())
record=dict(status='FOUR_COPIES_VERIFIED_2025_MIRROR_IO_PENDING',head=head,section='41.873',
    completed_at=datetime.now().astimezone().isoformat(),doc_sha256=expected[doc],doc_bytes=(repo/doc).stat().st_size,
    servers=[verified],owned_count=len(files),pending_mirror=previous['pending_mirror'],source_count=publication['source_count'],
    boundary='Original first prelaunch storage assertion failed before any NN/campaign. Additional three unused closed own binaries retired, explicit F2 R5 exception and failed N1 boundary preserved. 385 sources and RAW187 unchanged. Production initialization/M0/full still not run. Only26GPU0/1/no power/temp/25. GoalACTIVE_UNMET.')
(proof/'four_copy873_2025_pending.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='servers'},indent=2))
