"""Continue only the remaining sync after exact original path collision."""
from pathlib import Path
import hashlib,json,shlex,subprocess,urllib.request
import paramiko
repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
proof=Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
publication=json.loads((proof/'publication824_local.json').read_bytes())
previous=json.loads((proof/'four_copy823_2025_pending.json').read_bytes())
committed=json.loads((proof/'commit824.json').read_bytes())
head=committed['head']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==head
assert not (proof/'four_copy824_2025_pending.json').exists()
files=set()
for number in range(739,825):
    files.update(json.loads((proof/f'publication{number}_local.json').read_bytes())['files'])
files=sorted(files)
blob=subprocess.check_output(['git','cat-file','--batch'],cwd=repo,input=''.join(head+':'+name+'\n' for name in files).encode())
offset=0;expected={}
for name in files:
    end=blob.index(b'\n',offset);size=int(blob[offset:end].split()[-1]);offset=end+1
    data=blob[offset:offset+size];assert data==(repo/name).read_bytes(),name
    expected[name]=hashlib.sha256(data).hexdigest();offset+=size+1
doc='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
assert hashlib.sha256(urllib.request.urlopen('https://raw.githubusercontent.com/666666666666gao/Re-ID/'+head+'/'+doc,timeout=30).read()).hexdigest()==expected[doc]

def script(client,code):
    stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
    stdin.write(code);stdin.channel.shutdown_write();stdout.channel.settimeout(60)
    data,error=stdout.read(),stderr.read()
    assert stdout.channel.recv_exit_status()==0,error.decode()
    return data

def run(client,command):
    _,stdout,stderr=client.exec_command(command)
    data,error=stdout.read(),stderr.read()
    assert stdout.channel.recv_exit_status()==0,error.decode()
    return data

rows=[]
for port,root in ((2026,'/data/gaob/Re-ID/Trifusion'),):
    client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
    client.connect('172.19.12.138',port=port,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
    if port==2026:
        assert run(client,shlex.join(['git','-C',root,'rev-parse','HEAD'])).decode().strip()==publication['previous_head']
        run(client,shlex.join(['git','-C',root,'sparse-checkout','add','logs/global_task_role_preparation824_20261004','refine-logs/global_task_role_v1']))
        bundle=proof/'target824.bundle';destination='/tmp/trifusion_target824_20261004.bundle'
        script(client,f"from pathlib import Path;assert not Path({destination!r}).exists()")
        sftp=client.open_sftp();sftp.put(str(bundle),destination);sftp.close()
        script(client,f"from pathlib import Path;import hashlib;assert hashlib.sha256(Path({destination!r}).read_bytes()).hexdigest()=={hashlib.sha256(bundle.read_bytes()).hexdigest()!r}")
        run(client,shlex.join(['git','-C',root,'fetch',destination,'main']))
        run(client,shlex.join(['git','-C',root,'merge','--ff-only','FETCH_HEAD']))
        assert run(client,shlex.join(['git','-C',root,'rev-parse','HEAD'])).decode().strip()==head
    before=json.loads(script(client,f"from pathlib import Path;import hashlib,json;root=Path({root!r});print(json.dumps({{name:hashlib.sha256((root/name).read_bytes()).hexdigest() if (root/name).is_file() else None for name in {files!r}}}))"))
    prior=[row['verified_files'] for row in previous['servers'] if row['port']==port];assert len(prior)==1;prior=prior[0]
    assert all(before[name] in (None,prior.get(name),digest) for name,digest in expected.items())
    changed_here=[name for name,digest in expected.items() if before[name]!=digest]
    script(client,f"from pathlib import Path;root=Path({root!r});[(root/name).parent.mkdir(parents=True,exist_ok=True) for name in {changed_here!r}]")
    sftp=client.open_sftp()
    for name in changed_here:
        if before[name]!=expected[name]:
            sftp.put(str(repo/name),root+'/'+name)
    sftp.close()
    verification=json.loads(script(client,f"from pathlib import Path;import hashlib,json;root=Path({root!r});expected={expected!r};actual={{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in expected}};assert actual==expected;print(json.dumps({{'port':{port},'verified_files':actual,'text_sync_only':True}}))"))
    rows.append(verification);client.close()
assert hashlib.sha256((Path('C:/Users/gb/Desktop/document')/Path(doc).name).read_bytes()).hexdigest()==expected[doc]
record={'status':'FOUR_COPIES_VERIFIED_2025_MIRROR_IO_PENDING','head':head,'section':'41.824','completed_at':__import__('datetime').datetime.now().astimezone().isoformat(),
        'doc_sha256':expected[doc],'doc_bytes':(repo/doc).stat().st_size,'servers':rows,'owned_count':len(files),
        'pending_mirror':previous['pending_mirror'],'boundary':'Global versus fused author task ownership registered;CPU checks only,330 scientific sources pinned,no production M0/results yet. Four execution copies verified;2025 I/O mirror pending. Only26GPU0/1,no power/temp actions. Goal active/unmet.'}
(proof/'four_copy824_2025_pending.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:value for key,value in record.items() if key!='servers'},indent=2))
