from pathlib import Path
import subprocess,json,sys,os
os.environ["GIT_CONFIG_COUNT"]="1"
os.environ["GIT_CONFIG_KEY_0"]="core.longpaths"
os.environ["GIT_CONFIG_VALUE_0"]="true"
def read_local(path):return Path(chr(92)*2+"?"+chr(92)+str(path)).read_bytes()
p=Path('C:/Users/gb/.trifusion_github_publish_22c3bee');m=Path(sys.argv[1]);j=json.loads(m.read_bytes())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip()==j['old_head']
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=p).strip()
assert len(j['paths'])==len(set(j['paths']))
assert not any(x.startswith('.aris/') or x=='tools/run_trifusion_experiment.py' for x in j['paths'])
subprocess.run(['git','-c','core.autocrlf=false','add','-f','--pathspec-from-file=-','--pathspec-file-nul'],cwd=p,input=bytes([0]).join(x.encode() for x in j['paths'])+bytes([0]),check=True)
assert set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=p,text=True).splitlines())==set(j['paths'])
for x in j['paths']:assert subprocess.check_output(['git','show',':'+x],cwd=p)==read_local(p/x),x
subprocess.run(['git','-c','core.whitespace=cr-at-eol','diff','--cached','--check','--',*[x for x in j['paths'] if not x.startswith('evidence/')]],cwd=p,check=True)
subprocess.run(['git','commit','-m',sys.argv[2]],cwd=p,check=True,stdout=subprocess.DEVNULL)
j['new_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip()
m.write_text(json.dumps(j,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(head=j['new_head'],paths=len(j['paths']))))
