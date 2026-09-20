from pathlib import Path
import subprocess,sys,json,datetime
root=Path(__file__).parent
script=root/sys.argv[1]
output=root/sys.argv[2]
command=['C:/Users/gb/AppData/Local/Programs/ClawX/resources/bin/uv.exe','run','--offline','--with','numpy','--with','paramiko','python','-X','utf8','C:/Users/gb/.codex_tmp/role_set_remote_command_20260908.py',str(script),str(output)]
request=dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,script=str(script),stdout=str(output))
output.with_suffix(output.suffix+'.request.json').write_text(json.dumps(request,indent=2),encoding='utf-8')
with output.with_suffix(output.suffix+'.transport.txt').open('x',encoding='utf-8') as stream:
    result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,text=True)
print(json.dumps(dict(returncode=result.returncode,stdout=str(output),transport_log=str(output.with_suffix(output.suffix+'.transport.txt')),stdout_exists=output.exists())))
raise SystemExit(result.returncode)
