from datetime import datetime
from pathlib import Path
import json,subprocess,time

root=Path(__file__).resolve().parent
controller=Path('/proc/3854540')
samples=0
peaks={}
with (root/'samples.jsonl').open('x') as stream:
    while controller.exists():
        stat=(controller/'stat').read_text().split()
        assert stat[21]=='29297570'
        assert 'tools/queue_native_research.py' in (controller/'cmdline').read_bytes().decode()
        output=subprocess.check_output(['nvidia-smi','--id=0,1',
            '--query-gpu=index,uuid,temperature.gpu,power.draw,power.limit,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True)
        devices=[]
        for line in output.splitlines():
            fields=[value.strip() for value in line.split(',')]
            index=int(fields[0])
            assert index in (0,1)
            device={'index':index,'uuid':fields[1],'temperature_c':float(fields[2]),
                'power_draw_w':float(fields[3]),'power_limit_w':float(fields[4]),
                'memory_used_mib':int(fields[5]),'utilization_pct':int(fields[6])}
            devices.append(device)
            old=peaks.get(str(index),{'temperature_c':0.0,'power_draw_w':0.0})
            peaks[str(index)]={'temperature_c':max(old['temperature_c'],device['temperature_c']),
                'power_draw_w':max(old['power_draw_w'],device['power_draw_w'])}
        assert sorted(device['index'] for device in devices)==[0,1]
        samples+=1
        row={'at':datetime.now().astimezone().isoformat(),'controller_state':stat[2],'devices':devices}
        stream.write(json.dumps(row)+'\n')
        stream.flush()
        state={'status':'OBSERVING','sample_count':samples,'interval_seconds':30,'last':row,'observed_peaks':peaks,
            'scope':'Read-only GPU0/1 telemetry; no hardware setting, signal, pause, restart or model change.'}
        (root/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
        time.sleep(30)
    state.update(status='CONTROLLER_HANDLE_ENDED',ended_at=datetime.now().astimezone().isoformat())
    (root/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
