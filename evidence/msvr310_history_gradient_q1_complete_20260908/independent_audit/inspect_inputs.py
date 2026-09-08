import hashlib, json, sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
REGISTRY = OUT / 'audited_input_hashes.json'

def record(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    rows = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    rows[str(path)] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    REGISTRY.write_text(json.dumps(rows, indent=2)+'\n', encoding='utf-8')
    return data

def shape(obj, depth=1):
    if depth < 0:
        return ('dict '+str(list(obj))) if isinstance(obj,dict) else ('list '+str(len(obj))) if isinstance(obj,list) else obj
    if isinstance(obj,dict): return {k:shape(v,depth-1) for k,v in obj.items()}
    if isinstance(obj,list): return {'length':len(obj), 'first':shape(obj[0],depth-1) if obj else None}
    return obj

if __name__ == '__main__':
    mode, name, *rest = sys.argv[1:]
    path = Path(name)
    if not path.is_absolute(): path = REPO / path
    data = record(path)
    if mode == 'read':
        lines = data.decode('utf-8-sig').splitlines()
        start,end = map(int,rest) if rest else (1,len(lines))
        print(str(path), 'total_lines='+str(len(lines)))
        for i,line in enumerate(lines[start-1:end],start): print(f'{i}: {line}')
    elif mode == 'shape': print(json.dumps(shape(json.loads(data),int(rest[0]) if rest else 1),indent=2))
    elif mode == 'row':
        obj = json.loads(data.splitlines()[int(rest[0])-1])
        print(json.dumps(obj,indent=2))
    elif mode == 'intake':
        m = json.loads(data)
        checks=[]
        for r in m['files']:
            p=path.parent/r['path']; b=record(p)
            checks.append(dict(path=r['path'],bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),
                               match=len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']))
        result={'files':checks,'file_count':len(checks),'total_bytes':sum(x['bytes'] for x in checks),'all_match':all(x['match'] for x in checks)}
        (OUT/'intake_check.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='files'}))
