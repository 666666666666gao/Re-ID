from pathlib import Path
from datetime import datetime
from collections import Counter
import hashlib
import json

packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/geometry_reference895')
assert not packet.exists()
packet.mkdir()

def area(rect):
    y0,x0,y1,x1=rect
    return max(y1-y0,0)*max(x1-x0,0)

def intersect(a,b):
    return max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])

results=[]
for h,w in ((256,128),(128,256)):
    gh,gw=h//16,w//16
    support=(10,10,h-10,w-10)
    for flipped in (False,True):
        for dy,dx in ((0,0),(-10,10),(10,-10),(-3,-7),(7,3)):
            for erase in ((0,0,0,0),(h//3,w//4,h//3+38,w//4+31)):
                # Independent pixel enumeration of exact original->second coordinates.
                pixels=Counter()
                for y in range(10,h-10):
                    for x in range(10,w-10):
                        if erase[0]<=y<erase[2] and erase[1]<=x<erase[3]:
                            continue
                        y2=y-dy
                        x2=(w-1-x if flipped else x)-dx
                        if 0<=y2<h and 0<=x2<w:
                            p=(y//16)*gw+x//16
                            q=(y2//16)*gw+x2//16
                            pixels[p,q]+=1
                # Independent rectangle intersections, with horizontal flip using half-open bounds.
                rectangles={}
                for py in range(gh):
                    for px in range(gw):
                        student=(py*16,px*16,(py+1)*16,(px+1)*16)
                        for qy in range(gh):
                            for qx in range(gw):
                                teacher=(qy*16+dy,qx*16+dx,(qy+1)*16+dy,(qx+1)*16+dx)
                                if flipped:
                                    teacher=(teacher[0],w-teacher[3],teacher[2],w-teacher[1])
                                overlap=intersect(intersect(student,teacher),support)
                                count=area(overlap)-area(intersect(overlap,erase))
                                if count:
                                    rectangles[py*gw+px,qy*gw+qx]=count
                assert dict(pixels)==rectangles
                row_counts=Counter()
                column_counts=Counter()
                for (p,q),count in pixels.items():
                    row_counts[p]+=count
                    column_counts[q]+=count
                assert all(0<c<=256 for c in row_counts.values())
                results.append({'height':h,'width':w,'flip':flipped,'dy':dy,'dx':dx,'erase':erase,'supported_rows':len(row_counts),'supported_columns':len(column_counts),'overlap_pixels':sum(pixels.values()),'mass_sha256':hashlib.sha256(json.dumps(sorted((p,q,c) for (p,q),c in pixels.items())).encode()).hexdigest()})
report={'at':datetime.now().astimezone().isoformat(),'status':'EXACT_PIXEL_RECTANGLE_AGREEMENT','cases':len(results),'rows':results,'new_NN':0,'optimizer_updates':0,'boundary':'Pure CPU geometric reference only. Does not validate a production Torch implementation, actual loader/RNG, model M0, contextual token locality, cross-spectrum semantic correspondence, or retrieval gains. Conservative10px border avoids documented author pad/crop artificial boundary; also discards some real content.'}
(packet/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
