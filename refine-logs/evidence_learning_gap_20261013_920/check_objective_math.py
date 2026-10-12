"""Small real-arithmetic checks, not model inference or training."""
import hashlib
import json
import math
from pathlib import Path

root=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
out=Path(__file__).resolve().parent
s=math.sqrt(128)
sigmoid=lambda x:1/(1+math.exp(-x))
softplus=lambda x:math.log1p(math.exp(x))
def loss(g,r,w):
 return sum(a*softplus(-s*.5*(b+c)) for a,b,c in zip(w,g,r))/sum(w)
cases=[]
for g,r in (([.8,.8,.8],[.1,.2,.3]),([-.2,.1,.6],[.15,-.05,.2])):
 w=[sigmoid(-s*v) for v in g]
 uniform=loss(g,r,[1]*len(g))
 ambiguity=loss(g,r,w)
 scaled=loss(g,r,[a*.001 for a in w])
 assert abs(ambiguity-scaled)<1e-13
 if len(set(g))==1:assert abs(uniform-ambiguity)<1e-13
 errors=[]
 for i in range(len(r)):
  analytic=-s*.5*w[i]/sum(w)*sigmoid(-s*.5*(g[i]+r[i]))
  a,b=list(r),list(r);a[i]+=1e-5;b[i]-=1e-5
  numeric=(loss(g,a,w)-loss(g,b,w))/(2e-5)
  assert analytic<0 and abs(numeric-analytic)<1e-8
  errors.append(abs(numeric-analytic))
 cases.append(dict(global_margins=g,regional_margins=r,uniform_loss=uniform,ambiguity_loss=ambiguity,
  common_weight_scale_difference=abs(ambiguity-scaled),max_gradient_check_error=max(errors)))
assert sigmoid(-s*2)>1e-12
result=dict(status='SOURCE_FORMULA_REAL_ARITHMETIC_CHECKS_PASSED',source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
 for name in ('modeling/trifusion/region_use_objective.py','tools/run_region_use_probe.py')},
 cases=cases,minimum_ideal_single_relation_weight=sigmoid(-s*2),neural_calls=0,optimizer_updates=0,
 boundary='Two scalar examples and derivative/common-weight-invariance checks. No current feature or CUDA test, no observed weight distribution, model result, full-objective optimum, or failure cause.')
(out/'MATH_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
