import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from datetime import datetime
import json
import torch
torch.set_num_threads(2)
torch.manual_seed(42)
x = torch.randn(2, 4)
state = torch.nn.Linear(4, 4).state_dict()
rows = []
for teacher_cache in (True, False):
    model = torch.nn.Linear(4, 4)
    model.load_state_dict(state)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        with torch.no_grad(), torch.autocast('cpu', cache_enabled=teacher_cache):
            teacher = model(x)
        student = model(x)
    if student.requires_grad:
        student.float().sum().backward()
    rows.append({'teacher_cache_enabled': teacher_cache,
                 'teacher_dtype': str(teacher.dtype),
                 'student_dtype': str(student.dtype),
                 'teacher_requires_grad': teacher.requires_grad,
                 'student_requires_grad': student.requires_grad,
                 'weight_gradient_present': model.weight.grad is not None,
                 'bias_gradient_present': model.bias.grad is not None,
                 'weight_gradient_finite': bool(torch.isfinite(model.weight.grad).all()) if model.weight.grad is not None else None,
                 'teacher_student_forward_equal': torch.equal(teacher, student)})
print(json.dumps({'at': datetime.now().astimezone().isoformat(),
                  'torch_version': torch.__version__, 'device': 'cpu',
                  'status': 'FIXED_TWO_CASE_COMPONENT_MEASUREMENT',
                  'production_NN': 0, 'component_forwards': 4,
                  'component_backwards': sum(row['student_requires_grad'] for row in rows),
                  'optimizer_updates': 0, 'rows': rows,
                  'boundary': 'Existing Torch CPU bfloat16 Linear cache mechanism only; no CUDA execution, production model, M0, training, GPU/temperature/power/25 query or install.'}))
