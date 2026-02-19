from .adaround_ptq import AdaroundPTQ
from .noisyquant_ptq import NoisyQuantPTQ
from .ditas_ptq import DiTASPTQ
from .hyq_ptq import HyQPTQ
from .brecq_ptq import BRECQPTQ
from .repqvit_ptq import RepQViTPTQ
from .smoothquant_ptq import SmoothQuantPTQ
from .bc_ptq import BCPTQ


PTQ_METHODS = {
    "adaround": AdaroundPTQ,
    "noisyquant": NoisyQuantPTQ,
    "ditas": DiTASPTQ,
    "hyq": HyQPTQ,
    "brecq": BRECQPTQ,
    "repqvit": RepQViTPTQ,
    "smoothquant": SmoothQuantPTQ,
    "bc": BCPTQ,
}

def init_ptq(model, method, **kwargs):
    cls = PTQ_METHODS[method]
    if cls is None:
        raise ValueError(f"Unknown PTQ method: {method}")
    ptq = cls(model, **kwargs)
    return ptq.run()