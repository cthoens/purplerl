import numpy as np

def reload():
    import sys
    import importlib
    importlib.reload(sys.modules[__name__])


def unity_run(unity_env, dev_mode: bool = True, resume_lesson: int = None, resume_checkpoint: str = None):
    for i in range(10):
        unity_env.Reset()
        act = np.ones(5, dtype=np.float32);
        unity_env.Step(act);