# Tutorials

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"
.\.venv\Scripts\python examples\basic_run.py
```

## Building a custom search loop

```python
import numpy as np
from qes.permission import GenesisPermission
from qes.room import Room
from qes.space import QESSpace

seed = Room(
    x=np.zeros(2),
    x_star=np.zeros(2),
    lower=-np.ones(2),
    upper=np.ones(2),
    activation=np.ones(2),
)

space = QESSpace(
    permission_gate=GenesisPermission(theta=1.5),
    step_fn=lambda room, t, dt: room.x + 0.1 * (room.x_star - room.x),
)
space.add_room(seed)
space.step()
```

## Production runtime scheduling

```python
from qes.runtime import ProductionRuntime

runtime = ProductionRuntime(max_workers=2)
runtime.submit("job-a", lambda: {"score": 1}, priority=5)
runtime.submit("job-b", lambda: {"score": 2}, priority=8)
for job in runtime.run_all():
    print(job.status, job.result)
```

## Custom domain example

Use the same engine with a domain-specific interpretation of room state, such as:

- control gains
- design parameters
- scientific model coefficients
- resource allocation priorities

The framework only requires a bounded state vector, a step function, and an admissibility function.
