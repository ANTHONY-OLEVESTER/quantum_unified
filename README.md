# quantum-unified

Core utilities for the curvature–information invariant

Y = sqrt(d_eff - 1) * A^2 / I

- `A` = Bures (Uhlmann) angle
- `I` = mutual information (bits)
- `d_eff` = effective dimension = 1 / Tr(rho^2)

Install (from PyPI, when published)
- `pip install quantum-unified`

Install (from source)
- `pip install -r requirements.txt` (if present)
- `pip install .`

Usage
```python
import numpy as np
from quantum_unified import bures_angle, effective_dimension, compute_Y

rho0 = np.array([[1,0],[0,0]], dtype=complex)
rho1 = np.array([[0.9, 0.1],[0.1, 0.1]], dtype=complex)
A = bures_angle(rho0, rho1)
de = effective_dimension(rho1)
Y = compute_Y(rho0, rho1, Ibits=0.5)
print(A, de, Y)
```

Author
- Anthony Olevester (olevester.joram123@gmail.com)

License
- Add your preferred license here (e.g., MIT) and include a LICENSE file.
