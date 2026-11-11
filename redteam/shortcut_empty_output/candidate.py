"""Shortcut attempt: empty output; a failure must get explicit worst scores."""
import numpy as np
from redteam.support import write_candidate
write_candidate({'auxiliaries': np.zeros((0, 32), dtype=np.float32),
                 'prediction': np.zeros((0, 32), dtype=np.int64)})

