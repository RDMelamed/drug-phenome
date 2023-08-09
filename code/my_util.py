import pandas as pd
import numpy as np

def upper_tri(dots):
    return pd.DataFrame(dots).where(np.triu(np.ones(dots.shape),1).astype(np.bool)).stack()
