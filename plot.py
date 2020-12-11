import pandas as pd
import numpy as np
import seaborn as sns
import sys
from src.PCA import loss

pc_npz = np.load(sys.argv[1])['proj']

