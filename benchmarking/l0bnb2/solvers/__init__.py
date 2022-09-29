import warnings

from numba.core.errors import NumbaDeprecationWarning, \
    NumbaPendingDeprecationWarning, NumbaPerformanceWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

from .utils import get_ratio_threshold, compute_relative_gap