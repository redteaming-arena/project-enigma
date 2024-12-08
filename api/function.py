import numpy as np

def sigmoid(x : np.ndarray, k : float = 1, center : float = 0):
    """ sigmoid with a shift in center and depth """
    return 1 / (1 + np.exp(-k * (x - center)))

def softmax_1D(x : np.ndarray):
    """ project vector onto distribution function  
        
        e ^ x/ sum^i_{i=0}{e^{x_i}}
        f: R^{1xN} -> [0, 1]^{1xN}  
    """
    assert len(x.shape) == 1, f"x is not a 1d numpy vector: {x.shape}"
    return np.exp(x) / np.sum(np.exp(x))

