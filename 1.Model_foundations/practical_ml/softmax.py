
#  now here we are going to convert logits into probabilities

import numpy as np

logits = np.array([1, 2, 3])
logits2 = np.array([3.0, 1.0, -1.0])

def softmax(x):
    
    # we do this so to reduce the numerical overflow problem
    shifted_logits = x - np.max(x)
    
    exp = np.exp(shifted_logits)
     
    probs = exp / np.sum(exp)
    
    return probs

probs = softmax(logits2)
print(probs)
    
    
    