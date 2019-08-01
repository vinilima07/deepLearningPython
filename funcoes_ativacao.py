import numpy as np
#transfer function

def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0

def sigmoidFunction(soma):
    return 1/(1+np.exp(-soma))
 
def tahnFunction(soma):
    return (np.exp(soma) - np.exp(-soma))/(np.exp(soma) + np.exp(-soma))

def reluFunction(soma):
    if soma >= 0:
        return 1
    return 0

def linearFunction(soma):
    return soma

def softmaxFunction(soma):
    return np.exp(soma)/(np.exp(soma)).sum()

teste1 = stepFunction(1)
teste2 = sigmoidFunction(1)
teste3 = tahnFunction(1)
teste4 = reluFunction(1)
teste5 = linearFunction(1)

valores = [5, 2, 1.3]
teste6 = softmaxFunction(valores)