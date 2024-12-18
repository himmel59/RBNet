import torch


def Lossfunc_CrossEntropy(prediction, label):
    torch.set_default_dtype(torch.float64)

    loss = label * torch.log(prediction) + (1 - label) * torch.log(1 - prediction)
    return torch.mean(loss) * -1


def Lossfunc_CrossEntropy_Weighted(prediction, label, weight=5.):
    # label true but prediction false would be punished by weight
    torch.set_default_dtype(torch.float64)
    loss = label * torch.log(prediction) + (1 - label) * torch.log(1 - prediction) * weight
    return torch.mean(loss) * -1


def Lossfunc_Bayes(prediction, variance, original, label):
    '''
    predicition is the mu of full exc 
    variance is the variance of prediction
    label is the ground truth energy
    '''
    torch.set_default_dtype(torch.float64)

    del original
    s = torch.log(variance)
    partA = (prediction - label)**2 / variance
    partB = s
    loss = partA + partB
    return (torch.mean(loss), torch.mean(partA), torch.mean(partB))
