import numpy as np

min = 1.7 # minimum range of KL Divergence
max = 1.8 # maximum range of KL Divergence
num_label = 10 # This value should be set for the number of labels in dataset.
num_iter = 10 # This value is set for the number of clients.
alpha = 0.3 # This value is the skewness of the distribution distance which is used for Dirichlet distribution

# the values of below may not be change.
val_min = 9999
val_max = -9999
sum = []
kl_avg = 0


def kl_divergence(p,q):
    return np.sum(np.where(p!=0, p*np.log(p/q),0))

while len(sum) < 10:
    uniform = np.repeat(0.1, num_label)
    x = np.random.dirichlet(np.repeat(alpha, num_label))
    temp = kl_divergence(x,uniform)
    if temp >= min and temp <= max:
        print(len(sum))
        sum.append(x.tolist())
        kl_avg = kl_avg + temp
        if temp<=val_min:
            val_min = temp
        if temp>=val_max:
            val_max = temp



print('sum: ', np.einsum('ij->j', sum))
print('avg_val: ', kl_divergence((np.einsum('ij->j', sum)/np.sum((np.einsum('ij->j', sum)))),uniform))
print('min: ', val_min, 'max: ', val_max)
print('avg_kl: ', kl_avg/num_iter)
print(sum)