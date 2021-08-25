import numpy as np

n = 251

train_mask = np.zeros(n)
val_mask = np.zeros(n)
test_mask = np.zeros(n)
p = np.array([0.4, 0.6])

for i in range(n):
    train_mask[i] = np.random.choice([0, 1], p = p.ravel())

p = np.array([0.5, 0.5])
for i in range(n):
    if(train_mask[i] == 0):
        val_mask[i] = np.random.choice([0, 1], p = p.ravel())

for i in range(n):
    if(train_mask[i] == 0 and val_mask[i] == 0):
        test_mask[i] = 1   

print(train_mask.sum())
print(val_mask.sum())
print(test_mask.sum())
print(train_mask)
print(val_mask)
print(test_mask)
np.savez('wisconsin_split_3.npz',train_mask = train_mask, val_mask = val_mask, test_mask=test_mask)