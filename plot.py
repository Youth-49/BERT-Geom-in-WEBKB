# plot acc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

train_acc_GL = []
train_acc_G = []
train_acc_L = []
train_acc_Lnn = []
val_acc_GL = []
val_acc_G = []
val_acc_L = []
val_acc_Lnn = []

dataset = 'cornell'

model1 = 'GeomGCN_TwoLayers_BERT'
model2 = 'GeomGCN_TwoLayers_BERT_only_Graph'
model3 = 'GeomGCN_TwoLayers_BERT_only_Space'
model4 = 'Linear_TwoLayers_bert'

with open(f'list/{model1}_{dataset}_train_acc.txt') as file:
    for line in file:
        line = line.strip('[').rstrip(']')
        train_acc_GL = list(line.split(', '))
        train_acc_GL = list(map(float, train_acc_GL))
        
with open(f'list/{model1}_{dataset}_val_acc.txt') as file:
    for line in file:
        line = line.strip('[').rstrip(']')
        val_acc_GL = list(line.split(', '))
        val_acc_GL = list(map(float, val_acc_GL))
        
with open(f'list/{model2}_{dataset}_train_acc.txt') as file:
    for line in file:
        line = line.strip('[').rstrip(']')
        train_acc_G = list(line.split(', '))
        train_acc_G = list(map(float, train_acc_G))
        
with open(f'list/{model2}_{dataset}_val_acc.txt') as file:
    for line in file:
        line = line.strip('[').rstrip(']')
        val_acc_G = list(line.split(', '))
        val_acc_G = list(map(float, val_acc_G))
        
with open(f'list/{model3}_{dataset}_train_acc.txt') as file:
    for line in file:
        line = line.strip('[').rstrip(']')
        train_acc_L = list(line.split(', '))
        train_acc_L = list(map(float, train_acc_L))
        
with open(f'list/{model3}_{dataset}_val_acc.txt') as file:
    for line in file:
        line = line.strip('[').rstrip(']')
        val_acc_L = list(line.split(', '))
        val_acc_L = list(map(float, val_acc_L))

with open(f'list/{model4}_{dataset}_train_acc.txt') as file:
    for line in file:
        line = line.strip('[').rstrip(']')
        train_acc_Lnn = list(line.split(', '))
        train_acc_Lnn = list(map(float, train_acc_Lnn))
        
with open(f'list/{model4}_{dataset}_val_acc.txt') as file:
    for line in file:
        line = line.strip('[').rstrip(']')
        val_acc_Lnn = list(line.split(', '))
        val_acc_Lnn = list(map(float, val_acc_Lnn))
        

x_range = max(len(train_acc_GL), len(train_acc_G), len(train_acc_L), len(train_acc_Lnn),
                len(val_acc_G), len(val_acc_GL), len(val_acc_L), len(val_acc_Lnn))

plt.figure(dpi = 500)
plt.ylim((0.25, 1))
# plt.plot([i for i in range(1, len(train_acc_GL)+1)], train_acc_GL, 'r-', label='train acc BERT-Geom-GL')
# plt.plot([i for i in range(1, len(train_acc_G)+1)], train_acc_G, '-', label='train acc BERT-Geom-G')
# plt.plot([i for i in range(1, len(train_acc_L)+1)], train_acc_L, '-', label='train acc BERT-Geom-L')
# plt.plot([i for i in range(1, len(train_acc_Lnn)+1)], train_acc_Lnn, '-', label='train acc BERT-Linear')
plt.plot([i for i in range(1, len(val_acc_GL)+1)], val_acc_GL, '-', label='val acc BERT-Geom-GL')
plt.plot([i for i in range(1, len(val_acc_G)+1)], val_acc_G, '-', label='val acc BERT-Geom-G')
plt.plot([i for i in range(1, len(val_acc_L)+1)], val_acc_L, '-', label='val acc BERT-Geom-L')
plt.plot([i for i in range(1, len(val_acc_Lnn)+1)], val_acc_Lnn, '-', label='val acc BERT-Linear')


plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Cornell')
plt.legend()
plt.savefig(f'Cornell_all_acc.jpg')