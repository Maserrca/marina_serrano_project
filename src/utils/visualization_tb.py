import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dirr = os.path.dirname
path = dirr(dirr(dirr(__file__)))
print(path)
sys.path.append(path)

print('prueba')

def accuracy(x):
    plt.figure(figsize=(8,6))
    plt.title('Accuracy scores')
    plt.plot(x.x['accuracy'],'go-')
    plt.plot(x.x['val_accuracy'],'ro-')
    plt.legend(['accuracy', 'val_accuracy'])
    plt.show()
    plt.figure(figsize=(8,6))
    plt.title('Loss value')
    plt.plot(x.x['loss'],'go-')
    plt.plot(x.x['val_loss'],'ro-')
    plt.legend(['loss', 'val_loss'])
    plt.show()
