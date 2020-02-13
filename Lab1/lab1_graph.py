import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0.0, 25.0, 1)
loss_train = [2.111, 1.238, 0.909, 0.712,
              0.553,
              0.417,
              0.292,
              0.210,
              0.158,
              0.113,
              0.104,
              0.077,
              0.081,
              0.053,
              0.051,
              0.047,
              0.042,
              0.039, 0.045, 0.034, 0.029, 0.033, 0.022, 0.021, 0.019, ]

loss_val = [1.416,
            1.008,
            0.877,
            0.713,
            0.709,
            0.772,
            0.704,
            0.798,
            0.891,
            0.936,
            0.920,
            1.071,
            1.238,
            0.993,
            1.222,
            1.118,
            1.106,
            1.208,
            1.199,
            1.240,
            1.447,
            1.317,
            1.237,
            1.269,
            1.459]

train_acc = [49,
             66,
             73,
             82,
             85,
             88,
             94,
             95,
             96,
             98,
             98,
             97,
             98,
             98,
             98,
             99,
             99,
             99,
             99,
             99,
             98,
             99,
             99,
             99,
             99]

val_acc = [48,
           63,
           70,
           75,
           75,
           75,
           78,
           78,
           77,
           78,
           78,
           77,
           78,
           78,
           78,
           78,
           79,
           79,
           79,
           79,
           78,
           78,
           79,
           79,
           78]

fig, ax = plt.subplots()
ax.plot(t, loss_train, color='blue', label='Training loss')
ax.plot(t, loss_val, color='green', label='Validation loss')

#ax.plot(t, train_acc, color='red', label='Training accuracy')
#ax.plot(t, val_acc, color='orange', label='Validation accuracy')

ax.set(xlabel='Epoch', ylabel='Loss',
       title='Loss value')
ax.grid()
ax.legend()
fig.savefig("Loss.png")
plt.show()
