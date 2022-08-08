import numpy as np
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt

def my_plot(x1, y1, x2=[], y2=[]):
    if x1 and y1: plt.plot(x1, y1, color='blue', marker='o')
    if x2 and y2: plt.plot(x2, y2, color='red', marker='o')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    # plt.ylabel('mAP (%)')
    plt.title('testing result')
    # plt.xscale('log')
    plt.show()


epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # 
epochs2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

loss = [38.7, 10.7, 6.45, 4.53, 3.47, 2.85, 2.46, 2.1, 1.89, 1.74, 0.895, 0.695, 0.575, 0.566, 0.474, 0.442, 0.427, 0.404, 0.361]

test_no_obj_acc = [99.853294, 99.932129, 99.964600, 99.982033, 99.986176, 99.988686, 99.993149, 99.995979, 99.994370, 99.996506]
test_obj_acc = [99.966637, 99.766434, 99.733063, 99.165833, 99.332664, 99.232567, 99.299301, 98.798798, 99.666336, 98.965630]
mAP = [0.955919623374939, 0.9683190584182739, 0.9799336194992065, 0.9878032803535461, 0.9901291131973267, 0.9879307746887207, 0.9906947612762451, 0.9890689253807068, 0.9928376078605652, 0.9913437962532043]

# my_plot(epochs2, loss, epochs, mAP)
my_plot(epochs2, loss)
my_plot(epochs, mAP)
