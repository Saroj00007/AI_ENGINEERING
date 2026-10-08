# import numpy as np 
# 
# a = np.array([1,3,4])
# b = np.array([2,3,5])
# 
# print(a + b )

import numpy
from numpy.ma.core import mean 


#  what we are going to do is creating the tiny training model that will learn the y = 2x+1 system

x = numpy.array([1, 2, 3, 4, 5])
y = numpy.array([3, 5, 7, 9, 11])

w = 0.0
b = 0.0


learning_rate = 0.01

# y_das = wx + b

for i in range(100000):

    # forward pass
    y_das  = w*x + b 

    # loss

    loss = numpy.mean((y-y_das)**2)

    # gradient
    dw = numpy.mean(-2 * x * (y - y_das))
    db  = numpy.mean(-2 * (y- y_das))

    # parameter update

    w = w - learning_rate * dw 
    b = b - learning_rate * db 


# during learning our main agenda is simply reducing the loss and making the parameter close to for generating the output
print("w = ", w)    
print("b = ", b)

print("loss = ", loss)  





    



