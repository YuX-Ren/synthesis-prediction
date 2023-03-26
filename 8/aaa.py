from matplotlib import pyplot as plt
import numpy as np
a=np.linspace(1,100000,100000)
y=1-(1-1/a)**a
plt.plot(a,y)
plt.show()

# import random
# a=np.linspace(1,100,100)
# # print(a)
# j=k=0
# for i in range(10000):
#     if(random.randint(1,100)==4):
#         k=k+1
#         j=j+1
#         # print("hit")
#     if(i%100==0):
#         # print(k)
#         k=0
#         # print("%d"%(i/100))
# print(j)