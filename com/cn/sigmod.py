import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(0,900,num=901)

def sigmod300(x,low,high):
    w=(high-low)/5

    x1=x[0:450]
    x2=x[450:901]
    print(x1)
    print(x2)
    y1=low+(high-low)/(1+np.exp(-w*(x1-300)))
    y2=low+(high-low)/(1+np.exp(w*(x2-600)))
    y=np.r_[y1,y2]
    print(y)
    return y

y=sigmod300(x,4,2)
print(y)

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(x,y,'ro')
plt.show()
