
# coding: utf-8

# In[73]:

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel


# In[74]:

def feature_normalize(X):
    mean_r = []
    std_r = []
    X_norm = X
    nc = X.shape[1]
    for i in range(nc):
        m = np.mean(X[:,i])
        s = np.std(X[:,i])
        mean_r.append(m)
        std_r.append(m)
        X_norm[:,i] = (X_norm[:,i]-m)/s
        
    return X_norm,mean_r,std_r


# In[43]:

def compute_cost(X, y, theta):
    m = y.size
    predictions = X.dot(theta)
    sqErrors = (predictions - y)
    

    J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)
    return J


# In[75]:

def gradient_descent(X,y,theta,alpha,num_iters):
    m = y.size
    J_history = np.zeros(shape=(num_iters,1))
    for i in range(num_iters):
        predictions = X.dot(theta)
        theta_size = theta.size
        for it in range(theta_size):
            temp = X[:,it]
            temp.shape = (m,1)
            errors_x1 = (predictions-y)*temp
            theta[it][0] = theta[it][0]-alpha*(1.0/m)*errors_x1.sum()
            
        J_history[i,0] = compute_cost(X,y,theta)
        
    return theta,J_history


# In[76]:

data = np.loadtxt('ex1data2.txt',delimiter=',')


# In[77]:

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
n = 100
for c,m,zl,zh in [('r','o',-50,-25)]:
    xs = data[:,0]
    ys = data[:,1]
    zs = data[:,2]
    ax.scatter(xs,ys,zs,c=c,marker=m)
    
ax.set_xlabel('Size of house')
ax.set_ylabel('Number of Bedrooms')
ax.set_zlabel('Price of house')

plt.show()

X = data[:,:2]
y = data[:,2]


# In[78]:

m = y.size

y.shape = (m, 1)


# In[79]:

x, mean_r, std_r = feature_normalize(X)


# In[80]:

it = np.ones(shape=(m,3))
it[:,1:3] = x


# In[81]:

iterations = 100
alpha = 0.01


# In[82]:

theta = np.zeros(shape=(3,1))
theta,J_history = gradient_descent(it,y,theta,alpha,iterations)
print theta,J_history
plot(np.arange(iterations),J_history)
xlabel("Iterations")
ylabel("Cost Function")
show()


# In[83]:

price = np.array([1.0,   ((1650.0 - mean_r[0]) / std_r[0]), ((3 - mean_r[1]) / std_r[1])]).dot(theta)
print 'Predicted price of a 1650 sq-ft, 3 br house: %f' % (price)

