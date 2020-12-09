import pandas  as pd
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
#Generate a data set with non linear classification boundary
X,y=make_moons(noise=0.1, random_state=2)
data = pd.DataFrame(data = X, columns=['x1','x2'])
data['y']=y
data.head()
#replace 0,1 by -1,1
y = np.where(y==0,-1,1)
#Converting to polynomial with no intercept term. Its taken care off in the QP code. This can also be done by concatenating [x1,x2,x1x2,x1^2,x2*2,....].
poly = PolynomialFeatures(degree = 3, include_bias=False)
Xpoly = poly.fit_transform(X)
# standardize the data
scaler = StandardScaler()
Xpolystan = scaler.fit_transform(Xpoly)

#Calculate svm coeff and intercept using sklearn
svm_clf = LinearSVC(C=10,loss='hinge',max_iter=10000)
svm_clf.fit(Xpolystan,y)


#Using QP to solve dual form of svm
#define kernel
def kernel(x,z):
    return np.matmul(x,z.T)

def svm(X,Y,C=None):
          if C is not None: C = float(C) #Soft marging condition
          m,n = X.shape
          H = np.outer(Y,Y)*kernel(X,X)
          Y = Y.reshape(-1,1) * 1.
          #Converting into cvxopt format
          P = cvxopt_matrix(H)
          q = cvxopt_matrix(-np.ones((m, 1)))
          A = cvxopt_matrix(Y.reshape(1, -1))
          b = cvxopt_matrix(np.zeros(1))
          if C is None:
             G = cvxopt_matrix(-np.eye(m))
             h = cvxopt_matrix(np.zeros(m))
          else:
             G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
             h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))

          sol = cvxopt_solvers.qp(P, q, G, h, A, b)
          alphas = np.array(sol['x'])
          #Getting weights w = sum(y*alphas*x)
          w = ((Y * alphas).T @ X).reshape(-1,1)
          #Selecting the set of indices S corresponding to non zero parameters
          S = (alphas > 1e-4).flatten()
          #Computing b
          b = np.mean(Y[S] - np.dot(X[S], w))
          return w,b

def svm_predict(X,Y,w,b):
    Y_p = np.sign(np.dot(X,w[:,0]) + b)
    accuracy = metrics.accuracy_score(Y,Y_p)
    print('accuracy= '+str(accuracy))
    return Y_p
    
#Plot decision boundary of the classifier
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

#Input data
X_train = Xpolystan
Y_train = y
X_test = Xpolystan
Y_test = y

#Predict Y using qp
w,b = svm(X_train,Y_train)
Y_qp = svm_predict(X_test,Y_test,w,b)
#Predict Y using sklearn
w_sklearn,b_sklearn = svm_clf.coef_.reshape(-1,1),svm_clf.intercept_[0]
Y_sklearn = svm_predict(X_test,Y_test,w_sklearn,b_sklearn)



# create grids
X0, X1 = X[:, 0], X[:, 1]
xx0, xx1 = make_meshgrid(X0, X1)
# polynomial transformation and standardization on the grids
xgrid = np.c_[xx0.ravel(), xx1.ravel()]
xgridpoly = poly.transform(xgrid)
xgridpolystan = scaler.transform(xgridpoly)
# prediction
Z = xgridpolystan.dot(w[:,0].reshape(-1,1)) + b # wx + b
Z_sklearn = xgridpolystan.dot(w_sklearn[:,0].reshape(-1,1)) + b_sklearn # w_sklearn*x + b_sklearn
#Z = svm_clf.predict(xgridpolystan)
Z = Z.reshape(xx0.shape)
Z_sklearn = Z_sklearn.reshape(xx0.shape)

#This example using hard margin. You can play with the code by giveing different values of C for soft margin in line 81

plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1],c=y)
plt.contour(xx0,xx1,Z,lebels=[-1,0,1])
plt.contour(xx0, xx1, Z, alpha=0.5, levels=[-1,0,1])
plt.xlabel('x1')
plt.ylabel('x2')

plt.subplot(1,2,2)
plt.scatter(X[:,0],X[:,1],c=y)
plt.contour(xx0,xx1,Z_sklearn,lebels=[-1,0,1])
plt.contour(xx0, xx1, Z_sklearn, alpha=0.5, levels=[-1,0,1])
plt.xlabel('x1')
plt.ylabel('x2')
