import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
#standard scaling
def StandardScaler(Data):
    for feature in range (0,Data.shape[1]-1):
        Current_feature = Data.iloc[:,feature]
        feature_std = (Current_feature - np.mean(Data.iloc[:,feature]))/(np.std(Data.iloc[:,feature]))
        Data.iloc[:,feature].update(feature_std)
    return Data

#Train test split
def train_test_split(Data,train_frac):
    shuffle_data = Data.sample(frac=1)
    train_split = int(train_frac*len(shuffle_data))
    train_data = shuffle_data[:train_split]
    test_data = shuffle_data[train_split:]
    return train_data,test_data

df = pd.read_csv('/Users/nikhil/Data/ML_examples/iris.csv')
df = StandardScaler(df)
#we need to remove one class from the variety to make the dependent variable binary
df=df[df['variety']!='Virginica']
df = df.replace(to_replace=['Setosa','Versicolor'], value=[1, -1])
#Split data into training and testing sets
train_data,test_data = train_test_split(df,0.7)
X_train,X_test = train_data[['petal.length','petal.width']],test_data[['petal.length','petal.width']]
Y_train,Y_test = train_data['variety'],test_data['variety']
X_train,X_test = np.array(X_train),np.array(X_test)
Y_train,Y_test = np.array(Y_train),np.array(Y_test)

#Using sklearn to get svm coefficients
svc = LinearSVC(C=1,loss="hinge")
svc.fit(X_train,Y_train)

#define all kernels
def kernel(x,z):
    return np.matmul(x,z.T)

#Using QP to solve for matrices P,q,A,G
def svm_linear(X,Y,C=None):
          if C is not None: C = float(C)
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
          #w parameter in vectorized form
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

def make_meshgrid(x,y,h=.02):
    xmin,xmax = x.min()-1,x.max()+1
    ymin,ymax = y.min()-1,y.max()+1
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))
    return xx, yy

#Predict Y using QP
w,b = svm_linear(X_train,Y_train,1)
Y_qp = svm_predict(X_test,Y_test,w,b)
#Predict Y using sklearn
w_sklearn,b_sklearn = svc.coef_.reshape(-1,1),svc.intercept_[0]
Y_sklearn = svm_predict(X_test,Y_test,w_sklearn,b_sklearn)


    
    
X0,X1 = X_test[:,0],X_test[:,1]
xx0, xx1 = make_meshgrid(X0, X1)
xgrid = np.c_[xx0.ravel(), xx1.ravel()]
Z = xgrid.dot(w[:,0].reshape(-1,1)) + b # wx + b
Z_sklearn = xgrid.dot(w_sklearn[:,0].reshape(-1,1)) + b_sklearn
Z_sklearn = Z_sklearn.reshape(xx0.shape)
Z = Z.reshape(xx0.shape)


plt.subplot(1,2,1)
plt.scatter(X_test[:,0],X_test[:,1],c=Y_test)
plt.contour(xx0,xx1,Z,levels=[-1,0,1])
plt.xlabel('petal length')
plt.ylabel('petal width')

plt.subplot(1,2,2)
plt.scatter(X_test[:,0],X_test[:,1],c=Y_test)
plt.contour(xx0,xx1,Z_sklearn,levels=[-1,0,1])
plt.xlabel('petal length')
plt.ylabel('petal width')

