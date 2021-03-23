import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

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
df.insert(loc=0,column='intercept',value=1)#Insert 1 in X so that intercept b is absorbed in coefficients w
#we need to remove one class frrrom the variety to make the variable binary
df=df[df['variety']!='Virginica']
df = df.replace(to_replace=['Setosa','Versicolor'], value=[1, -1])
#Train test split
train_data,test_data = train_test_split(df,0.7)
X_train,X_test = train_data[['intercept','petal.length','petal.width']],test_data[['intercept','petal.length','petal.width']]
Y_train,Y_test = train_data['variety'],test_data['variety']
X_train,X_test = np.array(X_train),np.array(X_test)
Y_train,Y_test = np.array(Y_train),np.array(Y_test)

#Solve svm using sklearn
svc = LinearSVC(C=0.1,loss="hinge")
svc.fit(X_train,Y_train)

#We will be using Stochastic Gradient descent method, specifically PEGASOS algorithm explained in Shwartz,Singer et al. 2000
#Define cost function
def CF(W,x,y):
    return (0.5*lambd*np.sum(np.dot(W,W)) + np.sum(np.max([0,1-y[i]*(np.dot(x[i],W))]) for i in range (x.shape[0])))/len(Y_train)
#Initialize weight array by zeors
W = np.zeros(np.shape(X_train)[1])
lambd=1/1000 #Define lambda
n_iters = 3500
Cost  = []
for step in range(1,n_iters):
    X_train,Y_train = shuffle(X_train,Y_train)
    for i in range (0,len(X_train)):
        distance = np.max([0,1 - Y_train[i]*np.dot(W,X_train[i])])
        if distance == 0:
            W -= (1/(step*lambd))*(2*lambd*W)
        else:
            W -= (1/(step*lambd))*((2*lambd*W)-np.dot(X_train[i],Y_train[i]))
    Cost.append(CF(W,X_train,Y_train))

def make_meshgrid(x,y,h=.02):
    xmin,xmax = x.min()-1,x.max()+1
    ymin,ymax = y.min()-1,y.max()+1
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))
    return xx, yy

Y_p = np.sign(np.dot(X_test,W))
Accuracy = metrics.accuracy_score(Y_p,Y_test)

w,b = W[1:],W[0]
w_sklearn,b_sklearn = svc.coef_[0][1:],svc.coef_[0][0]

X0,X1 = X_test[:,1],X_test[:,2]
xx0, xx1 = make_meshgrid(X0, X1)
xgrid = np.c_[xx0.ravel(), xx1.ravel()]
Z = xgrid.dot(w.reshape(-1,1)) + b
Z_sklearn = xgrid.dot(w_sklearn.reshape(-1,1)) + b_sklearn
Z_sklearn = Z_sklearn.reshape(xx0.shape)
Z = Z.reshape(xx0.shape)

plt.subplot(2,2,1)
plt.plot(np.log(Cost))
plt.xlabel('Iterations')
plt.ylabel('Log[$f_{cost function}$]')

plt.subplot(2,2,2)
plt.scatter(X_test[:,1],X_test[:,2],c=Y_test)
plt.contour(xx0,xx1,Z,levels=[-1,0,1])
plt.xlabel('petal length')
plt.ylabel('petal width')

plt.subplot(2,2,3)
plt.scatter(X_test[:,1],X_test[:,2],c=Y_test)
plt.contour(xx0,xx1,Z_sklearn,levels=[-1,0,1])
plt.xlabel('petal length')
plt.ylabel('petal width')


