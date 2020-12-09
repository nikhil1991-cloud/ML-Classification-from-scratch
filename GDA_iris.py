import pandas  as pd #Data manipulation
import numpy as np #Data manipulation

#We will use Gaussian Discriminant analysis to solve the iris classification problem.

df = pd.read_csv('/Users/nikhil/Data/ML_examples/iris.csv')
#we need to remove one class frrrom the variety to make the variable binary
df=df[df['variety']!='Versicolor']
df = df.replace(to_replace=['Setosa','Virginica'], value=[0, 1])

#Shuffle data set
shuffle_df = df.sample(frac=1)
train_size = int(0.7 * len(df))
#Split it in train and test set
train_df = shuffle_df[:train_size]
test_df = shuffle_df[train_size:]

y_train = np.array(train_df['variety'])
x_train = np.array(train_df.drop('variety',axis=1))
y_test = np.array(test_df['variety'])
x_test = np.array(test_df.drop('variety',axis=1))


phi = len(x_train[np.where(y_train==1)])/len(x_train)
mu1 = np.mean(x_train[np.where(y_train==0)],axis=0)
mu2 = np.mean(x_train[np.where(y_train==1)],axis=0)

diff1 = x_train[y_train==0] - mu1
diff2 = x_train[y_train==1] - mu2
diff = np.concatenate((diff1,diff2),axis=0)

sigma = np.matmul(diff.T,diff)/len(diff)

d = x_train.shape[1]
sigma_inv = np.linalg.inv(sigma)
det_sigma = np.linalg.det(sigma)
Py0 = (1/((2*np.pi)**(d/2)))*(1/(det_sigma**0.5))* np.exp(- 0.5*np.sum(((x_test-mu1)@sigma_inv)*(x_test-mu1), axis=1)) * (1 - phi)
Py1 = (1/((2*np.pi)**(d/2)))*(1/(det_sigma**0.5))* np.exp(- 0.5*np.sum(((x_test-mu2)@sigma_inv)*(x_test-mu2), axis=1)) * phi
Py0 = Py0.reshape(-1, 1)
Py1 = Py1.reshape(-1, 1)
Predictions = np.argmax(np.concatenate((Py0, Py1), axis=1), axis=1)
Error = np.sum((Predictions-y_test)**2)/len(y_test)

