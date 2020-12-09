import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import scipy
from scipy import optimize
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

#In this example we will classify if a recepie is a dessert or not based on some factors like its rating, calorie content, protien content, fats, sodium content. We will use Logistic regression classifier. We use two different approaches, 1st is using sklearns logistic regression solver and 2nd is solving logistic regression using Iteratively Reweighted Least Squares. We compare the confusion matrices from the two methods along with accuracy, sensitivity and F1 score.

df = pd.read_csv('/Users/nikhil/Data/ML_examples/epi_r.csv')
variables = df.columns
#restrict all recipies < 10,000 calories and drop NaN values
epicurious = df[df['calories']<10000].dropna()
#for speed regression lets only take first 500 data points
epicurious=epicurious.drop('title',axis=1)
sns.scatterplot(epicurious['calories'],epicurious['dessert'])
X = epicurious[['rating','calories','protein','fat','sodium']][:][0:500]
y = epicurious['dessert'][:][0:500]
X_test = X#epicurious.drop('dessert',axis=1)[:][500:900]
y_test = y#epicurious['dessert'][:][500:900]
#check for null values
epicurious.isnull().sum()

#Using sklearn for logistic regression
loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(X,y)
beta_sklearn = np.zeros(np.shape(X)[1]+1)
beta_sklearn[0] = loj_model.intercept_[0]
beta_sklearn[1:np.shape(beta_sklearn)[0]] = loj_model.coef_[0]


#Using IRLS (Efficient)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
X_train = np.c_[np.ones((np.shape(X)[0],1)),np.array(X)]
X_test = np.c_[np.ones((np.shape(X_test)[0],1)),np.array(X_test)]
Y_train = np.array(y)
        
beta_IRLS = np.zeros(np.shape(X_train)[1])
epochs_IRLS = 100
i=0
for i in range (0,epochs_IRLS):
    Y_pred = sigmoid(np.dot(X_train,beta_IRLS))  # The current predicted value of Y
    r = (1-Y_pred)*(Y_pred)
    R = np.diag(r)
    Z = X_train.dot(beta_IRLS) + np.dot((Y_train-Y_pred),np.linalg.inv(R))
    Xt = np.transpose(X_train)
    XtS = Xt.dot(R)
    XtSX = XtS.dot(X_train)
    inverse_of_XtSX = np.linalg.inv(XtSX)
    inverse_of_XtSX_Xt = inverse_of_XtSX.dot(Xt)
    inverse_of_XtSX_XtS = inverse_of_XtSX_Xt.dot(R)
    beta_IRLS = inverse_of_XtSX_XtS.dot(Z)
    print(beta_IRLS)

y_pred_irls = sigmoid(np.dot(X_test,beta_IRLS))
y_pred_sklearn = sigmoid(np.dot(X_test,beta_sklearn))
y_pred_irls = np.array([1 if i > 0.5 else 0 for i in y_pred_irls])
y_pred_sklearn = np.array([1 if i > 0.5 else 0 for i in y_pred_sklearn])

#Generate confusion matrix and accuracy,sensitivty, F1 score
CFM_sklearn = np.zeros((2,2))
CFM_irls = np.zeros((2,2))
y_test = np.array(y_test)
#confusion matrix for sklearn method
CFM_sklearn[0,0] = np.where((y_pred_sklearn==1)&(y_test==1))[0].shape[0]#TP
CFM_sklearn[0,1] = np.where((y_pred_sklearn==0)&(y_test==1))[0].shape[0]#FN
CFM_sklearn[1,0] = np.where((y_pred_sklearn==1)&(y_test==0))[0].shape[0]#FP
CFM_sklearn[1,1] = np.where((y_pred_sklearn==0)&(y_test==0))[0].shape[0]#TN
A_sklearn = (CFM_sklearn[0,0] + CFM_sklearn[1,1])/(CFM_sklearn[0,0] + CFM_sklearn[1,1]+CFM_sklearn[0,1] + CFM_sklearn[1,0])
P_sklearn = (CFM_sklearn[0,0])/(CFM_sklearn[0,0]+CFM_sklearn[1,0])
R_sklearn = (CFM_sklearn[0,0])/(CFM_sklearn[0,0]+CFM_sklearn[0,1])
F1_sklearn = 2*(R_sklearn * P_sklearn) / (R_sklearn + P_sklearn)
#confusion matrix for IRLS
CFM_irls[0,0] = np.where((y_pred_irls==1)&(y_test==1))[0].shape[0]
CFM_irls[0,1] = np.where((y_pred_irls==0)&(y_test==1))[0].shape[0]
CFM_irls[1,0] = np.where((y_pred_irls==1)&(y_test==0))[0].shape[0]
CFM_irls[1,1] = np.where((y_pred_irls==0)&(y_test==0))[0].shape[0]
A_irls = (CFM_irls[0,0] + CFM_irls[1,1])/(CFM_irls[0,0] + CFM_irls[1,1]+CFM_irls[0,1] + CFM_irls[1,0])
P_irls = (CFM_irls[0,0])/(CFM_irls[0,0]+CFM_irls[1,0])
R_irls = (CFM_irls[0,0])/(CFM_irls[0,0]+CFM_irls[0,1])
F1_irls = 2*(R_irls * P_irls) / (R_irls + P_irls)

