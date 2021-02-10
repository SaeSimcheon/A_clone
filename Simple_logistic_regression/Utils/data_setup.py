import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def add(a,b):
    return a + b 

def sample_data_call(direct):
    data=pd.read_csv(direct)
    data=data.drop(["sex","cp","fbs","restecg","exang","slope","ca"],axis=1)
    Y=data.target
    X=data.drop(["target"],axis=1)
    trainX,testX,trainY,testY=train_test_split(X,Y,test_size = 0.3,random_state = 1)
    
    return trainX,testX,trainY,testY


def normalize(X1,X2):
    st=StandardScaler()
    st.fit(X1)
    
    return st.transform(X1), st.transform(X2)
    