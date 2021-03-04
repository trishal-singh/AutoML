import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
st.title("AUTOML")
st.write("""Select a dataset and start creating models.""")
dataset =st.sidebar.selectbox(label="Dataset:",options=('Iris',"Breast Cancer","Wine"))
model=st.sidebar.selectbox(label="Model:",options=("SVM","KNN","Random Forest","Ada Boost"))
def load_data(dataset):
    if dataset=='Iris':
        data=datasets.load_iris()
    elif dataset=='Breast Cancer':
        data=datasets.load_breast_cancer()
    else:
        data=datasets.load_wine()
    X=data.data
    Y=data.target
    return X,Y
X,y=load_data(dataset)
st.write("No of Rows: ",X.shape[0] )
st.write("No of Features: ",X.shape[1] )
def select_parameter(clf):
    if clf=='KNN':
        K=st.sidebar.number_input(label="Enter an Integer",value=1)
        params=dict()
        params["n_neighbors"]=K
    
    if clf=='SVM':
        C=st.sidebar.number_input(label="Regularization parameter",value=1.0)
        kernel=st.sidebar.selectbox(label="kernel_type",options=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])
        params=dict()
        params["C"]=C
        params["kernel"]=kernel
    if clf=="Random Forest":
        n_estim=st.sidebar.number_input(label="N Estimators",value=10)
        max_depth=st.sidebar.number_input(label="Maximum Depth",value=10)
        params=dict()
        params["n_estimators"]=n_estim
        params["max_depth"]=max_depth
    if clf=="Ada Boost":
        n_estim=st.sidebar.number_input(label="N Estimators",value=10)
        lr=st.sidebar.number_input(label="Learning Rate",value=0.1)
        params=dict()
        params["n_estimators"]=n_estim
        params["learning_rate"]=lr


    return params
param=select_parameter(model)
def create_model(clf,params):
    if clf=='KNN':
        model=KNeighborsClassifier(**params)
    if clf=='SVM':
        model=SVC(**params)
    if clf=="Random Forest":
        model=RandomForestClassifier(**params)
    if clf=="Ada Boost":
        model=AdaBoostClassifier(**params)
    return model
classifier=create_model(model,param)
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=4)
classifier.fit(train_x,train_y)
y_pred=classifier.predict(test_x)
accuracy=accuracy_score(test_y,y_pred)
st.write(f"Accuracy: {accuracy} ")
        

