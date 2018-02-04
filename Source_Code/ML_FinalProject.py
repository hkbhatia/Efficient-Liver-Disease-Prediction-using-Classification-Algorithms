import pandas as pd

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB

from sklearn import preprocessing
from sklearn import model_selection 
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import Data_Preparation
from time import time


# For reproducibility
RANDOM_SEED = 12345

# use get_clean_data function in Data_Preparation module to clean the dataset 
# df = Data_Preparation.get_clean_data()
# If the function get_clean_data is used, the cleaned dataset is used and
# assigning column names is not necessary here(already done in the function) and hence comment the below 2 statements 
# Read .csv from provided dataset
csv_filename="Indian Liver Patient Dataset (ILPD).csv"
df = pd.read_csv(csv_filename, names=["Age","Gender","TB","DB","Alkphos","Sgpt","Sgot","TP","ALB","A/G","LiverResult"])

#Convert Gender,Selector to numbericals
le = preprocessing.LabelEncoder()
df['Gender'] = le.fit_transform(df.Gender)
df['LiverResult'] = le.fit_transform(df.LiverResult)

#Removing the Selector column from the list of the features
features=(list(df.columns[:-1]))

#Retrieving the features and Selector data into X & y
X = df[features]
y = df['LiverResult']

#Fitting the features
X = Imputer().fit_transform(X)

#Split dataset to 60% training and 40% testing ran-dom_state
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=0)


correlation_matrix = df.corr()
plt.subplots(figsize = (10, 10))

sns.heatmap(correlation_matrix, vmax = .9, square = True)
plt.show()


########SVM
#Loading the SVM Classifier, Fitting the training sets to get the classification model, 
#Getting the prediction vetor 'svm_predictions' of the model, and Computing the Confusion Matrix
t0 = time()
svc = SVC()
clf_svc=svc.fit(X_train, y_train)
svm_predictions = clf_svc.predict(X_test)
svm_cnf_matrix = confusion_matrix(y_test, svm_predictions)
t1 = time()

#Printing the results of our SVM classifier 
print ("\nSVM")
print ("\nAcurracy:\n\t ", clf_svc.score(X_test,y_test) )
print("\nConfusion Matrix:\n",svm_cnf_matrix)
print("\nClassification Report:\n",classification_report(y_test,svm_predictions))
print ("Time Elapsed: ", t1-t0)

########Neural Network
#Loading the Neural Network Classifier, Fitting the training sets to get the classification model, 
#Getting the prediction vetor 'mlp_predictions' of the model, and Computing the Confusion Matrix
t0 = time()
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10),random_state=0)
clf_mlp = mlp.fit(X_train, y_train)
mlp_predictions = mlp.predict(X_test)
mlp_cnf_matrix = confusion_matrix(y_test,mlp_predictions)
t1 = time()


#Printing the results of our Neural Network classifier 
print ("\nNeural Network")
print ("\nAcurracy:\n\t ", clf_mlp.score(X_test,y_test) )
print("\nConfusion Matrix:\n",mlp_cnf_matrix)
print("\nClassification Report:\n",classification_report(y_test,mlp_predictions))
print ("Time Elapsed: ", t1-t0)


######## K Nearest Neighbors (KNN)
t0 = time()
knn = KNeighborsClassifier(n_neighbors=3)
clf_knn=knn.fit(X_train, y_train)
knn_pred = clf_knn.predict(X_test)
knn_cnf_matrix = confusion_matrix(y_test, knn_pred)
t1 = time()

#Printing the results of our classifier 
print ("\nKNN")
print ("\nAcurracy:\n\t ", clf_knn.score(X_test,y_test) )
print("\nConfusion Matrix:\n",knn_cnf_matrix)
print("\nClassification Report:\n",classification_report(y_test,knn_pred))
print ("Time Elapsed: ", t1-t0)

############Random Forest
t0 = time()
rf = RandomForestClassifier(n_estimators = 15, n_jobs = 2)
clf_rf = rf.fit(X_train, y_train)
rf_pred = clf_rf.predict(X_test)
rf_cnf_matrix = confusion_matrix(y_test, rf_pred)
t1 = time()

print ("\nRandom Forest")
print ("\nAcurracy:\n\t", clf_rf.score(X_test,y_test))
print("\nConfusion Matrix:\n",rf_cnf_matrix)
print("\nClassification Report:\n",classification_report(y_test,rf_pred))
print ("Time Elapsed: ", t1-t0)


############Log Regression
t0 = time()
lr = LogisticRegression(n_jobs=2, random_state=RANDOM_SEED,C=1,fit_intercept=True,penalty='l1')
clf_lr = lr.fit(X_train, y_train)
lr_pred = clf_lr.predict(X_test)
lr_cnf_matrix = confusion_matrix(y_test, lr_pred)
t1 = time()

print ("\nLog Regression")
print ("\nAcurracy:\n\t", clf_lr.score(X_test,y_test))
print("\nConfusion Matrix:\n",lr_cnf_matrix)
print("\nClassification Report:\n",classification_report(y_test,lr_pred))
print ("Time Elapsed: ", t1-t0)


##############Decision Tree
t0 = time()
dt = DecisionTreeClassifier(min_samples_split=20,random_state=99)
clf_dt=dt.fit(X_train, y_train)
dt_pred = clf_dt.predict(X_test)
dt_cnf_matrix = confusion_matrix(y_test, dt_pred)
t1 = time()

#Printing the results of our classifier 
print ("\nDecision Tree")
print ("\nAcurracy:\n\t ", clf_dt.score(X_test,y_test) )
print("\nConfusion Matrix:\n",dt_cnf_matrix)
print("\nClassification Report:\n",classification_report(y_test,dt_pred))
print ("Time Elapsed: ", t1-t0)


##############Naive Bayes
t0 = time()
nb = BernoulliNB()
clf_nb=nb.fit(X_train, y_train)
nb_pred = clf_nb.predict(X_test)
nb_cnf_matrix = confusion_matrix(y_test, nb_pred)
t1 = time()

#Printing the results of our classifier 
print ("\nNaive Bayes")
print ("\nAcurracy:\n\t ", clf_nb.score(X_test,y_test) )
print("\nConfusion Matrix:\n",nb_cnf_matrix)
print("\nClassification Report:\n",classification_report(y_test,nb_pred))
print ("Time Elapsed: ", t1-t0)


matrices = {
    "SVM":svm_cnf_matrix,
    "Neural Network": mlp_cnf_matrix,
    "K Nearest Neighbors":knn_cnf_matrix,
    "Random Forest":rf_cnf_matrix,
    "Log Regression":lr_cnf_matrix,
    "Decision Tree":dt_cnf_matrix,
    "Naive Bayes":nb_cnf_matrix,}
acurracies = {
    "SVM":clf_svc.score(X_test,y_test),
    "Neural Network": clf_mlp.score(X_test,y_test),
    "K Nearest Neighbors":clf_knn.score(X_test,y_test),
    "Random Forest":clf_rf.score(X_test,y_test),
    "Log Regression":clf_lr.score(X_test,y_test),
    "Decision Tree":clf_dt.score(X_test,y_test),
    "Naive Bayes":clf_nb.score(X_test,y_test),}

#We use the function to plot the Confusion Matrix of each model
def draw_confusion_matrices(cms, classes):
    fig = plt.figure(figsize = (10, 15))
    
    i = 1   # used to compute the matrix location
    for clf_name, cm in cms.items():
        thresh = cm.max() / 2   # used for the text color
        
        ax = fig.add_subplot(len(cms) / 2 + 1, 2, i,
                             title = 'Confusion Matrix of %s' % clf_name, 
                             xlabel = 'Predicted',
                             ylabel = 'True')
        cax = ax.matshow(cm, cmap = plt.cm.Blues)
        fig.colorbar(cax)
        i += 1
        
        # Ticks
        ax.set_xticklabels([''] + classes)
        ax.set_yticklabels([''] + classes)
        ax.tick_params(labelbottom = True, labelleft = True, labeltop = False)
        
        # Text
        for x in range(len(cm)):
            for y in range(len(cm[0])):
                ax.text(y, x, cm[x, y], 
                        horizontalalignment = 'center', 
                        color = 'black' if cm[x, y] < thresh else 'white')
        
    plt.tight_layout()
    plt.show()
    

labels = np.unique(y).tolist()
draw_confusion_matrices(matrices, labels)