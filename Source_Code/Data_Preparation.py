import os.path
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

DATA_FILE_PATH = "Indian Liver Patient Dataset (ILPD).csv"
CLEAN_DATA_FILE_PATH = "Cleaned_Indian Liver Patient Dataset (ILPD).csv"
_df = None
FEATURE_CORR_FILE_PATH = "featureCorrelation.png"
AUC_FILE_PATH = "AUCRFLR.png"
CROSS_VALIDATION_FOR_GRIDSEARCH = 5
RATIO_TRAINING_TESTING = 0.7

# For reproducibility.
RANDOM_SEED = 12345


def get_clean_data():

    if _df is None:
        _construct_clean_data()

    # Dangerous here! We should return a copy so that the original data is undisturbed.
    return _df.copy()


def _construct_clean_data():
    global _df
    if os.path.isfile(CLEAN_DATA_FILE_PATH):
        # Load the already cleansed data.
        _df = pd.read_csv(CLEAN_DATA_FILE_PATH)
    else:
        if os.path.isfile(DATA_FILE_PATH):
            temp_df = pd.read_csv(DATA_FILE_PATH, header=None)

            # Check numbers of rows and columns.
            (num_row, num_col) = temp_df.shape
            if num_row != 583 or num_col != 11:
                raise Exception("Dataset tampered")

            # Add meaningful columns' names.
            temp_df.columns = ["Age", "Gender", "TB", "DB", "Alkphos", "Sgpt", "Sgot", "TP", "ALB", "A/G", "LiverResult"]

            # Transform categorical target variable to numberic one. 1 means "positive", 0 means "negative".
            temp_df['LiverResult'] = temp_df['LiverResult'].apply(lambda x: 0 if (x == 2) else x)

            # Column "A/G" has some NULL values.
            index_null = temp_df['A/G'].index[temp_df['A/G'].isnull()]
            # 209, 241, 253, 312 ==> Remove these rows!
            # Even more, with NaN, seaborn library has problems!
            temp_df = temp_df.drop(temp_df.index[index_null])

            # Save clean data to file.
            temp_df.to_csv(CLEAN_DATA_FILE_PATH, index=False)

            _df = temp_df
        else:
            raise IOError("%s not found" % DATA_FILE_PATH)


# ******************* Preparing Data *******************
df = get_clean_data()

# Hint for PyCharm to recognize it as "Dataframe", so autocompletion feature.
assert isinstance(df, pd.DataFrame)

# ******************* Exploratory Data Analysis *******************
drawn_df = df.copy()

# seaborn requires "hue" column to be of non-number.
# All other columns must be of number.
if not os.path.isfile(FEATURE_CORR_FILE_PATH):
    drawn_df['Gender'] = drawn_df['Gender'].apply(lambda x: 1 if (x == "Male") else 0)
    drawn_df['LiverResult'] = drawn_df['LiverResult'].apply(lambda x: "LIVER" if (x == 1) else "NONE")

    sns_plot = sns.pairplot(drawn_df, hue="LiverResult")
    sns_plot.savefig(FEATURE_CORR_FILE_PATH)
    plt.close()

"""
Looking at the graph output.png, we can see that there are correlations between "direct_bilirubin" and "total_bilirubin"
(strong), "aspartate_aminotransferase" and "alamine_aminotransferase" (a little strong), "albumin" and "total_protiens"
(a little strong), "A/G" and "albumin" (a little strong).

Let's calculate the Standard correlation coefficient (pearson)!
"""

"""
drawn_df.dtypes:
age                                   int64
Gender                                int64
total_bilirubin                     float64
direct_bilirubin                    float64
alkaline_phosphotase                  int64
alamine_aminotransferase              int64
aspartate_aminotransferase            int64
total_protiens                      float64
albumin                             float64
ratio_albumin_and_globulin_ratio    float64
LiverResult                            object
dtype: object
"""

print("Correlation between features:\n%s\n" % str(drawn_df.corr()))

"""
Corr("direct_bilirubin" and "total_bilirubin") = 0.874481 => very strong
Corr("aspartate_aminotransferase" and "alamine_aminotransferase") = 0.791862 ==> strong
Corr("albumin" and "total_protiens") = 0.783112 ==> strong
Corr("ratio_albumin_and_globulin_ratio" and "albumin") = 0.689632 ==> quite strong
Corr("total_protiens" and "ratio_albumin_and_globulin_ratio") = 0.234887 ==> weak

So it makes sense to drop some columns of these to help not distort the model.
Columns deleted: direct_bilirubin (in favor of "total" sense), aspartate_aminotransferase (random choice),
                    albumin (to keep more columns (total_protiens, ratio_albumin_and_globulin_ratio)).
"""

# print("Dropping column: direct_bilirubin, aspartate_aminotransferase, albumin.")
# df.drop(['DB', 'Sgot', 'ALB'], axis=1, inplace=True)

# We should have 7 features and 1 target.
df['Gender'] = df['Gender'].apply(lambda x: 1 if (x == "Male") else 0)

"""
df.dtypes:
age                                   int64
Gender                                int64
total_bilirubin                     float64
alkaline_phosphotase                  int64
alamine_aminotransferase              int64
total_protiens                      float64
ratio_albumin_and_globulin_ratio    float64
LiverResult                             int64
dtype: object

df.shape:
(579, 8)
"""

#Convert Gender,Selector to numbericals
le = preprocessing.LabelEncoder()
# df['Gender'] = le.fit_transform(df.Gender)
df['LiverResult'] = le.fit_transform(df.LiverResult)

#Removing the Selector column from the list of the features
features = (list(df.columns[:-1]))

#Retrieving the features and Selector data into X & y
X = df[features]
y = df['LiverResult']

#Fitting the features
from sklearn.preprocessing import Imputer
X = Imputer().fit_transform(X)

#Split dataset to 60% training and 40% testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.4, random_state = 0)


# # ******************* Random Forest *******************
t2 = time()
print ("RandomForest")
rf = RandomForestClassifier(n_estimators = 15, n_jobs = 2)
# rf = RandomForestClassifier(n_jobs=2, random_state=RANDOM_SEED,
#                                criterion="entropy",
#                                max_features=5,
#                                n_estimators=15)
clf_rf = rf.fit(X_train, y_train)

y_pred = clf_rf.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
print ("Acurracy: ", clf_rf.score(X_test, y_test))
t3 = time()
print ("time elapsed: ", t3 - t2)
print("\nConfusion Matrix:\n", cnf_matrix)


# #Loading the SVM Classifier, Fitting the training sets to get the classification model,
# #Getting the prediction vetor 'y_pred' of the model, and Computing the Confusion Matrix
# svc = SVC(probability = True)
# clf_svc = svc.fit(X_train, y_train)
# y_pred = clf_svc.predict(X_test)
# cnf_matrix = confusion_matrix(y_test, y_pred)
#
# #Printing the results of our classifier
# print ("\nSVM")
# print ("\nAcurracy:\n\t ", clf_svc.score(X_test, y_test))
# print("\nConfusion Matrix:\n", cnf_matrix)