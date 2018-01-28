import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

# Core Requirement 1: Download the census dataset. Install Pandas and use it to read in the data.
# Set the column names to something descriptive/appropriate.
# Ensure that everything is working by printing out some summaries of the data.

headers = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
          "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                header=None, names=headers, na_values='?')

print(df.describe())
print()
print(df.head(15))

# Core Requirement 2: Appropriately handle the missing data.
df_nan = df.replace('[?]', np.nan, regex=True)
print()
print(df_nan.head(15))

# After replacing '?' values with NaN, we count how many rows contain NaN values
# to see if removing all rows containing NaNs is viable.
count_nan_rows = 0
for num_nans in df_nan.isnull().sum(axis=1).tolist():
   if num_nans > 0:
       count_nan_rows += 1

num_rows_df_nan = df_nan.shape[0]

print()
print("Percentage of rows containing at least 1 NaN value: %" + str(round(count_nan_rows / num_rows_df_nan * 100, 2)))

df_nan_deleted = df_nan.dropna(axis='rows')
print()
print("Deleted " + str(num_rows_df_nan - df_nan_deleted.shape[0]) + " rows from df_nan.")
print()
print(df_nan_deleted.head(15))

# Core Requirement 3: Convert the categorical attributes into some form of numeric attributes.
df_nan_deleted_cat = \
   pd.DataFrame(
       {col: df_nan_deleted[col].astype('category').cat.codes for col in df_nan_deleted}, index=df_nan_deleted.index
   )
print(df_nan_deleted_cat.head(15))

# Stretch Challenge 1: Normalize the numeric attributes using z-Score normalization.
std_scale = preprocessing.StandardScaler().fit(df_nan_deleted_cat[headers])
df_nan_deleted_cat_std_numpyarray = std_scale.transform(df_nan_deleted_cat[headers])

# Stretch Challenge 2: Convert the dataset into a NumPy array, and use an sklearn classifier to classify it.
# Conversion to NumPy array done after normalizing with preprocessing module.
print(type(df_nan_deleted_cat_std_numpyarray))
print(df_nan_deleted_cat_std_numpyarray.shape)

# Target class, income, at column # 6
y = df_nan_deleted_cat_std_numpyarray[ : , 6:7]

# Data is all else
X_left = df_nan_deleted_cat_std_numpyarray[ : , :6]
X_right = df_nan_deleted_cat_std_numpyarray[ : , 7:]
X = np.append(X_left, X_right, 1)

print(y.shape)
print(X.shape)

lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{'alpha': alphas}]

# Stretch Challenge 3: Use 10-fold cross validation to verify the accuracy of your predictions.
n_folds = 10

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
clf.fit(X, y)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']
plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])

plt.show()
