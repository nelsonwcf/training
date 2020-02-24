# -*- coding: utf-8 -*-
"""Chapter 2"""

import hashlib
import os
import ssl
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from six.moves import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

ssl._create_default_https_context = ssl._create_unverified_context

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """Download data for the project"""
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    """Load the housing data into pandas"""
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def split_train_test(data, test_ratio, seed=42):
    """Splits the data into train and test using the test_ratio"""
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(identifier, test_ratio, hash_):
    """Check if the current identifier is in the test or the train set"""
    return hash_(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash_=hashlib.md5):
    """Split using the hashing and some identifier"""
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash_))
    return data.loc[~in_test_set], data.loc[in_test_set]

fetch_housing_data()

housing = load_housing_data()

print(housing.head(), '\n')
print(housing.columns, '\n')
print(housing.info(), '\n')
print(housing["ocean_proximity"].value_counts(), '\n')
print(housing.describe(), '\n')

## dataframe.hist() historgram of all numericals
## check how to select matplotlib backend!
# housing.hist(bins=50, figsize=(20, 15))

## Regular split of test and train dataset
train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")

## Split using unique identifiers
housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

## Split using unique identifiers
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

## Using scikit train/test splitter
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

## discretizing median income
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

## Splitting the data by strata
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

## Checking if the proportions are correct in the three groups
print(housing["income_cat"].value_counts()/len(housing))
print(strat_train_set["income_cat"].value_counts()/len(strat_train_set))
print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))

## Removing the strata category from the data
for set_ in (strat_train_set, strat_test_set):
    set_.drop(["income_cat"], axis=1, inplace=True)

## with the sets created, replace original hosing with train set
housing = strat_train_set.copy()

## plotting directly with pandas
housing.plot(kind="scatter",
             x="longitude",
             y="latitude",
             alpha=0.4,
             s=housing["population"]/100,
             label="population",
             c="median_house_value",
             cmap=plt.get_cmap("jet"),
             colorbar=True
             )
plt.legend()

## getting correlation using only pandas
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

## plotting the SPLOM with pandas
attributes = ["median_house_value",
              "median_income",
              "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

# scatterplot using pandas again
housing.plot(kind="scatter",
             x="median_income",
             y="median_house_value",
             alpha=0.1)

## adding some attribute combinations
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

## check the new correlation_matrix
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
attributes = ["median_house_value",
              "median_income",
              "housing_median_age",
              "population_per_household",
              "bedrooms_per_room",
              "rooms_per_household"]
scatter_matrix(housing[attributes], figsize=(12, 8))

## Restore data and separate labels from regressors
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy() # copy not point

## Data Cleaning using Scikit tools
# housing.dropna(subset=["total_bedrooms"]) # drop rows based on N/As on column
# housing.drop("total_bedrooms", axis = 1) # drop the entire column
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median)

## Creating and training an SimpleInputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
print(imputer.statistics_, "\n", housing_num.median().values)

## Replacing all the values using the SimpleImputer
X = imputer.transform(housing_num)

## Converting it back to pandas DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

## Handling Text and Categorical Variables
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
print(housing_cat_encoded)
print(encoder.classes_)

## Alternative: convert to binary variables
encoder = OneHotEncoder(categories='auto')
housing_cat_1hot = encoder.fit_transform(np.array(housing_cat).reshape(-1, 1))
print(housing_cat_1hot)

encoder = LabelBinarizer(sparse_output=True) # default is dense matrix
housing_cat_1hot = encoder.fit_transform(housing_cat)
print(housing_cat_1hot)

## creating a custom transformer (python - duck typing)
## column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """Helper Class"""
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        """Constructor"""
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        """Trivial fit"""
        return self  # nothing else to do
    def transform(self, X, y=None):
        """Combine attributes"""
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

## testing basic scalers
bedrooms = np.array(housing_tr['total_rooms']).reshape(-1, 1)
scaler = MinMaxScaler()
bedrooms = scaler.fit_transform(bedrooms)

## this section is imported directly from github
## book is inconsistent/incorrect in pipeline usages
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"])
housing_extra_attribs.head()

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

class OldDataFrameSelector(BaseEstimator, TransformerMixin):
    """For Compatibility"""
    def __init__(self, attribute_names):
        """Constructor"""
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        """Trivial fit"""
        return self
    def transform(self, X):
        """Trivial transform"""
        return X[self.attribute_names].values

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

old_num_pipeline = Pipeline([
        ('selector', OldDataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

old_cat_pipeline = Pipeline([
        ('selector', OldDataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])

old_full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", old_num_pipeline),
        ("cat_pipeline", old_cat_pipeline),
    ])

old_housing_prepared = old_full_pipeline.fit_transform(housing)
print(old_housing_prepared)
np.allclose(housing_prepared, old_housing_prepared)

## using a simple Linear Regression with scikit
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:\t", lin_reg.predict(some_data_prepared))
print("Labels:\t\t", list(some_labels))

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

## using a decision tree instead
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_mse)

## cross-validating the data
## K-FOLD cross-validation
def display_scores(scores):
    print("Scores:", scores.round(0))
    print("Mean:", round(scores.mean(), 0))
    print("Standard deviation:", round(scores.std(), 0))

## using decision tree
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error",
                         cv=10)
rmse_scores = np.sqrt(-scores)
display_scores(rmse_scores)

## using linear regression
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

## using random forest regressor
forest_reg = RandomForestRegressor(max_features=6, n_estimators=30)
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

## Doing Grid-Search with Sci-Kit
param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
              {'bootstrap':[False], 'n_estimators':[3, 10],
               'max_features': [2, 3, 4]},
              ]
grid_search = GridSearchCV(forest_reg,
                           param_grid,
                           cv=5,
                           scoring='neg_mean_squared_error',
                           refit=True)
grid_search.fit(housing_prepared, housing_labels)
forest_reg = grid_search.best_estimator_
cvres = grid_search.cv_results_

## Grid-search scores - alternative is to use RandomizedSearchCV instead
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

## Comparing importance of features
feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

## Test set evaluation
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)
