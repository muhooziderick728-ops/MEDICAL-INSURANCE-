
print("#PART A: DATA ACQUISITION")

print("#Importing the required libraries")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from webencodings import Encoding
from IPython.display import display
print("#Loading the dataset into a pandas DataFrame")
df = pd.read_csv("C:\\Users\\HP\\Desktop\\medic\\Medical Cost Personal Dataset.csv")

print("#Displaying the first 5 rows to understand the data structure")
display(df.head())

print("#Showing number of rows and columns in the dataset / observations and variales.")
display(df.shape)

print("#Displaying the information about data types and non-null values")
display(df.info())

print("#Show summary statistics for numerical variables")
display(df.describe())

print("#PART B: DATA CLEANING")

print("#Check for missing values in each column")
display(df.isnull().sum())
print("#Check number of duplicate records")
display(df.duplicated().sum())

print("#Remove duplicate rows from dataset")
df = df.drop_duplicates()

print("#Confirm new shape after removing duplicates")
display(df.shape)

print("#Outlier Detection (IQR Method)")
print("#Define function to detect outliers using IQR method")
def detect_outliers(column):
    Q1 = df[column].quantile(0.25)   # First quartile
    Q3 = df[column].quantile(0.75)   # Third quartile
    IQR = Q3 - Q1                    # Interquartile range
    lower = Q1 - 1.5 * IQR           # Lower bound
    upper = Q3 + 1.5 * IQR           # Upper bound
    return df[(df[column] < lower) | (df[column] > upper)]

print("#Detect outliers in BMI")
print(detect_outliers("bmi"))

print("#Detect outliers in Charges")
print(detect_outliers("charges"))

print("#Encoding Categorical Variables")
print("#Convertpd.Categoricall variables into dummy variables")
df_encoded = pd.get_dummies(df, drop_first=True)

print("#Display first few rows after encoding")
print(df_encoded.head())

print("#PART C: EXPLORATORY DATA ANALYSIS")

print("#Plot histogram to see distribution of medical charges")
plt.hist(df['charges'], bins=30)
plt.title("Distribution of Charges")
plt.xlabel("Charges")
plt.ylabel("Frequency")
plt.show()


print("#Scatter plot to examine relationship between age and charges")
plt.scatter(df['age'], df['charges'])
plt.title("Age vs Charges")
plt.xlabel("Age")
plt.ylabel("Charges")
plt.show()


print("Scatter plot to examine relationship between BMI and charges")
plt.scatter(df['bmi'], df['charges'])
plt.title("BMI vs Charges")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.show()


print("#Boxplot to compare charges between smokers and non-smokers")
sns.boxplot(x=df['smoker'], y=df['charges'])
plt.title("Charges by Smoker Status")
plt.show()
print("correlation matrix as heatmap")
sns.heatmap(df_encoded.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()


print("#PART D: REGRESSION MODELING")
print("#Import regression libraries")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm


print("#PART D: REGRESSION MODELING")


import statsmodels.api as sm

print("#Make sure all encoded columns are numeric")
df_encoded = df_encoded.astype(float)


print("#SIMPLE LINEAR REGRESSION")


X_simple = df[['age']].astype(float)
y_simple = df['charges'].astype(float)

X_simple = sm.add_constant(X_simple)

model_simple = sm.OLS(y_simple, X_simple).fit()

print(model_simple.summary())



print("#MULTIPLE LINEAR REGRESSION")


print("#Define predictors")
X = df_encoded.drop('charges', axis=1)

print("#Define target")
y = df_encoded['charges']

print("#Convert everything to float (IMPORTANT STEP)")
X = X.astype(float)
y = y.astype(float)

print("#Add constant")
X = sm.add_constant(X)

print("#Fit model")
model = sm.OLS(y, X).fit()

print(model.summary())


print("#PART E: MODEL EVALUATION")


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("#Split data")
X_train, X_test, y_train, y_test = train_test_split(
    df_encoded.drop('charges', axis=1).astype(float),
    df_encoded['charges'].astype(float),
    test_size=0.2,
    random_state=42
)

print("#Create model")
lr = LinearRegression()

print("#Train model")
lr.fit(X_train, y_train)

print("#Predict")
y_pred = lr.predict(X_test)

print("#Evaluation metrics")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("MAE:", mae)
print("R²:", r2)

print("#Calculate residuals (actual - predicted)")
residuals = y_test - y_pred

print("#Plot residuals to check homoscedasticity")
plt.scatter(y_pred, residuals)
plt.axhline(y=0)
plt.title("Residual Plot")
plt.xlabel("Predicted Charges")
plt.ylabel("Residuals")
plt.show()

print("#Normality of Residuals")
print("#Plot histogram of residuals to check normality")
plt.hist(residuals, bins=30)
plt.title("Histogram of Residuals")
plt.show()

print("#Multicollinearity (VIF)")
print("#Import VIF function")
from statsmodels.stats.outliers_influence import variance_inflation_factor

print("#Add constant to predictors")
X_vif = sm.add_constant(df_encoded.drop('charges', axis=1))

print("#Create dataframe to store VIF values")
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns

print("#Calculate VIF for each feature")
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i)
                   for i in range(X_vif.shape[1])]
print("#Display VIF results")
print(vif_data)
