# Importing necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns



# Load the Iris dataset

iris = load_iris()

X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating and fitting the model
model = LogisticRegression(max_iter=1000) # Using Logistic regression for  classification
model.fit(X_train, y_train)


# Making predictions on the test set

predictions = model.predict(X_test)

#Calculating accuracy
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')
print("Accuracy: ", accuracy)
print("precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels = iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# Cross-validation

cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print("Cross-Validation Mean Accuracy: ", np.mean(cv_scores))

# Hyperparameter tuning with grid search

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_
print("Best Hyperparameters: ", best_params)


#Sample data
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2,4,5,4,5])

#Creating and fitting the model
model1 = LinearRegression()
model1.fit(X, y)

# Predicting values
X_test = np.array([[6],[7]])
predictions = model1.predict(X_test)

# Printing predictions
print("Predictions: ", predictions)

# Save the trained model
import joblib
joblib.dump(model, 'iris_model.pkl')
print("Model dave successfully!")