#Credit Card Fraud Detection model using Scikit Learn

import pandas as pd #for data manipulation and dataframe creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #scales features to have a mean 0 and standard deviation of 1
from sklearn.ensemble import RandomForestClassifier #for binary classification
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#Dataset- https://www.kaggle.com/datasets/ahmedjawed/creditscardfrauddataset
#Load the datset
df = pd.read_csv('creditcard.csv')

#Display the first few rows of the dataset
print("Dataset Sample:\n", df.head())

#Separate features (X) and target (y)
X= df.drop('Class', axis=1) #'Class' column is the target, with 0 for non-fraud and 1 for fraud; axis=1 means dropping a column
y= df['Class']

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify= y) #Stratify used to ensure both classes are proportionally represented in training and test sets; random_state sets a random seed for reproducibility

#Scale the features for numerical stability
scaler = StandardScaler() #standardizes features to improve numerical stability
X_train = scaler.fit_transform(X_train) #fits the scaler on X_train and transforms it
X_test = scaler.transform(X_test) #transforms X_test using scaling parameters from X_train

#Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42) #initializes classifier with 100 decision trees

#Train the model on training data
model.fit(X_train, y_train)

#Make predictions on the test set
y_pred = model.predict(X_test) #uses the trained model to predict labels for X_test

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred) #provides precision, recall, and f1 score for each class fraud and non-fraud
conf_matrix = confusion_matrix(y_test, y_pred) #shows true and false positives and negatives

print(f"\nAccuracy: {accuracy}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)