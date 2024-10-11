# Re-import necessary libraries and load the data again
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample

# Load the previously loaded Excel file
file_path = 'newdata/data.xlsx'
data = pd.read_excel(file_path)

# Define features (X) and target (Y)
X = data[['SPIC', 'SPWS', 'LSSSE', 'Insolvent', 'LIC', 'BPI', 'Size', 'EM1', 'DER']]
Y = data['Y']

# Address class imbalance using upsampling
# Combine X and Y for resampling
combined_data = pd.concat([X, Y], axis=1)
majority_class = combined_data[combined_data['Y'] == combined_data['Y'].mode()[0]]
minority_classes = [combined_data[combined_data['Y'] == cls] for cls in combined_data['Y'].unique() if cls != combined_data['Y'].mode()[0]]

# Upsample minority classes to match the size of the majority class
upsampled_minority_classes = [
    resample(minority, replace=True, n_samples=len(majority_class), random_state=42)
    for minority in minority_classes
]

# Combine majority class with upsampled minority classes
balanced_data = pd.concat([majority_class] + upsampled_minority_classes)

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split back into features and target
X = balanced_data[['SPIC', 'SPWS', 'LSSSE', 'Insolvent', 'LIC', 'BPI', 'Size', 'EM1', 'DER']]
Y = balanced_data['Y']

# Split the data into 70% training and 30% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42, ccp_alpha=0.01)  # ccp_alpha is used for pruning
clf.fit(X_train, Y_train)

# Predict on the test set
Y_pred = clf.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy on test set:", accuracy)
print("Classification Report:")
print(classification_report(Y_test, Y_pred))

# Cross-validation to evaluate the model
cv_scores = cross_val_score(clf, X, Y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))

# Plot the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=True)
plt.title("Decision Tree with Pruning")
plt.show()