

# ========================================SVM=======================================

# import pandas as pd
# from sklearn.svm import SVC
# from sklearn.model_selection import cross_validate, StratifiedKFold

# # Load the combined features
# combined_df = pd.read_csv('E:\\2ndSemester\\MIA\\Project\\features\\combined_features.csv')

# # Separate features and labels
# X = combined_df.drop(columns=['Image', 'Is_Stroke'])
# y = combined_df['Is_Stroke']

# # Create a StratifiedKFold object
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# # Initialize the SVM classifier
# svm = SVC(kernel='linear', random_state=42)

# # Perform cross-validation and collect all relevant metrics
# scoring = ['accuracy', 'precision', 'recall', 'f1']
# cv_results = cross_validate(svm, X, y, cv=skf, scoring=scoring)

# # Print the cross-validation scores for each metric
# print(f"Cross-validation accuracy scores: {cv_results['test_accuracy']}")
# print(f"Maximum accuracy: {cv_results['test_accuracy'].max()}")

# print(f"\nCross-validation precision scores: {cv_results['test_precision']}")
# print(f"Maximum precision: {cv_results['test_precision'].max()}")

# print(f"\nCross-validation recall scores: {cv_results['test_recall']}")
# print(f"Maximum recall: {cv_results['test_recall'].max()}")

# print(f"\nCross-validation F1 scores: {cv_results['test_f1']}")
# print(f"Maximum F1-score: {cv_results['test_f1'].max()}")

# ========================================================================================

# ========================================Random Forest=======================================
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

# Load the combined features
combined_df = pd.read_csv('E:\\2ndSemester\\MIA\\Project\\features\\combined_features.csv')

# Separate features and labels
X = combined_df.drop(columns=['Image', 'Is_Stroke'])
y = combined_df['Is_Stroke']

# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Perform cross-validation and collect all relevant metrics
scoring = ['accuracy', 'precision', 'recall', 'f1']
cv_results = cross_validate(rf, X, y, cv=skf, scoring=scoring)

# Print the cross-validation scores for each metric
print(f"Cross-validation accuracy scores: {cv_results['test_accuracy']}")
print(f"Maximum accuracy: {cv_results['test_accuracy'].max()}")

print(f"\nCross-validation precision scores: {cv_results['test_precision']}")
print(f"Maximum precision: {cv_results['test_precision'].max()}")

print(f"\nCross-validation recall scores: {cv_results['test_recall']}")
print(f"Maximum recall: {cv_results['test_recall'].max()}")

print(f"\nCross-validation F1 scores: {cv_results['test_f1']}")
print(f"Maximum F1-score: {cv_results['test_f1'].max()}")

# ========================================================================================


# ========================================Decision Tree =======================================

# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import cross_validate, StratifiedKFold

# # Load the combined features
# combined_df = pd.read_csv('E:\\2ndSemester\\MIA\\Project\\features\\combined_features.csv')

# # Separate features and labels
# X = combined_df.drop(columns=['Image', 'Is_Stroke'])
# y = combined_df['Is_Stroke']

# # Create a StratifiedKFold object
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# # Initialize the Decision Tree classifier
# dt = DecisionTreeClassifier(random_state=42)

# # Perform cross-validation and collect all relevant metrics
# scoring = ['accuracy', 'precision', 'recall', 'f1']
# cv_results = cross_validate(dt, X, y, cv=skf, scoring=scoring)

# print('decision Tree')
# # Print the cross-validation scores for each metric
# print(f"Cross-validation accuracy scores: {cv_results['test_accuracy']}")
# print(f"Maximum accuracy: {cv_results['test_accuracy'].max()}")

# print(f"\nCross-validation precision scores: {cv_results['test_precision']}")
# print(f"Maximum precision: {cv_results['test_precision'].max()}")

# print(f"\nCross-validation recall scores: {cv_results['test_recall']}")
# print(f"Maximum recall: {cv_results['test_recall'].max()}")

# print(f"\nCross-validation F1 scores: {cv_results['test_f1']}")
# print(f"Maximum F1-score: {cv_results['test_f1'].max()}")
# ========================================================================================


# ========================================Logistic Regression =======================================

# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_validate, StratifiedKFold

# # Load the combined features
# combined_df = pd.read_csv('E:\\2ndSemester\\MIA\\Project\\features\\combined_features.csv')

# # Separate features and labels
# X = combined_df.drop(columns=['Image', 'Is_Stroke'])
# y = combined_df['Is_Stroke']

# # Create a StratifiedKFold object
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# # Initialize the Logistic Regression classifier
# lr = LogisticRegression(random_state=42, max_iter=10000)

# # Perform cross-validation and collect all relevant metrics
# scoring = ['accuracy', 'precision', 'recall', 'f1']
# cv_results = cross_validate(lr, X, y, cv=skf, scoring=scoring)

# # Print the cross-validation scores for each metric
# print('LogisticRegression')
# print(f"Cross-validation accuracy scores: {cv_results['test_accuracy']}")
# print(f"Maximum accuracy: {cv_results['test_accuracy'].max()}")

# print(f"\nCross-validation precision scores: {cv_results['test_precision']}")
# print(f"Maximum precision: {cv_results['test_precision'].max()}")

# print(f"\nCross-validation recall scores: {cv_results['test_recall']}")
# print(f"Maximum recall: {cv_results['test_recall'].max()}")

# print(f"\nCross-validation F1 scores: {cv_results['test_f1']}")
# print(f"Maximum F1-score: {cv_results['test_f1'].max()}")
# ========================================================================================


# ========================================KNN =======================================

# import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import cross_validate, StratifiedKFold

# # Load the combined features
# combined_df = pd.read_csv('E:\\2ndSemester\\MIA\\Project\\features\\combined_features.csv')

# # Separate features and labels
# X = combined_df.drop(columns=['Image', 'Is_Stroke'])
# y = combined_df['Is_Stroke']

# # Create a StratifiedKFold object
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# # Initialize the K-Nearest Neighbors classifier
# knn = KNeighborsClassifier()

# # Perform cross-validation and collect all relevant metrics
# scoring = ['accuracy', 'precision', 'recall', 'f1']
# cv_results = cross_validate(knn, X, y, cv=skf, scoring=scoring)

# # Print the cross-validation scores for each metric
# print(f"Cross-validation accuracy scores: {cv_results['test_accuracy']}")
# print(f"Maximum accuracy: {cv_results['test_accuracy'].max()}")

# print(f"\nCross-validation precision scores: {cv_results['test_precision']}")
# print(f"Maximum precision: {cv_results['test_precision'].max()}")

# print(f"\nCross-validation recall scores: {cv_results['test_recall']}")
# print(f"Maximum recall: {cv_results['test_recall'].max()}")

# print(f"\nCross-validation F1 scores: {cv_results['test_f1']}")
# print(f"Maximum F1-score: {cv_results['test_f1'].max()}")

