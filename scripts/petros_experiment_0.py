import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

features_metadata_df = pd.read_csv("../metadata/merged_metadata_popularity_features.csv")
cols = ['log_views_norm', 'emotion_angry_mean',
       'emotion_angry_90p', 'emotion_happy_mean', 'emotion_happy_90p',
       'emotion_sad_mean', 'emotion_sad_90p', 'emotion_neutral_mean',
       'emotion_neutral_90p', 'strength_weak_mean', 'strength_weak_90p',
       'strength_neutral_mean', 'strength_neutral_90p', 'strength_strong_mean',
       'strength_strong_90p', 'positivity_negative_mean',
       'positivity_negative_90p', 'positivity_neutral_mean',
       'positivity_neutral_90p', 'positivity_positive_mean',
       'positivity_positive_90p', 'pauses_mean', 'pauses_10p', 'pauses_90p',
       'turn_durations_mean', 'turn_durations_10p', 'turn_durations_90p']


X = features_metadata_df.dropna()[cols].drop('log_views_norm', axis=1)  # Features
y = features_metadata_df.dropna()['log_views_norm']  # Target variable
y_cat = features_metadata_df.dropna()['log_views_norm_cat']

X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = train_test_split(X, y, y_cat, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

scores = cross_val_score(regressor, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validated MSE: {-scores.mean()}")

# classifier = RandomForestClassifier(random_state=42)
# classifier.fit(X_train, y_train_cat)


from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the labels to numerical values
y_cat_encoded = label_encoder.fit_transform(y_cat)
y_train_cat_encoded = label_encoder.fit_transform(y_train_cat)
y_test_cat_encoded = label_encoder.fit_transform(y_test_cat)

classifier = xgb.XGBClassifier(eval_metric='mlogloss')
classifier.fit(X_train, y_train_cat_encoded)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test_cat_encoded, y_pred)
f1 = f1_score(y_test_cat_encoded, y_pred, average="weighted")
print(f"Accuracy: {accuracy}, F1: {f1}")

scores = cross_val_score(classifier, X, y_cat_encoded, cv=5, scoring='f1_weighted')
print(f"Cross-validated F1: {scores.mean()}")




from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'y_test' contains the true class labels and 'y_pred' contains the predicted class labels
cm = confusion_matrix(y_test_cat_encoded, y_pred)

# Plotting the confusion matrix as a heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=True, yticklabels=True)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
