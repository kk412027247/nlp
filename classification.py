from sklearn.datasets import fetch_openml
from datetime import datetime
from joblib import Memory
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

memory = Memory('./tmp')
fetch_openml_cached = memory.cache(fetch_openml)

mnist = fetch_openml_cached('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target
print(X)
print(X.shape)
print(y)
print(y.shape)


def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap='binary')
    plt.axis('off')


some_digit = X[0]
plot_digit(some_digit)
# plt.show()

print(y[0])

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
predict_r = sgd_clf.predict([some_digit])
print(predict_r)

cross_val_score_r = cross_val_score(
    sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
print(cross_val_score_r)

# dummy_clf = DummyClassifier()
# dummy_clf.fit(X_train, y_train_5)
# print(any(dummy_clf.predict(X_train)))
# cross_val_score_r = cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring='accuracy')
# print(cross_val_score_r)

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)

r = precision_score(y_train_5, y_train_pred)
print(r)

r = recall_score(y_train_5, y_train_pred)
print(r)

r = f1_score(y_train_5, y_train_pred)
print(r)

y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)

threshold = 0
y_some_digit_pred = (y_scores > threshold)
print("y_some_digit_pred", y_some_digit_pred)

threshold = 3000
y_some_digit_pred = (y_scores > threshold)
print("y_some_digit_pred", y_some_digit_pred)

y_scores = cross_val_predict(
    sgd_clf, X_train, y_train_5, cv=3, method='decision_function')

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

plt.figure(figsize=(8, 4))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")
plt.axis([-50000, 50000, 0, 1])
plt.grid()
plt.xlabel("Threshold")
plt.legend(loc="center right")
# plt.show()


plt.figure(figsize=(6, 5))
plt.plot(recalls, precisions, linewidth=2, label='Precision/Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc="lower left")
# plt.show()

idx_for_90_precision = (precisions >= 0.90).argmax()

threshold_for_90_precision = thresholds[idx_for_90_precision]

y_train_pred_90 = (y_scores >= threshold_for_90_precision)
# return [False False False ...  True False False]

r = precision_score(y_train_5, y_train_pred_90)
print('precision_score', r)

recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
print('recall_at_90_precision', recall_at_90_precision)

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()

tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
# plt.show()

r = roc_auc_score(y_train_5, y_scores)
print('roc_auc_score', r)

forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(
    forest_clf, X_train, y_train_5, cv=3, method='predict_proba')

print(y_probas_forest[:2])

y_scores_forest = y_probas_forest[:, 1]

precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
    y_train_5, y_scores_forest)


plt.figure(figsize=(5, 5))
plt.plot(recalls_forest, precisions_forest, 'b-',
         linewidth=2, label='Random Forest')

plt.plot(recalls, precisions, '--', linewidth=2, label='SGD')
plt.show()

y_train_pred_forest = y_probas_forest[:, 1] >= 0.5
r = f1_score(y_train_5, y_train_pred_forest)
print('forest_f1_score', r)

r = roc_auc_score(y_train_5, y_scores_forest)
print('forest_roc_auc_score', r)
