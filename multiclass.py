from sklearn.svm import SVC
from joblib import Memory
from sklearn.datasets import fetch_openml
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

memory = Memory('./tmp')
fetch_openml_cached = memory.cache(fetch_openml)

mnist = fetch_openml_cached('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target
print(X)
print(X.shape)
print(y)
print(y.shape)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])

some_digit = X[0]

r = svm_clf.predict([some_digit])

print(r)

some_digit_scores = svm_clf.decision_function([some_digit])
r = some_digit_scores.round(2)
print(r)
class_id = some_digit_scores.argmax()
print(class_id)

print(svm_clf.classes_)
print(svm_clf.classes_[class_id])

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])
print(ovr_clf.predict([some_digit]))
print(len(ovr_clf.estimators_))

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
r = sgd_clf.predict([some_digit])
print(r)
r = sgd_clf.decision_function([some_digit]).round()
print('sgd_clf', r)

r = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy')
print(r)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train.astype('float64'))
r = cross_val_score(sgd_clf, x_train_scaled, y_train, cv=3, scoring='accuracy')
print(r)

y_train_pred = cross_val_predict(sgd_clf, x_train_scaled, y_train, cv=3)

# ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
# plt.show()

# ConfusionMatrixDisplay.from_predictions(
#     y_train, y_train_pred, normalize='true', values_format='.0%')
# plt.show()

# sample_weight = (y_train_pred != y_train)
# ConfusionMatrixDisplay.from_predictions(
#     y_train, y_train_pred, sample_weight=sample_weight, normalize='true', values_format='.0%')
# plt.show()

# ConfusionMatrixDisplay.from_predictions(
#     y_train, y_train_pred, sample_weight=sample_weight, normalize='pred', values_format='.0%')
# plt.show()

cl_a, cl_b = '3', '5'
X_aa =X_train[(y_train == cl_a)&(y_train_pred == cl_a)]
X_ab =X_train[(y_train == cl_a)&(y_train_pred == cl_b)]
X_ba =X_train[(y_train == cl_b)&(y_train_pred == cl_a)]
X_bb =X_train[(y_train == cl_b)&(y_train_pred == cl_b)]


size = 5
pad = 0.2
plt.figure(figsize=(size, size))
for images, (label_col, label_row) in [(X_ba, (0, 0)), (X_bb, (1, 0)),
                                       (X_aa, (0, 1)), (X_ab, (1, 1))]:
    for idx, image_data in enumerate(images[:size*size]):
        x = idx % size + label_col * (size + pad)
        y = idx // size + label_row * (size + pad)
        plt.imshow(image_data.reshape(28, 28), cmap="binary",
                   extent=(x, x + 1, y, y + 1))
plt.xticks([size / 2, size + pad + size / 2], [str(cl_a), str(cl_b)])
plt.yticks([size / 2, size + pad + size / 2], [str(cl_b), str(cl_a)])
plt.plot([size + pad / 2, size + pad / 2], [0, 2 * size + pad], "k:")
plt.plot([0, 2 * size + pad], [size + pad / 2, size + pad / 2], "k:")
plt.axis([0, 2 * size + pad, 0, 2 * size + pad])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()