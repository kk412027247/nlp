from sklearn.datasets import fetch_openml
from datetime import datetime
from joblib import Memory
import matplotlib.pyplot as plt

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
plt.show()

print(y[0])

X_train, X_test, y_train, y_test = X[:6000], X[6000:], y[:6000], y[6000:]
