# Import Libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("=" * 50)
print("      IRIS FLOWER CLASSIFICATION SYSTEM")
print("=" * 50)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# Create Model
model = KNeighborsClassifier(n_neighbors=3)

# Train Model
model.fit(X_train, y_train)

print("Model Status     : Trained Successfully")

# Make Predictions
predictions = model.predict(X_test)

print("\nModel Predictions:")
print(predictions)

print("\nActual Values:")
print(y_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy   : {accuracy * 100:.2f}%")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Dataset Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 2], c=y, cmap="viridis")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Iris Dataset Visualization")
plt.colorbar(label="Flower Class")
plt.show()

# User Input
print("\nEnter Flower Measurements")

sl = float(input("Sepal Length : "))
sw = float(input("Sepal Width  : "))
pl = float(input("Petal Length : "))
pw = float(input("Petal Width  : "))

prediction = model.predict([[sl, sw, pl, pw]])

print("\n" + "=" * 50)
print("            PREDICTION RESULT")
print("=" * 50)
print("Predicted Flower :", iris.target_names[prediction][0].capitalize())
print("=" * 50)