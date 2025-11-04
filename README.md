# 🤖 Logistic Regression from Scratch (Python/NumPy)

> A simple, custom implementation of the **Logistic Regression** algorithm built purely with Python and **NumPy**. This project focuses on understanding the core mechanics of binary classification, particularly the **Gradient Descent** optimization process with the **Sigmoid activation function**.

---

## 📁 Repository Structure

The core implementation is contained within the Jupyter Notebook:
* **`Logistic_Regression_Model.ipynb`**: Contains the `Logistic_Regression` class definition

---

## 🔑 The Model: `Logistic_Regression` Class

The model is defined by a class that encapsulates the initialization, training, and prediction logic for binary classification tasks.
```python
class Logistic_Regression:
    def __init__(self, learning_rate, no_of_iterations):
        # Initializes hyperparameters
        
    def fit(self, X, Y):
        # Initializes weights and runs the training loop
        
    def update_weights(self):
        # Performs a single step of Gradient Descent
        
    def predict(self, X):
        # Calculates predictions and applies threshold for binary classification
```

---

## ✨ Key Method Explanations

### 1. `fit(self, X, Y)`: Training the Model

The `fit` method orchestrates the training process:
1. **Initialization**: It first initializes the model's parameters—the weight vector (`self.w`) and the bias (`self.b`)—to zero.
2. **Shape Extraction**: Extracts the number of training examples (`m`) and number of features (`n`) from the input data.
3. **Iteration**: It then enters a loop defined by the `no_of_iterations` hyperparameter (epochs).
4. **Optimization**: In each iteration, it calls `update_weights()` to adjust the parameters based on the prediction error.

### 2. `update_weights()`: Gradient Descent Optimization

This is the core method for learning. It performs a single step of Gradient Descent to minimize the **Binary Cross-Entropy Loss Function**.

#### 🔄 Sigmoid Activation Function

Logistic Regression uses the **Sigmoid function** to transform the linear output into a probability between 0 and 1:
```
σ(z) = 1 / (1 + e^(-z))

where z = Xw + b
```

This ensures that predictions are probabilities, making it suitable for binary classification.

#### 📉 Partial Derivatives of the Binary Cross-Entropy Loss

To find the minimum cost, we calculate the gradient of the Binary Cross-Entropy loss function with respect to each parameter:

* **Partial Derivative with respect to Weights (`dw`)**:
```
  dw = (1/m) * X^T · (Ŷ - Y)
```

* **Partial Derivative with respect to Bias (`db`)**:
```
  db = (1/m) * Σ(Ŷ - Y)
```

(where `m` is the number of training examples, `Y` is the true label, and `Ŷ` is the predicted probability.)

#### 🎯 Parameter Update Rule

Once the gradients (`dw` and `db`) are calculated, the parameters are updated by moving in the opposite direction (descent):
```
w = w - learning_rate · dw
b = b - learning_rate · db
```

### 3. `predict(self, X)`: Making Predictions

This method uses the final, optimized weights (`self.w`) and bias (`self.b`) to:
1. Calculate the predicted probability (`Ŷ`) for new input data (`X`) using the Sigmoid function.
2. Apply a **threshold of 0.5** to convert probabilities into binary predictions (0 or 1).
```
Ŷ = σ(Xw + b)

Final Prediction = 1 if Ŷ > 0.5, else 0
```

**Important Note**: During training (`update_weights()`), we use continuous probability values for precise gradient calculations. During prediction, we convert these probabilities to discrete class labels (0 or 1) for final classification output.

---

## 🚀 Example Usage

The notebook demonstrates initializing and training the model with specific hyperparameters:
```python
# Initialize the model with a learning rate and number of iterations
model = Logistic_Regression(learning_rate=0.01, no_of_iterations=1000)

# Train the model
model.fit(X_train, Y_train)

# Make predictions
predictions = model.predict(X_test)

# Predictions will be binary: 0 or 1
```

---

## 🧑‍💻 Setup and Run

To run this implementation locally, follow these steps:

1. **Clone the repository**:
```bash
   git clone https://github.com/PranavbalajiGit/Logistic-Regression-From-Scratch.git
   cd Logistic-Regression-From-Scratch
```

2. **Install dependencies** (NumPy is the only required library):
```bash
   pip install numpy pandas matplotlib scikit-learn
```

3. **Open the notebook**: Open `Logistic_Regression_Model.ipynb` in a Jupyter environment (like VS Code or Jupyter Notebook/Lab) and execute the cells.

---

## 📊 Features

- ✅ Pure NumPy implementation (no scikit-learn for the model)
- ✅ Sigmoid activation function for binary classification
- ✅ Binary Cross-Entropy loss optimization
- ✅ Gradient Descent with configurable learning rate
- ✅ Threshold-based prediction (0.5 cutoff)
- ✅ Educational focus on understanding classification fundamentals

---

## 🔍 Key Concepts

### Binary Classification
Logistic Regression is used for problems where the output belongs to one of two classes (0 or 1, Yes or No, True or False).

### Sigmoid Function
Transforms any real-valued number into a value between 0 and 1, which can be interpreted as a probability.

### Binary Cross-Entropy Loss
The loss function used in Logistic Regression that measures the difference between predicted probabilities and actual labels.

### Decision Boundary
The model learns a linear decision boundary that separates the two classes in the feature space.

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/PranavbalajiGit/Logistic-Regression-From-Scratch/issues).

---

## 👤 Author

**PRANAV BALAJI P MA**
- GitHub: [@PranavbalajiGit](https://github.com/PranavbalajiGit)

---

**⭐ Star this repo if you find it helpful!**