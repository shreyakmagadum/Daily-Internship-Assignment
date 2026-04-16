import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

st.title("📊 Logistic Regression Visualizer")

# ============================
# PART 1: DATA EXPLORATION
# ============================
st.header("Part 1: Data Exploration")

# Create simple dataset
np.random.seed(42)
X = np.linspace(-10, 10, 100)
y = (X > 0).astype(int)   # binary output

st.write("Dataset created with 1 feature and binary output")

fig1, ax1 = plt.subplots()
ax1.scatter(X, y)
ax1.set_title("Dataset Distribution")
st.pyplot(fig1)


# ============================
# PART 2: LINEAR MODEL LIMITATION
# ============================
st.header("Part 2: Linear Model Limitation")

# Linear prediction
y_linear = 0.1 * X + 0.5

fig2, ax2 = plt.subplots()
ax2.scatter(X, y, label="Actual")
ax2.plot(X, y_linear, color='red', label="Linear Model")
ax2.legend()
ax2.set_title("Linear Model Limitation")
st.pyplot(fig2)

st.write("Linear model outputs are not valid probabilities")


# ============================
# PART 3: SIGMOID FUNCTION
# ============================
st.header("Part 3: Sigmoid Function")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
sig = sigmoid(z)

fig3, ax3 = plt.subplots()
ax3.plot(z, sig)
ax3.set_title("Sigmoid Curve")
st.pyplot(fig3)


# ============================
# PART 4: LOGISTIC MODEL
# ============================
st.header("Part 4: Logistic Model")

w = 1
b = 0

logits = w * X + b
probs = sigmoid(logits)

fig4, ax4 = plt.subplots()
ax4.scatter(X, y, label="Actual")
ax4.plot(X, probs, color='green', label="Logistic Curve")
ax4.legend()
st.pyplot(fig4)


# ============================
# PART 5: MODEL TRAINING
# ============================
st.header("Part 5: Model Training")

lr = st.slider("Learning Rate", 0.01, 1.0, 0.1)
epochs = st.slider("Iterations", 10, 200, 50)

w, b = 0, 0

losses = []

for i in range(epochs):
    logits = w * X + b
    preds = sigmoid(logits)

    loss = -np.mean(y*np.log(preds+1e-9) + (1-y)*np.log(1-preds+1e-9))
    losses.append(loss)

    dw = np.mean((preds - y) * X)
    db = np.mean(preds - y)

    w -= lr * dw
    b -= lr * db

fig5, ax5 = plt.subplots()
ax5.plot(losses)
ax5.set_title("Loss vs Iterations")
st.pyplot(fig5)


# ============================
# PART 6: DECISION BOUNDARY
# ============================
st.header("Part 6: Decision Boundary")

threshold = 0.5
boundary = -b / w

st.write(f"Decision Boundary at X = {boundary:.2f}")

fig6, ax6 = plt.subplots()
ax6.scatter(X, y)
ax6.axvline(boundary, color='red')
ax6.set_title("Decision Boundary")
st.pyplot(fig6)


# ============================
# PART 7: PREDICTIONS
# ============================
st.header("Part 7: Predictions")

preds_final = sigmoid(w * X + b)
classes = (preds_final >= threshold).astype(int)

accuracy = np.mean(classes == y)

st.write(f"Accuracy: {accuracy:.2f}")

fig7, ax7 = plt.subplots()
ax7.scatter(X, y, label="Actual")
ax7.scatter(X, classes, marker='x', label="Predicted")
ax7.legend()
st.pyplot(fig7)


# ============================
# PART 8: LIBRARY COMPARISON
# ============================
st.header("Part 8: Library Comparison")

model = LogisticRegression()
model.fit(X.reshape(-1,1), y)

sk_preds = model.predict(X.reshape(-1,1))

sk_acc = np.mean(sk_preds == y)

st.write(f"Sklearn Accuracy: {sk_acc:.2f}")

fig8, ax8 = plt.subplots()
ax8.scatter(X, y, label="Actual")
ax8.scatter(X, sk_preds, marker='x', label="Sklearn")
ax8.legend()
st.pyplot(fig8)
