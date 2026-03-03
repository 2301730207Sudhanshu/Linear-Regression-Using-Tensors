# Linear-Regression-Using-Tensors
📌 Objective

Build a simple Linear Regression model using:

Tensors

Variables

GradientTape

Manual Gradient Descent

No high-level APIs.
No Keras.
Only pure TensorFlow basics.

This project is built with the mindset of “talking to the computer” — understanding what each line tells the machine and how the machine responds.

🧠 Problem Statement

We want the model to learn the equation:

y=3x+2

Instead of telling the computer the formula, we let it discover the values of:

Weight (W)

Bias (b)

🗣 Talking to the Computer – Line by Line Explanation
🔹 Import TensorFlow
import tensorflow as tf

🧠 You tell the computer:

Load the TensorFlow engine so I can use tensors, variables, and automatic differentiation.

💻 Computer:

TensorFlow tools loaded into memory.

🔹 Create Dataset
X = tf.constant([1., 2., 3., 4., 5.])
Y = tf.constant([5., 8., 11., 14., 17.])

🧠 You tell the computer:

Store input data in X.
Store output data in Y.
These values must NOT change.

💻 Computer stores:

X = [1 2 3 4 5]
Y = [5 8 11 14 17]

These are constant tensors.

🔹 Create Variables (What Will Learn)
W = tf.Variable(0.0)
b = tf.Variable(0.0)

🧠 You tell the computer:

Create a weight W.
Create a bias b.
Both start at 0.
These must be allowed to change.

💻 Computer:

W = 0
b = 0

These are trainable parameters.

🔹 Define Learning Rate
learning_rate = 0.01

🧠 You tell the computer:

When updating weights, move in small steps.
Don’t jump too much.

🔹 Training Loop
for epoch in range(1000):

🧠 You tell the computer:

Repeat learning process 1000 times.
Improve little by little.

🔹 Start Recording Operations
with tf.GradientTape() as tape:

🧠 You tell the computer:

Record all math operations.
I will later ask for derivatives.

💻 Computer enters recording mode.

🔹 Forward Pass (Prediction)
predictions = W * X + b

🧠 You tell the computer:

For every input X, calculate:
Prediction = W × X + b

Initially:

W = 0
b = 0

So predictions are:

[0, 0, 0, 0, 0]
🔹 Compute Loss
loss = tf.reduce_mean((predictions - Y) ** 2)

🧠 You tell the computer:

Subtract prediction from actual value.

Square the differences.

Take average.

💻 Computer computes one number:

This is total error.

🔹 Compute Gradients
grad_W, grad_b = tape.gradient(loss, [W, b])

🧠 You tell the computer:

Tell me how much W caused the error.
Tell me how much b caused the error.

💻 Computer:
Uses chain rule (backpropagation).
Returns:

∂Loss/∂W
∂Loss/∂b

These are slopes.

🔹 Update Variables
W.assign_sub(learning_rate * grad_W)
b.assign_sub(learning_rate * grad_b)

🧠 You tell the computer:

Move W in opposite direction of gradient.
Move b in opposite direction of gradient.

Formula:

New Parameter = Old Parameter − LearningRate × Gradient

This is gradient descent.

🔹 Final Output
print("Final W:", W.numpy())
print("Final b:", b.numpy())

🧠 You tell the computer:

Show me what you learned.

💻 Computer prints approximately:

Final W ≈ 3
Final b ≈ 2

The model discovered the hidden pattern.

🎯 What This Project Teaches

✅ What tensors are
✅ What variables are
✅ How gradients are computed
✅ How backpropagation works
✅ How gradient descent updates weights
✅ How learning actually happens

🧠 Deep Insight

Learning = Updating Variables.

Data does not learn.
Graph does not learn.
Loss does not learn.

Only weights change.

And that change is guided by gradients.
