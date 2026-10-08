Absolutely. Let’s start **Chapter 5 — Practical ML Experiments**.

This is the final chapter of **Phase 1**. The goal is not to become a NumPy expert. We’re going to make the concepts from Chapters 1–4 actually **run**.

Our core loop will be:

> **Input → Prediction → Loss → Gradient → Parameter Update → Better Prediction**

### Chapter 5 roadmap

1. **Vector operations with NumPy**
2. **Vector similarity / cosine similarity**
3. **Softmax**
4. **Build a tiny model**
5. **Calculate loss**
6. **Calculate gradients**
7. **Update parameters**
8. **Run a complete training loop**
9. **Watch the model actually learn**

We'll use a tiny problem where the model has to learn:

$$
y = 2x + 1
$$

So instead of us giving the model `2` and `1`, we'll start with bad parameters and let training discover them.

---

# 5.1 — From mathematical notation to code

You've already seen:

$$
\hat y = wx + b
$$

Where:

* `x` → input
* `w` → weight/parameter
* `b` → bias/parameter
* `ŷ` → model prediction

Let's implement exactly that.

```python
import numpy as np

x = 5

w = 0.5
b = 0.2

y_hat = w * x + b

print(y_hat)
```

Output:

```text
2.7
```

The model predicted:

$$
\hat y = 0.5(5)+0.2=2.7
$$

But suppose the correct answer is:

```python
y = 11
```

because:

$$
y = 2(5)+1 = 11
$$

Our model is obviously wrong.

So now we have:

```text
x = 5
        ↓
   [ model ]
   w = 0.5
   b = 0.2
        ↓
prediction = 2.7

actual = 11
```

And this brings us to the next important part.

---

# 5.2 — Loss in code

We need to measure **how wrong** the model is.

We'll use squared error:

$$
L=(y-\hat y)^2
$$

```python
y = 11

loss = (y - y_hat) ** 2

print(loss)
```

We get:

$$
(11-2.7)^2 = 68.89
$$

So:

```text
Prediction = 2.7
Actual     = 11
Loss       = 68.89
```

The important idea is:

> **Loss converts "the model is wrong" into a numerical value that training can use.**

---

# 5.3 — Now the interesting part: changing the parameter

Remember what we learned earlier:

The model's prediction depends on its parameters.

$$
\hat y = wx+b
$$

Therefore, if we change `w` or `b`, the prediction changes.

For example:

```python
w = 1.5
b = 0.2

y_hat = w * x + b

print(y_hat)
```

Now:

$$
\hat y=1.5(5)+0.2=7.7
$$

That's much closer to 11.

So we have discovered something fundamental:

```text
parameters
    ↓
prediction
    ↓
loss
```

Training is essentially figuring out:

> **How should I change my parameters so that the loss becomes smaller?**

That's where the **gradient** enters.

---

# 5.4 — Gradient in practice

For our simple model:

$$
\hat y = wx+b
$$

and

$$
L=(y-\hat y)^2
$$

we can calculate how the loss changes when `w` changes.

For this particular example:

$$
\frac{\partial L}{\partial w}
=
-2x(y-\hat y)
$$

You don't need to memorize this formula.

What matters is what the result tells us.

Suppose:

```text
x = 5
y = 11
w = 0.5
b = 0.2
```

Prediction:

$$
\hat y=2.7
$$

Gradient with respect to `w`:

$$
\frac{\partial L}{\partial w}
=
-2(5)(11-2.7)
$$

$$
=-83
$$

So the gradient is:

```text
-83
```

That tells the optimizer:

> "Changing `w` in this direction can significantly reduce the loss."

Then gradient descent says:

$$
w_{new}=w-\eta\frac{\partial L}{\partial w}
$$

Suppose:

$$
\eta=0.01
$$

Then:

$$
w_{new}=0.5-(0.01)(-83)
$$

$$
w_{new}=1.33
$$

Look what happened:

```text
Before:
w = 0.5

Gradient:
-83

After update:
w = 1.33
```

The parameter moved substantially toward the value we actually need:

$$
w=2
$$

That's **learning**.

---

# 5.5 — Let's make NumPy do the training

Now we'll create a tiny dataset:

```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])
```

This dataset follows:

$$
y=2x+1
$$

Our model doesn't know that.

We'll start with:

```python
w = 0.0
b = 0.0
```

Then train.

```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

w = 0.0
b = 0.0

learning_rate = 0.01

for epoch in range(1000):

    # Forward pass
    y_hat = w * x + b

    # Loss
    loss = np.mean((y - y_hat) ** 2)

    # Gradients
    dw = np.mean(-2 * x * (y - y_hat))
    db = np.mean(-2 * (y - y_hat))

    # Parameter update
    w = w - learning_rate * dw
    b = b - learning_rate * db

print("w:", w)
print("b:", b)
print("loss:", loss)
```

After training, you'll see values approximately:

```text
w: 2
b: 1
loss: very close to 0
```

And that is our entire learning process.

---

# 5.6 — What just happened?

This tiny piece of code contains the fundamental structure of neural-network training.

```text
              ┌───────────────┐
x ───────────►│    MODEL      │
              │ ŷ = wx + b     │
              └───────┬───────┘
                      │
                      ▼
                  prediction
                      │
                      ▼
              ┌───────────────┐
y ───────────►│     LOSS      │
              └───────┬───────┘
                      │
                      ▼
                  gradients
                      │
                      ▼
              parameter update
                      │
                      ▼
                w and b change
                      │
                      ▼
                  repeat
```

And remember the distinction:

### During training

```text
parameters change
```

### During inference

```text
parameters are fixed
```

This connects directly back to **Phase 0**.

---

# 5.7 — The bigger picture

You've now seen the same concepts at three levels.

### Level 1 — Mathematical idea

$$
\hat y=wx+b
$$

### Level 2 — Training mechanism

$$
prediction
\rightarrow
loss
\rightarrow
gradient
\rightarrow
update
$$

### Level 3 — Actual code

```python
y_hat = w * x + b

loss = np.mean((y - y_hat) ** 2)

dw = ...
db = ...

w = w - learning_rate * dw
b = b - learning_rate * db
```

And this is the foundation underneath much larger neural networks.

A modern neural network may have **millions or billions of parameters**, but the fundamental idea remains:

> **Use the loss to determine how parameters should change, then repeatedly update them.**

---

## Your checkpoint

Before we move further in Chapter 5, explain these **without looking back**:

1. Why does changing `w` change the model's prediction?
2. Why can't we simply look at the loss and know how to update `w`?
3. What information does the gradient give us?
4. What does this line actually do?

```python
w = w - learning_rate * dw
```

5. In the training loop, what happens in this order?

```text
?
→
?
→
?
→
?
```

Answer in your own words. I'll review it like a senior engineer, then we'll continue with the remaining practical experiments.
