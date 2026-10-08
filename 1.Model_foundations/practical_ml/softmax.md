Great. Since you've already completed and had it reviewed, let's move on.

# 5.2 — Softmax: From Logits to Probabilities

You already learned the theory in Chapter 4. Now we're going to **implement it ourselves** so the transformation becomes concrete.

Suppose a model produces these logits:

```python
logits = np.array([2.0, 1.0, 0.1])
```

Remember:

> **Logits are raw scores, not probabilities.**

We want to convert them into a probability distribution.

---

## Step 1 — The basic softmax formula

For each logit:

$$
P_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

In simple terms:

```text
logits
  ↓
exponentiate each score
  ↓
divide by total
  ↓
probabilities
```

Let's implement it:

```python
import numpy as np

logits = np.array([2.0, 1.0, 0.1])

exp_logits = np.exp(logits)

probabilities = exp_logits / np.sum(exp_logits)

print(probabilities)
```

You'll get approximately:

```text
[0.659 0.242 0.099]
```

Notice:

$$
0.659 + 0.242 + 0.099 \approx 1
$$

So we've converted:

```text
[2.0, 1.0, 0.1]
```

into approximately:

```text
[65.9%, 24.2%, 9.9%]
```

---

# Why exponentiation?

You don't need to memorize the mathematical motivation deeply here.

The useful intuition is:

> Exponentiation makes differences between logits more pronounced while keeping the values positive.

For example:

```text
logit      exp(logit)

2.0   →      7.39
1.0   →      2.72
0.1   →      1.11
```

Then we normalize them.

This gives us a probability distribution.

---

# Step 2 — A very important numerical issue

The implementation above works for small numbers.

But imagine:

```python
logits = np.array([1000, 1001, 1002])
```

If we do:

```python
np.exp(logits)
```

we can run into numerical overflow.

This is a real engineering problem.

The standard solution is:

$$
\text{softmax}(z)
=
\text{softmax}(z-\max(z))
$$

So:

```python
logits = np.array([1000, 1001, 1002])

shifted = logits - np.max(logits)

exp_logits = np.exp(shifted)

probabilities = exp_logits / np.sum(exp_logits)

print(probabilities)
```

Why does this work?

Because subtracting the same value from every logit **doesn't change their relative differences**.

Original:

```text
[1000, 1001, 1002]
```

Shifted:

```text
[-2, -1, 0]
```

The relative differences are still:

```text
1000 → 1001 = +1
1001 → 1002 = +1
```

So the resulting probability distribution is the same, but the computation is much safer.

This is called **numerically stable softmax**.

---

# Step 3 — Put it into a function

Now let's write the version you'd actually reuse:

```python
def softmax(logits):
    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits)
```

Then:

```python
logits = np.array([2.0, 1.0, 0.1])

probs = softmax(logits)

print(probs)
```

---

# Step 4 — Connect this to an LLM

Suppose a tiny language model has three possible next tokens:

```text
Token       Logit
-----------------
"cat"        2.0
"dog"        1.0
"car"        0.1
```

The model produces:

```text
logits
[2.0, 1.0, 0.1]
       ↓
     softmax
       ↓
[0.659, 0.242, 0.099]
```

So:

```text
cat → 65.9%
dog → 24.2%
car →  9.9%
```

With **greedy decoding**, we'd choose:

```text
cat
```

because it has the highest probability.

With **sampling**, we could potentially select `"dog"` or even `"car"` according to the distribution.

Then that selected token becomes part of the context, and the model generates the next token.

So the larger pipeline is:

```text
Context
   ↓
Model
   ↓
Logits
   ↓
Softmax
   ↓
Probability distribution
   ↓
Sampling / Greedy selection
   ↓
Next token
   ↓
Updated context
   ↓
Repeat
```

You've now implemented an actual piece of the mechanism behind LLM generation.

---

# Your exercise

Implement this yourself:

```python
logits = np.array([3.0, 1.0, -1.0])
```

Then answer:

1. Which logit gets the highest probability?
2. Why don't logits themselves need to add up to 1?
3. Why do the softmax outputs add up to approximately 1?
4. Why do we subtract `np.max(logits)` before applying `np.exp()`?
5. If the model produces probabilities `[0.7, 0.2, 0.1]`, which token does **greedy decoding** select?
6. Can **sampling** select the token with probability `0.1`?

After this, we'll build the **tiny model + complete training loop**, which is the final major experiment of Chapter 5.
