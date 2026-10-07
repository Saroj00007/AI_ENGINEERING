# Phase 1 — Chapter 1: Representations

Since you already know linear algebra, we're going to skip the textbook definitions and focus on **how these mathematical objects are used by AI models**.

The core idea of this chapter is:

> **AI models need a numerical representation of information before they can perform mathematical operations on it.**

We'll build this in four steps:

```text
Information
    ↓
Numbers
    ↓
Vectors
    ↓
Operations on vectors
    ↓
Meaningful computation
```

## 1. Why does AI need numbers?

Suppose we give a model:

> `"This movie is excellent"`

A neural network cannot directly perform:

```text
"excellent" × 0.72
```

It needs a numerical representation.

For example, imagine we represent a sentence using three features:

```text
[positive, negative, excitement]
```

and assign numerical values:

```text
[0.9, 0.0, 0.8]
```

Now the model has something it can mathematically operate on.

This is the fundamental idea behind **representations**.

---

# 2. Scalar

A scalar is simply one number:

```text
5
0.72
-3.4
```

In AI, individual numbers can represent things such as:

```text
learning rate = 0.001
weight = 0.72
bias = -0.3
probability = 0.85
```

So scalars are the **individual numerical pieces** used throughout a model.

But one number isn't enough to represent something complex.

---

# 3. Vector

A vector is a collection of numbers:

```text
[0.2, 0.7, -0.4, 1.3]
```

You already know the mathematical definition.

The important AI question is:

> **What do those numbers represent?**

They can represent features of something.

For example, suppose we're representing a house:

```text
house = [size, bedrooms, age, distance_from_city]
```

Maybe:

```text
[2000, 3, 10, 5]
```

Now the house has a numerical representation.

A model can take this vector as input.

---

# 4. The important shift: features → dimensions

When we say:

```text
x = [2, 5, 1]
```

we can think of it as:

```text
        dimension
           ↓
x = [ 2,  5,  1 ]
      ↑   ↑   ↑
     f1  f2  f3
```

Each dimension can represent some information.

But here's where modern AI gets more interesting.

For traditional ML, humans often explicitly define the features.

For example:

```text
house → [size, bedrooms, age]
```

But with neural networks and especially LLMs, representations can be **learned**.

That's a very important concept.

---

# 5. Learned representations

Consider the word:

> `king`

An LLM doesn't store the concept of `king` as:

```text
king = male + royal + human + ...
```

Instead, words/tokens are represented using learned numerical vectors.

Conceptually:

```text
king → [0.21, -0.73, 0.14, ..., 0.52]
queen → [0.19, -0.70, 0.17, ..., 0.49]
apple → [-0.82, 0.31, ..., -0.12]
```

These numbers aren't manually assigned semantic labels.

They are **learned during training**.

And this is where our Phase 0 concept of parameters comes back.

The model learns numerical structures that allow useful relationships to emerge in these representations.

We'll go much deeper into this in **Phase 2 when we study embeddings and transformers**.

For now, remember:

> **A representation is a numerical form of information that a model can compute with.**

---

# 6. Why vectors are so important

Once information becomes a vector, we can perform mathematical operations on it.

For example:

```text
A = [1, 2, 3]
B = [4, 5, 6]
```

We can calculate:

### Addition

```text
A + B = [5, 7, 9]
```

### Scaling

```text
2A = [2, 4, 6]
```

### Dot product

```text
A · B = 1×4 + 2×5 + 3×6
      = 32
```

The interesting question is:

> **Why does AI care about the dot product?**

Because dot products are one of the fundamental operations used to combine **input representations with learned parameters**.

For example, a very simple model might calculate:

```text
y = w · x + b
```

where:

```text
x = input vector
w = learned weight vector
b = bias
```

This tiny equation is extremely important.

A huge amount of neural-network computation ultimately consists of variations of this kind of operation.

We'll unpack **exactly why** in Chapter 2.

---

# 7. Vector similarity

Vectors also let us ask:

> **How similar are these two representations?**

Suppose:

```text
A = [0.9, 0.8]
B = [0.8, 0.7]
C = [-0.9, -0.8]
```

A and B point in similar directions.

A and C point in opposite directions.

This gives us the concept of **vector similarity**.

One common measure is **cosine similarity**:

$$
\cos(\theta)=\frac{A\cdot B}{||A||||B||}
$$

You already know the formula.

What matters for AI is:

> **Cosine similarity measures how similarly two vectors are oriented, rather than simply comparing their raw magnitudes.**

This becomes extremely important later in:

* embeddings
* semantic search
* RAG
* vector databases
* retrieval

For example, conceptually:

```text
"How do I reset my password?"
             ↓
        embedding
             ↓
       [0.12, -0.4, ...]
```

and:

```text
"I forgot my password"
             ↓
        embedding
             ↓
       [0.11, -0.39, ...]
```

Their vectors may be close because their **meaning is similar**, even though the actual words differ.

That's the foundation behind semantic retrieval.

---

# 8. The big picture

So far:

```text
Real-world information
        ↓
Numerical representation
        ↓
      Vector
        ↓
 ┌──────┼────────┐
 ↓      ↓        ↓
Dot   Similarity  Matrix operations
product
 ↓
Model computation
```

And eventually:

```text
Input representation
        ↓
     weights
        ↓
  mathematical operations
        ↓
    prediction
        ↓
      loss
        ↓
 parameter updates
```

That brings us directly to **Chapter 2: Model Fundamentals**.

But before moving there, I want to make sure Chapter 1 is actually understood rather than just read.

### Checkpoint

Answer these in your own words:

1. **Why does an AI model need numerical representations?**
2. What does a vector represent in an AI system?
3. What's the difference between a vector's **mathematical definition** and its **meaning in an AI application**?
4. Why is the dot product useful inside a model?
5. Why can cosine similarity be useful for comparing embeddings?
6. What does it mean when we say a representation is **learned** rather than manually designed?

Don't worry about mathematical calculations. I'm interested in whether you understand **how the math is being used by AI**.
