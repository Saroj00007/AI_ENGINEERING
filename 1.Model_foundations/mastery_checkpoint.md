Phase 1 — Mastery Checkpoint
Practical ML Foundations

You should be able to explain the complete chain:

$$ \boxed{ \text{Representation} \rightarrow \text{Model} \rightarrow \text{Prediction} \rightarrow \text{Loss} \rightarrow \text{Gradient} \rightarrow \text{Parameter Update} \rightarrow \text{Learning} } $$
Part 1 — Representations

1. Why do machine-learning models need data represented as numbers?

2. What is the difference between a vector and a scalar?

3. What does a dot product do, and why is it useful inside a neural network?

4. What does cosine similarity tell us about two vectors?

Part 2 — Model Fundamentals

Consider:

$$ \hat y = wx+b $$

5. What are w and b?

6. What is the difference between a parameter and a hyperparameter?

7. What does the loss function tell us?

8. Why isn't the loss alone enough to update the parameters?

9. What information does the gradient provide?

10. Explain:

$$ w_{new}=w-\eta\frac{\partial L}{\partial w} $$

in plain English.

Part 3 — Neural Networks

11. What does a single neuron calculate?

12. Why do we need activation functions?

13. What is a forward pass?

14. What is backpropagation actually doing?

15. Explain why a neural network with many layers can learn more complex patterns than a single linear transformation.

Part 4 — Probability and LLMs

Suppose an LLM produces:

Token A → logit 5.2
Token B → logit 3.1
Token C → logit 1.4

16. What are these logits?

17. Why can't we directly interpret the logits as probabilities?

18. What does softmax do?

19. After softmax, suppose we get:

A → 0.70
B → 0.20
C → 0.10

What would greedy decoding select?

Could sampling select C?

20. What does temperature change during generation? Does it change the model's learned parameters?

Part 5 — The Most Important Question

This is the mastery gate.

Imagine you have a neural network with millions of parameters.

Explain this entire process from beginning to end:

Training data
      ↓
      ?
      ↓
      ?
      ↓
      ?
      ↓
      ?
      ↓
      ?
      ↓
Updated parameters

Then explain what happens when a new input arrives after training is finished.

Your explanation should include:

representations
parameters
forward pass
prediction
loss
gradients
parameter updates
repeated training
inference
logits/probabilities where appropriate
Final challenge

Give me your own explanation of:

"What does it actually mean for a neural network to learn?"