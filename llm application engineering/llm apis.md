Good. Your answers show that you understand the basic architecture of an LLM-powered application. Since your senior has reviewed and approved the checkpoint, we'll continue. ✅

One small refinement to answer 3: hosted LLMs *can* support conversational context and memory-like features, depending on the API and application design. The important point is that **we must not assume the model automatically remembers previous requests**. RepoSage needs a deliberate strategy for managing context.

# Phase 3 — Chapter 1, Part 2: Making Our First LLM API Call

Now we'll move from understanding the concept to implementing it.

Our goal is to make RepoSage send a question to a real hosted model and receive its answer.

The basic flow is:

```text
User's question
      |
      v
RepoSage application
      |
      v
Hosted LLM API
      |
      v
Model generates an answer
      |
      v
RepoSage receives the response
      |
      v
Answer returned to the user
```

We'll start with a minimal API call. We won't add complex prompting, tools, RAG, or agents yet. Those concepts will come later in our roadmap.

## First, choose our LLM provider

We need to choose a provider before writing the actual integration because each provider has its own SDK, API conventions, available models, and pricing.

Which provider would you like to use for RepoSage?

1. **OpenAI API**
2. **Google Gemini API**
3. **Anthropic Claude API**

We'll use the selected provider for our first implementation, while learning the underlying concepts in a way that transfers to other providers.