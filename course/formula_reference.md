# Formula Reference

Quick-reference for key formulas, metrics, and patterns used throughout the course.

---

## 1. Token & Cost Math

**Modules: 1, 5, 22**

### Tokens per Word

```
tokens ≈ words × 1.3
```

Rule of thumb: 1 token ≈ 4 characters in English. Exact count varies by tokenizer (cl100k for GPT-4, Claude's tokenizer, etc.).

### Cost Calculation

```
cost = (input_tokens × input_price_per_token) + (output_tokens × output_price_per_token)
```

Example (Claude Sonnet):

```
input_price  = $3.00 / 1M tokens  → $0.000003 per token
output_price = $15.00 / 1M tokens → $0.000015 per token

cost = (2000 × 0.000003) + (500 × 0.000015) = $0.006 + $0.0075 = $0.0135
```

### Context Window Utilization

```
utilization = total_tokens_used / context_window_size
effective_capacity = context_window - system_prompt_tokens - output_reserve
```

---

## 2. Prompt Patterns

**Modules: 2, 3, 7**

### System Prompt Anatomy

```
[Role Definition]     → Who the model is
[Context/Background]  → What it knows
[Instructions]        → What to do
[Constraints]         → What NOT to do
[Output Format]       → How to respond
```

### Few-Shot Template

```
System: {role_and_instructions}

User: {example_input_1}
Assistant: {example_output_1}

User: {example_input_2}
Assistant: {example_output_2}

User: {actual_input}
```

Optimal shot count: 3-5 examples for most tasks. More examples improve consistency but consume context.

### Chain-of-Thought Template

```
System: Think step by step. Show your reasoning before giving a final answer.

User: {question}
Assistant: Let me work through this step by step.
Step 1: ...
Step 2: ...
Therefore: {answer}
```

---

## 3. Embedding Math

**Modules: 8, 9, 10**

### Cosine Similarity

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)

where:
  A · B = Σ(a_i × b_i)        # dot product
  |A|   = √(Σ(a_i²))          # L2 norm (magnitude)
```

Range: [-1, 1] for general vectors, [0, 1] for normalized embeddings.

- 1.0 = identical direction (most similar)
- 0.0 = orthogonal (unrelated)
- -1.0 = opposite direction

### Dot Product

```
dot_product(A, B) = Σ(a_i × b_i)
```

Equivalent to cosine similarity when vectors are normalized (|A| = |B| = 1).

### L2 (Euclidean) Distance

```
L2(A, B) = √(Σ(a_i - b_i)²)
```

Lower = more similar. Relationship to cosine for normalized vectors:

```
L2² = 2 × (1 - cosine_similarity)
```

### Normalization

```
normalized(A) = A / |A|
```

After normalization, |A| = 1, so cosine similarity = dot product.

---

## 4. RAG Metrics

**Modules: 9, 10, 19**

### Precision@k

```
precision@k = |relevant docs in top k| / k
```

Of the k documents retrieved, what fraction is relevant?

### Recall@k

```
recall@k = |relevant docs in top k| / |total relevant docs|
```

Of all relevant documents, what fraction did we retrieve in the top k?

### Mean Reciprocal Rank (MRR)

```
MRR = (1/N) × Σ(1 / rank_i)

where rank_i = position of first relevant result for query i
```

Example: First relevant result at position 3 → reciprocal rank = 1/3.

### Normalized Discounted Cumulative Gain (NDCG)

```
DCG@k  = Σ(rel_i / log₂(i + 1))    for i = 1..k
IDCG@k = DCG@k for ideal ranking
NDCG@k = DCG@k / IDCG@k
```

Range: [0, 1]. Accounts for graded relevance and position.

### Faithfulness (RAG-specific)

```
faithfulness = |claims supported by context| / |total claims in response|
```

Measures whether the LLM's response is grounded in the retrieved context.

### Relevance Scoring

```
relevance = avg(score_i)    for each chunk in context

where score_i ∈ {0, 1} or [0, 1] based on relevance to the query
```

---

## 5. Chunking Formulas

**Modules: 9, 11**

### Chunk Size with Overlap

```
total_chunks = ceil((doc_tokens - overlap) / (chunk_size - overlap))
chunk_i_start = i × (chunk_size - overlap)
chunk_i_end   = chunk_i_start + chunk_size
```

Common defaults: chunk_size = 512 tokens, overlap = 50-100 tokens.

### Token Budget Allocation

```
token_budget = context_window - system_prompt - output_reserve

available_for_retrieval = token_budget - user_query_tokens
max_chunks = floor(available_for_retrieval / avg_chunk_size)
```

Example:

```
context_window  = 200,000 tokens
system_prompt   = 1,000 tokens
output_reserve  = 4,096 tokens
user_query      = 100 tokens
available       = 200,000 - 1,000 - 4,096 - 100 = 194,804 tokens
max_chunks      = floor(194,804 / 512) = 380 chunks
```

### Recursive Chunking Strategy

```
1. Split on headings (## → sections)
2. If section > max_chunk_size → split on paragraphs
3. If paragraph > max_chunk_size → split on sentences
4. If sentence > max_chunk_size → split on tokens
```

---

## 6. Eval Metrics

**Modules: 19, 20**

### Accuracy

```
accuracy = correct_predictions / total_predictions
```

### Precision

```
precision = true_positives / (true_positives + false_positives)
```

Of all positive predictions, how many were actually positive?

### Recall

```
recall = true_positives / (true_positives + false_negatives)
```

Of all actual positives, how many did we predict correctly?

### F1 Score

```
F1 = 2 × (precision × recall) / (precision + recall)
```

Harmonic mean of precision and recall. Range: [0, 1].

### BLEU Score (Bilingual Evaluation Understudy)

```
BLEU = BP × exp(Σ(w_n × log(p_n)))    for n = 1..N

where:
  p_n = modified n-gram precision
  w_n = 1/N (uniform weights, typically N=4)
  BP  = min(1, exp(1 - reference_length / candidate_length))
```

Measures n-gram overlap between generated and reference text. Range: [0, 1].

### ROUGE-L (Longest Common Subsequence)

```
LCS_length = length of longest common subsequence

ROUGE-L_precision = LCS_length / generated_length
ROUGE-L_recall    = LCS_length / reference_length
ROUGE-L_F1        = 2 × (precision × recall) / (precision + recall)
```

---

## 7. Cost Optimization

**Modules: 5, 22**

### Cache Hit Rate

```
cache_hit_rate = cache_hits / total_requests
```

### Expected Cost with Caching

```
expected_cost = (p_cache × cached_price) + ((1 - p_cache) × full_price)

savings = full_price - expected_cost
savings_pct = savings / full_price × 100
```

Example (prompt caching):

```
full_price   = $3.00 / 1M input tokens
cached_price = $0.30 / 1M input tokens (90% discount)
p_cache      = 0.7 (70% hit rate)

expected = (0.7 × 0.30) + (0.3 × 3.00) = $0.21 + $0.90 = $1.11 / 1M tokens
savings  = ($3.00 - $1.11) / $3.00 = 63%
```

### Model Routing Savings

```
routing_cost = (p_small × cost_small) + (p_large × cost_large)

where:
  p_small = fraction of queries routed to smaller/cheaper model
  p_large = 1 - p_small
```

### Token Budget Enforcement

```
estimated_cost = (est_input × input_price) + (max_tokens × output_price)
within_budget  = estimated_cost ≤ budget_limit
```

---

## 8. Agent Metrics

**Modules: 14, 15, 16**

### Task Completion Rate

```
task_completion_rate = successful_tasks / total_tasks_attempted
```

### Average Tool Calls per Task

```
avg_tool_calls = total_tool_calls / total_tasks_completed
```

Lower is generally better (more efficient). Track alongside success rate.

### Planning Accuracy

```
planning_accuracy = tasks_completed_as_planned / total_planned_tasks
```

Measures how often the agent's initial plan leads to successful execution without replanning.

### Agent Cost per Task

```
cost_per_task = total_token_cost / tasks_completed

total_token_cost = Σ(step_input_tokens × input_price + step_output_tokens × output_price)
    for each step in agent loop
```

### Loop Efficiency

```
loop_efficiency = min_steps_needed / actual_steps_taken
```

Range: (0, 1]. A value of 1.0 means the agent took the optimal number of steps.

### Multi-Agent Communication Overhead

```
overhead = coordination_tokens / total_tokens
useful_work_ratio = 1 - overhead
```

Track to ensure agents spend more tokens on task execution than on inter-agent communication.
