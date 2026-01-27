# Conversation Topic Classifier

An LLM-powered classifier that assigns topic labels to customer support conversations using a configurable taxonomy. Built with LangChain and OpenAI's structured output for consistent, validated results.

## Why Taxonomy Matters

A well-designed topic taxonomy is foundational for support automation. It enables:
- **Routing**: Auto-assign tickets to the right team (billing vs. technical)
- **Analytics**: Understand volume trends, identify emerging issues, measure resolution times by topic
- **Training data**: Build supervised models or fine-tune LLMs for domain-specific classification
- **Quality monitoring**: Detect drift in customer needs over time

This project demonstrates a practical approach: a human-designed taxonomy combined with LLM classification that produces explainable, auditable labels.

---

## Deliverables Checklist

| Deliverable | Location | Description |
|-------------|----------|-------------|
| Topic taxonomy | `data/taxonomy.csv` | 10 mutually exclusive categories with definitions |
| Labeled conversations | `data/conversations_ai_classified.csv` | 1,000 conversations with topic, confidence, rationale |
| Labeling script | `main.py` | LangChain + OpenAI classifier with structured output |
| Performance review | `model_review.ipynb` | Performs a quick model analysis |
---

## Repository Structure

```
conversation-topic-classifier/
├── main.py                  # Main labeling script (CLI)
├── data/
│   ├── conversations.csv    # Input: 1,000 customer support conversations
│   ├── taxonomy.csv         # Topic taxonomy (editable)
│   ├── conversations_manually_classified.csv     # Manually labeled conversations
│   └── conversations_ai_classified.csv           # Output: labeled conversations
├── tests/
│   └── test_taxonomy.py     # Taxonomy validation tests
├── model_review.ipynb       # Model performance review notebook
├── eda.ipynb                # Exploratory data analysis
├── requirements.txt         # Python dependencies
├── Makefile                 # Common commands
├── .env                     # API keys (not committed)
├── .env.example             # Template for .env
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd conversation-topic-classifier

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root (see `.env.example`):

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional: LangSmith tracing (recommended for debugging)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=conversation-classifier
```

### 3. Run the classifier

```bash
# Run on all conversations
python main.py

# Or use make
make run

# Test with a small sample first
make run-sample  # processes first 10 conversations
```

### CLI Options

```bash
python main.py --help

Options:
  --input PATH      Input conversations CSV (default: data/conversations.csv)
  --taxonomy PATH   Taxonomy CSV (default: data/taxonomy.csv)
  --output PATH     Output conversations with their AI label CSV (default: data/conversations_ai_classified.csv)
  --limit N         Process only first N conversations
  --model NAME      OpenAI model to use (default: gpt-4o-mini)
```

### 4. Output

The script produces `data/conversations_ai_classified.csv` with columns:
- `conversation_id`: Original identifier
- `conversation`: Full conversation text
- `topic`: Assigned taxonomy label (exactly one)
- `confidence`: `low`, `medium`, or `high`
- `rationale`: LLM's explanation for the classification

---

## How the Classifier Works

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  taxonomy.csv   │────>│   Prompt with    │────>│   Structured    │
│  (categories)   │     │   topic defs     │     │   Output        │
└─────────────────┘     └──────────────────┘     │   (Pydantic)    │
                                │                └────────┬────────┘
┌─────────────────────┐         │                         │
│ conversations_raw   │─────────┘                         v
│     .csv            │                ┌──────────────────────────────────────┐
└─────────────────────┘                │   conversations_ai_classified.csv    │
                                       └──────────────────────────────────────┘
```

### Prompt Strategy

1. **System prompt**: Instructs the model to act as a taxonomy classifier, pick exactly one topic, and use the provided descriptions
2. **Topic injection**: All taxonomy names and descriptions are included in every request, enabling zero-shot classification
3. **Fallback rule**: If intent is unclear or multi-topic, classify as "General Enquiries & Multi-Intent"

### Structured Output

The classifier uses LangChain's `with_structured_output()` with a dynamically generated Pydantic model:

```python
TopicLabel = create_model(
    "TopicLabel",
    topic=(Literal["Orders, Shipping & Delivery", ...], ...),  # Dynamic from CSV
    confidence=(Literal["low", "medium", "high"], ...),
    rationale=(str, ...),
)
```

This guarantees:
- **Type safety**: `topic` must be an exact match from the taxonomy
- **Consistency**: Every response includes confidence and rationale
- **Validation**: Invalid outputs are rejected by Pydantic

### Model Choice

- **gpt-5.2**: Chosen for the best model for accuracy
- **temperature=0**: Deterministic outputs for reproducibility

---

## Taxonomy Design Principles

The taxonomy in `data/taxonomy.csv` follows these principles:

| Principle | Implementation |
|-----------|----------------|
| **Mutually exclusive** | Each category has clear "Includes" and "Excludes" boundaries |
| **Collectively exhaustive** | "General Enquiries & Multi-Intent" catches edge cases |
| **Action-oriented** | Categories map to business actions (routing, escalation, ops) |
| **Defined by examples** | Each category lists concrete scenarios |

### Current Categories (10)

1. Orders, Shipping & Delivery
2. Returns, Refunds & Exchanges
3. Product Defects & Fulfillment Errors
4. Billing, Charges & Price Discrepancies
5. Account Access & Customer Profile
6. Technical & Platform Issues
7. Product Information & Availability
8. Promotions, Discounts & Loyalty
9. Complaints, Escalations & Negative Feedback
10. General Enquiries & Multi-Intent

### Modifying the Taxonomy

To add, remove, or edit categories:

1. **Edit `data/taxonomy.csv`** with columns: `taxonomy_id`, `taxonomy_name`, `taxonomy_description`
2. **Keep names unique** — the script validates for duplicates
3. **Keep "General Enquiries & Multi-Intent"** (or similar) as a catch-all
4. **Update descriptions** with clear Includes/Excludes sections
5. **Re-run the classifier** — taxonomy changes are loaded dynamically

---

## Evaluation & Monitoring

### Measuring Classification Quality

| Method | How to Implement |
|--------|------------------|
| **Gold set accuracy** | Manually label 50–100 conversations, compare to model predictions |
| **Confusion matrix** | Identify which categories get confused (e.g., Returns vs. Product Defects) |
| **Low-confidence rate** | Track % of `confidence=low` — if >15%, review taxonomy clarity |
| **Spot checks** | Randomly sample 10–20 labels per category, verify correctness |
| **Rationale review** | Read rationales for disagreements to identify prompt improvements |

### Monitoring for Drift

In production, monitor these signals:

- **Topic distribution shift**: Compare weekly/monthly topic proportions
- **New vocabulary**: Cluster low-confidence conversations to find emerging topics
- **Confidence trends**: Declining average confidence may signal taxonomy gaps
- **Rationale patterns**: LLM hedging language ("could be", "might") suggests ambiguity

---

## Limitations

- **Single-label only**: Conversations with multiple intents get the "primary" one or fall into General
- **No fine-tuning**: Uses zero-shot prompting; domain-specific training could improve accuracy
- **English only**: Taxonomy and prompts are in English
- **Cost at scale**: ~$0.01–0.02 per conversation with gpt-4o-mini; consider batching or caching for large datasets
- **No retry logic**: Rate limits or API errors will stop the script (see Troubleshooting)

---

## Next Steps

If extending this project:

1. **Add CLI arguments**: Parameterize input/output paths, model selection
2. **Implement batching**: Process conversations in parallel with rate limiting
3. **Build evaluation pipeline**: Automated gold-set comparison with metrics
4. **Add multi-label support**: For conversations with multiple clear intents
5. **Fine-tune a smaller model**: Use labels as training data for a cheaper classifier
6. **Deploy as API**: Wrap in FastAPI for real-time classification

---

## Example Usage

### Quick test with a subset

```bash
# Process only the first 10 conversations
python main.py --limit 10

# Or use make
make run-sample
```

### Use a different model

```bash
python main.py --model gpt-4o --limit 50
```

### Custom input/output paths

```bash
python main.py --input my_data.csv --output my_labels.csv
```

### Run taxonomy validation tests

```bash
make test

# Or directly with pytest
pytest tests/ -v
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `OPENAI_API_KEY not set` | Ensure `.env` file exists and contains `OPENAI_API_KEY=sk-...` |
| `Rate limit exceeded` | Add delays between requests or implement exponential backoff |
| `taxonomy.csv missing columns` | Ensure columns are exactly: `taxonomy_id`, `taxonomy_name`, `taxonomy_description` |
| `Duplicate taxonomy names` | Each `taxonomy_name` must be unique; check for copy-paste errors |
| `Empty taxonomy` | Ensure `taxonomy.csv` has at least one data row |
| `Invalid topic in output` | The structured output should prevent this; check Pydantic model if it occurs |
| `LangSmith not logging` | Verify `LANGCHAIN_TRACING_V2=true` and valid `LANGCHAIN_API_KEY` |

---

## License

MIT
