# Conversation Topic Classifier

An LLM-powered classifier that assigns **topic labels AND operational metadata** to customer support conversations using a configurable taxonomy. Built with LangChain and OpenAI's structured output for consistent, validated results.

In a **single LLM call per conversation**, the classifier extracts:
- Topic classification with confidence and rationale
- Call-handler summary
- Customer emotion and resolution difficulty
- Recommended operational actions
- Risk assessment and escalation flags
- Root cause analysis

## Why Taxonomy Matters

A well-designed topic taxonomy is foundational for support automation. It enables:
- **Routing**: Auto-assign tickets to the right team or AI workflow
- **Escalation**: Flag high-risk conversations for immediate human review
- **Analytics**: Understand contact drivers, root causes, and operational patterns
- **Training data**: Build supervised models or fine-tune LLMs for domain-specific tasks
- **Quality monitoring**: Detect drift in customer needs over time

This project demonstrates a practical approach: a human-designed taxonomy combined with LLM classification that produces explainable, auditable, and operationally actionable outputs.

---

## Deliverables Checklist

| Deliverable | Location | Description |
|-------------|----------|-------------|
| Topic taxonomy | `data/taxonomy.csv` | 10 mutually exclusive categories with definitions |
| Labeled conversations | `data/conversations_ai_classified_.csv` | 1,000 conversations with full classification + operational metadata |
| Labeling script | `main.py` | LangChain + OpenAI classifier with structured output |
| Performance review | `model_review.ipynb` | Comprehensive model analysis with routing/escalation views |

---

## Repository Structure

```
conversation-topic-classifier/
├── main.py                  # Main labeling script (CLI)
├── data/
│   ├── conversations_raw.csv                   # Input: raw conversations
│   ├── taxonomy.csv                            # Topic taxonomy (editable)
│   ├── conversations_manually_classified.csv   # Gold-set manual labels
│   └── conversations_ai_classified_.csv        # Output: full enriched labels
├── tests/
│   └── test_taxonomy.py     # Taxonomy validation tests
├── model_review.ipynb       # Model performance & operational analysis
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
python main.py --limit 10
```

### CLI Options

```bash
python main.py --help

Options:
  --input PATH      Input conversations CSV (default: data/conversations_raw.csv)
  --taxonomy PATH   Taxonomy CSV (default: data/taxonomy.csv)
  --output PATH     Output CSV (default: data/conversations_ai_classified_.csv)
  --limit N         Process only first N conversations
  --model NAME      OpenAI model to use (default: gpt-5.2)
```

---

## Input/Output Schemas

### Input: Conversations CSV

| Column | Type | Description |
|--------|------|-------------|
| `conversation_id` | int/string | Unique identifier |
| `conversation` | string | Full conversation text (Agent + Customer turns) |

### Input: Taxonomy CSV

| Column | Type | Description |
|--------|------|-------------|
| `taxonomy_id` | int | Unique numeric ID |
| `taxonomy_name` | string | Category name (used as output label) |
| `taxonomy_description` | string | Detailed definition with Includes/Excludes |

### Output: Enriched Classifications CSV

The classifier produces a CSV with **14 columns**:

| Column | Type | Constrained? | Description |
|--------|------|--------------|-------------|
| `conversation_id` | string | — | Original identifier |
| `conversation` | string | — | Full conversation text |
| `topic` | string | **Yes** (taxonomy) | Assigned topic from taxonomy |
| `confidence` | string | **Yes** (enum) | `low`, `medium`, or `high` |
| `rationale` | string | No (free-text) | LLM explanation for classification |
| `handler_summary` | string | No (free-text) | Call-handler-friendly summary (≤35 words) |
| `emotion` | string | **Yes** (enum) | Customer emotional state |
| `difficulty` | string | **Yes** (enum) | Resolution difficulty estimate |
| `operational_actions` | list | **Yes** (enum list) | Recommended SOP actions |
| `risk_level` | string | **Yes** (enum) | Risk assessment |
| `escalation_required` | bool | **Yes** | Whether escalation is needed |
| `escalation_flags` | list | **Yes** (enum list) | Escalation trigger flags |
| `root_cause_code` | string | **Yes** (enum) | Primary root cause |
| `root_cause_detail` | string | No (free-text) | Root cause specificity (≤12 words) |

#### Enum Values

**emotion**: `calm`, `confused`, `frustrated`, `angry`, `anxious`, `urgent`

**difficulty**: `low`, `medium`, `high`

**risk_level**: `none`, `low`, `medium`, `high`

**operational_actions** (18 values):
```
reset_password_or_otp, resend_otp_or_verification, reactivate_account,
update_customer_details, cancel_order, check_order_status, update_delivery_address,
reschedule_delivery_attempt, provide_tracking_link_or_update, initiate_return,
initiate_refund, initiate_exchange_replacement, troubleshoot_product_setup,
initiate_warranty_claim, escalate_to_tier2_support, escalate_to_ops,
provide_product_availability_alternatives, share_policy_timeline
```

**escalation_flags** (10 values):
```
suspected_fraud_or_scam, chargeback_threat, legal_threat, safety_risk,
high_value_order, repeat_contact, vip_customer, severe_dissatisfaction,
abuse_or_harassment, data_privacy_risk
```

**root_cause_code** (16 values):
```
otp_not_received, otp_attempts_exceeded_or_lockout, account_deactivated_inactive,
login_credentials_mismatch, cancel_button_or_ui_bug, payment_or_checkout_error,
delivery_address_issue, delivery_attempt_failed, tracking_unavailable_or_stale,
product_defective_or_doa, missing_parts_or_wrong_item, warranty_card_missing,
out_of_stock, recall_or_safety_return, refund_timeline_cod, unknown_or_multi_intent
```

---

## How the Classifier Works

### Architecture

```
┌─────────────────┐     ┌──────────────────────────┐     ┌─────────────────────┐
│  taxonomy.csv   │────>│   Prompt with:           │────>│   Structured Output │
│  (categories)   │     │   • Topic definitions    │     │   (Pydantic model)  │
└─────────────────┘     │   • Enum constraints     │     └──────────┬──────────┘
                        │   • Extraction rules     │                │
┌─────────────────┐     └──────────────────────────┘                ▼
│ conversations   │────────────────┘              ┌─────────────────────────────┐
│     _raw.csv    │                               │ conversations_ai_classified │
└─────────────────┘                               │         _.csv               │
                                                  └─────────────────────────────┘
```

### Structured Output Strategy

The classifier uses LangChain's `with_structured_output()` with a dynamically generated Pydantic model. This approach:

1. **Enforces schema**: OpenAI's function calling validates the output structure
2. **Constrains enums**: Invalid values are rejected at the API level
3. **Guarantees completeness**: All required fields must be present

#### Why Constrained Enums vs. Free Text?

| Field Type | Purpose | Examples |
|------------|---------|----------|
| **Constrained (enum)** | Routing rules, dashboards, alerting, analytics pipelines | `topic`, `emotion`, `risk_level`, `operational_actions` |
| **Free text** | Human readability, nuanced context that enums can't capture | `handler_summary`, `rationale`, `root_cause_detail` |

**Design principle**: Use enums for anything that feeds into automated systems (routing, dashboards, alerts). Use free text for human consumption where nuance matters.

```python
TopicLabel = create_model(
    "TopicLabel",
    # Constrained: Must match taxonomy exactly
    topic=(Literal["Orders, Shipping & Delivery", ...], ...),
    # Constrained: Fixed enum for routing
    emotion=(Literal["calm", "confused", "frustrated", ...], ...),
    # Free text: Human-readable explanation
    rationale=(str, ...),
    handler_summary=(str, Field(max_length=250)),
)
```

### Model Configuration

- **Model**: Configurable via CLI (default: `gpt-5.2`)
- **Temperature**: `0` for deterministic outputs
- **Seed**: `42` for reproducibility
- **Retry logic**: Exponential backoff (3 attempts, 2-60s wait)

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

### Taxonomy CSV Format

```csv
taxonomy_id,taxonomy_name,taxonomy_description
1,"Orders, Shipping & Delivery","Definition: Customer enquiries related to...

Includes:
- Order tracking and shipment updates
- Delivery delays or missed delivery windows
...

Excludes:
- Damaged or defective items (see Product Issues)
..."
```

---

## Measuring Quality

Evaluating taxonomy and classification quality requires both quantitative metrics and qualitative review.

### Quantitative Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Gold set accuracy** | >90% | Compare AI labels to manual labels on 50-100 conversations |
| **Confidence calibration** | High conf = high accuracy | Accuracy should increase with confidence level |
| **Low-confidence rate** | <15% | Track `confidence=low` proportion |
| **Escalation precision** | >85% | Of flagged escalations, how many truly needed escalation? |
| **Escalation recall** | >95% | Of true escalations, how many did we catch? |
| **Label distribution stability** | <5% week-over-week shift | Monitor topic proportions over time |
| **Catch-all rate** | <20% | "General Enquiries" shouldn't be a dumping ground |

### Qualitative Review Process

1. **Weekly spot checks**: Sample 5 conversations per topic, verify correctness
2. **Rationale review**: Read rationales for low-confidence or disputed labels
3. **Handler feedback loop**: Collect feedback from agents using `handler_summary`
4. **Escalation audit**: Review all `escalation_required=True` cases weekly

### Human Audit / Inter-Annotator Agreement

For taxonomy changes or quality assessment:

1. **Sample selection**: Stratified sample of 50-100 conversations (5-10 per topic)
2. **Dual annotation**: Two humans label independently
3. **Agreement measurement**: Calculate Cohen's Kappa (target: >0.8)
4. **Disagreement review**: Discuss edge cases to refine taxonomy descriptions

### Sampling Protocol

```
Weekly Quality Check:
├── Random sample: 50 conversations
├── Stratified by topic: ~5 per category
├── Stratified by confidence: oversample low-confidence
├── Include: all escalation_required=True from past week
└── Review checklist:
    ├── Topic correct? (Y/N)
    ├── Emotion accurate? (Y/N)
    ├── Actions appropriate? (Y/N)
    ├── Escalation decision correct? (Y/N)
    └── Handler summary useful? (1-5 scale)
```

### A/B Testing Taxonomy Changes

When modifying the taxonomy:

1. **Baseline**: Run current taxonomy on test set, record metrics
2. **Variant**: Run modified taxonomy on same test set
3. **Compare**:
   - Accuracy on gold set (if available)
   - Confidence distribution (should shift toward higher confidence)
   - Catch-all rate (should decrease if taxonomy improved)
   - Inter-annotator agreement (should increase)
4. **Rollout**: If metrics improve, deploy new taxonomy
5. **Monitor**: Watch for unexpected shifts in first 48 hours

### Routing Correctness

Validate that topic → workflow mappings are working:

| Topic | Expected Workflow | Validation |
|-------|-------------------|------------|
| Account Access | Auth Support Bot | Check resolution rate |
| Billing | Billing Specialist | Check CSAT scores |
| Complaints | Senior Agent | Check escalation resolution |

### Drift Monitoring Triggers

Set alerts for:
- Topic distribution shift >10% week-over-week
- Low-confidence rate >20%
- Catch-all rate >25%
- New vocabulary clusters in low-confidence conversations
- Escalation rate spike >2x baseline

---

## Keeping the Taxonomy Current

Taxonomies must evolve as products, policies, and customer needs change.

### Change Control Process

```
Taxonomy Change Workflow:
1. Identify need (metrics trigger, business request, new product)
2. Draft change (add/modify/merge/split category)
3. Test on sample (50-100 conversations)
4. Review with stakeholders (support ops, analytics, product)
5. Approve and version
6. Deploy and monitor
```

### Versioning Strategy

1. **Version taxonomy.csv**: Use semantic versioning in filename or header
   ```
   # taxonomy_version: 2.1.0
   # effective_date: 2024-01-15
   # changelog: Added "Subscription Management" category
   ```

2. **Tag releases**: Git tag each taxonomy version
   ```bash
   git tag taxonomy-v2.1.0
   ```

3. **Archive old versions**: Keep `data/taxonomy_archive/taxonomy_v2.0.0.csv`

### Backwards Compatibility

When changing taxonomy:

| Change Type | Backward Compatible? | Migration Strategy |
|-------------|---------------------|-------------------|
| Add category | Yes | No action needed |
| Rename category | No | Map old → new in analytics |
| Merge categories | No | Map old → new in analytics |
| Split category | No | Re-label historical data OR accept mixed labels |
| Remove category | No | Map to remaining category |

**Label mapping table** (maintain in `data/taxonomy_mappings.csv`):
```csv
old_label,new_label,effective_date
"Technical Issues","Technical & Platform Issues",2024-01-15
"Website Bugs","Technical & Platform Issues",2024-01-15
```

### Handling Emerging Topics

When new topics appear:

1. **Detection**: Monitor "General Enquiries" for clusters
   - High volume of similar low-confidence conversations
   - Repeated root causes not covered by existing categories
   - Handler feedback: "This doesn't fit any category"

2. **Evaluation**: Before adding a new category, ask:
   - Volume: >5% of total conversations?
   - Distinct: Different from existing categories?
   - Actionable: Different routing/handling needed?
   - Stable: Not a temporary spike?

3. **Implementation**:
   - Add to taxonomy with clear description
   - Test on historical data
   - Monitor for 2 weeks before declaring stable

### Avoiding Taxonomy Bloat

**Warning signs**:
- >15 categories (harder for LLM to distinguish)
- Categories with <3% volume (consider merging)
- Overlapping descriptions causing confusion
- Low inter-annotator agreement on specific categories

**Remediation**:
- Merge low-volume categories into related ones
- Consolidate overlapping categories
- Use hierarchical taxonomy (parent → child) if needed
- Review annually: "Is this category still needed?"

### Recommended Cadence

| Activity | Frequency |
|----------|-----------|
| Spot checks | Weekly |
| Metrics review | Weekly |
| Taxonomy review meeting | Monthly |
| Full taxonomy audit | Quarterly |
| Major taxonomy revision | As needed (target: ≤2x/year) |

---

## Limitations

- **Single-label only**: Conversations with multiple intents get the "primary" one
- **No fine-tuning**: Uses zero-shot prompting; domain-specific training could improve accuracy
- **English only**: Taxonomy and prompts are in English
- **Cost at scale**: ~$0.02-0.05 per conversation with full enrichment
- **Latency**: Single LLM call adds ~1-3s per conversation

---

## Example Usage

### Quick test with a subset

```bash
python main.py --limit 10
```

### Use a different model

```bash
python main.py --model gpt-4o --limit 50
```

### Custom input/output paths

```bash
python main.py --input my_data.csv --output my_labels.csv --taxonomy my_taxonomy.csv
```

### Run taxonomy validation tests

```bash
pytest tests/ -v
```

### View analysis notebook

```bash
jupyter notebook model_review.ipynb
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `OPENAI_API_KEY not set` | Ensure `.env` file exists and contains `OPENAI_API_KEY=sk-...` |
| `Rate limit exceeded` | Script has built-in retry; for persistent issues, add `--limit` or wait |
| `taxonomy.csv missing columns` | Ensure columns are: `taxonomy_id`, `taxonomy_name`, `taxonomy_description` |
| `Duplicate taxonomy names` | Each `taxonomy_name` must be unique |
| `Empty operational_actions` | This is valid; not all conversations need specific actions |
| `Lists showing as strings` | Pandas serializes lists; use `ast.literal_eval()` to parse |

---

## License

MIT
