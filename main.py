"""
Conversation Topic Classifier with Operational Enrichment

An LLM-powered classifier that assigns topic labels AND operational metadata to
customer support conversations using a configurable taxonomy. Uses LangChain with
OpenAI's structured output to ensure consistent, validated results.

Usage:
    python main.py                          # Run on all conversations
    python main.py --limit 10               # Test with first 10 conversations
    python main.py --model gpt-4o           # Use a different model
    python main.py --input data/custom.csv  # Use custom input file

Architecture:
    1. Load taxonomy from CSV (categories with descriptions)
    2. Build a dynamic Pydantic model constraining output to valid topics + operational enums
    3. For each conversation, invoke LLM with structured output
    4. Write results to CSV with all classification and operational fields

Output Fields:
    CLASSIFICATION (existing):
        - topic: Constrained to taxonomy names (Literal enum)
        - confidence: low/medium/high
        - rationale: Free-text explanation (for auditability)

    OPERATIONAL (new):
        - handler_summary: Free-text summary for call handlers (<=35 words)
        - emotion: Constrained enum for sentiment routing
        - difficulty: Constrained enum for workload estimation
        - operational_actions: List of constrained action codes
        - risk_level: Constrained enum for prioritization
        - escalation_required: Boolean flag
        - escalation_flags: List of constrained flag codes
        - root_cause_code: Constrained enum for analytics
        - root_cause_detail: Free-text specificity (<=12 words)

Design Rationale:
    - Constrained enums (emotion, difficulty, actions, etc.) ensure consistent
      analytics, dashboards, and routing rules across all conversations.
    - Free-text fields (handler_summary, rationale, root_cause_detail) allow
      nuanced, human-readable context that enums cannot capture.
"""

import argparse
import logging
import os
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from pydantic import Field, create_model
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# =============================================================================
# Operational Enums (Fixed for Analytics/Routing Consistency)
# =============================================================================
# These enums are constrained to ensure downstream systems (dashboards, routing,
# alerting) receive predictable values. DO NOT use free-text for these fields.

# Customer emotional state - used for agent preparation and tone matching
EmotionType = Literal[
    "calm",
    "confused",
    "frustrated",
    "angry",
    "anxious",
    "urgent"
]

# Resolution difficulty estimate - used for workload balancing and SLA prediction
DifficultyType = Literal["low", "medium", "high"]

# Risk assessment - used for prioritization queues and manager alerts
RiskLevelType = Literal["none", "low", "medium", "high"]

# Operational actions the handler should take - maps to runbooks/SOPs
OperationalActionType = Literal[
    "reset_password_or_otp",
    "resend_otp_or_verification",
    "reactivate_account",
    "update_customer_details",
    "cancel_order",
    "check_order_status",
    "update_delivery_address",
    "reschedule_delivery_attempt",
    "provide_tracking_link_or_update",
    "initiate_return",
    "initiate_refund",
    "initiate_exchange_replacement",
    "troubleshoot_product_setup",
    "initiate_warranty_claim",
    "escalate_to_tier2_support",
    "escalate_to_ops",
    "provide_product_availability_alternatives",
    "share_policy_timeline"
]

# Escalation trigger flags - used for compliance, risk management, VIP handling
EscalationFlagType = Literal[
    "suspected_fraud_or_scam",
    "chargeback_threat",
    "legal_threat",
    "safety_risk",
    "high_value_order",
    "repeat_contact",
    "vip_customer",
    "severe_dissatisfaction",
    "abuse_or_harassment",
    "data_privacy_risk"
]

# Root cause codes - used for product/ops feedback loops and trend analysis
RootCauseCodeType = Literal[
    "otp_not_received",
    "otp_attempts_exceeded_or_lockout",
    "account_deactivated_inactive",
    "login_credentials_mismatch",
    "cancel_button_or_ui_bug",
    "payment_or_checkout_error",
    "delivery_address_issue",
    "delivery_attempt_failed",
    "tracking_unavailable_or_stale",
    "product_defective_or_doa",
    "missing_parts_or_wrong_item",
    "warranty_card_missing",
    "out_of_stock",
    "recall_or_safety_return",
    "refund_timeline_cod",
    "unknown_or_multi_intent"
]


# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# Silence noisy third-party HTTP / OpenAI / LangChain logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# =============================================================================
# CLI Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the classifier.

    Returns:
        argparse.Namespace: Parsed arguments with the following attributes:
            - input (str): Path to input conversations CSV
            - taxonomy (str): Path to taxonomy definitions CSV
            - output (str): Path for output labels CSV
            - limit (int|None): Optional limit on number of conversations to process
            - model (str): OpenAI model identifier to use

    Example:
        >>> args = parse_args()
        >>> print(args.input)
        'data/conversations_raw.csv'
    """
    parser = argparse.ArgumentParser(
        description="Classify customer support conversations by topic using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              Run on all conversations
  python main.py --limit 10                   Test with first 10 rows
  python main.py --model gpt-4o --limit 50    Use GPT-4o on 50 conversations
  python main.py --input custom.csv           Use custom input file
        """
    )
    parser.add_argument(
        "--input",
        default="data/conversations_raw.csv",
        help="Path to input conversations CSV (default: data/conversations_raw.csv)"
    )
    parser.add_argument(
        "--taxonomy",
        default="data/taxonomy.csv",
        help="Path to taxonomy CSV (default: data/taxonomy.csv)"
    )
    parser.add_argument(
        "--output",
        default="data/conversations_ai_classified.csv",
        help="Path to output labels CSV (default: data/conversations_ai_classified.csv)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N conversations (default: all)"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    return parser.parse_args()


# =============================================================================
# Taxonomy Loading and Validation
# =============================================================================

def load_taxonomy(taxonomy_path: str) -> tuple[list[str], list[str]]:
    """
    Load and validate the topic taxonomy from a CSV file.

    The taxonomy CSV must contain:
        - taxonomy_name: Unique identifier for each topic (used in output)
        - taxonomy_description: Detailed definition used to guide LLM classification

    Validation checks performed:
        - Required columns exist
        - At least one category defined
        - No duplicate taxonomy names
        - Warning if catch-all category is missing

    Args:
        taxonomy_path: Path to the taxonomy CSV file

    Returns:
        tuple: (taxonomy_names, taxonomy_descriptions) as parallel lists

    Raises:
        ValueError: If required columns are missing, taxonomy is empty,
                    or duplicate names are found

    Example:
        >>> names, descs = load_taxonomy("data/taxonomy.csv")
        >>> print(names[0])
        'Orders, Shipping & Delivery'
    """
    tax_df = pd.read_csv(taxonomy_path).fillna("")

    # Validate required columns exist
    required_cols = {"taxonomy_name", "taxonomy_description"}
    if not required_cols.issubset(set(tax_df.columns)):
        missing = required_cols - set(tax_df.columns)
        raise ValueError(f"taxonomy.csv missing required columns: {missing}")

    # Extract and clean taxonomy data
    taxonomy_names = tax_df["taxonomy_name"].astype(str).str.strip().tolist()
    taxonomy_descs = tax_df["taxonomy_description"].astype(str).str.strip().tolist()

    # Validate taxonomy is not empty
    if len(taxonomy_names) == 0:
        raise ValueError("taxonomy.csv has no rows.")

    # Validate uniqueness of taxonomy names (required for Literal type)
    if len(set(taxonomy_names)) != len(taxonomy_names):
        dupes = pd.Series(taxonomy_names)[pd.Series(taxonomy_names).duplicated()].unique().tolist()
        raise ValueError(f"taxonomy_name must be unique. Duplicates found: {dupes}")

    # Warn if catch-all category is missing (recommended for ambiguous cases)
    if "General Enquiries & Multi-Intent" not in taxonomy_names:
        logger.warning(
            "'General Enquiries & Multi-Intent' not found in taxonomy.csv. "
            "Consider adding it so the model has an 'unclear' bucket."
        )

    return taxonomy_names, taxonomy_descs


# =============================================================================
# Classifier Construction
# =============================================================================

def create_classifier(taxonomy_names: list[str], taxonomy_descs: list[str], model: str):
    """
    Create a LangChain classifier with structured output constrained to the taxonomy
    and operational enums.

    This function builds:
        1. A topics block string combining names and descriptions for the prompt
        2. A dynamic Pydantic model with:
           - topic constrained to Literal[taxonomy_names]
           - confidence constrained to low/medium/high
           - rationale as free-text
           - NEW: operational fields (emotion, difficulty, actions, risk, escalation, root cause)
        3. A LangChain pipeline: prompt -> LLM -> structured output parser

    Field Design Philosophy:
        - CONSTRAINED fields (enums/Literals): For routing, analytics, dashboards.
          Must be predictable for downstream automation.
        - FREE-TEXT fields: For human readability and nuance that enums can't capture.
          Limited to: handler_summary, rationale, root_cause_detail.

    Args:
        taxonomy_names: List of valid topic names (e.g., ["Billing", "Returns", ...])
        taxonomy_descs: Parallel list of topic descriptions for LLM context
        model: OpenAI model identifier (e.g., "gpt-4o-mini", "gpt-4o")

    Returns:
        tuple: (classifier_chain, topics_block)
            - classifier_chain: LangChain runnable that takes {"topics": str, "conversation": str}
            - topics_block: Formatted string of all topics for prompt injection
    """
    # Build the topics block: each entry is "Name\nDescription\n\n"
    # This format helps the LLM understand what each category means
    topics_block = "\n".join(
        [f"{name}\n{desc}\n\n" for name, desc in zip(taxonomy_names, taxonomy_descs)]
    )

    # -------------------------------------------------------------------------
    # Dynamic Pydantic Model with ALL Output Fields
    # -------------------------------------------------------------------------
    # We use create_model() because taxonomy names are loaded at runtime.
    # The Literal type ensures the LLM can ONLY output valid taxonomy values.

    TopicLabel = create_model(
        "TopicLabel",
        # ----- EXISTING CLASSIFICATION FIELDS (unchanged) -----
        topic=(
            Literal[tuple(taxonomy_names)],
            Field(..., description="The assigned topic label from the taxonomy. Must match exactly.")
        ),
        confidence=(
            Literal["low", "medium", "high"],
            Field(..., description="Classification confidence: low if ambiguous, high if clear single intent.")
        ),
        rationale=(
            str,
            Field(..., description="Brief explanation (1-2 sentences) for why this topic was chosen. Free-text for auditability.")
        ),

        # Free-text: Human-readable summary for call handlers
        handler_summary=(
            str,
            Field(
                ...,
                description="Call-handler-friendly summary (max 35 words): what's happening + what customer wants. Natural language.",
                max_length=250  # ~35 words buffer
            )
        ),

        # Constrained: Customer emotional state for agent preparation
        emotion=(
            EmotionType,
            Field(..., description="Customer's emotional state: calm, confused, frustrated, angry, anxious, or urgent.")
        ),

        # Constrained: Resolution difficulty for workload balancing
        difficulty=(
            DifficultyType,
            Field(..., description="How hard to resolve: low (simple/quick), medium (some investigation), high (complex/multi-step).")
        ),

        # Constrained list: Recommended operational actions (can be empty)
        operational_actions=(
            list[OperationalActionType],
            Field(
                default_factory=list,
                description="List of recommended actions from the fixed set. Include all applicable. Empty list if none apply."
            )
        ),

        # Constrained: Risk level for prioritization
        risk_level=(
            RiskLevelType,
            Field(..., description="Risk level: none, low, medium, high. Consider financial, legal, reputational, safety risks.")
        ),

        # Boolean: Escalation flag
        escalation_required=(
            bool,
            Field(..., description="True if this conversation should be escalated beyond tier-1 support.")
        ),

        # Constrained list: Escalation trigger flags (can be empty)
        escalation_flags=(
            list[EscalationFlagType],
            Field(
                default_factory=list,
                description="List of escalation flags from the fixed set. Empty list if no flags apply."
            )
        ),

        # Constrained: Root cause code for analytics
        root_cause_code=(
            RootCauseCodeType,
            Field(..., description="Primary root cause code. Use 'unknown_or_multi_intent' if unclear or multiple causes.")
        ),

        # Free-text: Additional root cause specificity
        root_cause_detail=(
            str,
            Field(
                ...,
                description="Short detail (max 12 words) adding specificity to root cause. Natural language.",
                max_length=100  # ~12 words buffer
            )
        ),
    )

    # -------------------------------------------------------------------------
    # Enhanced System Prompt with Operational Instructions
    # -------------------------------------------------------------------------
    system_prompt = """You are a customer support conversation analyzer. For each conversation, you must extract:

1. TOPIC CLASSIFICATION:
   - Pick exactly ONE topic from the allowed taxonomy list
   - Use the topic descriptions to choose correctly
   - If multiple apply, choose the PRIMARY customer intent
   - If unclear, choose 'General Enquiries & Multi-Intent'

2. CONFIDENCE ASSESSMENT:
   - high: Clear single intent, obvious topic match
   - medium: Some ambiguity but reasonable classification
   - low: Multiple intents, unclear, or edge case

3. HANDLER SUMMARY (free-text, max 35 words):
   - Write a natural, call-handler-friendly summary
   - Include: what's happening + what the customer wants
   - Be concise and actionable

4. EMOTIONAL STATE:
   - Assess customer's emotional state from the conversation
   - Choose: calm, confused, frustrated, angry, anxious, urgent

5. DIFFICULTY ASSESSMENT:
   - low: Simple, quick resolution (info lookup, basic action)
   - medium: Requires investigation or multiple steps
   - high: Complex issue, multiple systems, potential exceptions

6. OPERATIONAL ACTIONS:
   - Select ALL applicable actions from the fixed list
   - These map to standard operating procedures
   - Return empty list if no specific actions needed

7. RISK ASSESSMENT:
   - none: No risk indicators
   - low: Minor inconvenience, low financial impact
   - medium: Potential churn, moderate financial impact
   - high: Legal/safety/major financial/reputational risk

8. ESCALATION:
   - Set escalation_required=true if tier-2+ involvement needed
   - Add relevant escalation_flags from the fixed list
   - Return empty list if no flags apply

9. ROOT CAUSE:
   - Select the primary root_cause_code from the fixed list
   - Add root_cause_detail (max 12 words) for specificity
   - Use 'unknown_or_multi_intent' if cause is unclear

IMPORTANT:
- All enum/list fields MUST use values from the provided fixed sets
- Only handler_summary, rationale, and root_cause_detail are free-text
- Lists can be empty but must always be lists (not null)"""

    human_prompt = """Allowed topics (with descriptions):
{topics}

---
OPERATIONAL ACTIONS (choose from):
reset_password_or_otp, resend_otp_or_verification, reactivate_account, update_customer_details,
cancel_order, check_order_status, update_delivery_address, reschedule_delivery_attempt,
provide_tracking_link_or_update, initiate_return, initiate_refund, initiate_exchange_replacement,
troubleshoot_product_setup, initiate_warranty_claim, escalate_to_tier2_support, escalate_to_ops,
provide_product_availability_alternatives, share_policy_timeline

ESCALATION FLAGS (choose from):
suspected_fraud_or_scam, chargeback_threat, legal_threat, safety_risk, high_value_order,
repeat_contact, vip_customer, severe_dissatisfaction, abuse_or_harassment, data_privacy_risk

ROOT CAUSE CODES (choose from):
otp_not_received, otp_attempts_exceeded_or_lockout, account_deactivated_inactive,
login_credentials_mismatch, cancel_button_or_ui_bug, payment_or_checkout_error,
delivery_address_issue, delivery_attempt_failed, tracking_unavailable_or_stale,
product_defective_or_doa, missing_parts_or_wrong_item, warranty_card_missing,
out_of_stock, recall_or_safety_return, refund_timeline_cod, unknown_or_multi_intent

---
Conversation to analyze:
{conversation}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    # Initialize the LLM with temperature=0 for deterministic outputs
    llm = ChatOpenAI(
        model=model,
        temperature=0,
        seed=42
    )

    # Chain: prompt -> LLM with structured output
    # with_structured_output() uses OpenAI's function calling to enforce the schema
    classifier = prompt | llm.with_structured_output(TopicLabel)

    return classifier, topics_block


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """
    Main entry point for the conversation classifier.

    Workflow:
        1. Load environment variables (API keys from .env)
        2. Parse CLI arguments
        3. Load and validate conversations and taxonomy
        4. Create the classifier chain
        5. Process each conversation with retry logic
        6. Write results to output CSV
        7. Log summary statistics

    The classifier produces a CSV with columns:
        - conversation_id: Original identifier from input
        - conversation: Full conversation text
        - topic: Assigned taxonomy label
        - confidence: low/medium/high
        - rationale: LLM's explanation for the classification
        - handler_summary: Call-handler-friendly summary
        - emotion: Customer emotional state
        - difficulty: Resolution difficulty estimate
        - operational_actions: Recommended actions (JSON list)
        - risk_level: Risk assessment
        - escalation_required: Boolean escalation flag
        - escalation_flags: Escalation triggers (JSON list)
        - root_cause_code: Primary root cause
        - root_cause_detail: Root cause specificity

    Error handling:
        - Rate limits: Automatic retry with exponential backoff (up to 3 attempts)
        - Other errors: Logged and marked as "ERROR" in output, processing continues
    """
    # Load API keys from .env file
    load_dotenv()
    args = parse_args()

    # -------------------------------------------------------------------------
    # Load input data
    # -------------------------------------------------------------------------
    logger.info(f"Loading conversations from {args.input}")
    df = pd.read_csv(args.input)

    logger.info(f"Loading taxonomy from {args.taxonomy}")
    taxonomy_names, taxonomy_descs = load_taxonomy(args.taxonomy)
    logger.info(f"Loaded {len(df)} conversations and {len(taxonomy_names)} taxonomy categories")

    # -------------------------------------------------------------------------
    # Build classifier
    # -------------------------------------------------------------------------
    classifier, topics_block = create_classifier(taxonomy_names, taxonomy_descs, args.model)

    # Wrap classifier invocation with retry logic for rate limit handling
    # Uses exponential backoff: 2s -> 4s -> 8s... up to 60s max
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"Rate limited, retrying in {retry_state.next_action.sleep} seconds..."
        )
    )
    def classify(conversation: str):
        """Classify a single conversation with automatic retry on failure."""
        return classifier.invoke({
            "topics": topics_block,
            "conversation": conversation
        })

    # -------------------------------------------------------------------------
    # Apply optional limit for testing
    # -------------------------------------------------------------------------
    if args.limit:
        df = df.iloc[:args.limit]
        logger.info(f"Processing limited to first {args.limit} conversations")

    # -------------------------------------------------------------------------
    # Process all conversations
    # -------------------------------------------------------------------------
    rows = []
    total = len(df)
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        try:
            out = classify(r["conversation"])
            rows.append({
                # Original identifiers
                "conversation_id": r["conversation_id"],
                "conversation": r["conversation"],
                # Classification fields (existing)
                "topic": out.topic,
                "confidence": out.confidence,
                "rationale": out.rationale,
                # Operational fields (new)
                "handler_summary": out.handler_summary,
                "emotion": out.emotion,
                "difficulty": out.difficulty,
                "operational_actions": out.operational_actions,  # List -> stored as-is, pandas will serialize
                "risk_level": out.risk_level,
                "escalation_required": out.escalation_required,
                "escalation_flags": out.escalation_flags,  # List -> stored as-is
                "root_cause_code": out.root_cause_code,
                "root_cause_detail": out.root_cause_detail,
            })
        except Exception as e:
            # Log error but continue processing remaining conversations
            logger.error(f"Failed to classify conversation {r['conversation_id']}: {e}")
            rows.append({
                "conversation_id": r["conversation_id"],
                "conversation": r["conversation"],
                "topic": "ERROR",
                "confidence": "low",
                "rationale": str(e),
                "handler_summary": "Classification failed",
                "emotion": "calm",
                "difficulty": "low",
                "operational_actions": [],
                "risk_level": "none",
                "escalation_required": False,
                "escalation_flags": [],
                "root_cause_code": "unknown_or_multi_intent",
                "root_cause_detail": "Classification error",
            })

        # Print progress every 25 rows (adjust as needed)
        if i % 25 == 0 or i == total or i == 1:
            percent = (i / total) * 100
            logger.info(f"Progress: {i}/{total} ({percent:.0f}%)")

    # -------------------------------------------------------------------------
    # Write output and log summary
    # -------------------------------------------------------------------------
    output_df = pd.DataFrame(rows)
    output_df.to_csv(args.output, index=False)
    logger.info(f"Wrote {len(rows)} labels to {args.output}")

    # Print distribution summaries for quick quality check
    topic_counts = output_df["topic"].value_counts()
    confidence_counts = output_df["confidence"].value_counts()
    emotion_counts = output_df["emotion"].value_counts()
    difficulty_counts = output_df["difficulty"].value_counts()
    escalation_counts = output_df["escalation_required"].value_counts()

    logger.info(f"Topic distribution:\n{topic_counts.to_string()}")
    logger.info(f"Confidence distribution:\n{confidence_counts.to_string()}")
    logger.info(f"Emotion distribution:\n{emotion_counts.to_string()}")
    logger.info(f"Difficulty distribution:\n{difficulty_counts.to_string()}")
    logger.info(f"Escalation required:\n{escalation_counts.to_string()}")


if __name__ == "__main__":
    main()
