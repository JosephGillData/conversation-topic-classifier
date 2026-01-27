"""
Conversation Topic Classifier

An LLM-powered classifier that assigns topic labels to customer support conversations
using a configurable taxonomy. Uses LangChain with OpenAI's structured   to ensure
consistent, validated results.

Usage:
    python main.py                          # Run on all conversations
    python main.py --limit 10               # Test with first 10 conversations
    python main.py --model gpt-4o           # Use a different model
    python main.py --input data/custom.csv  # Use custom input file

Architecture:
    1. Load taxonomy from CSV (categories with descriptions)
    2. Build a dynamic Pydantic model constraining output to valid topics
    3. For each conversation, invoke LLM with structured output
    4. Write results to CSV with topic, confidence, and rationale

The structured output approach guarantees:
    - Topic is always one of the defined taxonomy values (type-safe enum)
    - Confidence is always low/medium/high
    - Rationale is always provided for auditability
"""

import argparse
import logging
import os

import pandas as pd
from dotenv import load_dotenv
from pydantic import Field, create_model
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


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
        default="data/labels.csv",
        help="Path to output labels CSV (default: data/labels.csv)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N conversations (default: all)"
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="OpenAI model to use (default: gpt-5.2)"
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
    Create a LangChain classifier with structured output constrained to the taxonomy.

    This function builds:
        1. A topics block string combining names and descriptions for the prompt
        2. A dynamic Pydantic model with topic constrained to Literal[taxonomy_names]
        3. A LangChain pipeline: prompt -> LLM -> structured output parser

    The structured output ensures:
        - 'topic' is always one of the valid taxonomy names (enforced by Pydantic)
        - 'confidence' is always one of: low, medium, high
        - 'rationale' is always a non-empty explanation string

    Args:
        taxonomy_names: List of valid topic names (e.g., ["Billing", "Returns", ...])
        taxonomy_descs: Parallel list of topic descriptions for LLM context
        model: OpenAI model identifier (e.g., "gpt-4o-mini", "gpt-4o")

    Returns:
        tuple: (classifier_chain, topics_block)
            - classifier_chain: LangChain runnable that takes {"topics": str, "conversation": str}
            - topics_block: Formatted string of all topics for prompt injection

    Example:
        >>> classifier, topics = create_classifier(["Billing", "Returns"], ["...", "..."], "gpt-4o-mini")
        >>> result = classifier.invoke({"topics": topics, "conversation": "I need a refund"})
        >>> print(result.topic)
        'Returns'
    """
    # Build the topics block: each line is "- Name: Description"
    # This format helps the LLM understand what each category means
    topics_block = "\n".join(
        [f"{name}\n{desc}\n\n" for name, desc in zip(taxonomy_names, taxonomy_descs)]
    )

    # Create a dynamic Pydantic model with taxonomy names as a Literal type
    # This constrains the LLM output to only valid taxonomy values
    AllowedTopic = Literal[tuple(taxonomy_names)]  # type: ignore

    TopicLabel = create_model(
        "TopicLabel",
        topic=(AllowedTopic, Field(..., description="The assigned topic label from the taxonomy")),
        confidence=(Literal["low", "medium", "high"], Field(..., description="Classification confidence level")),
        rationale=(str, Field(..., description="Brief explanation for why this topic was chosen")),
    )

    # Build the prompt template with system instructions and human message
    # System prompt establishes the classifier role and rules
    # Human message injects the taxonomy and conversation for each request
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a customer support taxonomy classifier. "
         "Pick exactly ONE topic from the allowed list. "
         "Use the provided topic descriptions to choose correctly. "
         "If multiple apply, choose the PRIMARY customer intent. "
         "If unclear, choose 'General Enquiries & Multi-Intent'."),
        ("human",
         "Allowed topics (name: description):\n{topics}\n\nConversation:\n{conversation}")
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
                "conversation_id": r["conversation_id"],
                "conversation": r["conversation"],
                "topic": out.topic,
                "confidence": out.confidence,
                "rationale": out.rationale
            })
        except Exception as e:
            # Log error but continue processing remaining conversations
            logger.error(f"Failed to classify conversation {r['conversation_id']}: {e}")
            rows.append({
                "conversation_id": r["conversation_id"],
                "conversation": r["conversation"],
                "topic": "ERROR",
                "confidence": "low",
                "rationale": str(e)
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
    logger.info(f"Topic distribution:\n{topic_counts.to_string()}")
    logger.info(f"Confidence distribution:\n{confidence_counts.to_string()}")


if __name__ == "__main__":
    main()
