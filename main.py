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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Classify customer support conversations by topic using LLM"
    )
    parser.add_argument(
        "--input",
        default="data/conversations.csv",
        help="Path to input conversations CSV (default: data/conversations.csv)"
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
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    return parser.parse_args()


def load_taxonomy(taxonomy_path: str) -> tuple[list[str], list[str]]:
    """Load and validate taxonomy from CSV."""
    tax_df = pd.read_csv(taxonomy_path).fillna("")

    required_cols = {"taxonomy_name", "taxonomy_description"}
    if not required_cols.issubset(set(tax_df.columns)):
        missing = required_cols - set(tax_df.columns)
        raise ValueError(f"taxonomy.csv missing required columns: {missing}")

    taxonomy_names = tax_df["taxonomy_name"].astype(str).str.strip().tolist()
    taxonomy_descs = tax_df["taxonomy_description"].astype(str).str.strip().tolist()

    if len(taxonomy_names) == 0:
        raise ValueError("taxonomy.csv has no rows.")

    if len(set(taxonomy_names)) != len(taxonomy_names):
        dupes = pd.Series(taxonomy_names)[pd.Series(taxonomy_names).duplicated()].unique().tolist()
        raise ValueError(f"taxonomy_name must be unique. Duplicates found: {dupes}")

    if "General Enquiries & Multi-Intent" not in taxonomy_names:
        logger.warning(
            "'General Enquiries & Multi-Intent' not found in taxonomy.csv. "
            "Consider adding it so the model has an 'unclear' bucket."
        )

    return taxonomy_names, taxonomy_descs


def create_classifier(taxonomy_names: list[str], taxonomy_descs: list[str], model: str):
    """Create the LangChain classifier with structured output."""
    # Build the topics block using BOTH name + description
    topics_block = "\n".join(
        [f"- {name}: {desc}" for name, desc in zip(taxonomy_names, taxonomy_descs)]
    )

    # Dynamic Pydantic model with taxonomy as Literal enum
    AllowedTopic = Literal[tuple(taxonomy_names)]  # type: ignore

    TopicLabel = create_model(
        "TopicLabel",
        topic=(AllowedTopic, Field(..., description="The assigned topic label")),
        confidence=(Literal["low", "medium", "high"], Field(..., description="Confidence level")),
        rationale=(str, Field(..., description="Brief explanation for the classification")),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a customer support taxonomy classifier. "
         "Pick exactly ONE topic from the allowed list. "
         "Use the provided topic descriptions to choose correctly. "
         "If multiple apply, choose the PRIMARY customer intent. "
         "If unclear, choose 'General Enquiries & Multi-Intent' (if present in the list)."),
        ("human",
         "Allowed topics (name: description):\n{topics}\n\nConversation:\n{conversation}")
    ])

    llm = ChatOpenAI(model=model, temperature=0)
    classifier = prompt | llm.with_structured_output(TopicLabel)

    return classifier, topics_block


def main():
    load_dotenv()
    args = parse_args()

    logger.info(f"Loading conversations from {args.input}")
    df = pd.read_csv(args.input)

    logger.info(f"Loading taxonomy from {args.taxonomy}")
    taxonomy_names, taxonomy_descs = load_taxonomy(args.taxonomy)
    logger.info(f"Loaded {len(df)} conversations and {len(taxonomy_names)} taxonomy categories")

    classifier, topics_block = create_classifier(taxonomy_names, taxonomy_descs, args.model)

    # Retry wrapper for rate limit handling
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"Rate limited, retrying in {retry_state.next_action.sleep} seconds..."
        )
    )
    def classify(conversation: str):
        return classifier.invoke({
            "topics": topics_block,
            "conversation": conversation
        })

    # Apply limit if specified
    if args.limit:
        df = df.iloc[:args.limit]
        logger.info(f"Processing limited to first {args.limit} conversations")

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
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
            logger.error(f"Failed to classify conversation {r['conversation_id']}: {e}")
            rows.append({
                "conversation_id": r["conversation_id"],
                "conversation": r["conversation"],
                "topic": "ERROR",
                "confidence": "low",
                "rationale": str(e)
            })

    output_df = pd.DataFrame(rows)
    output_df.to_csv(args.output, index=False)
    logger.info(f"Wrote {len(rows)} labels to {args.output}")

    # Summary statistics
    topic_counts = output_df["topic"].value_counts()
    confidence_counts = output_df["confidence"].value_counts()
    logger.info(f"Topic distribution:\n{topic_counts.to_string()}")
    logger.info(f"Confidence distribution:\n{confidence_counts.to_string()}")


if __name__ == "__main__":
    main()
