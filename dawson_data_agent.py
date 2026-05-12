#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dawson Data — Executive Business Intelligence AI Agent

ISYS 573: AI Agent / Agentic AI | Group 3 | Track A
Team: Antonia Briones · Brea Robinson · Amanda Acosta · Stefan Jurado · Steven Vu

This is an auto-generated Python script from the Jupyter notebook.
Run with: python dawson_data_agent.py
"""

# ── Standard library ─────────────────────────────────────────────────────────
import os
import re
import json
import warnings
import textwrap
from pathlib import Path
from typing import TypedDict, Literal, Optional

# ── Data ─────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np

# ── LangChain / LangGraph ─────────────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# ── UI ────────────────────────────────────────────────────────────────────────
import gradio as gr

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
# Set your Anthropic API key here OR export it as an environment variable:
#   export ANTHROPIC_API_KEY="sk-ant-..."
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY") or "sk-ant-api03-g3IQAA"

MODEL_NAME        = "claude-sonnet-4-5"          # Claude model to use
EMBED_MODEL       = "all-MiniLM-L6-v2"           # Local HuggingFace embedding model
CHROMA_DIR        = "./chroma_db"                 # Persistent vector store location
COLLECTION_NAME   = "dawson_data_kb"              # ChromaDB collection name
TOP_K             = 4                             # Number of RAG chunks to retrieve
MAX_TOKENS        = 1024                          # Max response tokens
CHUNK_SIZE        = 500                           # RAG chunk size (tokens approx.)
CHUNK_OVERLAP     = 50                            # Overlap between chunks

# ── Dataset paths (update these to point to your downloaded Kaggle CSVs) ──────
DATA_DIR          = Path("./data")
SALES_CSV         = DATA_DIR / "amazon_sales.csv"       # karkavelrajaj/amazon-sales-dataset
REVIEWS_CSV       = DATA_DIR / "amazon_reviews.csv"     # kritanjalijain/amazon-reviews
SALES_2025_CSV    = DATA_DIR / "amazon_sales_2025.csv"  # zahidmughal2343/amazon-sales-2025

print("✅ Imports and configuration loaded.")
print(f"   Model     : {MODEL_NAME}")
print(f"   Embeddings: {EMBED_MODEL}")
print(f"   Vector DB : {CHROMA_DIR}")


# ── Helper: generate synthetic sample data ────────────────────────────────────
def _make_sample_sales() -> pd.DataFrame:
    """Generates a small synthetic sales DataFrame for demo/testing."""
    rng = np.random.default_rng(42)
    categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports"]
    records = []
    for month in range(1, 13):
        for cat in categories:
            records.append({
                "month":          month,
                "category":       cat,
                "product_name":   f"{cat} Product {rng.integers(100, 999)}",
                "units_sold":     int(rng.integers(50, 500)),
                "revenue_usd":    round(float(rng.uniform(1000, 50000)), 2),
                "avg_price_usd":  round(float(rng.uniform(10, 200)), 2),
                "rating":         round(float(rng.uniform(3.0, 5.0)), 1),
                "region":         rng.choice(["North", "South", "East", "West"]),
            })
    return pd.DataFrame(records)


def _make_sample_reviews() -> pd.DataFrame:
    """Generates a small synthetic reviews DataFrame for demo/testing."""
    templates = [
        ("Electronics", "Great battery life and fast delivery. Very satisfied.", 5),
        ("Clothing",    "Sizing runs small but quality is excellent.",           4),
        ("Home & Kitchen", "Stopped working after 2 months. Disappointed.",     2),
        ("Books",       "Insightful and well-written. Highly recommend.",        5),
        ("Sports",      "Decent quality for the price. Would buy again.",        4),
        ("Electronics", "Poor customer support when the item arrived damaged.",  1),
        ("Clothing",    "Fast shipping and exactly as described.",               5),
        ("Home & Kitchen", "Works perfectly. Best purchase this year.",          5),
    ]
    return pd.DataFrame(templates, columns=["category", "review_text", "star_rating"])


def _make_sample_sales_2025() -> pd.DataFrame:
    """Generates synthetic 2025 sales snapshot."""
    rng = np.random.default_rng(7)
    categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports"]
    records = []
    for q in ["Q1 2025", "Q2 2025"]:
        for cat in categories:
            records.append({
                "quarter":       q,
                "category":      cat,
                "total_revenue": round(float(rng.uniform(80000, 300000)), 2),
                "units_sold":    int(rng.integers(500, 5000)),
                "yoy_growth_pct":round(float(rng.uniform(-10, 40)), 1),
            })
    return pd.DataFrame(records)


# ── Load datasets ─────────────────────────────────────────────────────────────
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the three Kaggle datasets.  Falls back to synthetic data if CSVs
    are not present so the script is fully runnable out of the box.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _load(path: Path, fallback_fn, label: str) -> pd.DataFrame:
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            print(f"  ✅ Loaded   {label}: {len(df):,} rows  ← {path}")
        else:
            df = fallback_fn()
            print(f"  ⚠️  Synthetic {label}: {len(df):,} rows  (CSV not found at {path})")
        return df

    print("Loading datasets…")
    sales    = _load(SALES_CSV,      _make_sample_sales,      "Amazon Sales")
    reviews  = _load(REVIEWS_CSV,    _make_sample_reviews,    "Amazon Reviews")
    sales25  = _load(SALES_2025_CSV, _make_sample_sales_2025, "Amazon Sales 2025")
    return sales, reviews, sales25


df_sales, df_reviews, df_sales25 = load_datasets()

print("\nSales columns  :", list(df_sales.columns))
print("Reviews columns:", list(df_reviews.columns))
print("Sales25 columns:", list(df_sales25.columns))


# ────────────────────────────────────────────────────────────────────────────────
# STRUCTURED DATA ANALYSIS TOOLS
# ────────────────────────────────────────────────────────────────────────────────

class StructuredAnalyzer:
    """
    Provides structured analytical capabilities over the loaded DataFrames.
    All methods return formatted plain-text suitable for LLM consumption.
    """

    def __init__(self, sales: pd.DataFrame, reviews: pd.DataFrame, sales25: pd.DataFrame):
        self.sales   = sales.copy()
        self.reviews = reviews.copy()
        self.s25     = sales25.copy()
        self._normalise()

    def _normalise(self):
        """Standardise column names across datasets."""
        # Sales: ensure numeric columns are numeric
        for col in ["units_sold", "revenue_usd", "avg_price_usd", "rating"]:
            if col in self.sales.columns:
                self.sales[col] = pd.to_numeric(self.sales[col], errors="coerce")

        # Reviews: ensure star_rating is numeric
        if "star_rating" in self.reviews.columns:
            self.reviews["star_rating"] = pd.to_numeric(
                self.reviews["star_rating"], errors="coerce"
            )

        # Sales 2025: ensure numeric
        for col in ["total_revenue", "units_sold", "yoy_growth_pct"]:
            if col in self.s25.columns:
                self.s25[col] = pd.to_numeric(self.s25[col], errors="coerce")

    # ── Individual analysis methods ────────────────────────────────────────────

    def top_categories_by_revenue(self, n: int = 5) -> str:
        """Returns the top-N categories ranked by total revenue."""
        if "category" not in self.sales.columns or "revenue_usd" not in self.sales.columns:
            return "Revenue data by category is unavailable in the current dataset."
        top = (
            self.sales.groupby("category")["revenue_usd"]
            .sum()
            .sort_values(ascending=False)
            .head(n)
        )
        lines = [f"Top {n} Categories by Total Revenue:"]
        for rank, (cat, rev) in enumerate(top.items(), 1):
            lines.append(f"  {rank}. {cat}: ${rev:,.2f}")
        return "\n".join(lines)

    def monthly_revenue_trend(self) -> str:
        """Returns month-by-month revenue trend."""
        if "month" not in self.sales.columns or "revenue_usd" not in self.sales.columns:
            return "Monthly revenue trend data is unavailable."
        trend = (
            self.sales.groupby("month")["revenue_usd"]
            .sum()
            .sort_index()
        )
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        lines = ["Monthly Revenue Trend:"]
        for m, rev in trend.items():
            label = month_names[int(m) - 1] if 1 <= int(m) <= 12 else str(m)
            lines.append(f"  {label}: ${rev:,.2f}")
        return "\n".join(lines)

    def top_products_by_revenue(self, n: int = 5) -> str:
        """Returns the top-N products by total revenue."""
        col = next(
            (c for c in self.sales.columns if "product" in c.lower() and "name" in c.lower()),
            next((c for c in self.sales.columns if "product" in c.lower()), None)
        )
        if col is None or "revenue_usd" not in self.sales.columns:
            return "Product-level revenue data is unavailable."
        top = (
            self.sales.groupby(col)["revenue_usd"]
            .sum()
            .sort_values(ascending=False)
            .head(n)
        )
        lines = [f"Top {n} Products by Revenue:"]
        for rank, (prod, rev) in enumerate(top.items(), 1):
            lines.append(f"  {rank}. {prod}: ${rev:,.2f}")
        return "\n".join(lines)

    def average_rating_by_category(self) -> str:
        """Returns average product rating per category."""
        if "category" not in self.sales.columns or "rating" not in self.sales.columns:
            return "Rating data by category is unavailable."
        avg = (
            self.sales.groupby("category")["rating"]
            .mean()
            .sort_values(ascending=False)
        )
        lines = ["Average Customer Rating by Category:"]
        for cat, r in avg.items():
            lines.append(f"  {cat}: {r:.2f} / 5.0")
        return "\n".join(lines)

    def yoy_growth_2025(self) -> str:
        """Returns 2025 year-over-year growth summary."""
        if "yoy_growth_pct" not in self.s25.columns:
            return "2025 YoY growth data is unavailable."
        lines = ["2025 Year-over-Year Revenue Growth by Category:"]
        for _, row in self.s25.sort_values("yoy_growth_pct", ascending=False).iterrows():
            cat    = row.get("category", "Unknown")
            qtr    = row.get("quarter",  "")
            growth = row.get("yoy_growth_pct", 0)
            sign   = "+" if growth >= 0 else ""
            lines.append(f"  {qtr} | {cat}: {sign}{growth:.1f}%")
        return "\n".join(lines)

    def underperforming_categories(self, threshold_pct: float = 25.0) -> str:
        """Identifies categories in the bottom revenue percentile."""
        if "category" not in self.sales.columns or "revenue_usd" not in self.sales.columns:
            return "Cannot determine underperforming categories without revenue data."
        cat_rev = self.sales.groupby("category")["revenue_usd"].sum()
        cutoff  = cat_rev.quantile(threshold_pct / 100)
        under   = cat_rev[cat_rev <= cutoff].sort_values()
        if under.empty:
            return "No categories are significantly underperforming."
        lines = [f"Underperforming Categories (bottom {threshold_pct:.0f}%):"]
        for cat, rev in under.items():
            lines.append(f"  ⚠️  {cat}: ${rev:,.2f} total revenue")
        return "\n".join(lines)

    def sentiment_summary(self) -> str:
        """Summarises review sentiment by category."""
        if "star_rating" not in self.reviews.columns:
            return "Review sentiment data is unavailable."
        avg = (
            self.reviews.groupby("category")["star_rating"]
            .mean()
            .sort_values(ascending=False)
        )
        lines = ["Customer Sentiment by Category (avg. star rating):"]
        for cat, score in avg.items():
            emoji = "😊" if score >= 4 else ("😐" if score >= 3 else "😞")
            lines.append(f"  {emoji} {cat}: {score:.2f} / 5.0")
        return "\n".join(lines)

    def kpi_dashboard(self) -> str:
        """Returns a high-level KPI summary."""
        parts = []
        if "revenue_usd" in self.sales.columns:
            total = self.sales["revenue_usd"].sum()
            parts.append(f"Total Revenue: ${total:,.2f}")
        if "units_sold" in self.sales.columns:
            units = self.sales["units_sold"].sum()
            parts.append(f"Total Units Sold: {units:,}")
        if "rating" in self.sales.columns:
            avg_r = self.sales["rating"].mean()
            parts.append(f"Average Product Rating: {avg_r:.2f}")
        if "category" in self.sales.columns:
            n_cat = self.sales["category"].nunique()
            parts.append(f"Number of Categories: {n_cat}")
        if "star_rating" in self.reviews.columns:
            avg_sent = self.reviews["star_rating"].mean()
            parts.append(f"Average Review Sentiment: {avg_sent:.2f} / 5.0")
        return "Executive KPI Dashboard:\n" + "\n".join(f"  • {p}" for p in parts)

    def run_all(self) -> str:
        """Runs all analysis methods and concatenates results."""
        sections = [
            self.kpi_dashboard(),
            self.top_categories_by_revenue(),
            self.top_products_by_revenue(),
            self.monthly_revenue_trend(),
            self.underperforming_categories(),
            self.average_rating_by_category(),
            self.yoy_growth_2025(),
            self.sentiment_summary(),
        ]
        return "\n\n".join(s for s in sections if s)


# ── Instantiate the analyzer ──────────────────────────────────────────────────
analyzer = StructuredAnalyzer(df_sales, df_reviews, df_sales25)

# Quick smoke test
print(analyzer.kpi_dashboard())
print()
print(analyzer.top_categories_by_revenue())
print()


# ────────────────────────────────────────────────────────────────────────────────
# RAG PIPELINE: BUILD & PERSIST CHROMADB
# ────────────────────────────────────────────────────────────────────────────────

def build_document_corpus(
    sales: pd.DataFrame,
    reviews: pd.DataFrame,
    sales25: pd.DataFrame,
) -> list[Document]:
    """
    Converts all three datasets into LangChain Document objects.
    Each document carries metadata for filtered retrieval.
    """
    docs: list[Document] = []

    # ── Reviews → one doc per row ──────────────────────────────────────────────
    text_col = next(
        (c for c in reviews.columns if "review" in c.lower() and "text" in c.lower()),
        next((c for c in reviews.columns if "review" in c.lower()), None)
    )
    if text_col:
        for _, row in reviews.iterrows():
            text = str(row[text_col]).strip()
            if len(text) < 10:
                continue
            meta = {
                "source":   "amazon_reviews",
                "category": str(row.get("category", "unknown")),
                "rating":   str(row.get("star_rating", "")),
            }
            docs.append(Document(page_content=text, metadata=meta))

    # ── Sales → category-level narrative summaries ─────────────────────────────
    if "category" in sales.columns and "revenue_usd" in sales.columns:
        for cat, grp in sales.groupby("category"):
            rev   = grp["revenue_usd"].sum()
            units = grp["units_sold"].sum() if "units_sold" in grp.columns else "N/A"
            rating = grp["rating"].mean() if "rating" in grp.columns else "N/A"
            text = (
                f"Category: {cat}. "
                f"Total revenue: ${rev:,.2f}. "
                f"Total units sold: {units}. "
                f"Average rating: {rating if isinstance(rating, str) else f'{rating:.2f}'}/5.0. "
            )
            docs.append(Document(
                page_content=text,
                metadata={"source": "amazon_sales", "category": str(cat)}
            ))

    # ── Sales 2025 → quarter-level narrative summaries ─────────────────────────
    if "quarter" in sales25.columns and "total_revenue" in sales25.columns:
        for _, row in sales25.iterrows():
            growth = row.get("yoy_growth_pct", "N/A")
            text = (
                f"In {row['quarter']}, the {row.get('category', 'overall')} category "
                f"generated ${row['total_revenue']:,.2f} in revenue with "
                f"{row.get('units_sold', 'N/A')} units sold. "
                f"Year-over-year growth: {growth}%."
            )
            docs.append(Document(
                page_content=text,
                metadata={
                    "source":   "amazon_sales_2025",
                    "quarter":  str(row.get("quarter", "")),
                    "category": str(row.get("category", "")),
                }
            ))

    # ── Static internal business documents ─────────────────────────────────────
    business_docs = [
        "Amazon Historical Revenue Trend (2005-2025): Net revenue grew from modest levels "
        "in 2005 to approximately 717 billion U.S. dollars in 2025. The company's revenue "
        "model includes e-commerce retail (largest contributor), third-party seller services, "
        "cloud computing (AWS), and subscription services. North America generated 423 billion "
        "dollars in 2025 compared to 161 billion internationally. Amazon's brand value exceeds "
        "338 billion dollars, making it one of the world's most valuable companies.",

        "Dawson Data Q4 Strategic Memo: Management has prioritized expansion into "
        "the Electronics and Home & Kitchen categories following strong YoY growth. "
        "Budget allocations for Q1 2025 reflect a 20% increase in marketing spend "
        "for top-performing SKUs. Underperforming categories will undergo SKU rationalization.",

        "Inventory Policy Document: Dawson Data maintains a 30-day safety stock for "
        "high-velocity products. Reorder triggers are set at 15% below the 90-day "
        "moving average. Overstock situations in Clothing have resulted in margin "
        "compression of approximately 8% in H2.",

        "Customer Experience Report: Net Promoter Score (NPS) stands at 42 for the "
        "current quarter. Primary complaints center on delivery delays (34% of negative "
        "reviews) and product quality inconsistency in the Clothing category (28%). "
        "Electronics maintains the highest NPS segment at 61.",

        "Regional Performance Summary: The West region leads in total revenue contribution "
        "at 34%, followed by East (28%), North (22%), and South (16%). The South region "
        "shows the highest growth rate at 18% YoY despite the lowest absolute volume.",

        "Pricing Strategy Overview: Dynamic pricing is applied to Electronics and Books. "
        "Clothing and Sports categories operate on fixed seasonal pricing with end-of-season "
        "markdowns of up to 40%. Home & Kitchen pricing is reviewed quarterly based on "
        "competitor benchmarking and demand elasticity analysis.",
    ]
    for i, text in enumerate(business_docs):
        docs.append(Document(
            page_content=text,
            metadata={"source": "internal_documents", "doc_id": str(i)}
        ))

    return docs


# ── Chunk documents ────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],
)

raw_docs = build_document_corpus(df_sales, df_reviews, df_sales25)
chunks   = splitter.split_documents(raw_docs)
print(f"📄 Raw documents : {len(raw_docs):,}")
print(f"✂️  After chunking : {len(chunks):,} chunks")


# ── Embed & persist to ChromaDB ───────────────────────────────────────────────
print(f"\nLoading embedding model: {EMBED_MODEL} …")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("Building ChromaDB vector store … (this may take 1–2 minutes on first run)")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
print(f"✅ Vector store ready — {vectorstore._collection.count():,} vectors indexed.")
print()


# ────────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ────────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the Dawson Data Executive Business Intelligence AI Agent — a trusted analytical \
advisor for senior business stakeholders. Your role is to transform business data and \
internal documents into clear, actionable insights.

CORE PRINCIPLES:
1. GROUND every claim in the provided data context. Never fabricate numbers or trends.
2. STRUCTURE responses clearly: lead with the direct answer, then support with evidence.
3. DISTINGUISH between structured metrics (sales data) and qualitative signals (reviews/docs).
4. RECOMMEND specific, actionable next steps when appropriate.
5. FLAG uncertainty explicitly when data is incomplete or ambiguous.
6. AVOID jargon — responses must be understandable to non-technical executives.

RESPONSE FORMAT:
- Start with a concise executive summary (2–3 sentences).
- Provide supporting evidence with specific numbers.
- End with 2–3 actionable recommendations.
- Use bullet points for lists; keep prose paragraphs to 3 sentences max.

SAFETY CONSTRAINTS:
- You are a decision-SUPPORT tool. Do not make autonomous business decisions.
- If a query falls outside your data scope, say so clearly rather than guessing.
- Do not disclose raw database schemas or internal system architecture details.
"""


def build_structured_prompt(user_query: str, structured_context: str) -> str:
    return f"""\
The user has asked a question that requires structured data analysis.

USER QUERY:
{user_query}

STRUCTURED DATA CONTEXT (from Dawson Data sales and performance datasets):
{structured_context}

Based solely on the structured data above, provide a clear executive-level response.
Include specific numbers and identify the most important business insight.
"""


def build_rag_prompt(user_query: str, rag_context: str) -> str:
    return f"""\
The user has asked a question that requires analysis of internal documents and qualitative data.

USER QUERY:
{user_query}

RETRIEVED DOCUMENT CONTEXT (from internal reports, reviews, and business documents):
{rag_context}

Based on the retrieved documents above, provide a clear executive-level response.
Cite the source type (e.g., 'customer reviews', 'strategic memo') when referencing evidence.
"""


def build_hybrid_prompt(user_query: str, structured_context: str, rag_context: str) -> str:
    return f"""\
The user has asked a complex question requiring both quantitative data and qualitative insights.

USER QUERY:
{user_query}

STRUCTURED DATA CONTEXT (sales metrics, KPIs):
{structured_context}

QUALITATIVE DOCUMENT CONTEXT (reviews, internal reports, strategic documents):
{rag_context}

Synthesize both data sources into a coherent executive-level response.
Clearly indicate when insights come from quantitative data vs. qualitative sources.
Identify where the two sources corroborate or contradict each other.
"""


def build_clarification_prompt(user_query: str) -> str:
    return f"""\
The user submitted the following query, but it is ambiguous or outside the available data scope:

USER QUERY:
{user_query}

Politely explain what information IS available, and ask one clarifying question to \
better understand what the user needs. Suggest 2–3 example questions you CAN answer.
"""


print("✅ Prompt templates defined.")
print()


# ────────────────────────────────────────────────────────────────────────────────
# LANGGRAPH AGENT WORKFLOW
# ────────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Typed state dictionary passed between LangGraph nodes."""
    user_query:          str                            # Original user input
    intent:              str                            # classified intent
    structured_context:  str                            # output of structured_retriever
    rag_context:         str                            # output of rag_retriever
    synthesized_context: str                            # merged context
    raw_response:        str                            # LLM output before validation
    final_response:      str                            # validated, formatted output
    confidence:          str                            # "high" | "medium" | "low"
    error:               Optional[str]                  # error message if any node fails
    node_trace:          list[str]                      # execution trace for debugging


# ── LLM client ────────────────────────────────────────────────────────────────
llm = ChatAnthropic(
    model=MODEL_NAME,
    anthropic_api_key=ANTHROPIC_API_KEY,
    max_tokens=MAX_TOKENS,
    temperature=0.3,     # low temperature for factual business analysis
)


# Keywords that signal each query type (fast rule-based classifier)
_STRUCTURED_KEYWORDS = [
    "revenue", "sales", "units", "kpi", "metric", "top", "best", "worst",
    "trend", "growth", "decline", "performance", "rank", "profit", "price",
    "category", "product", "region", "month", "quarter", "dashboard", "total",
    "average", "highest", "lowest", "increase", "decrease",
]
_UNSTRUCTURED_KEYWORDS = [
    "review", "feedback", "complaint", "sentiment", "customer", "opinion",
    "report", "document", "strategy", "policy", "memo", "nps", "satisfaction",
    "quality", "experience", "recommend",
]


def intent_classifier(state: AgentState) -> AgentState:
    """Classifies the user query into: 'structured', 'unstructured', 'hybrid', or 'clarification_needed'."""
    query = state["user_query"].lower()

    has_structured   = any(kw in query for kw in _STRUCTURED_KEYWORDS)
    has_unstructured = any(kw in query for kw in _UNSTRUCTURED_KEYWORDS)

    if has_structured and has_unstructured:
        intent = "hybrid"
    elif has_structured:
        intent = "structured"
    elif has_unstructured:
        intent = "unstructured"
    elif len(query.split()) < 3:
        intent = "clarification_needed"
    else:
        intent = "hybrid"

    trace = state.get("node_trace", []) + [f"intent_classifier → {intent}"]
    return {**state, "intent": intent, "node_trace": trace}


def structured_retriever(state: AgentState) -> AgentState:
    """Runs structured analysis based on query keywords."""
    query = state["user_query"].lower()
    sections: list[str] = []

    try:
        sections.append(analyzer.kpi_dashboard())

        if any(kw in query for kw in ["category", "categories"]):
            sections.append(analyzer.top_categories_by_revenue())
            sections.append(analyzer.underperforming_categories())

        if any(kw in query for kw in ["product", "products", "top", "best"]):
            sections.append(analyzer.top_products_by_revenue())

        if any(kw in query for kw in ["trend", "month", "monthly", "time"]):
            sections.append(analyzer.monthly_revenue_trend())

        if any(kw in query for kw in ["growth", "2025", "yoy", "year-over-year"]):
            sections.append(analyzer.yoy_growth_2025())

        if any(kw in query for kw in ["rating", "score", "quality"]):
            sections.append(analyzer.average_rating_by_category())

        if any(kw in query for kw in ["sentiment", "review", "feedback"]):
            sections.append(analyzer.sentiment_summary())

        if len(sections) <= 1:
            sections = [analyzer.run_all()]

        context = "\n\n".join(sections)
        trace   = state["node_trace"] + ["structured_retriever → success"]
        return {**state, "structured_context": context, "node_trace": trace}

    except Exception as e:
        error_str = str(e).encode('utf-8', errors='replace').decode('utf-8')
        trace = state["node_trace"] + [f"structured_retriever → ERROR: {error_str}"]
        return {**state, "structured_context": "", "error": error_str, "node_trace": trace}


def rag_retriever(state: AgentState) -> AgentState:
    """Performs semantic search against ChromaDB."""
    try:
        docs = retriever.invoke(state["user_query"])
        if not docs:
            context = "No relevant documents found in the knowledge base for this query."
        else:
            parts = []
            for i, doc in enumerate(docs, 1):
                src = doc.metadata.get("source", "unknown")
                cat = doc.metadata.get("category", "")
                label = f"[{i}] Source: {src}" + (f" | Category: {cat}" if cat else "")
                parts.append(f"{label}\n{doc.page_content}")
            context = "\n\n".join(parts)

        trace = state["node_trace"] + [f"rag_retriever → {len(docs)} chunks retrieved"]
        return {**state, "rag_context": context, "node_trace": trace}

    except Exception as e:
        error_str = str(e).encode('utf-8', errors='replace').decode('utf-8')
        trace = state["node_trace"] + [f"rag_retriever → ERROR: {error_str}"]
        return {**state, "rag_context": "", "error": error_str, "node_trace": trace}


def synthesizer(state: AgentState) -> AgentState:
    """Merges contexts and selects appropriate prompt template."""
    intent = state["intent"]
    query  = state["user_query"]
    s_ctx  = state.get("structured_context", "")
    r_ctx  = state.get("rag_context", "")

    if intent == "structured":
        merged = build_structured_prompt(query, s_ctx or "No structured data available.")
    elif intent == "unstructured":
        merged = build_rag_prompt(query, r_ctx or "No documents retrieved.")
    elif intent == "clarification_needed":
        merged = build_clarification_prompt(query)
    else:
        merged = build_hybrid_prompt(
            query,
            s_ctx or "No structured data available.",
            r_ctx or "No documents retrieved.",
        )

    trace = state["node_trace"] + [f"synthesizer → prompt built ({intent})"]
    return {**state, "synthesized_context": merged, "node_trace": trace}


def reasoning_engine(state: AgentState) -> AgentState:
    """Sends prompt to Claude LLM."""
    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=state["synthesized_context"]),
        ]
        response = llm.invoke(messages)
        raw_response = response.content
        trace = state["node_trace"] + ["reasoning_engine → LLM response received"]
        return {**state, "raw_response": raw_response, "node_trace": trace}

    except Exception as e:
        # Encode error message safely, handling Unicode characters
        error_str = str(e).encode('utf-8', errors='replace').decode('utf-8')
        error_msg = f"LLM call failed: {type(e).__name__}: {error_str}"
        trace = state["node_trace"] + [f"reasoning_engine → ERROR: {error_msg}"]
        return {**state, "raw_response": "", "error": error_msg, "node_trace": trace}


_HALLUCINATION_FLAGS = [
    "i don't have access", "i cannot access", "as an ai", "i'm not able to",
    "i do not have real-time", "my knowledge cutoff", "i don't know",
]


def validator(state: AgentState) -> AgentState:
    """Validates response for completeness and hallucination."""
    raw = state.get("raw_response", "")
    err = state.get("error", None)

    if err or not raw:
        fallback = (
            "⚠️ The agent encountered an error processing your request. "
            "Please try rephrasing your question or contact your system administrator.\n"
            f"Technical detail: {err or 'empty response'}"
        )
        trace = state["node_trace"] + ["validator → FAIL (error passthrough)"]
        return {**state, "raw_response": fallback, "confidence": "low", "node_trace": trace}

    raw_lower = raw.lower()

    if any(flag in raw_lower for flag in _HALLUCINATION_FLAGS):
        confidence = "low"
        trace_msg = "validator → low confidence (hallucination signal detected)"
    elif len(raw) < 100:
        confidence = "low"
        trace_msg = "validator → low confidence (response too short)"
    elif len(raw) < 300:
        confidence = "medium"
        trace_msg = "validator → medium confidence"
    else:
        confidence = "high"
        trace_msg = "validator → high confidence"

    trace = state["node_trace"] + [trace_msg]
    return {**state, "confidence": confidence, "node_trace": trace}


def output_formatter(state: AgentState) -> AgentState:
    """Formats final output with confidence badge and footer."""
    raw = state.get("raw_response", "")
    confidence = state.get("confidence", "medium")
    intent = state.get("intent", "hybrid")

    badge_map = {
        "high": "🟢 High Confidence",
        "medium": "🟡 Medium Confidence",
        "low": "🔴 Low Confidence — review recommended",
    }
    source_map = {
        "structured": "📊 Structured Data (Sales KPIs)",
        "unstructured": "📄 Document Knowledge Base (RAG)",
        "hybrid": "📊 Structured Data + 📄 Document Knowledge Base",
        "clarification_needed": "ℹ️  Agent Guidance",
    }

    header = f"**{badge_map.get(confidence, '')}**  |  Source: {source_map.get(intent, 'Combined')}"
    footer = "\n\n---\n*This response is generated by the Dawson Data BI Agent.*"

    final = f"{header}\n\n{raw}{footer}"
    trace = state["node_trace"] + ["output_formatter → done"]
    return {**state, "final_response": final, "node_trace": trace}


def route_after_classifier(state: AgentState) -> Literal["structured_retriever", "rag_retriever", "both", "synthesizer"]:
    """Routes to appropriate retriever based on intent."""
    intent = state["intent"]
    if intent == "structured":
        return "structured_retriever"
    elif intent == "unstructured":
        return "rag_retriever"
    elif intent == "clarification_needed":
        return "synthesizer"
    else:
        return "both"


def route_after_structured(state: AgentState) -> Literal["rag_retriever", "synthesizer"]:
    """Routes after structured retrieval for hybrid queries."""
    return "rag_retriever" if state["intent"] == "hybrid" else "synthesizer"


# ── Build the LangGraph state machine ─────────────────────────────────────────
workflow = StateGraph(AgentState)

# Register nodes
workflow.add_node("intent_classifier", intent_classifier)
workflow.add_node("structured_retriever", structured_retriever)
workflow.add_node("rag_retriever", rag_retriever)
workflow.add_node("synthesizer", synthesizer)
workflow.add_node("reasoning_engine", reasoning_engine)
workflow.add_node("validator", validator)
workflow.add_node("output_formatter", output_formatter)

# Connections
workflow.set_entry_point("intent_classifier")

workflow.add_conditional_edges(
    "intent_classifier",
    route_after_classifier,
    {
        "structured_retriever": "structured_retriever",
        "rag_retriever": "rag_retriever",
        "both": "structured_retriever",
        "synthesizer": "synthesizer",
    },
)

workflow.add_conditional_edges(
    "structured_retriever",
    route_after_structured,
    {"rag_retriever": "rag_retriever", "synthesizer": "synthesizer"},
)

workflow.add_edge("rag_retriever", "synthesizer")
workflow.add_edge("synthesizer", "reasoning_engine")
workflow.add_edge("reasoning_engine", "validator")
workflow.add_edge("validator", "output_formatter")
workflow.add_edge("output_formatter", END)

# Compile
agent = workflow.compile()

print("✅ LangGraph agent compiled successfully.")
print(f"   Nodes: {list(workflow.nodes.keys())}")
print()


# ────────────────────────────────────────────────────────────────────────────────
# AGENT RUNNER
# ────────────────────────────────────────────────────────────────────────────────

def run_agent(user_query: str, verbose: bool = False) -> str:
    """Runs the full LangGraph workflow for a given user query."""
    query = user_query.strip()
    if not query:
        return "⚠️ Please enter a question."
    if len(query) > 2000:
        return "⚠️ Query is too long (max 2,000 characters)."

    _INJECTION_PATTERNS = [
        r"ignore (previous|all|above) instructions?",
        r"system prompt",
        r"jailbreak",
        r"disregard .{0,30} instructions?",
    ]
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return "⚠️ Your query contains patterns that cannot be processed."

    initial_state: AgentState = {
        "user_query": query,
        "intent": "",
        "structured_context": "",
        "rag_context": "",
        "synthesized_context": "",
        "raw_response": "",
        "final_response": "",
        "confidence": "",
        "error": None,
        "node_trace": [],
    }

    result = agent.invoke(initial_state)

    if verbose:
        print("\n── Node Execution Trace ──")
        for step in result["node_trace"]:
            print(f"  {step}")
        print()

    return result["final_response"]


print("✅ run_agent() helper ready.")
print()


# ────────────────────────────────────────────────────────────────────────────────
# GRADIO UI
# ────────────────────────────────────────────────────────────────────────────────

def chat_handler(message: str, history: list) -> str:
    """Gradio chat callback."""
    return run_agent(message)


EXAMPLE_QUERIES = [
    "What are the top 5 revenue-generating categories?",
    "Which categories are underperforming and why?",
    "What is our year-over-year growth in 2025?",
    "What are customers saying about the Clothing category?",
    "Which categories have high sales but low customer ratings?",
    "Give me an executive KPI dashboard summary.",
]

CSS = """
.gradio-container { max-width: 900px !important; margin: auto !important; }
.chat-header { background: linear-gradient(135deg, #1E3A5F, #2563EB);
               color: white; padding: 20px 24px; border-radius: 12px;
               margin-bottom: 16px; }
.chat-header h2 { margin: 0; font-size: 1.4em; }
.chat-header p  { margin: 4px 0 0; opacity: 0.85; font-size: 0.9em; }
"""

with gr.Blocks(css=CSS, title="Dawson Data BI Agent") as demo:

    gr.HTML("""
        <div class="chat-header">
          <h2>📊 Dawson Data — Executive Business Intelligence Agent</h2>
          <p>Ask any business question about sales, revenue, trends, customer sentiment.</p>
        </div>
    """)

    chatbot = gr.ChatInterface(
        fn=chat_handler,
        chatbot=gr.Chatbot(height=480, show_label=False, render_markdown=True),
        textbox=gr.Textbox(
            placeholder="Ask a business question...",
            container=False,
            scale=7,
        ),
        examples=EXAMPLE_QUERIES,
        cache_examples=False,
    )

    gr.Markdown("""
    ---
    **Data Sources:** Amazon Sales Dataset · Amazon Reviews · Amazon Sales 2025  
    **Tech Stack:** Anthropic Claude · LangGraph · ChromaDB · Gradio  
    **ISYS 573 Group 3**
    """)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Launching Gradio UI at http://127.0.0.1:7860")
    print("="*60 + "\n")
    demo.launch(share=False)
