# ===============================
# Imports & Setup
# ===============================

import os
from dotenv import load_dotenv

import pandas as pd
import gradio as gr

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# ===============================
# UX Validation (NOT ranking logic)
# ===============================

def validate_query(query: str) -> tuple[bool, str]:
    """Minimal UX validation. Does not affect ranking."""
    clean = query.strip()

    if len(clean) < 5:
        return False, "Please describe the book you're looking for."

    if not any(c.isalpha() for c in clean):
        return False, "Please enter a natural language description."

    words = clean.split()
    if len(words) >= 3:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return False, "Your query seems repetitive. Try rephrasing it."

    return True, ""

# ===============================
# UX Warnings (intentionally minimal)
# ===============================

def check_contradictions(query: str, category: str, tone: str) -> str:
    """
    UX hints only.
    Most contradictions (e.g., dark romance) are legitimate,
    so we rely on semantic search instead.
    """
    return ""

# ===============================
# Category Classification (OFFLINE)
# ===============================

def classify_category(raw_category):
    """Run once during preprocessing. Never at runtime."""
    if pd.isna(raw_category):
        return "Other"

    cat = str(raw_category).lower()

    if any(x in cat for x in ["young adult", "ya", "children"]):
        return "Children & YA"
    if any(x in cat for x in ["mystery", "thriller", "crime", "detective", "horror"]):
        return "Mystery & Thriller"
    if "romance" in cat:
        return "Romance"
    if any(x in cat for x in ["fantasy", "science fiction", "sci-fi", "dystopian"]):
        return "Fantasy & Sci-Fi"
    if any(x in cat for x in ["history", "biography", "memoir"]):
        return "History & Biography"
    if any(x in cat for x in ["business", "self-help", "finance"]):
        return "Self-Help & Business"
    if any(x in cat for x in ["philosophy", "psychology", "physics", "biology"]):
        return "Science & Ideas"

    return "Literary & Contemporary"

# ===============================
# Data Loading (Preprocessed)
# ===============================

ENRICHED_FILE = "books_enriched_production.csv"

if os.path.exists(ENRICHED_FILE):
    books = pd.read_csv(ENRICHED_FILE)
else:
    books = pd.read_csv("books_with_emotions.csv")
    books["category"] = books["categories"].apply(classify_category)
    books.to_csv(ENRICHED_FILE, index=False)

print(f"Loaded {len(books)} books")

books["large_thumbnail"] = books["thumbnail"].astype(str) + "&fife=w800"
books["large_thumbnail"] = books["large_thumbnail"].replace(
    {"nan&fife=w800": "cover-not-found.jpg"}
)

# ===============================
# Vector Database
# ===============================

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not set")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=api_key,
)

db_books = Chroma(
    persist_directory="chroma_books_Database",
    embedding_function=embeddings,
)

# ===============================
# Tone Matching (5-tone system)
# ===============================

TONE_PROFILES = {
    "Happy": {"joy": 1.0},
    "Sad": {"sadness": 1.0},
    "Calm": {"neutral": 1.0},
    "Dark": {"fear": 0.6, "sadness": 0.4},
    "Intense": {"anger": 0.5, "fear": 0.5},
}

def compute_tone_score(row, tone):
    if tone == "All" or tone not in TONE_PROFILES:
        return 0.5

    profile = TONE_PROFILES[tone]
    score = sum(row.get(e, 0.5) * w for e, w in profile.items())
    return score / sum(profile.values())

# ===============================
# Keyword Boost (Exact-match tiebreaker)
# ===============================

def compute_keyword_boost(desc, query):
    """
    Exact keyword match bonus: 5% per matching term, max 20%.
    Helps technical queries like 'quantum mechanics textbook'.
    """
    if pd.isna(desc):
        return 0.0

    desc_lower = desc.lower()
    terms = [t for t in query.lower().split() if len(t) >= 3]

    if not terms:
        return 0.0

    matches = sum(1 for t in terms if t in desc_lower)
    return min(0.2, matches * 0.05)

# ===============================
# Core Retrieval & Scoring
# ===============================

def retrieve_recommendations(query, category, tone, top_k=16):
    is_valid, error_msg = validate_query(query)
    if not is_valid:
        return pd.DataFrame(), error_msg

    results = db_books.similarity_search_with_relevance_scores(query, k=800)

    scores = {}
    for doc, score in results:
        try:
            isbn = int(doc.page_content.split()[0].replace('"', ""))
            scores[isbn] = score
        except:
            continue

    df = books[books["isbn13"].isin(scores.keys())].copy()
    if df.empty:
        return df, ""

    # ---- Semantic similarity (normalized) ----
    df["semantic"] = df["isbn13"].map(scores)
    sem_min, sem_max = df["semantic"].min(), df["semantic"].max()
    if sem_max > sem_min:
        df["semantic"] = (df["semantic"] - sem_min) / (sem_max - sem_min)
    else:
        df["semantic"] = 0.5

    # ---- Tone match ----
    df["tone"] = df.apply(lambda r: compute_tone_score(r, tone), axis=1)

    # ---- Category match ----
    if category != "All":
        df["category_match"] = df["category"].eq(category).astype(float)
    else:
        df["category_match"] = 0.0

    # ---- Keyword boost ----
    df["keyword"] = df["description"].apply(
        lambda d: compute_keyword_boost(d, query)
    )

    # ---- Final score ----
    if category == "All":
        df["final_score"] = (
            0.85 * df["semantic"]
            + 0.10 * df["tone"]
            + 0.05 * df["keyword"]
        )
    else:
        df["final_score"] = (
            0.65 * df["semantic"]
            + 0.20 * df["category_match"]
            + 0.10 * df["tone"]
            + 0.05 * df["keyword"]
        )

    return df.sort_values("final_score", ascending=False).head(top_k), ""

# ===============================
# Gradio UI
# ===============================

current_recommendations = []

def search_books(query, category, tone):
    global current_recommendations

    warning = check_contradictions(query, category, tone)
    df, error_msg = retrieve_recommendations(query, category, tone)

    if error_msg:
        gr.Warning(error_msg)
        return [None] * 16

    if warning:
        gr.Warning(warning)

    current_recommendations = df.reset_index(drop=True)
    return [
        df.iloc[i]["large_thumbnail"] if i < len(df) else None
        for i in range(16)
    ]

def show_details(index):
    idx = int(index)
    if idx >= len(current_recommendations):
        return [None, "", "", "", "", "", "", gr.update(visible=False)]

    row = current_recommendations.iloc[idx]

    return (
        row["large_thumbnail"],
        f"## {row['title']}",
        f"**Authors:** {row['authors']}",
        f"**Rating:** {row['average_rating']}",
        f"**Published:** {row['published_year']}",
        f"**Category:** {row['category']}",
        f"### Description\n{row.get('description','')[:800]}...",
        gr.update(visible=True),
    )

# ===============================
# Launch App
# ===============================

categories = ["All"] + sorted(books["category"].dropna().unique())
tones = ["All"] + list(TONE_PROFILES.keys())

with gr.Blocks(theme=gr.themes.Ocean()) as app:
    gr.Markdown("# LitMatch: Semantic Book Recommendations")

    with gr.Row():
        query = gr.Textbox(label="Describe a book")
        category = gr.Dropdown(categories, value="All")
        tone = gr.Dropdown(tones, value="All")
        btn = gr.Button("Search")

    images = [gr.Image(height=250) for _ in range(16)]
    btn.click(search_books, [query, category, tone], images)

if __name__ == "__main__":
    app.launch()
