import os
import re
from dotenv import load_dotenv

import pandas as pd
import gradio as gr

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_query(query: str) -> tuple[bool, str]:
    """
    Validates user query to catch edge cases.

    Returns:
        (is_valid, error_message)
    """
    clean = query.strip()

    # Check minimum length
    if len(clean) < 3:
        return False, "Query too short. Please describe the book you want."

    words = clean.lower().split()

    # Check for excessive repetition (spam detection)
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return (
                False,
                "Your query seems repetitive. Please describe what you're looking for more naturally.",
            )

    # Gibberish detection - must have at least one common word
    common_words = {
        # Basic words
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "about",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "can",
        "may",
        "might",
        "must",
        # Book-related
        "book",
        "books",
        "story",
        "stories",
        "novel",
        "fiction",
        "read",
        "reading",
        "author",
        "writer",
        "character",
        "characters",
        "plot",
        "genre",
        # Genre terms
        "mystery",
        "romance",
        "fantasy",
        "science",
        "detective",
        "love",
        "adventure",
        "thriller",
        "horror",
        "historical",
        "biography",
        "war",
        "family",
        "life",
        "death",
        "journey",
        "quest",
        "magic",
        "space",
        "time",
        "world",
        "man",
        "woman",
        "child",
        "children",
        "hero",
        "villain",
        "dark",
        "light",
        "new",
        "old",
        "find",
        "looking",
        "want",
        "need",
        "like",
        "search",
        "recommend",
        # Scientific/technical terms
        "physics",
        "quantum",
        "chemistry",
        "biology",
        "mathematics",
        "math",
        "theory",
        "atom",
        "molecule",
        "equation",
        "experiment",
        "particle",
        "research",
        "study",
        "analysis",
        "data",
        "algorithm",
        "computing",
        "boson",
        "collider",
        "neutrino",
        "cosmos",
        "relativity",
        "evolution",
    }

    if len(words) >= 2:
        has_common = any(w in common_words for w in words)
        if not has_common:
            return (
                False,
                "We couldn't understand your query. Please use regular words to describe the book.",
            )

    return True, ""


# =============================================================================
# CONTRADICTION DETECTION
# =============================================================================

INCOMPATIBLE_PAIRS = [
    (
        "Romance",
        [
            "corporate thriller",
            "bank",
            "corruption",
            "financial crisis",
            "serial killer",
            "murder mystery",
            "detective",
            "investigation",
            "space opera",
            "alien invasion",
            "dystopia",
            "war strategy",
            "quantum physics",
            "mathematics",
            "scientific theory",
            "horror",
            "demon",
            "terrifying",
            "haunted",
        ],
    ),
    (
        "Children & YA",
        [
            "corporate",
            "erotic",
            "explicit",
            "graphic violence",
            "gore",
            "serial killer",
            "brutal",
            "torture",
            "sexual content",
            "financial markets",
            "stock trading",
            "business strategy",
            "philosophical treatise",
            "academic",
            "dense theory",
            "adult thriller",
            "noir",
        ],
    ),
    (
        "Science & Ideas",
        [
            "romance",
            "love story",
            "romantic comedy",
            "dating",
            "detective story",
            "murder mystery",
            "crime thriller",
            "fantasy quest",
            "magic system",
            "dragon",
            "wizard",
            "space battles",
            "alien invasion",
            "horror",
            "scary",
            "terrifying",
        ],
    ),
    (
        "Mystery & Thriller",
        [
            "peaceful village",
            "gentle humor",
            "cozy",
            "heartwarming",
            "self-help",
            "meditation",
            "spiritual journey",
            "romantic comedy",
            "lighthearted romance",
            "children's bedtime story",
            "picture book",
        ],
    ),
    (
        "Fantasy & Sci-Fi",
        [
            "realistic",
            "true story",
            "memoir",
            "autobiography",
            "historical accuracy",
            "documentary",
            "self-help guide",
            "business manual",
            "how-to",
            "textbook",
            "academic study",
        ],
    ),
    (
        "History & Biography",
        [
            "fantasy",
            "magic",
            "dragons",
            "wizards",
            "science fiction",
            "alien",
            "space travel",
            "time machine",
            "fictional character",
            "imaginary world",
        ],
    ),
    (
        "Self-Help & Business",
        [
            "fiction",
            "novel",
            "story",
            "narrative",
            "fantasy adventure",
            "mystery plot",
            "thriller",
            "science fiction",
            "horror story",
            "children's tale",
            "fairy tale",
        ],
    ),
    (
        "Literary & Contemporary",
        [
            "self-help tips",
            "business strategy",
            "how-to guide",
            "textbook",
            "manual",
            "instruction",
        ],
    ),
]

TONE_CONTRADICTIONS = [
    (
        "Happy",
        [
            "murder",
            "death",
            "killing",
            "tragedy",
            "grief",
            "loss",
            "depression",
            "suicide",
            "apocalypse",
            "war",
            "genocide",
            "torture",
            "abuse",
            "horror",
            "terrifying",
            "brutal",
            "dark",
            "bleak",
            "hopeless",
            "despair",
        ],
    ),
    (
        "Sad",
        [
            "comedy",
            "hilarious",
            "funny",
            "humor",
            "laughing",
            "uplifting",
            "joyful",
            "celebration",
            "triumph",
            "happy ending",
            "feel-good",
            "lighthearted",
        ],
    ),
    (
        "Calm",
        [
            "intense",
            "thrilling",
            "action-packed",
            "explosive",
            "fast-paced",
            "adrenaline",
            "battle",
            "war",
            "horror",
            "terrifying",
            "shocking",
            "brutal",
            "serial killer",
            "murder spree",
            "violent",
        ],
    ),
    (
        "Intense",
        [
            "peaceful",
            "calm",
            "gentle",
            "relaxing",
            "meditative",
            "quiet",
            "slow-paced",
            "contemplative",
            "serene",
            "cozy",
            "comfortable",
            "soothing",
        ],
    ),
    (
        "Dark",
        [
            "uplifting",
            "hopeful",
            "optimistic",
            "joyful",
            "lighthearted",
            "cheerful",
            "bright",
            "positive",
            "feel-good",
            "heartwarming",
            "inspiring",
        ],
    ),
    (
        "Hopeful",
        [
            "hopeless",
            "despair",
            "bleak",
            "nihilistic",
            "pointless",
            "meaningless",
            "futile",
            "doomed",
            "inevitable doom",
        ],
    ),
    (
        "Wholesome",
        [
            "graphic violence",
            "gore",
            "brutal",
            "torture",
            "horror",
            "terrifying",
            "disturbing",
            "explicit",
            "dark",
            "gritty",
        ],
    ),
]


def check_contradictions(query: str, category: str, tone: str) -> str:
    """
    Detects contradictory user inputs (e.g., "murder mystery" + "Romance" category).

    Returns:
        Warning message string, or empty string if no contradictions.
    """
    warnings = []
    query_lower = query.lower()

    # Check category contradictions
    if category != "All":
        for cat, keywords in INCOMPATIBLE_PAIRS:
            if category == cat:
                for keyword in keywords:
                    if keyword in query_lower:
                        warnings.append(
                            f"⚠️ Your query mentions '{keyword}' but you selected {category}. Consider changing category or query."
                        )
                        break
                if warnings:
                    break

    # Check tone contradictions
    if tone != "All":
        for tone_name, keywords in TONE_CONTRADICTIONS:
            if tone == tone_name:
                for keyword in keywords:
                    if keyword in query_lower:
                        warnings.append(
                            f"⚠️ Your query mentions '{keyword}' but you selected {tone} tone. Consider changing tone or query."
                        )
                        break
                if len(warnings) > 1:
                    break

    return " ".join(warnings[:2])  # Max 2 warnings


# =============================================================================
# CATEGORY CLASSIFICATION
# =============================================================================

def classify_category(raw_category):
    """
    Rule-based category mapper - deterministic and fast.

    Handles:
    - Multi-part categories (e.g., "Fiction / Mystery / Thriller")
    - YA override (Children's books take precedence)
    - False positive prevention (e.g., "Political Science" ≠ Sci-Fi)

    Returns:
        One of 9 standard categories
    """
    if raw_category is None or pd.isna(raw_category):
        return "Other"

    # Normalize and split on common delimiters
    cat = str(raw_category).lower()
    parts = re.split(r"[\/;,>|]", cat)
    parts = [p.strip() for p in parts if p.strip()]

    # Detection flags
    has_ya = False
    has_mystery = False
    has_romance = False
    has_fantasy = False
    has_scifi = False
    has_horror = False
    has_history = False
    has_bio = False
    has_business = False
    has_science_nonfiction = False
    has_fiction_tag = False
    has_literary = False

    # Scan all parts
    for p in parts:
        if any(
            x in p for x in ["young adult", "ya", "juvenile", "children", "childrens"]
        ):
            has_ya = True
        if any(
            x in p
            for x in ["mystery", "thriller", "crime", "detective", "suspense", "noir"]
        ):
            has_mystery = True
        if "horror" in p:
            has_horror = True
            has_mystery = True  # Horror buckets into thriller
        if "fantasy" in p:
            has_fantasy = True
        if any(
            x in p
            for x in [
                "sci-fi",
                "science fiction",
                "space opera",
                "space adventure",
                "dystopian",
                "cyberpunk",
            ]
        ):
            has_scifi = True
        if "romance" in p:
            has_romance = True
        if any(x in p for x in ["history", "historical"]):
            has_history = True
        if any(x in p for x in ["biography", "autobiography", "memoir"]):
            has_bio = True
        if any(
            x in p
            for x in [
                "business",
                "self-help",
                "self help",
                "management",
                "finance",
                "economics",
            ]
        ):
            has_business = True
        if any(
            x in p
            for x in [
                "philosophy",
                "psychology",
                "religion",
                "mathematics",
                "physics",
                "chemistry",
                "astronomy",
                "biology",
                "neuroscience",
            ]
        ):
            if "fiction" not in p:  # Avoid misclassifying sci-fi
                has_science_nonfiction = True
        if "fiction" in p:
            has_fiction_tag = True
        if any(
            x in p
            for x in [
                "literary",
                "contemporary",
                "classic",
                "drama",
                "poetry",
                "literature",
            ]
        ):
            has_literary = True

    # Category decision logic (priority order matters)
    if has_ya:
        return "Children & YA"
    if has_mystery or has_horror:
        return "Mystery & Thriller"
    if has_romance:
        return "Romance"
    if has_fantasy or has_scifi:
        # Prevent false positives like "Political Science"
        if "science" in cat and not ("fiction" in cat or "sci-fi" in cat):
            pass  # Not actually sci-fi
        else:
            return "Fantasy & Sci-Fi"
    if has_history or has_bio:
        return "History & Biography"
    if has_science_nonfiction:
        return "Science & Ideas"
    if has_business:
        return "Self-Help & Business"
    if has_fiction_tag or has_literary:
        return "Literary & Contemporary"

    return "Other"


# =============================================================================
# DATA LOADING
# =============================================================================

ENRICHED_FILE = "books_enriched_production.csv"

if os.path.exists(ENRICHED_FILE):
    books = pd.read_csv(ENRICHED_FILE)
else:
    # First run - process raw data
    books = pd.read_csv("books_with_emotions.csv")
    books["category"] = books["categories"].apply(classify_category)
    books.to_csv(ENRICHED_FILE, index=False)

print(f"✔ Loaded {len(books)} books")

# Prepare thumbnails
books["large_thumbnail"] = books["thumbnail"].astype(str) + "&fife=w800"
books["large_thumbnail"] = books["large_thumbnail"].replace(
    {"nan&fife=w800": "cover-not-found.jpg"}
)


# =============================================================================
# VECTOR DATABASE
# =============================================================================

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "⚠️ GOOGLE_API_KEY not found! "
        "Set it in .env locally or in HuggingFace Spaces secrets."
    )

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", google_api_key=api_key
)
db_books = Chroma(
    persist_directory="chroma_books_Database", embedding_function=embeddings
)
print("✔ Vector DB ready")


# =============================================================================
# SCORING & RETRIEVAL
# =============================================================================


# =============================================================================
# SCORING & RETRIEVAL
# =============================================================================

# Tone profiles - emotion weight blends for nuanced matching
TONE_PROFILES = {
    "Happy": {
        "joy": 0.8,
        "surprise": 0.15,
        "neutral": 0.05,
    },
    "Sad": {
        "sadness": 0.7,
        "fear": 0.2,
        "neutral": 0.1,
    },
    "Calm": {
        "neutral": 0.7,
        "joy": 0.3,
    },
    "Intense": {
        "anger": 0.4,
        "fear": 0.4,
        "surprise": 0.2,
    },
    "Dark": {
        "fear": 0.5,
        "sadness": 0.3,
        "anger": 0.2,
    },
    "Hopeful": {
        "joy": 0.4,
        "neutral": 0.3,
        "sadness": 0.3,
    },
    "Wholesome": {
        "joy": 0.5,
        "neutral": 0.4,
        "sadness": 0.1,
    },
}


def compute_tone_score(row, tone):
    """
    Compute tone match using weighted emotion profiles.
    
    Args:
        row: Book row with emotion columns (joy, sadness, anger, fear, surprise, neutral)
        tone: Selected tone name
    
    Returns:
        Weighted tone similarity score (0-1)
    """
    if tone == "All" or tone not in TONE_PROFILES:
        return 0.5
    
    profile = TONE_PROFILES[tone]
    score = 0.0
    weight_sum = 0.0
    
    for emotion, weight in profile.items():
        emotion_value = row.get(emotion, 0.5)
        if pd.isna(emotion_value):
            emotion_value = 0.5
        score += emotion_value * weight
        weight_sum += weight
    
    return score / weight_sum if weight_sum > 0 else 0.5

def compute_metadata_score(row, category, tone):
    """Compute category + tone match score using weighted emotion profiles."""
    selected_cat = row.get("category", "Other")

    # Category score
    if category and category != "All":
        cat_score = 0.90 if selected_cat == category else 0.30
    else:
        cat_score = 0.70

    # Tone score (using emotion blend)
    tone_score = compute_tone_score(row, tone)

    return 0.4 * cat_score + 0.6 * tone_score


def compute_keyword_boost(desc, query):
    """Simple keyword matching with stemming."""
    if pd.isna(desc):
        return 0.0

    desc = desc.lower()
    terms = re.findall(r"\w+", query.lower())

    score = 0
    for t in terms:
        if len(t) <= 2:
            continue
        # Match word stems (crude stemming by dropping last char)
        pattern = r"\b" + re.escape(t[:-1]) + r"\w*\b"
        if re.search(pattern, desc):
            score += 1

    return min(1.0, (score / max(len(terms), 1))) * 0.5


def retrieve_recommendations(query, category, tone, top_k=16):
    """
    Main retrieval function - semantic search + filtering + ranking.

    Returns:
        (results_dataframe, error_message)
    """
    # Validate input
    is_valid, error_msg = validate_query(query)
    if not is_valid:
        return pd.DataFrame(), error_msg

    # Semantic search with Chroma
    results = db_books.similarity_search_with_relevance_scores(query, k=800)

    # Extract ISBNs and scores
    sem_scores = {}
    isbn_list = []
    for doc, score in results:
        try:
            isbn = int(doc.page_content.split()[0].replace('"', ""))
            sem_scores[isbn] = score
            isbn_list.append(isbn)
        except:
            continue

    # Get matching books
    results_df = books[books["isbn13"].isin(isbn_list)].copy()

    if results_df.empty:
        return results_df, ""

    # Normalize semantic scores to 0-1
    results_df["semantic"] = results_df["isbn13"].map(sem_scores).fillna(0)
    sem_min, sem_max = results_df["semantic"].min(), results_df["semantic"].max()
    results_df["semantic_norm"] = (
        (results_df["semantic"] - sem_min) / (sem_max - sem_min)
        if sem_max > sem_min
        else 0.5
    )

    # Compute metadata and keyword scores
    results_df["meta"] = results_df.apply(
        lambda r: compute_metadata_score(r, category, tone), axis=1
    )
    results_df["keyword"] = results_df["description"].apply(
        lambda d: compute_keyword_boost(d, query)
    )

    # Final weighted score
    if category and category != "All":
        # Higher weight on metadata when category is specified
        results_df["final_score"] = (
            0.75 * results_df["semantic_norm"]
            + 0.15 * results_df["meta"]
            + 0.10 * results_df["keyword"]
        )
    else:
        # Prioritize semantic similarity when no category filter
        results_df["final_score"] = (
            0.85 * results_df["semantic_norm"]
            + 0.05 * results_df["meta"]
            + 0.10 * results_df["keyword"]
        )

    return results_df.sort_values("final_score", ascending=False).head(top_k), ""


# =============================================================================
# GRADIO UI
# =============================================================================

current_recommendations = []


def search_books(query, category, tone):
    """UI callback for search button."""
    global current_recommendations

    # Check for contradictions
    warning = check_contradictions(query, category, tone)

    # Get recommendations
    df, error_msg = retrieve_recommendations(query, category, tone)

    # Handle errors
    if error_msg:
        gr.Warning(error_msg)
        return [None] * 16

    if warning:
        gr.Warning(warning)

    # Store results
    current_recommendations = df.reset_index(drop=True)

    # Return thumbnail images
    imgs = []
    for i in range(16):
        if i < len(df):
            imgs.append(df.iloc[i]["large_thumbnail"])
        else:
            imgs.append(None)
    return imgs


def show_details(index):
    """UI callback for clicking a book thumbnail."""
    idx = int(index)
    if idx >= len(current_recommendations):
        return [None, "", "", "", "", "", "", "", gr.update(visible=False)]

    row = current_recommendations.iloc[idx]

    title_md = f"## {row['title']}"
    authors_md = f"**Authors:** {row['authors']}"
    rating_md = (
        f"**Rating:** {row['average_rating']} ({int(row['ratings_count'])} ratings)"
    )
    year_md = f"**Published:** {int(row['published_year']) if pd.notna(row['published_year']) else 'Unknown'}"
    category_md = f"**Category:** {row['category']}"

    # Truncate long descriptions
    desc = str(row.get("description", ""))
    wc = len(desc.split())
    if wc > 200:
        desc = " ".join(desc.split()[:200]) + f"...\n\n*({wc} words total)*"
    description_md = f"### Description\n{desc}"

    # Show scoring breakdown
    relevance_md = f"""
### Relevance Breakdown
- **Semantic Similarity:** {row['semantic_norm']:.3f}
- **Metadata Match:** {row['meta']:.3f}
- **Keyword Boost:** {row['keyword']:.3f}
- **Final Score:** **{row['final_score']:.3f}**
"""
    return (
        row["large_thumbnail"],
        title_md,
        authors_md,
        rating_md,
        year_md,
        category_md,
        description_md,
        relevance_md,
        gr.update(visible=True),
    )

# Build UI
categories = ["All"] + sorted(books["category"].dropna().unique())
tones = ["All", "Happy", "Sad", "Calm", "Intense", "Dark", "Hopeful", "Wholesome"]

with gr.Blocks(theme=gr.themes.Ocean()) as app:
    gr.Markdown("# StorySense: Where Your Next Book Finds You")

    with gr.Row():
        query_box = gr.Textbox(
            label="Describe a book you want",
            placeholder="e.g., A dark psychological thriller about obsession...",
            scale=3,
        )
        category_select = gr.Dropdown(
            choices=categories, label="Category", value="All", scale=1
        )
        tone_select = gr.Dropdown(choices=tones, label="Tone", value="All", scale=1)
        search_btn = gr.Button("Search", variant="primary", scale=1)

    # Results grid
    with gr.Row():
        result_images = []
        for i in range(16):
            img = gr.Image(
                height=250,
                show_label=False,
                interactive=False,
                sources=None,
                type="filepath",
            )
            result_images.append(img)

    # Detail panel (hidden by default)
    with gr.Column(visible=False) as detail_panel:
        gr.Markdown("## Book Details")

        with gr.Row():
            detail_cover = gr.Image(
                show_label=False,
                height=500,
                interactive=False,
                sources=[],
                type="filepath",
            )

            with gr.Column():
                detail_title = gr.Markdown()
                detail_authors = gr.Markdown()
                detail_rating = gr.Markdown()
                detail_year = gr.Markdown()
                detail_category = gr.Markdown()
                detail_description = gr.Markdown()

        detail_relevance = gr.Markdown()

    # Wire up events
    search_btn.click(
        fn=search_books,
        inputs=[query_box, category_select, tone_select],
        outputs=result_images,
    )

    for i, img in enumerate(result_images):
        img.select(
            fn=lambda idx=i: show_details(idx),
            inputs=[],
            outputs=[
                detail_cover,
                detail_title,
                detail_authors,
                detail_rating,
                detail_year,
                detail_category,
                detail_description,
                detail_relevance,
                detail_panel,
            ],
        )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",  # Allows external access
        server_port=7860,
        share=False,  # HF Spaces handles sharing
        show_error=True,  # Show detailed errors
    )
