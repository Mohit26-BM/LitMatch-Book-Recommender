# ===============================
# Imports & Setup
# ===============================

import os
import pandas as pd
import gradio as gr

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ===============================
# UX Validation
# ===============================


def validate_query(query: str):
    clean = query.strip()

    if len(clean) < 5:
        return False, "Please describe the book you're looking for."

    if not any(c.isalpha() for c in clean):
        return False, "Please enter a natural language description."

    words = clean.split()
    if len(words) >= 3:
        if len(set(words)) / len(words) < 0.3:
            return False, "Your query seems repetitive. Try rephrasing it."

    return True, ""


# ===============================
# Category Classification (Offline)
# ===============================


def classify_category(raw):
    if pd.isna(raw):
        return "Other"

    r = raw.lower()

    if "romance" in r:
        return "Romance"
    if any(x in r for x in ["thriller", "mystery", "crime", "horror"]):
        return "Mystery & Thriller"
    if any(x in r for x in ["fantasy", "science fiction", "sci-fi"]):
        return "Fantasy & Sci-Fi"
    if any(x in r for x in ["history", "biography", "memoir"]):
        return "History & Biography"
    if any(x in r for x in ["business", "self-help"]):
        return "Self-Help & Business"

    return "Literary & Contemporary"


# ===============================
# Load Data
# ===============================

if os.path.exists("books_enriched_production.csv"):
    books = pd.read_csv("books_enriched_production.csv")
else:
    books = pd.read_csv("books_with_emotions.csv")
    books["category"] = books["categories"].apply(classify_category)
    books.to_csv("books_enriched_production.csv", index=False)

books["thumbnail"] = books["thumbnail"].fillna("")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = books["large_thumbnail"].replace(
    {"&fife=w800": "cover-not-found.jpg"}
)


# ===============================
# Vector DB (HuggingFace)
# ===============================

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory="chroma_books_Database",
    embedding_function=embeddings,
)


# ===============================
# Tone Profiles
# ===============================

TONE_PROFILES = {
    "Happy": {"joy": 1.0},
    "Sad": {"sadness": 1.0},
    "Calm": {"neutral": 1.0},
    "Dark": {"fear": 0.6, "sadness": 0.4},
    "Intense": {"anger": 0.5, "fear": 0.5},
    "Hopeful": {"joy": 0.6, "neutral": 0.4},
    "Wholesome": {"joy": 0.7, "neutral": 0.3},
}


def compute_tone_score(row, tone):
    if tone == "All":
        return 0.5

    profile = TONE_PROFILES[tone]
    return sum(row.get(e, 0.5) * w for e, w in profile.items()) / sum(profile.values())


# ===============================
# Retrieval
# ===============================


def retrieve_recommendations(query, category, tone):
    valid, msg = validate_query(query)
    if not valid:
        return pd.DataFrame(), msg

    results = db.similarity_search_with_relevance_scores(query, k=800)
    scores = {}

    for doc, score in results:
        try:
            isbn = int(doc.page_content.split()[0])
            scores[isbn] = score
        except:
            continue

    df = books[books["isbn13"].isin(scores.keys())].copy()

    if df.empty:
        return df, ""

    df["semantic"] = df["isbn13"].map(scores)
    smin, smax = df["semantic"].min(), df["semantic"].max()
    df["semantic"] = (df["semantic"] - smin) / (smax - smin) if smax > smin else 0.5

    df["tone"] = df.apply(lambda r: compute_tone_score(r, tone), axis=1)

    df["category_match"] = (
        df["category"].eq(category).astype(float) if category != "All" else 0.0
    )

    if category == "All":
        df["final_score"] = 0.85 * df["semantic"] + 0.15 * df["tone"]
    else:
        df["final_score"] = (
            0.65 * df["semantic"] + 0.20 * df["category_match"] + 0.15 * df["tone"]
        )

    return df.sort_values("final_score", ascending=False).head(16), ""


# ===============================
# UI Styling
# ===============================

custom_css = """
/* Hide Gradio branding and footer */
footer {
    display: none !important;
}

.gradio-container .footer {
    display: none !important;
}

a[href*="gradio.app"] {
    display: none !important;
}

/* Hide recording button and related UI */
button[aria-label*="Record"] {
    display: none !important;
}

button[aria-label*="Stop"] {
    display: none !important;
}

.record-button {
    display: none !important;
}

/* Hide upload area completely */
.image-container .upload-container {
    display: none !important;
}

.image-container [data-testid="image-upload-button"] {
    display: none !important;
}

/* Hide all icon buttons */
.image-container .icon-buttons {
    display: none !important;
}

.image-container button[aria-label] {
    display: none !important;
}

div[data-testid="image"] button {
    display: none !important;
}

.image-container .download-button,
.image-container .share-button,
.image-container .fullscreen-button {
    display: none !important;
}

/* Make images fill space and clickable */
.image-container {
    padding: 0 !important;
}

.image-container img {
    object-fit: cover !important;
    width: 100% !important;
    height: 100% !important;
    border-radius: 12px;
    cursor: pointer;
}

.gr-column {
    padding: 4px !important;
}

/* Hover effect */
.image-container:hover img {
    transform: scale(1.02);
    transition: transform 0.2s ease;
}
"""


# ===============================
# UI Layout
# ===============================

categories = ["All"] + sorted(books["category"].unique())
tones = ["All"] + list(TONE_PROFILES.keys())

current_recommendations = []

with gr.Blocks(theme=gr.themes.Ocean(), css=custom_css) as app:

    gr.Markdown("# 📚 LitMatch: Semantic Book Recommendations")

    with gr.Row():
        q = gr.Textbox(
            label="Describe a book",
            placeholder="e.g., A mystery novel set in Victorian London",
            scale=3,
        )
        c = gr.Dropdown(categories, value="All", label="Genre")  # Changed from default
        t = gr.Dropdown(tones, value="All", label="Mood")  # Changed from default
        btn = gr.Button("Search", variant="primary")

    # ================= GRID VIEW =================
    with gr.Column() as grid_view:
        images = []

        for r in range(4):
            with gr.Row():
                for col in range(4):
                    idx = r * 4 + col
                    img = gr.Image(
                        height=320,  # Increased from 260
                        show_label=False,
                        show_download_button=False,
                        show_share_button=False,
                        show_fullscreen_button=False,
                        interactive=False,  # Changed to False to prevent upload
                        container=False,
                    )
                    images.append(img)

    # ================= DETAIL VIEW =================
    with gr.Column(visible=False) as detail_view:

        back_btn = gr.Button("← Back to Results")

        with gr.Row():
            with gr.Column(scale=1):
                cover = gr.Image(
                    height=600,  # Increased from 500
                    show_label=False,
                    show_download_button=False,
                    show_share_button=False,
                    show_fullscreen_button=False,
                    interactive=False,
                )

            with gr.Column(scale=2):
                title_md = gr.Markdown()
                authors_md = gr.Markdown()
                rating_md = gr.Markdown()
                year_md = gr.Markdown()
                category_md = gr.Markdown()
                gr.Markdown("---")
                desc_md = gr.Markdown()

    # ================= LOGIC =================

    def search_and_show(query, category, tone):
        global current_recommendations

        df, err = retrieve_recommendations(query, category, tone)

        if err:
            gr.Warning(err)
            return [None] * 16

        current_recommendations = df.reset_index(drop=True)

        return [
            df.iloc[i]["large_thumbnail"] if i < len(df) else None for i in range(16)
        ]

    def open_details(idx):
        if idx >= len(current_recommendations):
            return [gr.update()] * 9

        row = current_recommendations.iloc[idx]

        return (
            gr.update(visible=False),
            gr.update(visible=True),
            row["large_thumbnail"],
            f"## {row['title']}",
            f"**Authors:** {row['authors']}",
            f"**Rating:** {row['average_rating']:.1f} ⭐",
            f"**Published:** {row['published_year']}",
            f"**Category:** {row['category']}",
            row.get("description", "No description available."),
        )

    def go_back():
        return gr.update(visible=True), gr.update(visible=False)

    btn.click(search_and_show, [q, c, t], images)

    for i in range(16):
        images[i].select(
            lambda idx=i: open_details(idx),
            outputs=[
                grid_view,
                detail_view,
                cover,
                title_md,
                authors_md,
                rating_md,
                year_md,
                category_md,
                desc_md,
            ],
        )

    back_btn.click(go_back, outputs=[grid_view, detail_view])


if __name__ == "__main__":
    app.launch()
