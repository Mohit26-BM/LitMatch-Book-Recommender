# LitMatch - Find Your Next Book

An AI-powered book recommendation engine that uses semantic search and emotion-based filtering to help readers discover books they'll love.

## Live Demo
[Try it here](https://huggingface.co/spaces/Mohit26BM/LitMatch-Recommender)

## Features
- **Semantic Search**: Natural language queries like "a thriller about AI"
- **Emotion Filtering**: Find books by mood (Happy, Dark, Intense, etc.)
- **5,000+ Books**: Curated dataset with ratings and descriptions
- **Smart Ranking**: Hybrid algorithm combining similarity, genre, and tone

## Tech Stack
- **Frontend**: Gradio
- **ML/NLP**: LangChain, Sentence Transformers, ChromaDB
- **Embeddings**: all-MiniLM-L6-v2
- **Deployment**: Hugging Face Spaces

## How It Works
1. User enters natural language query
2. Query embedded using Sentence Transformers
3. Vector similarity search in ChromaDB
4. Results ranked by semantic match + genre + emotional tone
5. Top 16 recommendations displayed

## Key Achievements
- Processed 5,000+ book descriptions into vector embeddings
- Achieved sub-second query response times
- Built intuitive UI with zero ML knowledge required for users
