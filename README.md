# Semantic Book Recommendation System

An AI-powered book recommendation system that understands natural language queries using semantic embeddings instead of simple keyword matching.

## Overview
This project allows users to search for books by describing what they want to read. The system retrieves relevant books using vector similarity search and refines results using category classification, emotion-based tone matching, and lightweight rule-based logic.

## Key Features
- Semantic search using transformer-based embeddings
- Vector similarity search with ChromaDB
- Rule-based category classification (deterministic and fast)
- Emotion and tone-based filtering (Happy, Sad, Calm, Intense, Dark, Hopeful, Wholesome)
- Contradiction detection between query, category, and tone
- Interactive web interface built with Gradio
- Manual test suite for validating relevance and behavior

## Tech Stack
- Python
- LangChain
- ChromaDB
- Google Gemini Embeddings
- Pandas
- Gradio

## How It Works
1. User enters a natural language query describing a book
2. The query is validated to handle gibberish, repetition, and edge cases
3. The query is embedded using transformer-based embeddings
4. ChromaDB performs vector similarity search to retrieve candidate books
5. Results are re-ranked using:
   - Category match score
   - Emotion-based tone similarity
   - Keyword overlap boost
6. Final ranked recommendations are displayed in the UI

## Project Structure
- app.py: Main application logic and Gradio UI
- books_with_emotions.csv: Source dataset with metadata and emotion scores
- books_enriched_production.csv: Cached dataset with classified categories
- chroma_books_Database/: Vector database (not tracked in GitHub)
- requirements.txt: Python dependencies
- .env: API keys (not committed)
