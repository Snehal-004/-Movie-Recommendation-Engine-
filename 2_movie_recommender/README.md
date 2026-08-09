# Content-Based Movie Recommendation System

Recommends movies similar to one you already like, using TF-IDF on
genres/tags and cosine similarity — a simplified content-based version
of "Because you watched..." style recommenders.

## How to run in VS Code

1. Open this folder in VS Code.
2. Open a terminal (Terminal → New Terminal).
3. Create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Run the app:
   ```
   streamlit run app.py
   ```
6. It will open automatically in your browser at `http://localhost:8501`.

## Project Structure
```
2_movie_recommender/
├── app.py              # Streamlit UI + recommendation logic
├── data/movies.csv      # 30 sample movies with genres & tags
├── requirements.txt
└── README.md
```

## How it works
1. Each movie's genres + tags are combined into one text "profile".
2. TF-IDF vectorizes all movie profiles.
3. Cosine similarity is computed between every pair of movies, forming
   an N x N similarity matrix (computed once, cached for speed).
4. For a selected movie, the most similar movies (excluding itself) are
   returned as recommendations.

## Talking points for interviews
- **Content-based filtering** (this project) uses item features (genre,
  tags) — works even for brand-new movies with no ratings yet ("cold
  start" friendly).
- **Collaborative filtering** (the alternative approach) uses patterns
  in *other users'* ratings instead — better personalization but needs
  a lot of user interaction data.
- A production system (e.g. Netflix) typically combines both in a
  **hybrid recommender**.
- Possible extensions: add a user ratings table and layer in
  collaborative filtering; use `TruncatedSVD` for dimensionality
  reduction on larger datasets.
