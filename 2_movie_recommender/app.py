"""
Content-Based Movie Recommendation System
------------------------------------------
Pick a movie you like and get similar recommendations based on genre
and thematic tags — the same core idea behind "Because you watched..."
on Netflix (a simplified, content-based version).

How it works (interview talking points):
1. Feature Combination: genres + tags are combined into one text "profile"
   per movie.
2. Vectorization: TF-IDF converts each movie's profile into a numeric
   vector representing the importance of each word.
3. Similarity Matrix: cosine similarity is computed between every pair of
   movies once, giving an N x N similarity matrix.
4. Recommendation: for a selected movie, we sort all other movies by
   similarity score and return the top-k most similar ones.

This is "content-based filtering" — as opposed to "collaborative
filtering" which uses other users' ratings instead of item features.
Good to mention both if asked to compare approaches.
"""

import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_data
def load_data():
    data_path = BASE_DIR / "data" / "movies.csv"
    df = pd.read_csv(data_path)
    df["profile"] = (df["genres"] + " " + df["tags"]).str.lower()
    return df


@st.cache_resource
def build_similarity_matrix(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["profile"])
    sim_matrix = cosine_similarity(tfidf_matrix)
    return sim_matrix


def recommend(df, sim_matrix, movie_title, top_n=5):
    idx = df.index[df["title"] == movie_title][0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]
    result = df.iloc[[i for i, _ in scores]].copy()
    result["similarity"] = [round(score * 100, 2) for _, score in scores]
    return result


def main():
    st.title("🎬 Content-Based Movie Recommender")
    st.caption("TF-IDF + Cosine Similarity on genres & tags")

    df = load_data()
    sim_matrix = build_similarity_matrix(df)

    col1, col2 = st.columns([2, 1])
    with col1:
        movie_choice = st.selectbox("Pick a movie you like:", sorted(df["title"]))
    with col2:
        top_n = st.slider("Number of recommendations", 3, 10, 5)

    selected = df[df["title"] == movie_choice].iloc[0]
    st.info(f"**Genres:** {selected['genres']}  \n**Tags:** {selected['tags']}")

    if st.button("Recommend Similar Movies", type="primary"):
        recs = recommend(df, sim_matrix, movie_choice, top_n=top_n)

        st.subheader(f"Because you liked '{movie_choice}':")
        for _, row in recs.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f"### {row['title']}")
                    st.write(f"Genres: {row['genres']}")
                    st.write(f"Tags: {row['tags']}")
                with c2:
                    st.metric("Similarity", f"{row['similarity']}%")


if __name__ == "__main__":
    main()
