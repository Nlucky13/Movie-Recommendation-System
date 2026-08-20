import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Load saved files
# -----------------------------

df = pd.read_csv("cleaned_movies.csv")

with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

with open("title_index.pkl", "rb") as f:
    title_to_idx = pickle.load(f)


# -----------------------------
# Recommendation function
# -----------------------------

def recommend(title=None, genre=None, n=10):

    # -------------------------
    # Title + Genre
    # -------------------------
    if title and genre:

        title = title.lower().strip()
        genre = genre.lower().strip()

        if title not in title_to_idx:
            return None, f"Movie '{title}' was not found."

        idx = title_to_idx[title]

        genre_matches = df[
            df["genres"].str.lower().str.contains(
                genre,
                na=False
            )
        ].copy()

        if genre_matches.empty:
            return None, f"No movies found for genre '{genre}'."

        movie_vec = tfidf_matrix[idx]

        sim_scores = cosine_similarity(
            movie_vec,
            tfidf_matrix
        ).flatten()

        genre_matches["similarity"] = genre_matches.index.map(
            lambda x: sim_scores[x]
        )

        genre_matches = genre_matches[
            genre_matches.index != idx
        ]

        max_score = genre_matches["score"].max()

        if max_score > 0:
            genre_matches["normalized_score"] = (
                genre_matches["score"] / max_score
            )
        else:
            genre_matches["normalized_score"] = 0

        genre_matches["final_rank"] = (
            0.6 * genre_matches["similarity"]
            + 0.4 * genre_matches["normalized_score"]
        )

        result = genre_matches.sort_values(
            "final_rank",
            ascending=False
        ).head(n)

        return result[
            [
                "title",
                "genres",
                "averageRating",
                "numVotes",
                "score",
                "release_year",
                "poster_url",
                "similarity",
                "final_rank"
            ]
        ], None


    # -------------------------
    # Title only
    # -------------------------
    elif title:

        title = title.lower().strip()

        if title not in title_to_idx:
            return None, f"Movie '{title}' was not found."

        idx = title_to_idx[title]

        movie_vec = tfidf_matrix[idx]

        sim_scores = cosine_similarity(
            movie_vec,
            tfidf_matrix
        ).flatten()

        sim_indices = sim_scores.argsort()[::-1][1:50]

        candidates = df.iloc[sim_indices][
            [
                "title",
                "genres",
                "averageRating",
                "numVotes",
                "score",
                "release_year",
                "poster_url"
            ]
        ].copy()

        candidates["similarity"] = sim_scores[sim_indices]

        candidates["final_rank"] = (
            0.6 * candidates["similarity"]
            + 0.4 * (
                candidates["score"]
                / candidates["score"].max()
            )
        )

        result = candidates.sort_values(
            "final_rank",
            ascending=False
        ).head(n)

        return result, None


    # -------------------------
    # Genre only
    # -------------------------
    elif genre:

        genre = genre.lower().strip()

        genre_matches = df[
            df["genres"].str.lower().str.contains(
                genre,
                na=False
            )
        ].copy()

        if genre_matches.empty:
            return None, f"No movies found for genre '{genre}'."

        result = genre_matches.sort_values(
            "score",
            ascending=False
        ).head(n)

        return result[
            [
                "title",
                "genres",
                "averageRating",
                "numVotes",
                "score",
                "release_year",
                "poster_url"
            ]
        ], None


    return None, "Please enter a movie title or genre."

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")

st.write(
    "Find movies based on a movie title, genre, or both."
)

# -----------------------------
# User inputs
# -----------------------------

title = st.text_input(
    "Movie Title",
    placeholder="Example: Inception"
)

genre = st.text_input(
    "Genre",
    placeholder="Example: Action"
)


# -----------------------------
# Recommend button
# -----------------------------

if st.button("Recommend Movies"):

    results, error = recommend(
        title=title,
        genre=genre,
        n=10
    )

    if error:
        st.error(error)

    else:

        st.subheader("Recommended Movies")

        for i, (_, movie) in enumerate(results.iterrows(), start=1):
            col1, col2 = st.columns([1, 4])

            with col1:
                if movie['poster_url']:
                    st.image(
                    movie['poster_url'],
                    width=150
                )
                else:
                    st.write("No poster available")

            with col2:
                st.markdown(f"### {i}. {movie['title']}")

                st.write(
                    f"**Genre:** {movie['genres']}"
                )

                st.write(
                f"⭐ **Rating:** {movie['averageRating']:.1f}"
                )

                st.write(
                    f"📅 **Release Year:** {int(movie['release_year'])}"
                )

                st.write(
                    f"👥 **Votes:** {int(movie['numVotes'])}"
                )

                if 'similarity' in movie:
                    st.write(
                        f"🎯 **Similarity:** "
                        f"{movie['similarity']:.2%}"
                    )

            st.divider()