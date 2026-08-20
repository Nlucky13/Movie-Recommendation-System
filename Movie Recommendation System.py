import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("TMDB  IMDB Movies Dataset.csv")

df = df[df['status'] == 'Released']
df = df[df['vote_count'] >= 50]
df = df[df['numVotes'] >= 100]

df = df.dropna(subset=['overview'])

for col in ['genres', 'keywords', 'cast', 'directors', 'tagline']:
    df[col] = df[col].fillna('')

df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

df = df.reset_index(drop=True)
print("Cleaning Dataset Done")

def build_soup(row):
    directors = ' '.join([row['directors']] * 3)
    cast = row['cast'].replace(',', ' ')
    genres = row['genres'].replace(',', ' ')
    keywords = row['keywords'].replace(',', ' ')
    overview = row['overview']
    return f"{directors} {cast} {genres} {keywords} {overview}"

df['soup'] = df.apply(build_soup, axis=1)
df['soup'] = df['soup'].str.lower()
print("Weights Added to Text")

from sklearn.feature_extraction.text import TfidfVectorizer

df = df.drop_duplicates(subset='title')
df = df.reset_index(drop=True)

tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=20000,   
    ngram_range=(1, 2)    
)

tfidf_matrix = tfidf.fit_transform(df['soup'])
print("Vectorized")

from sklearn.metrics.pairwise import cosine_similarity
import pickle
with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)

title_to_idx = pd.Series(df.index, index=df['title'].str.lower())

C = df['averageRating'].mean()      
m = df['numVotes'].quantile(0.70)   

def weighted_rating(row, C=C, m=m):
    v = row['numVotes']
    R = row['averageRating']
    return (v / (v + m)) * R + (m / (v + m)) * C

df['score'] = df.apply(weighted_rating, axis=1)

from sklearn.metrics.pairwise import cosine_similarity

def recommend(search, n=10):
    search = search.lower().strip()

    if search in title_to_idx:
        idx = title_to_idx[search]
        movie_vec = tfidf_matrix[idx]
        sim_scores = cosine_similarity(movie_vec, tfidf_matrix).flatten()

        sim_indices = sim_scores.argsort()[::-1][1:50]

        candidates = df.iloc[sim_indices][
            ['title', 'genres', 'averageRating',
             'numVotes', 'score', 'release_year']
        ].copy()

        candidates['similarity'] = sim_scores[sim_indices]

        candidates['final_rank'] = (
            0.6 * candidates['similarity'] +
            0.4 * (candidates['score'] / candidates['score'].max())
        )

        return candidates.sort_values(
            'final_rank',
            ascending=False
        ).head(n)

    else:
        genre_matches = df[
            df['genres'].str.lower().str.contains(search, na=False)
        ].copy()

        if genre_matches.empty:
            print(f"'{search}' not found as a title or genre.")
            return

        genre_matches = genre_matches.sort_values(
            'score',
            ascending=False
        )

        return genre_matches[
            ['title', 'genres', 'averageRating',
             'numVotes', 'score', 'release_year']
        ].head(n)

df['poster_url'] = df['poster_path'].apply(
    lambda x: f"https://image.tmdb.org/t/p/w500{x}"
    if pd.notna(x) and x != ''
    else ''
)

df.to_csv('cleaned_movies.csv', index=False)

with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)

with open('title_index.pkl', 'wb') as f:
    pickle.dump(title_to_idx, f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

print("Program reached the end")
print()