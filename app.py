import streamlit as st
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from model import CollaborativeFiltering, ContentBasedFiltering, HybridModel

# Load data
ratings_df = pd.read_csv('ratings.csv')
tmdb_df = pd.read_csv('tmdb_5000_movies.csv')
tmdb_df['genres'] = tmdb_df['genres'].apply(
    lambda x: " ".join([i['name'] for i in eval(x)]))
tmdb_df['keywords'] = tmdb_df['keywords'].apply(
    lambda x: " ".join([i['name'] for i in eval(x)]))
tmdb_df['features'] = tmdb_df['genres'] + " " + tmdb_df['keywords']

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(tmdb_df['features'])

# Load models
num_users = ratings_df['userId'].nunique()
num_movies = ratings_df['movieId'].nunique()
cf_model = CollaborativeFiltering(num_users, num_movies, embedding_dim=32)
cb_model = ContentBasedFiltering(num_features=20, embedding_dim=32)
hybrid_model = HybridModel(cf_model, cb_model)

cf_model.load_state_dict(torch.load('cf_model.pth'))
cb_model.load_state_dict(torch.load('cb_model.pth'))
hybrid_model.load_state_dict(torch.load('hybrid_model.pth'))

# Streamlit app
st.title("Movie Recommendation System")

# User input
movie_name = st.text_input("Enter a movie name:")
user_id = st.number_input("Enter your User ID (if available):",
                          min_value=0, max_value=num_users-1, value=0)

# Get recommendations
if st.button("Get Recommendations"):
    if movie_name:
        # Content-based recommendations
        movie_index = tmdb_df[tmdb_df['title'] == movie_name].index
        if len(movie_index) == 0:
            st.write("Movie not found in the database.")
        else:
            cosine_sim = cosine_similarity(
                tfidf_matrix[movie_index], tfidf_matrix)
            sim_scores = list(enumerate(cosine_sim[0]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_movies = sim_scores[1:6]  # Top 5 similar movies
            content_recommendations = [
                tmdb_df.iloc[i[0]]['title'] for i in top_movies]

            # Collaborative filtering recommendations (if user ID is provided)
            if user_id > 0:
                movie_ids = torch.arange(num_movies)
                user_ids = torch.full((num_movies,), user_id)
                with torch.no_grad():
                    cf_scores = hybrid_model(
                        user_ids, movie_ids, None, use_cf=True)
                top_movies = torch.topk(
                    cf_scores.squeeze(), k=5).indices.tolist()
                cf_recommendations = [tmdb_df.iloc[movie_id]
                                      ['title'] for movie_id in top_movies]

                # Combine recommendations
                recommendations = list(
                    set(content_recommendations + cf_recommendations))
            else:
                recommendations = content_recommendations

            # Display recommendations
            st.write("Recommended Movies:")
            for movie in recommendations:
                st.write(f"- {movie}")
    else:
        st.write("Please enter a movie name.")
