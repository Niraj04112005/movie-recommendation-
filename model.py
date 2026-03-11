import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Collaborative Filtering Model


class CollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim):
        super(CollaborativeFiltering, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user_ids, movie_ids):
        user_embedded = self.user_embedding(user_ids)
        movie_embedded = self.movie_embedding(movie_ids)
        concatenated = torch.cat([user_embedded, movie_embedded], dim=1)
        output = self.fc(concatenated)
        return torch.sigmoid(output)

# Content-Based Filtering Model


class ContentBasedFiltering(nn.Module):
    def __init__(self, num_features, embedding_dim):
        super(ContentBasedFiltering, self).__init__()
        self.fc1 = nn.Linear(num_features, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, 1)

    def forward(self, movie_features):
        hidden = torch.relu(self.fc1(movie_features))
        output = self.fc2(hidden)
        return torch.sigmoid(output)

# Hybrid Model


class HybridModel(nn.Module):
    def __init__(self, cf_model, cb_model):
        super(HybridModel, self).__init__()
        self.cf_model = cf_model
        self.cb_model = cb_model

    def forward(self, user_ids, movie_ids, movie_features, use_cf=True):
        if use_cf:
            return self.cf_model(user_ids, movie_ids)
        else:
            return self.cb_model(movie_features)

# Load and preprocess data


def load_and_preprocess_data(ratings_path, tmdb_path):
    # Load ratings data
    ratings_df = pd.read_csv(ratings_path)
    num_users = ratings_df['userId'].nunique()
    num_movies = ratings_df['movieId'].nunique()

    # Load TMDB data
    tmdb_df = pd.read_csv(tmdb_path)
    tmdb_df['genres'] = tmdb_df['genres'].apply(
        lambda x: " ".join([i['name'] for i in eval(x)]))
    tmdb_df['keywords'] = tmdb_df['keywords'].apply(
        lambda x: " ".join([i['name'] for i in eval(x)]))
    tmdb_df['features'] = tmdb_df['genres'] + " " + tmdb_df['keywords']

    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(tmdb_df['features'])

    return ratings_df, tmdb_df, tfidf_matrix, num_users, num_movies

# Train Collaborative Filtering Model


def train_collaborative_filtering(num_users, num_movies, embedding_dim, epochs=10):
    cf_model = CollaborativeFiltering(num_users, num_movies, embedding_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(cf_model.parameters(), lr=0.001)

    for epoch in range(epochs):
        user_ids = torch.randint(0, num_users, (32,))
        movie_ids = torch.randint(0, num_movies, (32,))
        ratings = torch.randint(0, 2, (32,)).float()

        optimizer.zero_grad()
        outputs = cf_model(user_ids, movie_ids)
        loss = criterion(outputs.squeeze(), ratings)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    return cf_model

# Train Content-Based Filtering Model


def train_content_based_filtering(num_features, embedding_dim, epochs=10):
    cb_model = ContentBasedFiltering(num_features, embedding_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(cb_model.parameters(), lr=0.001)

    for epoch in range(epochs):
        movie_features = torch.randn(32, num_features)
        ratings = torch.randint(0, 2, (32,)).float()

        optimizer.zero_grad()
        outputs = cb_model(movie_features)
        loss = criterion(outputs.squeeze(), ratings)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    return cb_model

# Main function to train and save models


def main():
    # Load and preprocess data
    ratings_df, tmdb_df, tfidf_matrix, num_users, num_movies = load_and_preprocess_data(
        'ratings.csv', 'tmdb_5000_movies.csv')

    # Train Collaborative Filtering Model
    print("Training Collaborative Filtering Model...")
    cf_model = train_collaborative_filtering(
        num_users, num_movies, embedding_dim=32)

    # Train Content-Based Filtering Model
    print("Training Content-Based Filtering Model...")
    cb_model = train_content_based_filtering(num_features=20, embedding_dim=32)

    # Create Hybrid Model
    hybrid_model = HybridModel(cf_model, cb_model)

    # Save models
    torch.save(cf_model.state_dict(), 'cf_model.pth')
    torch.save(cb_model.state_dict(), 'cb_model.pth')
    torch.save(hybrid_model.state_dict(), 'hybrid_model.pth')
    print("Models saved successfully.")


# Run the main function if this script is executed
if __name__ == "__main__":
    main()
