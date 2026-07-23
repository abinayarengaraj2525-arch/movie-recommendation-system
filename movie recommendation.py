import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Convert genres into vectors
cv = CountVectorizer()
count_matrix = cv.fit_transform(movies["genre"])

# Calculate similarity
similarity = cosine_similarity(count_matrix)

# Recommendation function
def recommend(movie_name):
    if movie_name not in movies["title"].values:
        print("Movie not found!")
        return

    movie_index = movies[movies["title"] == movie_name].index[0]
    distances = list(enumerate(similarity[movie_index]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)

    print(f"\nMovies similar to {movie_name}:\n")

    count = 0
    for movie in sorted_movies[1:]:
        print(movies.iloc[movie[0]].title)
        count += 1
        if count == 5:
            break

movie = input("Enter a movie name: ")
recommend(movie)