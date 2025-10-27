import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st


# LOAD AND PREPROCESS

@st.cache_data
def load_data():
    data = pd.read_csv("movies.csv")
    data['genres'] = data['genres'].fillna('')
    return data

movies = load_data()


# FEATURE EXTRACTION

cv = CountVectorizer(stop_words='english')
genre_matrix = cv.fit_transform(movies['genres'])
similarity = cosine_similarity(genre_matrix)


# RECOMMENDATION FUNCTION

def recommend(movie_title):
    movie_title = movie_title.lower()
    if movie_title not in movies['title'].str.lower().values:
        return ["Movie not found in dataset."]
    
    index = movies[movies['title'].str.lower() == movie_title].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommended = []
    for i in scores[1:6]:  # top 5 similar movies
        recommended.append(movies.iloc[i[0]].title)
    return recommended


# STREAMLIT UI

st.title("🎬 Simple Movie Recommendation System")
st.write("Get top 5 similar movies based on genres")

movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie you like:", movie_list)

if st.button("Show Recommendations"):
    recs = recommend(selected_movie)
    st.subheader("Recommended Movies:")
    for i, movie in enumerate(recs, start=1):
        st.write(f"{i}. {movie}")
