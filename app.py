
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("📚 Smart Book Recommendation System")
st.write("Find books similar to your favourites!")

data = {
    "Title": [
        "Harry Potter", "The Hobbit", "Percy Jackson",
        "Pride and Prejudice", "Jane Eyre", "Me Before You",
        "Gone Girl", "The Silent Patient", "Sherlock Holmes",
        "The Hunger Games", "Divergent", "The Fault in Our Stars",
        "Little Women", "Sapiens", "Wings of Fire",
        "Atomic Habits", "The Power of Now", "The Alchemist"
    ],
    "Genre": [
        "Fantasy", "Fantasy", "Fantasy",
        "Romance", "Romance", "Romance",
        "Mystery", "Mystery", "Mystery",
        "Science Fiction", "Science Fiction", "Drama",
        "Drama", "History", "History",
        "Self Help", "Self Help", "Adventure"
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(df["Genre"])
similarity = cosine_similarity(matrix)

book = st.selectbox("Choose a book you like:", df["Title"])

if st.button("Recommend Books"):
    i = df.index[df["Title"] == book][0]

    scores = sorted(
        enumerate(similarity[i]),
        key=lambda x: x[1],
        reverse=True
    )

    st.subheader("Recommended Books")

    results = [
        (df.iloc[j]["Title"], score)
        for j, score in scores if j != i and score > 0
    ][:5]

    if results:
        for title, score in results:
            st.write(f"📖 **{title}**")
            st.caption(f"Genre similarity: {score * 100:.1f}%")
    else:
        st.write("No similar books found.")
