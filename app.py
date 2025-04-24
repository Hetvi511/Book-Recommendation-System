from flask import Flask, render_template, request
import pickle
import numpy as np

# Loading pickle files
popular_df = pickle.load(open('popular.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
model = pickle.load(open('knn_model.pkl', 'rb'))  # Renaming similarity_scores to model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                           book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values)
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html', book_titles=list(pt.index))

@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')

    if user_input not in pt.index:
        return render_template('recommend.html', error="Book not found. Please select from suggestions.", book_titles=list(pt.index))

    # Get similar books using KNN
    distances, indices = model.kneighbors(pt.loc[[user_input]], n_neighbors=6)

    data = []
    for i in range(1, len(indices[0])):  # Skip the first one (it's the input book itself)
        index = indices[0][i]
        score = 1 - distances[0][i]

        similar_book_title = pt.index[index]
        temp_df = books[books['Book-Title'] == similar_book_title]

        title = temp_df['Book-Title'].dropna().values[0] if not temp_df['Book-Title'].dropna().empty else 'Unknown'
        author = temp_df['Book-Author'].dropna().values[0] if not temp_df['Book-Author'].dropna().empty else 'Unknown'
        image_url = temp_df['Image-URL-M'].dropna().values[0] if not temp_df['Image-URL-M'].dropna().empty else ''

        data.append([title, author, image_url, round(score, 4)])

    return render_template('recommend.html', data=data, book_titles=list(pt.index))

if __name__ == '__main__':
    app.run(debug=True)
