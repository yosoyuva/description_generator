from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load your Joconde database CSV
df = pd.read_csv('/path/to/joconde.csv')  # Adjust path as necessary
vectorizer = TfidfVectorizer()

# Pre-compute TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(df['Artwork_Description'])  # Adjust column name as necessary

@app.route('/find_artwork', methods=['POST'])
def find_artwork():
    text = request.json['text']
    text_tfidf = vectorizer.transform([text])
    cosine_similarities = cosine_similarity(text_tfidf, tfidf_matrix)
    max_sim_index = cosine_similarities.argmax()
    max_sim_score = cosine_similarities.max()

    if max_sim_score > 0.5:  # Adjust threshold as needed
        matched_artwork = df.iloc[max_sim_index].to_dict()
        return jsonify(matched_artwork)
    else:
        return jsonify({"message": "No matching artwork found"}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Adjust port as necessary
