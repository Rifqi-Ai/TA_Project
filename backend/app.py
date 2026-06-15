import os, json
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import os
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_folder='../frontend')

# Load TF‑IDF vectorizer and recipe matrix on startup
vectorizer_path = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')
matrix_path = os.path.join(os.path.dirname(__file__), 'recipe_matrix.pkl')
vectorizer = load(vectorizer_path)
recipe_matrix = load(matrix_path)

# Load recipe dataframe (keep for result lookup)
csv_path = os.path.join(os.path.dirname(__file__), 'Indonesian_Food_Recipes.csv')
recipes_df = pd.read_csv(csv_path)

def ingredients_to_vector(ing_list):
    # ing_list: list of strings (lowercased)
    # Join into a space‑separated string and transform with the stored vectorizer
    text = ' '.join(ing_list)
    return vectorizer.transform([text])
@app.route('/')
def index():
    # Serve frontend index.html
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json(force=True)
    ingredients = data.get('ingredients', [])
    if not isinstance(ingredients, list):
        return jsonify({'error': 'ingredients must be a list'}), 400
    # Normalise
    ing_norm = [i.strip().lower() for i in ingredients if i.strip()]
    X = ingredients_to_vector(ing_norm)
    # Compute cosine similarity between query vector and all recipe vectors
    sims = cosine_similarity(X, recipe_matrix).flatten()
    # Get top 5 indices
    top_idx = sims.argsort()[-5:][::-1]
    results = []
    for idx in top_idx:
        recipe = recipes_df.iloc[idx]
        results.append({
            'recipe': recipe.get('Recipe') or recipe.get('Name') or f'Row {idx}',
            'ingredients': recipe.get('Ingredients'),
            'steps': recipe.get('Steps') or recipe.get('Method'),
            'score': float(sims[idx])
        })
    return jsonify({'recommendations': results})

if __name__ == '__main__':
    # If vectorizer or matrix missing, (re)train the model
    if not os.path.exists(vectorizer_path) or not os.path.exists(matrix_path):
        print('Vectorizer or matrix not found, training now...')
        os.system('python3 train_model.py')
        vectorizer = load(vectorizer_path)
        recipe_matrix = load(matrix_path)
    app.run(host='0.0.0.0', port=5000)
