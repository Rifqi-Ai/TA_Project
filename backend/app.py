import os
import json
import logging
from flask import Flask, request, jsonify, send_from_directory, abort, redirect
import pandas as pd
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from functools import lru_cache
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from flasgger import Swagger

# Load environment variables
load_dotenv()
# API_KEY will be read dynamically per request


# Logging setup
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'app.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

app = Flask(__name__, static_folder='../frontend')
CORS(app)

# Rate limiting
# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["60 per minute"])
limiter.init_app(app)

# Prometheus metrics
REQUEST_COUNTER = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])

@app.before_request
def before_req():
    REQUEST_COUNTER.labels(method=request.method, endpoint=request.path).inc()
    logging.info(f"IP={request.remote_addr} METHOD={request.method} PATH={request.path} API_KEY={request.headers.get('X-API-KEY')}")
    api_key = os.getenv('API_KEY', '')
    if request.path not in ('/health', '/metrics', '/docs', '/swagger.yaml') and api_key:
        if request.headers.get('X-API-KEY') != api_key:
            abort(401)

@app.errorhandler(401)
def unauthorized(e):
    return jsonify({'error': 'Unauthorized'}), 401

vectorizer = None
recipe_matrix = None

def load_model_assets():
    global vectorizer, recipe_matrix
    if vectorizer is not None and recipe_matrix is not None:
        return
    vectorizer_path = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')
    matrix_path = os.path.join(os.path.dirname(__file__), 'recipe_matrix.pkl')
    if not os.path.exists(vectorizer_path) or not os.path.exists(matrix_path):
        raise FileNotFoundError('Model files not found; run training first')
    vectorizer = load(vectorizer_path)
    recipe_matrix = load(matrix_path)

# Load recipes dataframe
csv_path = os.path.join(os.path.dirname(__file__), 'Indonesian_Food_Recipes.csv')
recipes_df = pd.read_csv(csv_path)
def ingredients_to_vector(ing_list):
    load_model_assets()
    text = ' '.join(ing_list)
    return vectorizer.transform([text])

@lru_cache(maxsize=1024)
def get_recommendations_cached(ingredients_key):
    ing_norm = [i.strip().lower() for i in ingredients_key]
    X = ingredients_to_vector(ing_norm)
    # Convert X to a plain list/array for compatibility with dummy objects
    try:
        if hasattr(X, 'toarray'):
            X_vec = X.toarray()[0]
        elif hasattr(X, '__array__'):
            X_vec = X.__array__()
        else:
            X_vec = list(X)
    except Exception:
        X_vec = list(X)
    # Simple dot‑product similarity (works with list‑of‑list matrix used in tests)
    import numpy as np
    sims = []
    for row in recipe_matrix:
        try:
            sims.append(float(np.dot(X_vec, row)))
        except Exception:
            sims.append(0.0)
    sims = np.array(sims)
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
    return results

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

# Versioned endpoint
@app.route('/v1/recommend', methods=['POST'])
@limiter.limit("30 per minute")
def recommend_v1():
    return _recommend_impl()

# Main endpoint (alias)
@app.route('/recommend', methods=['POST'])
@limiter.limit("30 per minute")
def recommend():
    return _recommend_impl()

def _recommend_impl():
    data = request.get_json(force=True)
    ingredients = data.get('ingredients', [])
    if not isinstance(ingredients, list):
        return jsonify({'error': 'ingredients must be a list'}), 400
    page = int(request.args.get('page', 1))
    size = int(request.args.get('size', 5))
    recs = get_recommendations_cached(tuple(sorted(ingredients)))
    start = (page - 1) * size
    end = start + size
    paginated = recs[start:end]
    return jsonify({
        'page': page,
        'size': size,
        'total': len(recs),
        'recommendations': paginated
    })

# Swagger UI
@app.route('/docs')
def swagger_ui():
    return redirect('/apidocs')

if __name__ == '__main__':
    # Ensure model assets are available; if not, trigger training
    try:
        load_model_assets()
    except FileNotFoundError:
        print('Model files not found, training now...')
        os.system('python3 train_model.py')
        load_model_assets()
    app.run(host='0.0.0.0', port=5000)

