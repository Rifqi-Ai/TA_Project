import os, re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load

# ---------- Load dataset ----------
csv_path = os.path.join(os.path.dirname(__file__), 'Indonesian_Food_Recipes.csv')
df = pd.read_csv(csv_path)
if 'Ingredients' not in df.columns:
    raise ValueError('CSV must contain an "Ingredients" column')

# ---------- Preprocess ingredients into a single string per recipe ----------
def normalize_ingredients(text):
    if pd.isna(text):
        return ''
    parts = re.split(r'[;,]', str(text))
    cleaned = [p.strip().lower() for p in parts if p.strip()]
    return ' '.join(cleaned)

df['ing_text'] = df['Ingredients'].apply(normalize_ingredients)

# ---------- TF‑IDF vectorizer (limit dimensions) ----------
vectorizer = TfidfVectorizer(max_features=2000, stop_words=None)
X = vectorizer.fit_transform(df['ing_text'])

# Save vectorizer and matrix for inference
model_path = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')
matrix_path = os.path.join(os.path.dirname(__file__), 'recipe_matrix.pkl')

dump(vectorizer, model_path)
dump(X, matrix_path)
print('TF‑IDF vectorizer saved to', model_path)
print('Recipe matrix saved to', matrix_path)
