import os
import json
from pathlib import Path

import pytest
from backend.app import app as flask_app

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client

def test_health(client):
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('status') == 'ok'

def test_recommend_unauthorized(client):
    # No API key provided – should get 401 if API_KEY is set in env
    # Ensure env variable is set for test
    os.environ['API_KEY'] = 'testkey'
    resp = client.post('/recommend', json={'ingredients': ['bawang']})
    assert resp.status_code == 401
    data = resp.get_json()
    assert data.get('error') == 'Unauthorized'

def test_recommend_success(client, monkeypatch):
    # Provide correct API key and mock model to avoid heavy load
    os.environ['API_KEY'] = 'testkey'
    # Monkeypatch vectorizer and matrix to simple dummy objects
    class DummyVec:
        def transform(self, texts):
            # return a 1x2 vector of ones
            class Res:
                def __array__(self):
                    return [[1, 1]]
            return Res()
    class DummyMat:
        def __len__(self):
            return 2
    from backend import app as app_mod
    monkeypatch.setattr(app_mod, 'vectorizer', DummyVec())
    monkeypatch.setattr(app_mod, 'recipe_matrix', [[1, 1], [1, 1]])
    # Also monkeypatch recipes_df to minimal dataframe
    import pandas as pd
    df = pd.DataFrame({
        'Recipe': ['R1', 'R2'],
        'Ingredients': ['bawang, cabai', 'ayam, bawang'],
        'Steps': ['Step1', 'Step2']
    })
    monkeypatch.setattr(app_mod, 'recipes_df', df)
    resp = client.post('/recommend', json={'ingredients': ['bawang']}, headers={'X-API-KEY': 'testkey'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'recommendations' in data
    assert len(data['recommendations']) == 2
    assert data['recommendations'][0]['recipe'] in ('R1', 'R2')
