# TA_Project — Sistem Rekomendasi Resep Masakan Indonesia

## Deskripsi
Proyek ini menyediakan API **/recommend** yang menerima daftar bahan (ingredients) dalam format JSON
 dan mengembalikan 5 resep paling cocok berdasarkan **TF‑IDF** dan *cosine similarity*.

## Fitur Utama
- **API key security** – endpoint `/recommend` membutuhkan header `X-API-KEY` (opsional, diatur melalui `.env`).
- **Health check** – endpoint `/health` mengembalikan `{"status":"ok"}`.
- **Logging** – semua request tercatat di `logs/app.log`.
- **Docker multi‑stage** dengan `HEALTHCHECK`.
- **CI/CD** – GitHub Actions menjalankan unit‑test (`pytest`) dan membangun image Docker.
- **Swagger/OpenAPI** – file `swagger.yaml` mendeskripsikan API.

## Cara Build & Jalankan
```bash
# Clone repo
git clone https://github.com/Rifqi-Ai/TA_Project.git
cd TA_Project/project

# Buat virtual env (optional, hanya untuk development)
python3 -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt

# Siapkan .env (lihat .env.example)
cp .env.example .env   # ubah API_KEY bila diperlukan

# Build dan jalankan dengan Docker
docker compose up -d --build
```

API tersedia pada `http://<IP_VM>:5000`. Frontend dapat diakses pada URL yang sama (root path).

## Contoh Request (curl)
```bash
curl -X POST http://localhost:5000/recommend \
     -H "Content-Type: application/json" \
     -H "X-API-KEY: $API_KEY" \
     -d '{"ingredients": ["bawang", "cabai", "ayam"]}'
```

## Testing
```bash
pytest tests/
```

## License
MIT – bebas pakai, modifikasi, dan distribusi.
