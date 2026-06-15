# ---------- Builder stage ----------
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY backend/ ./
COPY backend/frontend/ ./frontend/

# Train model (optional – also runs at runtime if missing)
RUN python train_model.py

# ---------- Runtime stage ----------
FROM python:3.12-slim
WORKDIR /app

# Copy only needed files from builder
COPY --from=builder /app /app

# Copy environment file (if exists) and entrypoint script
COPY backend/.env /app/.env
COPY backend/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5000
ENV FLASK_APP=app.py
# Healthcheck using the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:5000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "app.py"]
