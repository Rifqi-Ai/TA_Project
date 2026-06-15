FROM python:3.12-slim

# Set workdir
WORKDIR /app

# Copy backend code
COPY . ./
# Copy frontend (served as static files)
COPY frontend/ ./frontend/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Train model (runs script) – optional, will also run at runtime if missing
RUN python train_model.py

EXPOSE 5000
ENV FLASK_APP=app.py
CMD ["python", "app.py"]
