#!/bin/sh
# Load environment variables from .env if present
if [ -f /app/.env ]; then
  export $(grep -v '^#' /app/.env | xargs)
fi
# Execute the CMD passed from Dockerfile
exec "$@"
