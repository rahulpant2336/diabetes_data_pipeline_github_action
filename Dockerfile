FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# COPY . /app/
COPY . .

EXPOSE 8000


# CMD ["python", "src/model_building.py"]
# CMD ["python", "main_docker.py"]
CMD ["uvicorn", "backend.backend:app", "--host", "0.0.0.0", "--port", "8000"]