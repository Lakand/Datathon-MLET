FROM python:3.10-slim

WORKDIR /app

ENV PYTHONPATH=/app
# [SUGESTÃO] Define um local padrão para o banco caso não seja passado
ENV DB_PATH=/app/data/monitoring.db 

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p docs models data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]