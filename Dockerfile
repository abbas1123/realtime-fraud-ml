FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY fraud_ml ./fraud_ml
COPY models ./models
RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 8000

CMD ["uvicorn", "fraud_ml.serve:app", "--host", "0.0.0.0", "--port", "8000"]
