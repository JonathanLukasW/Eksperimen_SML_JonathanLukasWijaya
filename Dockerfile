FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir pandas scikit-learn Flask mlflow prometheus_client

COPY . /app

EXPOSE 8001

CMD ["python"]