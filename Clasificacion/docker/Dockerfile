FROM python:3.10.17-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mejor_modelo_nn.h5 .
COPY preprocess.pkl .
COPY inferencia.py .
COPY weatherAUS.csv .

RUN mkdir -p /app/resultados

CMD ["python", "inferencia.py"]

ENV PYTHONPATH=/app
ENV TZ=America/Argentina/Buenos_Aires