FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY main.py .
COPY carga_entreno.py .
COPY prediccion_maraton.py .
COPY *.csv .
COPY *.pkl .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "main.py"]