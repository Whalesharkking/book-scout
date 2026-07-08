FROM docker.io/library/python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent ./agent
COPY scripts ./scripts

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "agent.main"]
