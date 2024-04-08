FROM python:3.11.5-slim-bullseye

COPY . /src

WORKDIR /src

RUN pip install -r requirements-frozen.txt

CMD ["python3", "app.py"]