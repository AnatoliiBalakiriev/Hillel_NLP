# Hillel NLP Project

## Run Web API
### Local

```sh
$ ./run.sh
```

### Docker
```sh
$ docker build -f Dockerfile -t fastapi-ml .
$ docker run -p 9000:9000 --rm --name fastapi-ml -t -i fastapi-ml
```

### Docker Compose

```sh
$ docker compose up --build
```

## Request Commands

### Predict Endpoint

```sh 
$ curl --request POST --url http://127.0.0.1:9000/api/v1/predict --header 'Content-Type: application/json' --data '{"input_text": "test"}'
```

```sh
$ http POST http://127.0.0.1:9000/api/v1/predict input_text=test
```

### Similarity Endpoint

Send a POST request to `/similarity` with the following request body, using different methods like `jaccard`, `hamming`, or `cosine`:

#### Example Request

```sh
curl --request POST \
  --url http://127.0.0.1:9000/similarity \
  --header 'Content-Type: application/json' \
  --data '{
    "method": "levenshtein",
    "line1": "hello",
    "line2": "hallo"
  }'
```

#### Example Response

```json
{
  "method": "levenshtein",
  "line1": "hello",
  "line2": "hallo",
  "similarity": 0.8
}
```

### Classification Endpoint

Send a POST request to `/classify` with the following request body:

#### Example Request

```sh
curl --request POST \
  --url http://127.0.0.1:9000/classify \
  --header 'Content-Type: application/json' \
  --data '{
    "text": "I absolutely loved this movie!"
  }'
```

#### Example Response

```json
{
  "text": "I absolutely loved this movie!",
  "label": "positive"
}
```

### Group Sentences Endpoint

Send a POST request to `/group_sentences` with the following request body:

#### Example Request

```sh
curl --request POST \
  --url http://127.0.0.1:9000/group_sentences \
  --header 'Content-Type: application/json' \
  --data '{
    "sentences": [
      "The economy is growing at a rapid pace.",
      "Stocks are up in early morning trading.",
      "The government is planning a new infrastructure project.",
      "The movie received excellent reviews from critics.",
      "The actor won an award for his performance."
    ]
  }'
```

#### Example Response

```json
{
  "groups": {
    "subject": [
      "The economy is growing at a rapid pace.",
      "Scientists discovered a new species of bird."
    ],
    "line": [
      "Stocks are up in early morning trading."
    ],
    "state": [
      "The government is planning a new infrastructure project.",
      "The movie received excellent reviews from critics."
    ],
    "game": [
      "The actor won an award for his performance."
    ],
    "file": [
      "The research paper was published in a renowned journal."
    ]
  }
}
```

## Development
### Run Tests and Linter

```
$ poetry run tox
```

## Installation and Setup

1. **Clone the repository**:

```sh
git clone https://github.com/AnatoliiBalakiriev/Hillel_NLP.git
cd Hillel_NLP
```

2. **Install dependencies using Poetry**:

```sh
poetry install
```
3. **Load NLTK resources**:

```sh
poetry run python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

4. **Run the application**:

```sh
./run.sh
```

## Reference

- [tiangolo/full\\-stack\\-fastapi\\-postgresql: Full stack, modern web application generator\\. Using FastAPI, PostgreSQL as database, Docker, automatic HTTPS and more\\.](https://github.com/tiangolo/full-stack-fastapi-postgresql)
- [eightBEC/fastapi\\-ml\\-skeleton: FastAPI Skeleton App to serve machine learning models production\\-ready\\.](https://github.com/eightBEC/fastapi-ml-skeleton)