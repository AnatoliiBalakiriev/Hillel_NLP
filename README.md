# FastAPI ML Template

## Run Web API
### Local

```sh
$ sh run.sh
```

```sh
$ poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 9000
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

```sh 
$ curl --request POST --url http://127.0.0.1:9000/api/v1/predict --header 'Content-Type: application/json' --data '{"input_text": "test"}'
```

```sh
$ http POST http://127.0.0.1:9000/api/v1/predict input_text=テスト
```

## Development
### Run Tests and Linter

```
$ poetry run tox
```

## String Similarity Endpoint

This project includes an endpoint for calculating string similarity using the TextDistance library and FastAPI.

### Installation

Ensure you have Python 3.7 or higher installed. Install the necessary libraries by running:

```bash
pip install -r requirements.txt
```

### Running the Server

To run the FastAPI server, use the following command:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

#### Parameters

- `app.main:app` - Path to your FastAPI application. Here, `app` is the directory, `main` is the `main.py` file, and `app` is the FastAPI instance.
- `--reload` - Enables auto-reloading of the server on code changes (useful for development).
- `--host 0.0.0.0` - Makes the server accessible from any IP address (useful if you want to access the server from another device in the network).
- `--port 8001` - Port on which the server will run. You can use any available port.

### API Documentation

After starting the server, open your browser and go to:

```
http://localhost:8001/docs
```

This will open the Swagger UI where you can see all your endpoints and test them.

### Example Requests

#### Calculate String Similarity

Send a POST request to `/similarity` with the following request body, using different methods like `hamming`, `mlipns`, `levenshtein`, `damerau_levenshtein`, `jaro_winkler`, `jaro`, `strcmp95`, `jaccard`, `sorensen`, `sorensen_dice`, `tversky`, `overlap`, `tanimoto`, `overlap`, `cosine`, `monge_elkan`, `bag`, `lcsstr`, `tversky`, `ratcliff_obershelp`, `arith_ncd`, `rle_ncd`, `bwtrle_ncd`, `sqrt_ncd`, `entropy_ncd`, `bz2_ncd`, `lzma_ncd`, `zlib_ncd`, `editex`, `prefix`, `postfix`, `length`, `identity`, `matrix`:

##### Using Levenshtein

```json
{
    "method": "levenshtein",
    "line1": "hello",
    "line2": "hola"
}
```

The response will look like this:

```json
{
    "method": "levenshtein",
    "line1": "hello",
    "line2": "hola",
    "similarity": 0.6
}
```

##### Using Jaccard

```json
{
    "method": "jaccard",
    "line1": "hello",
    "line2": "hola"
}
```

The response will look like this:

```json
{
    "method": "jaccard",
    "line1": "hello",
    "line2": "hola",
    "similarity": 0.25
}
```

##### Using Hamming

```json
{
    "method": "hamming",
    "line1": "hello",
    "line2": "hullo"
}
```

The response will look like this:

```json
{
    "method": "hamming",
    "line1": "hello",
    "line2": "hullo",
    "similarity": 0.8
}
```

## Reference

- [tiangolo/full\-stack\-fastapi\-postgresql: Full stack, modern web application generator\. Using FastAPI, PostgreSQL as database, Docker, automatic HTTPS and more\.](https://github.com/tiangolo/full-stack-fastapi-postgresql)
- [eightBEC/fastapi\-ml\-skeleton: FastAPI Skeleton App to serve machine learning models production\-ready\.](https://github.com/eightBEC/fastapi-ml-skeleton)

