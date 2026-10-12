# Vector Database with Qdrant — arXiv Semantic Search

A small semantic search service over the arXiv papers dataset, built on
[Qdrant](https://qdrant.tech/) for vector storage/search and OpenAI's
`text-embedding-ada-002` for query embeddings. Exposes a single FastAPI
endpoint, `POST /search`, that takes a natural-language query (optionally
naming an author) and returns the most similar papers.

## Requirements

- Python 3.x with `venv`
- Docker (for the Qdrant instance)
- An OpenAI API key
- The dataset: [`arxiv-embeddings` (version 85, 2024/12/22)](https://www.kaggle.com/datasets/awester/arxiv-embeddings/versions/85) from Kaggle,
  downloaded as `ml-arxiv-embeddings.json`

Each line of the dataset is a JSON object with an arXiv paper's metadata and a
1536-dim embedding of its title/authors/year/abstract (see `main.py` for the
exact fields).

## Setup

1. **Place the dataset.** Download `ml-arxiv-embeddings.json` from Kaggle and
   put it in this folder (or update `DATA_FILE` in `main.py` to point
   elsewhere). Keep it out of version control — it's large and already
   covered by `.gitignore`.

2. **Create a virtual environment and install dependencies:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Set your OpenAI API key.** Create a `.env` file in this folder:

   ```
   OPENAI_API_KEY=sk-...
   ```

   Never commit `.env` or hardcode the key — it's excluded via `.gitignore`.

4. **Start Qdrant** (from this folder, so the storage volume lands here):

   ```bash
   docker run -p 6333:6333 -p 6334:6334 \
     -v "$(pwd)/qdrant_storage":/qdrant/storage:z \
     qdrant/qdrant
   ```

   Make sure nothing else is already listening on port 6333.

## Running

```bash
source venv/bin/activate
python main.py
# or: uvicorn main:app --reload
```

On first run, this creates the `arxiv_papers` collection (1536-dim,
Cosine distance) and streams/batches the dataset into it — this takes a
few minutes. On later runs, loading is skipped once the collection already
has points. The server then listens on `http://localhost:8000`.

Interactive API docs: `http://localhost:8000/docs`.

## API

### `POST /search`

**Request body:**

```json
{
  "query": "the attention mechanism in deep learning",
  "top_n": 5
}
```

- `query` — natural-language search text. If it contains `by <Author Name>`,
  results are additionally filtered to papers whose `authors` field contains
  that name.
- `top_n` — number of results to return (defaults to 5).

**Response body:**

```json
{
  "results": [
    {
      "id": "e56dda13-8e69-5a3e-beff-4979ae826edb",
      "payload": { "id": "1706.03762", "title": "...", "authors": "...", "...": "..." },
      "score": 0.87
    }
  ]
}
```

`id` is the Qdrant point ID; the arXiv paper ID is `payload.id`.

**Example:**

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "Papers on clustering by Andrew Ng", "top_n": 5}'
```

## How search works (`main.py`)

- `create_collection` / `load_vectors_to_qdrant` — one-time setup: create the
  collection and batch-upsert every record that has an embedding (point ID =
  UUID5 of the arXiv ID; payload = all fields except the embedding).
- `find_similar_papers` — given an existing paper's ID, find its nearest
  neighbors by embedding similarity.
- `get_query_embedding` — embed free-text queries with the same
  `text-embedding-ada-002` model used to build the dataset.
- `extract_author` — pull an author name out of queries shaped like
  `"... by <Author Name>"`.
- `query_papers` — the core search: embeds the query, and if an author was
  named, narrows to papers whose `authors` field contains that name (as a
  literal substring, not just matching tokens) before ranking by similarity.
- `search_by_text` — convenience wrapper returning just the matched paper IDs.
- `POST /search` — the FastAPI endpoint wiring the above into a validated
  request/response pair (`SearchRequest` / `SearchResponse`).
