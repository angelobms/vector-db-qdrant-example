import json
import os
import re
import uuid
from typing import Any, Dict, Generator, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchText, OptimizersConfigDiff, PointStruct, ScoredPoint
from tqdm import tqdm

COLLECTION_NAME = "arxiv_papers"

# https://www.kaggle.com/datasets/awester/arxiv-embeddings/versions/85?resource=download
DATA_FILE = "ml-arxiv-embeddings.json"

EMBEDDING_MODEL = "text-embedding-ada-002"

load_dotenv()

client = QdrantClient(host="localhost", port=6333, timeout=60)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()


def stream_json(file_path) -> Generator[Any, Any, None]:
    """
    Lazily yield one parsed JSON object per line from a JSONL file.

    Args:
        file_path: Path to the newline-delimited JSON file.

    Yields:
        The decoded JSON object for each line, in file order.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def create_collection() -> None:
    """
    Create the `arxiv_papers` Qdrant collection if it does not already exist.

    Uses 1536-dim/Cosine vectors and disables automatic indexing
    (`indexing_threshold=0`) so the initial bulk load isn't slowed down, or
    corrupted, by concurrent segment optimization.
    """
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"size": 1536, "distance": "Cosine"},
            optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
        )


def load_vectors_to_qdrant(file_path, batch_size=500) -> None:
    """
    Stream arXiv records from `file_path` and upsert them into Qdrant in batches.

    Each record becomes a `PointStruct` whose vector is the record's `embedding`
    and whose payload is every other field; the point ID is a UUID5 derived from
    the arXiv paper ID. Records with no embedding are skipped. Progress is
    reported with a tqdm bar and a final summary print.

    Args:
        file_path: Path to the JSONL dataset (see `stream_json`).
        batch_size: Number of points to accumulate before each `client.upsert` call.
    """
    batch = []
    total_records = 0
    skipped = 0

    with tqdm(desc="Loading records", unit="rec") as pbar:
        for record in stream_json(file_path):
            embedding = record.get("embedding")
            if not embedding:
                skipped += 1
                continue

            payload = {k: v for k, v in record.items() if k != "embedding"}

            point = PointStruct(
                id=str(uuid.uuid5(namespace=uuid.NAMESPACE_DNS, name=record["id"])),
                vector=embedding,
                payload=payload,
            )
            batch.append(point)

            if len(batch) >= batch_size:
                client.upsert(collection_name=COLLECTION_NAME, points=batch)
                total_records += len(batch)
                pbar.update(len(batch))
                batch = []

        if batch:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            total_records += len(batch)
            pbar.update(len(batch))
            batch = []

    print(f"Loaded {total_records} records, skipped {skipped} records without embeddings.")


def find_similar_papers(collection_name: str, paper_id: str, top_k: int = 5) -> List[str]:
    """
    Find the papers whose embeddings are most similar to a given paper's own embedding.

    Looks up the point whose `id` payload field equals `paper_id`, uses its stored
    vector as the query, and returns the IDs of the `top_k` nearest points
    (the paper itself is typically included as the closest match).

    Args:
        collection_name: Qdrant collection to search.
        paper_id: The arXiv paper ID to find neighbors for (e.g. "1311.5068").
        top_k: Number of results to return, including the paper itself.

    Returns:
        Paper IDs of the top-k most similar points, ordered by similarity.

    Raises:
        ValueError: If no point with that paper ID exists in the collection.
    """
    scroll_filter = {"must": [{"key": "id", "match": {"value": paper_id}}]}
    response = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=1,
        with_vectors=True,
    )
    points, _ = response
    if not points:
        raise ValueError(f"Point with paper ID '{paper_id}' was not found in collection '{collection_name}'.")

    query_vector = points[0].vector

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    ).points

    return [result.payload["id"] for result in results]


def get_query_embedding(query: str) -> List[float]:
    """
    Embed a natural-language query with the same OpenAI model used for the dataset.

    Args:
        query: Raw query text. Newlines are stripped before embedding, since the
            embedding API performs better on single-line input.

    Returns:
        The 1536-element embedding vector for the query.
    """
    query = query.replace("\n", " ")
    response = openai_client.embeddings.create(input=query, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def extract_author(query: str) -> Optional[str]:
    """
    Pull an author name out of a query of the form "... by <Author Name>".

    Args:
        query: Raw query text.

    Returns:
        The matched author name (stripped of surrounding whitespace), or
        `None` if the query contains no "by <name>" clause.
    """
    match = re.search(r"by\s+([A-Za-z\s\-]+)", query)
    if match:
        return match.group(1).strip()
    return None


def query_papers(collection_name: str, query: str, top_k: int = 5) -> List[ScoredPoint]:
    """
    Semantic-search the collection for a natural-language query, optionally
    filtered to a single author named in the query (e.g. "... by Jane Doe").

    Embeds `query` and runs a vector search. If `extract_author` finds an
    author clause, candidates are first narrowed with a token-based
    `MatchText` filter on the `authors` payload field, then further filtered
    client-side to only those whose `authors` field literally contains the
    author's name as a substring (since `MatchText` matches each word
    independently, e.g. "Andrew Hard ... Sara Ng" would otherwise falsely
    match a search for "Andrew Ng").

    Args:
        collection_name: Qdrant collection to search.
        query: Natural-language query, optionally containing "by <author>".
        top_k: Number of results to return.

    Returns:
        Up to `top_k` `ScoredPoint`s (with payload, without vectors),
        ordered by similarity score.
    """
    query_vector = get_query_embedding(query)

    author = extract_author(query)
    if not author:
        return client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        ).points

    query_filter = Filter(must=[FieldCondition(key="authors", match=MatchText(text=author))])
    candidates = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=max(top_k * 20, 200),
        with_payload=True,
        with_vectors=False,
    ).points

    author_lower = author.lower()
    matches = [c for c in candidates if author_lower in c.payload.get("authors", "").lower()]
    return matches[:top_k]


def search_by_text(collection_name: str, query: str, top_k: int = 5) -> List[str]:
    """
    Thin wrapper around `query_papers` that returns just the matched paper IDs.

    Args:
        collection_name: Qdrant collection to search.
        query: Natural-language query, optionally containing "by <author>".
        top_k: Number of results to return.

    Returns:
        Paper IDs of the matching points, ordered by similarity score.
    """
    results = query_papers(collection_name, query, top_k)
    return [result.payload["id"] for result in results]


class SearchRequest(BaseModel):
    """
    Model representing a search request sent by a client.

    Attributes:
        query (str): The search query string provided by the client.
        top_n (Optional[int]): The number of top search results to return.
            Defaults to 5 if not provided.
    """

    query: str
    top_n: Optional[int] = 5


class SearchResult(BaseModel):
    """
    Model representing an individual search result.

    Attributes:
        id (str): Unique identifier (Qdrant point ID) for the search result item.
        payload (Dict): A dictionary containing additional data or metadata about the item.
        score (float): The relevance score of the search result, typically used for ranking.
    """

    id: str
    payload: Dict
    score: float


class SearchResponse(BaseModel):
    """
    Model representing the response returned to a client after a search.

    Attributes:
        results (List[SearchResult]): A list containing the top matching search results.
    """

    results: List[SearchResult]


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """
    Search endpoint: embed `request.query`, run the author-aware vector
    search via `query_papers`, and return the top `request.top_n` matches.

    Args:
        request: The parsed `SearchRequest` body.

    Returns:
        A `SearchResponse` holding the matching results.
    """
    results = query_papers(COLLECTION_NAME, request.query, request.top_n)
    search_results = [
        SearchResult(id=str(result.id), payload=result.payload, score=result.score) for result in results
    ]
    return SearchResponse(results=search_results)


create_collection()
if client.count(collection_name=COLLECTION_NAME).count == 0:
    load_vectors_to_qdrant(DATA_FILE)
    client.update_collection(
        collection_name=COLLECTION_NAME,
        optimizers_config=OptimizersConfigDiff(indexing_threshold=10000),
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
