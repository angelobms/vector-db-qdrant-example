import json
import requests
from hstest import StageTest, CheckResult, dynamic_test

class QdrantCollectionTest(StageTest):
    @dynamic_test(time_limit=120_000)
    def test_search_endpoint(self):
        url = 'http://localhost:8000/search'

        request_payload = {
            "query": "Papers on clustering by Andrew Ng",
            "top_n": 5
        }

        try:
            response = requests.post(url, json=request_payload)
        except requests.exceptions.ConnectionError:
            return CheckResult.wrong("Cannot connect to the server at 'http://localhost:8000'. Ensure the FastAPI app is running.")

        if response.status_code != 200:
            return CheckResult.wrong(f"Expected status code 200, but got {response.status_code}.")

        try:
            response_data = response.json()
        except json.JSONDecodeError:
            return CheckResult.wrong("Response is not valid JSON.")

        if 'results' not in response_data:
            return CheckResult.wrong("Response JSON does not contain 'results' key.")

        results = response_data['results']

        if not isinstance(results, list):
            return CheckResult.wrong("'results' should be a list.")

        expected_top_n = request_payload['top_n']
        if len(results) != expected_top_n:
            return CheckResult.wrong(f"Expected {expected_top_n} results, but got {len(results)}.")

        for idx, result in enumerate(results):
            if not isinstance(result, dict):
                return CheckResult.wrong(f"Result at index {idx} is not a dictionary.")

            required_result_keys = {'id', 'payload', 'score'}
            if not required_result_keys.issubset(result.keys()):
                return CheckResult.wrong(f"Result at index {idx} does not contain the required keys: {required_result_keys}.")

            actual_id = result['id']
            if not isinstance(actual_id, str):
                return CheckResult.wrong(f"'id' at index {idx} should be a string.")

            if not isinstance(result['score'], (float, int)):
                return CheckResult.wrong(f"'score' at index {idx} should be a number.")

            payload = result['payload']
            if not isinstance(payload, dict):
                return CheckResult.wrong(f"'payload' at index {idx} is not a dictionary.")

            expected_payload_keys = {
                'id', 'submitter', 'authors', 'title', 'comments',
                'journal-ref', 'doi', 'report-no', 'categories',
                'license', 'abstract', 'versions', 'update_date',
                'authors_parsed'
            }
            missing_keys = expected_payload_keys - payload.keys()
            if missing_keys:
                return CheckResult.wrong(f"Payload at index {idx} is missing keys: {missing_keys}.")

            authors = payload.get('authors', '')
            if not isinstance(authors, str):
                return CheckResult.wrong(f"'authors' in payload at index {idx} should be a string.")

            if 'Andrew Ng' not in authors:
                return CheckResult.wrong(f"'Andrew Ng' not found in 'authors' of result at index {idx}.")

        return CheckResult.correct()