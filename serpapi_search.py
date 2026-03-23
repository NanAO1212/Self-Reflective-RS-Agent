import os
from typing import List

import requests


def _format_results(items: List[dict], k: int, with_url: bool = False, max_len: int = 1500) -> str:
    snippets = []
    for item in items[:k]:
        title = item.get("title", "")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        content = ""
        if title:
            content += f"{title}: "
        if link and with_url:
            content += f"({link})"
        if snippet:
            content += snippet
        snippets.append(content.strip())

    if not snippets:
        return "No good Google Search Result was found"

    result = ""
    for idx, item in enumerate(snippets, 1):
        result += f"{idx} - {item}\n\n"

    if len(result) > max_len:
        result = result[:max_len] + "..."
    return result


def serpapi_search(query: str, k: int = 10, timeout: int = 10) -> str:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("Please set SERPAPI_API_KEY for SerpApi.")

    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": k,
    }
    resp = requests.get("https://serpapi.com/search.json", params=params, timeout=timeout)
    if resp.status_code != 200:
        raise ConnectionError(f"Error {resp.status_code}: {resp.text}")

    data = resp.json()
    organic = data.get("organic_results", [])
    return _format_results(organic, k=k, with_url=False)
