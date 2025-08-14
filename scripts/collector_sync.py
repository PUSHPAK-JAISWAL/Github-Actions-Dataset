#!/usr/bin/env python3
"""
Synchronous GitHub Actions workflow scraper + Gemini question generator.

Works per-run:
- Searches GitHub code for .github/workflows/*.yml or .yaml
- Fetches file content
- For up to MAX_GEMINI_CALLS_PER_RUN files per run:
    - Calls Gemini to generate one concise question
    - Stores {"question":..., "answer": <raw workflow YAML>, meta...} in datasets/dataset_N.json
    - Marks the item processed in datasets/processed.json
"""

import requests
import os
import base64
import json
import time
from datetime import datetime
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError

# ---------- Config (from env)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
MAX_GEMINI_CALLS_PER_RUN = int(os.getenv("MAX_GEMINI_CALLS_PER_RUN", "10"))
DATASET_DIR = "datasets"
PROCESSED_PATH = os.path.join(DATASET_DIR, "processed.json")
MAX_ENTRIES_PER_FILE = 1000
GITHUB_SEARCH_PER_PAGE = 100  # GitHub max
SEARCH_STARS_FILTER = os.getenv("SEARCH_STARS_FILTER", "").strip()  # optional e.g. "stars:>10"

HEADERS = {}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"
HEADERS["Accept"] = "application/vnd.github.v3+json"

os.makedirs(DATASET_DIR, exist_ok=True)

# Configure Gemini API client
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
else:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')

# ---------- dataset helpers ----------
def list_dataset_files():
    files = [f for f in os.listdir(DATASET_DIR) if f.startswith("dataset_") and f.endswith(".json")]
    def idx(fn):
        try:
            return int(fn.split("_")[1].split(".")[0])
        except:
            return 0
    return sorted(files, key=idx)

def get_current_dataset_path():
    files = list_dataset_files()
    if not files:
        return os.path.join(DATASET_DIR, "dataset_1.json")
    last = files[-1]
    path = os.path.join(DATASET_DIR, last)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            arr = json.load(fh)
    except Exception:
        arr = []
    if len(arr) >= MAX_ENTRIES_PER_FILE:
        idx = int(last.split("_")[1].split(".")[0]) + 1
        return os.path.join(DATASET_DIR, f"dataset_{idx}.json")
    return path

def append_to_dataset(obj: dict):
    path = get_current_dataset_path()
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            data = []
    except Exception:
        data = []
    data.append(obj)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ---------- processed tracking ----------
def load_processed():
    if not os.path.exists(PROCESSED_PATH):
        return set()
    try:
        with open(PROCESSED_PATH, "r", encoding="utf-8") as fh:
            arr = json.load(fh)
            return set(arr)
    except Exception:
        return set()

def save_processed(s: set):
    with open(PROCESSED_PATH, "w", encoding="utf-8") as fh:
        json.dump(sorted(list(s)), fh, indent=2)


# ---------- GitHub helpers ----------
def github_search(page=1, per_page=GITHUB_SEARCH_PER_PAGE):
    # Build query; user can set SEARCH_STARS_FILTER in workflow env
    query_parts = ["path:.github/workflows", "extension:yml", "extension:yaml"]
    if SEARCH_STARS_FILTER:
        query_parts.append(SEARCH_STARS_FILTER)
    query = " ".join(query_parts)
    url = "https://api.github.com/search/code"
    params = {"q": query, "per_page": per_page, "page": page}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if resp.status_code == 403:
        # Rate limit or forbidden
        reset = resp.headers.get("X-RateLimit-Reset")
        print(f"[github_search] 403 rate limit. Reset: {reset}. Response: {resp.text}")
        resp.raise_for_status()
    resp.raise_for_status()
    return resp.json().get("items", [])

def fetch_file_contents(contents_url):
    resp = requests.get(contents_url, headers=HEADERS, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json()
    content_b64 = data.get("content")
    if not content_b64:
        return None
    try:
        return base64.b64decode(content_b64).decode("utf-8", errors="ignore")
    except Exception:
        return base64.b64decode(content_b64 + "===").decode("utf-8", errors="ignore")


# ---------- Gemini call ----------
def call_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        # Check if the response is a BlockedPromptError
        if response.candidates and response.candidates[0].safety_ratings and \
           any(rating.blocked for rating in response.candidates[0].safety_ratings):
            print("[gemini] response was blocked due to safety policy.")
            return ""
        return response.text.strip()
    except genai.types.BlockedPromptException:
        print("[gemini] prompt was blocked due to safety policy.")
        return ""
    except GoogleAPIError as e:
        print(f"[gemini] API error: {e}")
        return ""
    except Exception as e:
        print(f"[gemini] Unknown error during API call: {e}")
        return ""


# ---------- main flow ----------
def main():
    print("[start] collector_sync")
    processed = load_processed()
    gemini_used = 0
    page = 1
    total_added = 0

    # Keep searching pages until gemini budget exhausted or no more items
    while gemini_used < MAX_GEMINI_CALLS_PER_RUN:
        try:
            items = github_search(page=page)
        except Exception as e:
            print(f"[error] github_search page {page}: {e}")
            break

        if not items:
            print("[done] no more search items")
            break

        for item in items:
            if gemini_used >= MAX_GEMINI_CALLS_PER_RUN:
                break

            repo = item.get("repository", {}).get("full_name", "unknown")
            path = item.get("path")
            unique_id = f"{repo}:{path}"
            if unique_id in processed:
                # already handled previously
                continue

            contents_url = item.get("url")  # contents API url returned by search
            try:
                content = fetch_file_contents(contents_url)
            except Exception as e:
                print(f"[error] fetch_file_contents {unique_id}: {e}")
                continue

            if not content:
                print(f"[skip] empty content for {unique_id}")
                # do NOT mark as processed, so future runs can try again
                continue

            # Build prompt for Gemini: generate a single concise question, no answer
            prompt = (
                "You will be given the contents of a GitHub Actions workflow YAML file.\n"
                "Generate exactly one clear, single-sentence question (no answer) about how this workflow behaves or what it does.\n"
                "Keep it concise (preferably under 30 words).\n\n"
                f"--- WORKFLOW START ---\n{content}\n--- WORKFLOW END ---\n\n"
                "Question:"
            )

            try:
                q_text = call_gemini(prompt)
            except Exception as e:
                print(f"[error] gemini API call for {unique_id}: {e}")
                # do not mark as processed; allow retry in future
                continue

            if not q_text:
                print(f"[skip] empty gemini response for {unique_id}")
                continue

            # sanitize question: take first non-empty line
            q_line = next((line for line in q_text.splitlines() if line.strip()), q_text.strip())
            qa_obj = {
                "question": q_line.strip(),
                "answer": content,
                "source": repo,
                "path": path,
                "url": item.get("html_url", contents_url),
                "retrieved_at": datetime.utcnow().isoformat() + "Z"
            }

            try:
                append_to_dataset(qa_obj)
                processed.add(unique_id)
                gemini_used += 1
                total_added += 1
                print(f"[added] {unique_id} (gemini {gemini_used}/{MAX_GEMINI_CALLS_PER_RUN})")
            except Exception as e:
                print(f"[error] append_to_dataset for {unique_id}: {e}")
                # do not mark as processed if write fails
                continue

        page += 1
        time.sleep(1) # Add a small delay between pages to avoid hitting GitHub rate limits

    save_processed(processed)
    print(f"[finished] gemini_used={gemini_used}, total_added={total_added}")

    # optional: try to commit from script, but workflow will also commit
    try:
        import subprocess, shlex
        subprocess.run(shlex.split('git config --global user.name "github-actions[bot]"'), check=False)
        subprocess.run(shlex.split('git config --global user.email "github-actions[bot]@users.noreply.github.com"'), check=False)
        subprocess.run("git add datasets/*.json || true", shell=True, check=False)
        subprocess.run("git commit -m \"Update datasets via collector_sync\" || true", shell=True, check=False)
        subprocess.run("git push || true", shell=True, check=False)
    except Exception as e:
        print(f"[git fallback] {e}")


if __name__ == "__main__":
    main()
