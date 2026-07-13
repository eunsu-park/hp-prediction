# Vendored from setup-sw-db/core/download.py @ de72933 on 2026-04-19 — DO NOT EDIT.
# Subset retained: download, download_json.
# Re-sync: see src/_vendor/README.md.
"""HTTP download utilities with retry."""
import requests
import urllib3

# Suppress SSL warnings for JSOC self-signed certificates in upstream workflows.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def download_json(url: str, timeout: int = 30, max_retries: int = 3) -> dict | list | None:
    """Download JSON data from URL with retries.

    Args:
        url: URL to download from.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts.

    Returns:
        Parsed JSON object (dict or list) or None if download failed.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            is_retriable = not isinstance(e, requests.HTTPError) or \
                           (e.response is not None and e.response.status_code >= 500)
            if is_retriable and attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries}: {e}")
            else:
                print(f"  Download failed: {e}")
                return None
    return None


def download(url: str, timeout: int = 30, max_retries: int = 3) -> str | None:
    """Download text data from URL with retries.

    Args:
        url: URL to download from.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts.

    Returns:
        Response text or None if download failed.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            if not response.text.strip():
                print(f"  Empty response from {url}")
                return None
            return response.text
        except requests.RequestException as e:
            is_retriable = not isinstance(e, requests.HTTPError) or \
                           (e.response is not None and e.response.status_code >= 500)
            if is_retriable and attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries}: {e}")
            else:
                print(f"  Download failed: {e}")
                return None
    return None
