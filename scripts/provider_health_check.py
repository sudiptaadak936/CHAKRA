#!/usr/bin/env python3
"""
CHAKRA Backend & Infrastructure Dependency Health Check CLI (Step 0).
Queries the running CHAKRA FastAPI backend /health and /health/dependencies endpoints.
"""
import os
import sys
import json
import urllib.request
import urllib.error

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def check_endpoint(path: str) -> bool:
    url = f"{BACKEND_URL}{path}"
    print(f"[*] Checking {url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CHAKRA-CLI/0.1.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            status_code = response.getcode()
            body = response.read().decode("utf-8")
            parsed = json.loads(body)
            print(f"[+] Status {status_code}: OK")
            print(json.dumps(parsed, indent=2))
            return True
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        print(f"[-] HTTP Error {e.code} for {url}: {e.reason}")
        if body:
            print(body)
        return False
    except urllib.error.URLError as e:
        print(f"[-] Connection failed for {url}: {e.reason}")
        return False
    except Exception as e:
        print(f"[-] Unexpected error for {url}: {e}")
        return False


def main():
    print("==========================================")
    print("   CHAKRA Step 0 Health Verification      ")
    print("==========================================")
    h_ok = check_endpoint("/health")
    print()
    d_ok = check_endpoint("/health/dependencies")

    if h_ok and d_ok:
        print("\n[SUCCESS] All health checks passed.")
        sys.exit(0)
    else:
        print("\n[FAILURE] One or more health checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()