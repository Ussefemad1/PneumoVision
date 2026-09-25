"""Download the selected MIMIC-CXR images listed in cxr_download_list.csv.

Reads credentials from .env (PHYSIONET_COOKIE) and pulls each dicom_id's jpg
from PhysioNet into data/mimic-cxr-jpg/2.0.0/files/, preserving PhysioNet's
folder structure (which is what resize.py later flattens into resized/).

Notes on the endpoint, learned the hard way:
  - the account's DUA is signed for version 2.1.0, NOT 2.0.0, so requests must
    go to /files/mimic-cxr-jpg/2.1.0/... -- 2.0.0 returns 403 for everything.
    v2.1.0 contains the same 377,110 images; only an extra labels CSV differs.
  - HTTP Basic auth is rejected by PhysioNet for this account (403 even on
    2.1.0). A browser sessionid cookie works. Cookies expire, so if this
    script starts returning 403 everywhere, refresh PHYSIONET_COOKIE in .env.
  - the local folder stays named 2.0.0 because that is what the MedPatch
    code expects as cxr_data_dir.

Safe to interrupt and re-run: files already on disk are skipped, so it
resumes where it stopped. Never deletes anything.
"""
import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
LIST_FILE = REPO_ROOT / "data" / "cxr_download_lists" / "cxr_download_list.csv"
FILENAMES_FILE = REPO_ROOT / "data" / "mimic-cxr-jpg" / "2.0.0" / "IMAGE_FILENAMES.txt"
DEST_ROOT = REPO_ROOT / "data" / "mimic-cxr-jpg" / "2.0.0"
FAILED_LOG = REPO_ROOT / "data" / "cxr_download_lists" / "failed_downloads.txt"

BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

print_lock = threading.Lock()
NETWORK_TROUBLE = threading.Event()


def load_cookie():
    if not ENV_FILE.is_file():
        sys.exit(f"Missing {ENV_FILE}")
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith("PHYSIONET_COOKIE="):
            value = line.split("=", 1)[1].strip()
            if value:
                return value
    sys.exit("PHYSIONET_COOKIE not set in .env")


def load_paths(limit=None, list_file=None):
    """Map the wanted dicom_ids to their PhysioNet relative paths."""
    wanted = set(pd.read_csv(list_file or LIST_FILE)["dicom_id"])
    if limit:
        wanted = set(list(wanted)[:limit])

    paths = []
    found = set()
    with open(FILENAMES_FILE) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            stem = line.rsplit("/", 1)[-1][: -len(".jpg")]
            if stem in wanted:
                paths.append(line)
                found.add(stem)
                if len(found) == len(wanted):
                    break

    missing = wanted - found
    if missing:
        print(f"[warn] {len(missing)} dicom_ids not found in IMAGE_FILENAMES.txt")
    return paths


def wait_for_network(session, probe_timeout=15):
    """Block until PhysioNet is reachable again.

    Without this, a dropped wifi connection makes every queued request fail
    instantly, burning through thousands of images in seconds and ending the
    run. Instead we stall here until the network comes back.
    """
    delay = 10
    while True:
        with print_lock:
            print(f"  [network] unreachable -- waiting {delay}s before probing again")
        time.sleep(delay)
        try:
            response = session.get(f"{BASE_URL}/LICENSE.txt", timeout=probe_timeout)
            if response.status_code == 200:
                with print_lock:
                    print("  [network] back online, resuming")
                return
            if response.status_code in (401, 403):
                with print_lock:
                    print(f"  [network] reachable but HTTP {response.status_code} -- "
                          "session cookie has probably expired; refresh "
                          "PHYSIONET_COOKIE in .env and re-run")
                return
        except Exception:
            pass
        delay = min(delay * 2, 300)


def download_one(session, rel_path, timeout, attempts=3):
    dest = DEST_ROOT / rel_path
    if dest.is_file() and dest.stat().st_size > 0:
        return "skipped", rel_path, dest.stat().st_size

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = "unknown"

    for attempt in range(attempts):
        try:
            response = session.get(f"{BASE_URL}/{rel_path}", timeout=timeout, stream=True)

            if response.status_code in (401, 403):
                # auth problem -- retrying will not help
                return f"http_{response.status_code}", rel_path, 0
            if response.status_code != 200:
                last = f"http_{response.status_code}"
                time.sleep(2 * (attempt + 1))
                continue

            size = 0
            with open(tmp, "wb") as out:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        out.write(chunk)
                        size += len(chunk)

            if size == 0:
                tmp.unlink(missing_ok=True)
                last = "empty"
                time.sleep(2 * (attempt + 1))
                continue

            tmp.replace(dest)
            return "ok", rel_path, size

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as error:
            tmp.unlink(missing_ok=True)
            last = f"error:{type(error).__name__}"
            NETWORK_TROUBLE.set()
            time.sleep(2 * (attempt + 1))

        except Exception as error:
            tmp.unlink(missing_ok=True)
            return f"error:{type(error).__name__}", rel_path, 0

    return last, rel_path, 0


def main():
    parser = argparse.ArgumentParser(description="Download selected MIMIC-CXR images.")
    parser.add_argument("--limit", type=int, default=None,
                        help="only download the first N images (for speed testing)")
    parser.add_argument("--workers", type=int, default=8,
                        help="parallel download threads (default 8)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="per-image timeout in seconds (default 300)")
    parser.add_argument("--list", dest="list_file", default=None,
                        help="CSV of dicom_ids to fetch "
                             "(default: cxr_download_list.csv; "
                             "use task_a_studies.csv for Task A only)")
    args = parser.parse_args()

    cookie = load_cookie()
    paths = load_paths(limit=args.limit, list_file=args.list_file)
    print(f"Requested: {len(paths)} images")
    print(f"Workers:   {args.workers}")
    print(f"Target:    {DEST_ROOT / 'files'}")
    print()

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Cookie": f"sessionid={cookie}"})

    results = {}
    total_bytes = 0
    done = 0
    start = time.time()

    consecutive_failures = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, session, p, args.timeout): p for p in paths}
        for future in as_completed(futures):
            status, rel_path, size = future.result()
            results[status] = results.get(status, 0) + 1
            total_bytes += size
            done += 1

            # A run of network errors means the connection is down. Stall the
            # main loop until it recovers, otherwise the remaining queue fails
            # instantly and the whole run is lost to one brief outage.
            if status.startswith("error:"):
                consecutive_failures += 1
                if consecutive_failures >= args.workers and NETWORK_TROUBLE.is_set():
                    wait_for_network(session)
                    NETWORK_TROUBLE.clear()
                    consecutive_failures = 0
            elif status in ("ok", "skipped"):
                consecutive_failures = 0

            if done % 10 == 0 or done == len(paths):
                elapsed = time.time() - start
                rate = total_bytes / elapsed / 1024 if elapsed else 0
                with print_lock:
                    print(f"  {done}/{len(paths)}  "
                          f"{total_bytes/1024/1024:.0f} MB  "
                          f"{rate:.0f} KB/s  "
                          f"{elapsed:.0f}s elapsed",
                          flush=True)

    elapsed = time.time() - start
    print()
    print("=" * 60)
    for status, count in sorted(results.items()):
        print(f"  {status:20s} {count}")
    print("=" * 60)

    on_disk = sum(1 for p in paths if (DEST_ROOT / p).is_file() and (DEST_ROOT / p).stat().st_size > 0)
    print(f"Requested:      {len(paths)}")
    print(f"On disk now:    {on_disk}")
    print(f"Downloaded:     {total_bytes/1024/1024:.1f} MB in {elapsed:.0f}s "
          f"({total_bytes/elapsed/1024:.0f} KB/s)" if elapsed else "")

    failures = [p for p in paths if not ((DEST_ROOT / p).is_file() and (DEST_ROOT / p).stat().st_size > 0)]
    if failures:
        FAILED_LOG.write_text("\n".join(failures) + "\n")
        print(f"\n{len(failures)} failed -- written to {FAILED_LOG}")
        print("Re-run this script to retry them (completed files are skipped).")
    else:
        print("\nAll requested images present.")

    if on_disk and elapsed and args.limit:
        per_image = elapsed / on_disk
        full = 15181
        print(f"\nProjection: {per_image:.1f}s/image -> "
              f"{full * per_image / 3600:.1f} hours for all {full}")


if __name__ == "__main__":
    main()
