"""DEAD END -- KEPT AS DOCUMENTATION. DO NOT RUN. See the warning below.

=============================================================================
HTTP Basic auth does NOT work for PhysioNet downloads on our accounts.
=============================================================================

This script uses `wget --user/--password`-style HTTP Basic auth, which is the
method PhysioNet's own documentation describes. It returns **403 on every
file**, including files we can demonstrably download through the browser.

What we established, so nobody repeats the day this cost:

  * Basic auth fails with 403 -- and identically for correct credentials,
    deliberately wrong credentials, and no credentials at all. The 403 is
    therefore useless for diagnosing the problem.
  * PhysioNet never sends a 401 challenge, so Python's HTTPBasicAuthHandler
    never even attaches the credentials. Sending the Authorization header
    preemptively does not help either.
  * A browser **session cookie works**. Copy `sessionid` from the browser
    (F12 -> Application -> Cookies -> physionet.org) into .env as
    PHYSIONET_COOKIE, and send it as `Cookie: sessionid=...`.
    Cookies expire after roughly two weeks; refresh it when downloads start
    failing with 403 again.
  * Separately: our DUA is signed for **mimic-cxr-jpg v2.1.0, not v2.0.0**.
    Every request to /files/mimic-cxr-jpg/2.0.0/... returns 403 regardless of
    auth method. v2.1.0 holds the same 377,110 images and still ships the
    metadata under the original `mimic-cxr-2.0.0-*` filenames.

The working implementation is scripts/download_cxr_images.py.
The metadata CSVs were ultimately downloaded by hand through the browser.

=============================================================================

Original docstring follows.

Weeks 1-2 (Norhan): fetch the MIMIC-CXR-JPG metadata list, not the images.

Downloads mimic-cxr-2.0.0-metadata.csv and mimic-cxr-2.0.0-chexpert.csv from
PhysioNet (a few MB total) and copies in the bundled train/validate/test split
file. This is everything cxr_dataset.py needs to know *which* studies exist
and which patient they belong to -- the ~11,000 actual jpgs get pulled later,
once the patient list exists (see the handbook, Weeks 4-5).

Credentials come from a local .env file (PHYSIONET_USER, PHYSIONET_PASSWORD)
that is gitignored and never printed by this script.
"""
import gzip
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
BUNDLED_SPLIT = REPO_ROOT / "medpatch" / "ehr_utils" / "mimic-cxr-ehr-split.csv"
OUT_DIR = REPO_ROOT / "data" / "mimic-cxr-jpg" / "2.0.0"

BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.0.0"
FILES = [
    "mimic-cxr-2.0.0-metadata.csv.gz",
    "mimic-cxr-2.0.0-chexpert.csv.gz",
]


def load_credentials():
    if not ENV_FILE.is_file():
        sys.exit(f"Missing {ENV_FILE}. Create it with PHYSIONET_USER and PHYSIONET_PASSWORD.")

    values = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()

    user = values.get("PHYSIONET_USER")
    password = values.get("PHYSIONET_PASSWORD")
    if not user or not password:
        sys.exit("PHYSIONET_USER and PHYSIONET_PASSWORD must both be set in .env")
    return user, password


def build_opener(user, password):
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, BASE_URL, user, password)
    handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
    return urllib.request.build_opener(handler)


def download_one(opener, filename, out_dir: Path):
    csv_path = out_dir / filename[: -len(".gz")]
    if csv_path.is_file():
        print(f"[skip] {csv_path.name} already present")
        return

    gz_path = out_dir / filename
    url = f"{BASE_URL}/{filename}"
    print(f"[get]  {filename}")

    with opener.open(url, timeout=60) as response, open(gz_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)

    with gzip.open(gz_path, "rb") as gz_in, open(csv_path, "wb") as csv_out:
        shutil.copyfileobj(gz_in, csv_out)
    gz_path.unlink()

    size_kb = csv_path.stat().st_size / 1024
    print(f"[ok]   {csv_path.name} ({size_kb:.0f} KB)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    user, password = load_credentials()
    opener = build_opener(user, password)

    try:
        for filename in FILES:
            download_one(opener, filename, OUT_DIR)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            sys.exit(
                "Authentication failed (HTTP "
                f"{error.code}). Check PHYSIONET_USER / PHYSIONET_PASSWORD in .env, "
                "and confirm your PhysioNet account has signed the MIMIC-CXR-JPG "
                "data use agreement (a separate step from just having an account)."
            )
        raise

    if BUNDLED_SPLIT.is_file():
        dest = OUT_DIR / BUNDLED_SPLIT.name
        shutil.copy(BUNDLED_SPLIT, dest)
        print(f"[ok]   copied bundled split file -> {dest.name}")
    else:
        print(f"[warn] bundled split file not found at {BUNDLED_SPLIT}")

    print()
    print(f"Metadata ready in {OUT_DIR}")
    print("This is the list only -- the actual jpgs (Weeks 4-5) come after matching")
    print("against the EHR patient list and filtering to frontal views.")


if __name__ == "__main__":
    main()
