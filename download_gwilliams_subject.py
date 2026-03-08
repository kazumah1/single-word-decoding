"""Download multiple subjects from the Gwilliams2022 (MEG-MASC) dataset.

Usage:
    python download_gwilliams_subject.py --subjects sub-02 sub-03 --path neural_data/gwilliams2022

Requirements:
    pip install osfclient mne_bids
"""

import argparse
from pathlib import Path

OSF_REPOS = ["ag3kj", "h2tzn", "u5327"]
TOP_LEVEL_FILES = {
    "dataset_description.json",
    "participants.json",
    "participants.tsv",
    "README.txt",
}


def download_subject(subject: str, dest: Path) -> None:
    import osfclient
    from tqdm import tqdm

    dl_dir = dest / "download"
    dl_dir.mkdir(parents=True, exist_ok=True)

    for repo_id in OSF_REPOS:
        print(f"\n--- OSF repo: {repo_id} ---")
        project = osfclient.OSF().project(repo_id)

        for storage in project.storages:
            files = list(storage.files)
            to_download = []

            for f in files:
                path = f.path.lstrip("/")
                if (
                    path in TOP_LEVEL_FILES
                    or path.startswith("stimuli/")
                    or path.startswith(f"{subject}/")
                ):
                    to_download.append((path, f))

            if not to_download:
                print(f"  No matching files found.")
                continue

            print(f"  {len(to_download)} files to download.")
            for path, f in tqdm(to_download, desc=repo_id):
                out = dl_dir / path
                if out.exists():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("wb") as fb:
                    f.write_to(fb)

    print(f"\nDone. Data saved to: {dl_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=["sub-02", "sub-03"], help="Subject IDs, e.g. sub-02 sub-03")
    parser.add_argument("--path", default="neural_data/gwilliams2022", help="Destination path")
    args = parser.parse_args()

    for subject in args.subjects:
        print(f"\n=== Downloading {subject} ===")
        download_subject(subject, Path(args.path))
