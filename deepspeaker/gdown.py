import os
from pathlib import Path
import requests
from tqdm import tqdm

GOOGLE_DOWNLOAD_URL = "https://docs.google.com/uc?export=download"


def download(
    id: str, file: str | os.PathLike, desc: str = "Downloading", verbose: bool = True
):
    with requests.Session() as session:
        token, response = None, None

        for _ in range(2):
            response = session.get(
                GOOGLE_DOWNLOAD_URL,
                params={"id": id, **({"confirm": token} if token else {})},
                stream=True,
            )
            response.raise_for_status()

            if not token:
                token = next(
                    (
                        value
                        for key, value in response.cookies.items()
                        if key.startswith("download_warning")
                    ),
                    None,
                )
                continue

        assert response
        size = int(response.headers.get("Content-Length", 0))
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {file} ({size // (1024 ** 2)} MB)")

        with open(file, "wb") as f:
            for chunk in tqdm(
                response.iter_content(1024**2),
                total=size // (1024**2),
                unit="MB",
                disable=not verbose,
                leave=False,
                desc=desc,
            ):
                if chunk:
                    f.write(chunk)
