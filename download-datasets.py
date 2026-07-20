import os
import shutil
import requests
import zipfile
import io


def download_and_unzip_phase1(url, extract_to):
    """
    Download a zip file from the specified URL and extract its contents directly
    to a folder, without saving the zip file to disk.
    """
    # Send HTTP request
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Load response content into an in-memory bytes buffer
    zip_bytes = io.BytesIO()
    for chunk in response.iter_content(chunk_size=8192):
        zip_bytes.write(chunk)

    # Move pointer to start of the buffer
    zip_bytes.seek(0)

    # Create target folder if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)

    # Open the zipfile from memory and extract
    with zipfile.ZipFile(zip_bytes, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"Downloaded and extracted zip file from {url} to: {extract_to}")


def download_and_unzip_phase2(url, extract_to):
    """
    Download a zip file from the specified URL and extract its contents directly
    to a folder, without saving the zip file to disk.
    """
    # Send HTTP request
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Load response content into an in-memory bytes buffer
    zip_bytes = io.BytesIO()
    for chunk in response.iter_content(chunk_size=8192):
        zip_bytes.write(chunk)

    # Move pointer to start of the buffer
    zip_bytes.seek(0)

    # Create target folder if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)

    # Hardcoded for the specific structure of the zip file, extract only the "test-phase2/yelp" directory
    with zipfile.ZipFile(zip_bytes, "r") as zip_ref:
        for member in zip_ref.infolist():
            name = member.filename

            if not name.startswith("text-phase2/yelp/"):
                continue

            if member.is_dir():
                continue

            # remove prefix
            relative_path = name.split("text-phase2/")[-1]

            target_path = os.path.join(extract_to, relative_path)
            target_path = os.path.normpath(target_path)

            # ensure folder exists
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            # write file
            with zip_ref.open(member) as source, open(target_path, "wb") as target:
                target.write(source.read())

    print(f"Downloaded and extracted zip file from {url} to: {extract_to}")


def main():
    phase1_url = "https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-26-text/text-phase1.zip"
    phase2_url = "https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-26-text/text-phase2.zip"
    extract_folder = "data"

    download_and_unzip_phase1(phase1_url, extract_folder)
    download_and_unzip_phase2(phase2_url, extract_folder)


if __name__ == "__main__":
    main()
