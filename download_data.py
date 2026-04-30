# pip install gdown
import gdown

# folder_url = "https://drive.google.com/drive/folders/1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7"
# gdown.download_folder(url=folder_url, output="data", quiet=False)


import os
import tarfile

workdir = "downloaded_files"
datadir = "data"
os.makedirs(datadir, exist_ok=True)

for fname in os.listdir(workdir):
    if not fname.endswith(".tar.gz"):
        continue
    archive = os.path.join(workdir, fname)
    name = fname[:-7]  # strip .tar.gz
    dest = os.path.join(datadir, name)
    
    if os.path.isdir(dest):
        print(f" {name} already extracted, skipping")
        continue
    
    print(f" Extracting {archive} → {dest}/")
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=dest)
