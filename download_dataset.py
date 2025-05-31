import os
import zipfile
from pathlib import Path
from SoccerNet.Downloader import SoccerNetDownloader as SNdl

# Create dataset directory
dataset_dir = Path(".")
mvfouls_dir = dataset_dir / "mvfouls"
mvfouls_dir.mkdir(exist_ok=True)

print(f"Downloading MVFouls dataset to: {mvfouls_dir}")

# Download the dataset
mySNdl = SNdl(LocalDirectory=str(dataset_dir))
mySNdl.downloadDataTask(task="mvfouls", split=["train","valid","test","challenge"], password="s0cc3rn3t")

print("Download completed. Extracting archived folders...")

# Define the expected zip files and their corresponding folder names
zip_splits = ["train", "valid", "test", "challenge"]

for split in zip_splits:
    zip_file = mvfouls_dir / f"{split}.zip"
    extract_dir = mvfouls_dir / split
    
    if zip_file.exists():
        print(f"Extracting {split}.zip to {split}/ folder...")
        
        # Create the extraction directory
        extract_dir.mkdir(exist_ok=True)
        
        # Extract the zip file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Remove the zip file after extraction
        os.remove(zip_file)
        print(f"✓ Extracted and cleaned up {split}.zip")
    else:
        print(f"⚠ Warning: {split}.zip not found")

print("\nAll archives extracted successfully!")
print(f"Dataset structure in: {mvfouls_dir}")

# List the final directory structure
extracted_dirs = [d for d in mvfouls_dir.iterdir() if d.is_dir()]
if extracted_dirs:
    print("\nFinal directory structure:")
    for dir_path in sorted(extracted_dirs):
        file_count = len(list(dir_path.rglob("*"))) if dir_path.exists() else 0
        print(f"  📁 {dir_path.name}/ ({file_count} files)")
else:
    print("⚠ No directories found after extraction")