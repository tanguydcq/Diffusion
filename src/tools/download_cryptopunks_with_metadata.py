"""
Download CryptoPunks dataset with metadata (type and accessories).
Uses the Kaggle dataset for images and fetches metadata from CSV.
"""

import os
import json
import requests
from pathlib import Path
from tqdm import tqdm
import shutil
import csv
import kagglehub


def get_cryptopunks_metadata():
    """
    Fetch metadata for all 10,000 CryptoPunks from CSV file.
    """
    # GitHub CSV file with all punk attributes (classic format: id, type, count, accessories)
    url = "https://raw.githubusercontent.com/cryptopunksnotdead/punks.attributes/master/original/cryptopunks-classic.csv"
    
    print("Fetching CryptoPunks metadata from CSV...")
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch metadata: {response.status_code}")
    
    # Parse CSV - format: id, type, count, accessories
    lines = response.text.strip().split('\n')
    reader = csv.reader(lines)
    header = next(reader)  # Skip header: id, type, count, accessories
    
    punks = []
    for row in reader:
        if len(row) >= 3:
            punk = {
                'id': int(row[0].strip()),
                'type': row[1].strip(),  # Male, Female, Zombie, Ape, Alien
                'count': int(row[2].strip()) if row[2].strip() else 0,
                'accessories': [a.strip() for a in row[3].split('/') if a.strip()] if len(row) > 3 and row[3].strip() else []
            }
            punks.append(punk)
    
    return punks


def build_attribute_mappings(all_punks_data):
    """
    Build mappings for types and accessories.
    Returns:
        type_to_idx: dict mapping type name to index
        accessory_to_idx: dict mapping accessory name to index
    """
    types = set()
    accessories = set()
    
    for punk in all_punks_data:
        # Combine type and gender for more specific categorization
        # Or just use type: Human, Zombie, Ape, Alien
        types.add(punk['type'])
        accessories.update(punk['accessories'])
    
    # Sort for consistent ordering
    type_to_idx = {t: i for i, t in enumerate(sorted(types))}
    accessory_to_idx = {a: i for i, a in enumerate(sorted(accessories))}
    
    print(f"Found {len(type_to_idx)} types: {list(type_to_idx.keys())}")
    print(f"Found {len(accessory_to_idx)} unique accessories")
    
    return type_to_idx, accessory_to_idx


def download_and_prepare_cryptopunks(output_dir="./data/CRYPTOPUNKS_CLASSES"):
    """
    Download CryptoPunks images and prepare metadata for class-conditional training.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Download images from Kaggle
    print("Downloading CryptoPunks images from Kaggle...")
    kaggle_path = kagglehub.dataset_download("chwasiq0569/cryptopunks-pixel-art-dataset")
    print(f"Images downloaded to: {kaggle_path}")
    
    # Find the images folder
    kaggle_path = Path(kaggle_path)
    if (kaggle_path / "cryptopunks").exists():
        images_source = kaggle_path / "cryptopunks"
    else:
        images_source = kaggle_path
    
    # 2. Get metadata
    try:
        metadata_list = get_cryptopunks_metadata()
        # Convert list to dict by punk ID
        metadata = {punk['id']: punk for punk in metadata_list}
    except Exception as e:
        print(f"Warning: Could not fetch metadata: {e}")
        print("Creating dummy metadata structure...")
        metadata = {}
    
    # 3. Build attribute mappings
    type_to_idx, accessory_to_idx = build_attribute_mappings(list(metadata.values()) if metadata else [])
    
    # 4. Create structured dataset
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Copy images and build labels
    labels = {}
    
    print("Processing images and building labels...")
    image_files = list(images_source.glob("*.png"))
    
    for img_file in tqdm(image_files):
        # Extract punk ID from filename (e.g., "0.png" -> 0)
        try:
            punk_id = int(img_file.stem)
        except ValueError:
            continue
        
        # Copy image
        dest_path = images_dir / img_file.name
        if not dest_path.exists():
            shutil.copy(img_file, dest_path)
        
        # Get attributes
        if punk_id in metadata:
            punk_data = metadata[punk_id]
            punk_type = punk_data['type']
            accessories = punk_data['accessories']
        else:
            punk_type = "Human"
            accessories = []
        
        # Create label vector
        type_idx = type_to_idx.get(punk_type, 0)
        accessory_vector = [0] * len(accessory_to_idx)
        for acc in accessories:
            if acc in accessory_to_idx:
                accessory_vector[accessory_to_idx[acc]] = 1
        
        labels[punk_id] = {
            "type": punk_type,
            "type_idx": type_idx,
            "accessories": accessories,
            "accessory_vector": accessory_vector
        }
    
    # 5. Save metadata
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            "type_to_idx": type_to_idx,
            "idx_to_type": {v: k for k, v in type_to_idx.items()},
            "accessory_to_idx": accessory_to_idx,
            "idx_to_accessory": {v: k for k, v in accessory_to_idx.items()},
            "num_types": len(type_to_idx),
            "num_accessories": len(accessory_to_idx),
            "labels": {str(k): v for k, v in labels.items()}
        }, f, indent=2)
    
    print(f"\nDataset prepared at: {output_path}")
    print(f"  - Images: {len(list(images_dir.glob('*.png')))} files")
    print(f"  - Metadata: {metadata_file}")
    print(f"  - Types: {len(type_to_idx)}")
    print(f"  - Accessories: {len(accessory_to_idx)}")
    
    return output_path


if __name__ == "__main__":
    download_and_prepare_cryptopunks()
