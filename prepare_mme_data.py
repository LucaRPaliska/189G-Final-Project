# prepare_mme_data.py
import json
import os
import zipfile
import requests
from PIL import Image
from tqdm import tqdm
from io import BytesIO

# ============================================================
# CONFIG
# ============================================================

MME_DATA_URL = "https://github.com/BradyFU/MME/raw/main/data/MME_Benchmark_release_version.zip"
MME_DIR = "mme_raw_data"
OUTPUT_JSON = "data/mme.json"
IMAGE_DIR = "data/images"

# Hallucination-specific subsets from your proposal
HALLUCINATION_SUBSETS = [
    "existence",      # Does the object exist?
    "count",          # Count of objects
    "position",       # Object positions
    "color"           # Object colors
]

# ============================================================
# DOWNLOAD AND EXTRACT MME DATA
# ============================================================

def download_mme():
    """Download MME benchmark zip file"""
    os.makedirs(MME_DIR, exist_ok=True)
    
    zip_path = os.path.join(MME_DIR, "MME_Benchmark_release_version.zip")
    
    if not os.path.exists(zip_path):
        print(f"Downloading MME data from {MME_DATA_URL}...")
        response = requests.get(MME_DATA_URL, stream=True)
        
        with open(zip_path, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                f.write(chunk)
        print("Download complete.")
    else:
        print("MME zip already exists.")
    
    # Extract
    extract_dir = os.path.join(MME_DIR, "MME_Benchmark_release_version")
    if not os.path.exists(extract_dir):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(MME_DIR)
        print("Extraction complete.")
    
    return extract_dir

# ============================================================
# LOAD MME SUBSETS
# ============================================================

def load_mme_subsets(data_dir):
    """Load MME hallucination-specific subsets"""
    all_samples = []
    
    for subset in HALLUCINATION_SUBSETS:
        json_path = os.path.join(data_dir, f"{subset}.json")
        
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, skipping {subset}")
            continue
        
        print(f"Loading {subset} subset...")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            # MME format:
            # {
            #   "image_path": "path/to/image.jpg",
            #   "question": "Is there a car in the image?",
            #   "answer": "Yes" or "No"
            # }
            
            # Convert to POPE-compatible format
            sample = {
                "image": item["image_path"],  # Will handle path separately
                "text": item["question"],
                "label": item["answer"].strip().lower(),  # "yes" or "no"
                "subset": subset  # Track which subset for analysis
            }
            all_samples.append(sample)
    
    print(f"Loaded {len(all_samples)} total samples from MME")
    return all_samples

# ============================================================
# COPY IMAGES TO TARGET DIRECTORY
# ============================================================

def copy_mme_images(mme_data_dir, target_image_dir, samples):
    """Copy MME images to unified image directory"""
    os.makedirs(target_image_dir, exist_ok=True)
    
    # MME images are in subdirectories like:
    # MME_Benchmark_release_version/images/existence/xxx.jpg
    
    for idx, sample in enumerate(tqdm(samples, desc="Copying images")):
        original_path = os.path.join(mme_data_dir, "images", sample["subset"], sample["image"])
        
        # Handle different possible path structures
        if not os.path.exists(original_path):
            # Try alternative path
            original_path = os.path.join(mme_data_dir, sample["image"])
        
        if os.path.exists(original_path):
            # Create new filename
            new_filename = f"mme_{sample['subset']}_{idx}_{os.path.basename(sample['image'])}"
            new_path = os.path.join(target_image_dir, new_filename)
            
            # Copy image
            img = Image.open(original_path)
            img.save(new_path)
            
            # Update sample with new image path
            sample["image"] = new_filename
        else:
            print(f"Warning: Image not found: {original_path}")
    
    return samples

# ============================================================
# SAVE IN POPE FORMAT
# ============================================================

def save_as_pope_format(samples, output_json):
    """Save samples in same format as POPE JSON"""
    # Remove subset field if you don't want it in final JSON
    pope_format_samples = []
    for sample in samples:
        pope_format_samples.append({
            "image": sample["image"],
            "text": sample["text"],
            "label": sample["label"]
        })
    
    with open(output_json, "w") as f:
        json.dump(pope_format_samples, f, indent=2)
    
    print(f"Saved {len(pope_format_samples)} samples to {output_json}")
    
    # Also save with subset info for analysis
    subset_json = output_json.replace(".json", "_with_subsets.json")
    with open(subset_json, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved subset info to {subset_json}")

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Preparing MME Hallucination Dataset")
    print("=" * 60)
    
    # Step 1: Download MME
    print("\n[1/4] Downloading MME dataset...")
    mme_dir = download_mme()
    
    # Step 2: Load subsets
    print("\n[2/4] Loading hallucination subsets...")
    samples = load_mme_subsets(mme_dir)
    
    if len(samples) == 0:
        print("Error: No samples loaded. Check MME download.")
        return
    
    # Step 3: Copy images
    print("\n[3/4] Copying images to unified directory...")
    mme_images_dir = os.path.join(mme_dir, "MME_Benchmark_release_version")
    samples = copy_mme_images(mme_images_dir, IMAGE_DIR, samples)
    
    # Step 4: Save in POPE format
    print("\n[4/4] Saving in POPE-compatible format...")
    save_as_pope_format(samples, OUTPUT_JSON)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("MME Dataset Preparation Complete!")
    print("=" * 60)
    print(f"Total samples: {len(samples)}")
    print(f"Images saved to: {IMAGE_DIR}")
    print(f"Annotations saved to: {OUTPUT_JSON}")
    
    # Subset breakdown
    subset_counts = {}
    for sample in samples:
        subset = sample.get("subset", "unknown")
        subset_counts[subset] = subset_counts.get(subset, 0) + 1
    
    print("\nSamples per subset:")
    for subset, count in subset_counts.items():
        print(f"  {subset}: {count}")

# ============================================================
# VERIFICATION FUNCTION
# ============================================================

def verify_mme_data():
    """Quick verification that data is in correct format"""
    try:
        with open(OUTPUT_JSON, "r") as f:
            data = json.load(f)
        
        print("\nVerification:")
        print(f"  ✅ JSON file exists with {len(data)} entries")
        
        sample = data[0]
        required_keys = ["image", "text", "label"]
        for key in required_keys:
            if key in sample:
                print(f"  ✅ '{key}' field present")
            else:
                print(f"  ❌ '{key}' field missing")
        
        image_path = os.path.join(IMAGE_DIR, sample["image"])
        if os.path.exists(image_path):
            print(f"  ✅ First image exists: {sample['image']}")
        else:
            print(f"  ❌ First image missing: {image_path}")
        
        print("\nSample entry:")
        print(json.dumps(sample, indent=2))
        
    except Exception as e:
        print(f"Verification failed: {e}")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
    verify_mme_data()