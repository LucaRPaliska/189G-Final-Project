# prepare_pope_data.py - Fixed version
import json
from datasets import load_dataset
from PIL import Image
import os

os.makedirs("data/images", exist_ok=True)

print("Loading POPE dataset from lmms-lab/POPE...")
# Use 'test' split instead of 'val'
dataset = load_dataset("lmms-lab/POPE", split="test")

print(f"Loaded {len(dataset)} samples")

pope_data = []
for idx, item in enumerate(dataset):
    # Save image
    image_path = f"data/images/image_{idx}.jpg"
    
    # Handle image - could be PIL Image or path
    if hasattr(item['image'], 'save'):
        item['image'].save(image_path)
    else:
        img = Image.open(item['image'])
        img.save(image_path)
    
    # Get question and answer
    question = item['question']
    answer = item['answer'].lower() if 'answer' in item else item['label'].lower()
    
    pope_data.append({
        "image": f"image_{idx}.jpg",
        "text": question,
        "label": answer
    })
    
    if (idx + 1) % 20 == 0:
        print(f"Processed {idx + 1}/{len(dataset)}")

# Save JSON
with open("data/pope.json", "w") as f:
    json.dump(pope_data, f, indent=2)

print(f"\nSaved {len(pope_data)} samples to data/pope.json")
print(f"Images saved to data/images/")

# Print first 3 samples
print("\nFirst 3 samples:")
for i in range(min(3, len(pope_data))):
    print(f"  Q: {pope_data[i]['text']}")
    print(f"  A: {pope_data[i]['label']}\n")