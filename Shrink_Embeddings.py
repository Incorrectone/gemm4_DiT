import os
import torch
from pathlib import Path
from tqdm import tqdm

def shrink_shards():
        input_dir = Path("./gemma4_40")
    output_dir = Path("./gemma4_40_pooled")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Replicate train/val structure
    (output_dir / "train").mkdir(exist_ok=True)
    (output_dir / "val").mkdir(exist_ok=True)

    shard_files = list(input_dir.rglob("*.pt"))
    if not shard_files:
        print(f"No .pt files found in {input_dir}. Please ensure the shards are downloaded.")
        return

    print(f"Found {len(shard_files)} shards. Shrinking footprint...")

    total_original_size = 0
    total_new_size = 0

    for shard_path in tqdm(shard_files, desc="Pooling Embeddings"):
        total_original_size += shard_path.stat().st_size
        
        # Determine the relative path for saving
        rel_path = shard_path.relative_to(input_dir)
        save_path = output_dir / rel_path
        
        data = torch.load(shard_path, map_location="cpu", weights_only=False)
        
        shrunk_data = {}
        if isinstance(data, dict):
            for img_id, text_tensor in data.items():
                if isinstance(text_tensor, list):
                    pooled_list = []
                    for t in text_tensor:
                        if t.ndim >= 2:
                            t = t.mean(dim=0)
                        pooled_list.append(t.to(torch.float16))
                    shrunk_data[img_id] = torch.stack(pooled_list)
                else:
                    # Handle single tensor captions
                    if text_tensor.ndim == 3: # [5, seq_len, dim]
                        text_tensor = text_tensor.mean(dim=1)
                    elif text_tensor.ndim == 2: # [5, dim] or [seq_len, dim]
                        if text_tensor.shape[0] != 5:
                            text_tensor = text_tensor.mean(dim=0)
                    
                    shrunk_data[img_id] = text_tensor.to(torch.float16)
        
        # Save the optimized, pooled dictionary
        torch.save(shrunk_data, save_path)
        total_new_size += save_path.stat().st_size

    print(f"\nOptimization Complete!")
    print(f"Original Footprint: {total_original_size / 1e9:.2f} GB")
    print(f"New Footprint: {total_new_size / 1e9:.2f} GB")
    print("\nThe pooled embeddings are ready in the 'gemma4_40_pooled' directory.")

if __name__ == "__main__":
    shrink_shards()

