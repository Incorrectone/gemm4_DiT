# Flow Matching Diffusion Transformer (DiT)

A custom, highly optimized 33M parameter Diffusion Transformer trained from scratch using Optimal Transport Conditional Flow Matching (OT-CFM). This project implements state-of-the-art generative modeling techniques, completely bypassing disk I/O bottlenecks to train efficiently on a single consumer GPU (RTX 5060 Ti).

## Credits & Acknowledgements
- **Text Encoder:** [Google Gemma 4 (e4b) on Kaggle](https://www.kaggle.com/models/google/gemma-4)
- **Dataset:** [COCO 2017 Dataset by awsaf49](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)
- **Architecture Inspiration:** [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) by William Peebles and Saining Xie.

## Architecture & Features

- **Model:** DiT-Small (12 layers, 6 heads, 384 hidden dim, 33M parameters)
- **Text Encoder:** Gemma 4 (e4b) Layer 40 Embeddings (3584 dimensions)
- **Visual Latents:** COCO 2017 Dataset (4x32x32 VAE compressed latents)
- **Formulation:** Optimal Transport Conditional Flow Matching (Straight-line velocity prediction v = x_1 - x_0)
- **Conditioning:** Adaptive Layer Normalization (adaLN-Zero)

### Training Optimizations
- **Logit-Normal Timestep Sampling:** Heavily biases training toward the critical midpoint (t ≈ 0.5) for superior final FID scores.
- **Zero-Disk Bottleneck:** A custom dataloader dynamically caches pre-pooled float16 shards entirely into system RAM (~5GB footprint), keeping GPU utilization at 100%.
- **Regularization:** 10% Classifier-Free Guidance (CFG) text dropout for prompt adherence, coupled with nn.Dropout(0.1) in the semantic projection layers.
- **Stability:** Cosine Learning Rate Schedule with a 5% linear warmup, Gradient Clipping, and native bfloat16 precision.
- **EMA Checkpointing:** Maintains an Exponential Moving Average (EMA) shadow model (0.999 decay) for ultra-smooth inference generation.

## Project Structure

* `DiT_Train.ipynb`: The core training loop. Handles architecture definitions, RAM data loading, validation, and auto-resuming from checkpoints.
* `VAE_creator.ipynb`: Pipeline to compress raw RGB images into lightweight visual latents offline.
* `Extract_Gemma4_40.ipynb`: Distributed extraction of massive semantic text embeddings from Gemma 4 e4b.
* `Shrink_Embeddings.py`: Pre-processing utility to sequence-pool the massive 37GB raw text embeddings down to a 3GB RAM-friendly float16 footprint.
* `checkpoints/`: Auto-generated directory containing epoch states and the final inference .pt model.

## Inference

Inference is designed for a Dual-GPU setup (e.g., Kaggle Dual T4s). The heavy `gemma-4-e4b` model is hosted on `cuda:0`, while the lightweight DiT and VAE sit on `cuda:1`.

A standalone Python inference script handles the Layer 40 embedding extraction, Classifier-Free Guidance extrapolation, and a 50-step Euler ODE solver generation.

## Getting Started

1. Ensure the COCO VAE latents are located in `VAE_latents/` and the shrunk Gemma embeddings are in `gemma4_40_pooled/`.
2. Open `DiT_Train.ipynb`.
3. Run all cells. The script will automatically load the shards into RAM, initialize the models, and begin the OT-CFM training loop with auto-saving and evaluation every 5 epochs.


## Results (In Progress)

The model is currently undergoing a 40-epoch weekend training run. Final generated images and evaluation metrics will be posted here upon completion. 

## Roadmap & Future Plans

- **[In Progress] Baseline Training:** Completing the 40-epoch weekend run of the 33M parameter DiT-Small on the COCO dataset.
- **Dual-GPU Inference:** Deploying the final `ema_model` weights to a Kaggle dual-T4 environment to run the 50-step Euler ODE solver and generate the first batch of custom text-to-image generations.
- **High-LR Warm Start:** If the network plateaus but hasn't overfit by Epoch 40, we plan to warm-start a second training run from the final checkpoint using an aggressive learning rate (`3e-4` to `4e-4`) to break through into a deeper local minimum. 
- **Architecture Scaling (DiT-Base):** While the 33M parameter model should have more than enough mathematical capacity for a 118k image dataset, if semantic text adherence plateaus, the architecture is ready to seamlessly scale up to a **DiT-Base (130M parameters)** (hidden size 768, 24 layers).
- **Dataset Expansion:** Depending on baseline visual results, future iterations may integrate JourneyDB or synthetic captions to push the DiT to its theoretical limits.

## Disclaimer

*Portions of this project's architecture, data pipeline, and codebase were developed and optimized with the assistance of an AI coding agent.*
