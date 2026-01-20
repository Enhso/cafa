"""
ESM-2 Protein Embedding Extractor

This script extracts protein embeddings using the ESM-2 language model from
Facebook/Meta AI. It processes sequences from a FASTA file and saves the
embeddings as PyTorch tensors for later use in downstream tasks.

ESM-2 (Evolutionary Scale Modeling) is a protein language model trained on
millions of protein sequences. The embeddings capture evolutionary and
structural information useful for function prediction.

Model: facebook/esm2_t36_3B_UR50D
- 36 transformer layers
- 3 billion parameters  
- Trained on UniRef50 database
- Output dimension: 2560 per residue

Usage:
    python extract_embeddings.py --fasta data/raw/Train/train_sequences.fasta \
                                  --output data/embeddings/train \
                                  --batch_size 1
"""

import os
import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
from tqdm import tqdm


# ESM-2 has a maximum context length of 1024 tokens (including special tokens)
# [CLS] + sequence + [EOS] = 1024, so max sequence length is 1022
MAX_SEQUENCE_LENGTH = 1022

# Model configuration
MODEL_NAME = "facebook/esm2_t36_3B_UR50D"
EMBEDDING_DIM = 2560  # Output dimension for esm2_t36_3B


def load_model(
    model_name: str = MODEL_NAME,
    device: Optional[str] = None
) -> tuple:
    """
    Load the ESM-2 model and tokenizer.
    
    Args:
        model_name: HuggingFace model identifier.
        device: Device to load model on ('cuda', 'cpu', or None for auto).
    
    Returns:
        Tuple of (model, tokenizer, device).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params / 1e9:.2f}B parameters")
    
    return model, tokenizer, device


def process_sequence(
    sequence: str,
    model: torch.nn.Module,
    tokenizer,
    device: str
) -> torch.Tensor:
    """
    Process a single protein sequence to get its embedding.
    
    Steps:
    1. Truncate sequence to MAX_SEQUENCE_LENGTH (1022 residues)
    2. Tokenize the sequence (adds [CLS] and [EOS] tokens)
    3. Run through model in no_grad mode (inference only)
    4. Extract hidden states from last layer
    5. Remove [CLS] (position 0) and [EOS] (last position) tokens
    6. Apply MEAN pooling over residue positions
    
    The mean pooling aggregates per-residue embeddings into a single
    fixed-size vector representing the entire protein.
    
    Args:
        sequence: Amino acid sequence string.
        model: The ESM-2 model.
        tokenizer: The ESM-2 tokenizer.
        device: Device string ('cuda' or 'cpu').
    
    Returns:
        Tensor of shape (embedding_dim,) - the protein embedding.
    """
    # Step 1: Truncate sequence if too long
    # ESM-2 max length is 1024 tokens including [CLS] and [EOS]
    if len(sequence) > MAX_SEQUENCE_LENGTH:
        sequence = sequence[:MAX_SEQUENCE_LENGTH]
    
    # Step 2: Tokenize
    # The tokenizer automatically adds [CLS] at start and [EOS] at end
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH + 2  # +2 for special tokens
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Step 3: Run model in no_grad mode (no gradient computation needed)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Step 4: Get the last hidden state
    # Shape: (batch_size=1, seq_len, hidden_dim)
    last_hidden_state = outputs.last_hidden_state
    
    # Step 5: Remove [CLS] (first) and [EOS] (last) tokens
    # We only want embeddings for actual amino acid residues
    # Shape after: (1, seq_len-2, hidden_dim)
    sequence_embeddings = last_hidden_state[:, 1:-1, :]
    
    # Step 6: Mean pooling over the sequence dimension
    # This aggregates all residue embeddings into one protein embedding
    # Shape: (1, hidden_dim) -> (hidden_dim,)
    protein_embedding = sequence_embeddings.mean(dim=1).squeeze(0)
    
    return protein_embedding.cpu()


def process_fasta(
    fasta_path: str,
    output_dir: str,
    model: torch.nn.Module,
    tokenizer,
    device: str,
    batch_size: int = 1
) -> dict:
    """
    Process all sequences in a FASTA file and save embeddings.
    
    Features:
    - Resume capability: skips sequences with existing embedding files
    - Progress bar with tqdm
    - Saves each embedding as a separate .pt file
    
    Args:
        fasta_path: Path to input FASTA file.
        output_dir: Directory to save embedding .pt files.
        model: The ESM-2 model.
        tokenizer: The ESM-2 tokenizer.
        device: Device string.
        batch_size: Not used currently (processes one at a time for memory).
    
    Returns:
        Dictionary with processing statistics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Count sequences for progress bar
    print(f"Counting sequences in {fasta_path}...")
    total_sequences = sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))
    print(f"Found {total_sequences} sequences")
    
    stats = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "total": total_sequences
    }
    
    # Process each sequence
    with tqdm(total=total_sequences, desc="Extracting embeddings") as pbar:
        for record in SeqIO.parse(fasta_path, "fasta"):
            protein_id = record.id
            sequence = str(record.seq)
            
            # Output file path
            embedding_file = output_path / f"{protein_id}.pt"
            
            # Resume capability: skip if file already exists
            if embedding_file.exists():
                stats["skipped"] += 1
                pbar.update(1)
                pbar.set_postfix(
                    processed=stats["processed"],
                    skipped=stats["skipped"],
                    errors=stats["errors"]
                )
                continue
            
            try:
                # Process the sequence
                embedding = process_sequence(sequence, model, tokenizer, device)
                
                # Save the embedding
                torch.save(embedding, embedding_file)
                stats["processed"] += 1
                
            except Exception as e:
                print(f"\nError processing {protein_id}: {e}")
                stats["errors"] += 1
            
            pbar.update(1)
            pbar.set_postfix(
                processed=stats["processed"],
                skipped=stats["skipped"],
                errors=stats["errors"]
            )
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract ESM-2 protein embeddings from FASTA file"
    )
    parser.add_argument(
        "--fasta",
        type=str,
        required=True,
        help="Path to input FASTA file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for embedding files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"HuggingFace model name (default: {MODEL_NAME})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.fasta):
        raise FileNotFoundError(f"FASTA file not found: {args.fasta}")
    
    # Load model
    model, tokenizer, device = load_model(args.model, args.device)
    
    # Process FASTA file
    print(f"\nProcessing: {args.fasta}")
    print(f"Output directory: {args.output}")
    
    stats = process_fasta(
        fasta_path=args.fasta,
        output_dir=args.output,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("Processing Complete!")
    print("=" * 50)
    print(f"Total sequences:     {stats['total']}")
    print(f"Newly processed:     {stats['processed']}")
    print(f"Skipped (existing):  {stats['skipped']}")
    print(f"Errors:              {stats['errors']}")
    print(f"Embeddings saved to: {args.output}")


if __name__ == "__main__":
    main()
