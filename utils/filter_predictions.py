#!/usr/bin/env python3
"""
Filter low-probability predictions from CAFA 6 submission file.
This script processes large TSV files efficiently by streaming through them.
"""

import argparse
import sys
from pathlib import Path


def estimate_file_stats(input_file, sample_size=100000):
    """
    Sample the file to estimate probability distribution.
    """
    print(f"Sampling {sample_size} lines to analyze probability distribution...")
    probabilities = []
    
    with open(input_file, 'r') as f:
        # Skip header if present
        first_line = f.readline()
        if not first_line.strip().replace('.', '').replace('\t', '').replace('_', '').replace('-', '').isdigit():
            # Likely a header
            has_header = True
        else:
            has_header = False
            probabilities.append(float(first_line.strip().split('\t')[2]))
        
        # Sample lines
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                try:
                    probabilities.append(float(parts[2]))
                except ValueError:
                    continue
    
    probabilities.sort()
    n = len(probabilities)
    
    print(f"\nProbability distribution (from {n} samples):")
    print(f"  Min: {probabilities[0]:.6f}")
    print(f"  10th percentile: {probabilities[int(n*0.1)]:.6f}")
    print(f"  25th percentile: {probabilities[int(n*0.25)]:.6f}")
    print(f"  Median: {probabilities[int(n*0.5)]:.6f}")
    print(f"  75th percentile: {probabilities[int(n*0.75)]:.6f}")
    print(f"  90th percentile: {probabilities[int(n*0.9)]:.6f}")
    print(f"  95th percentile: {probabilities[int(n*0.95)]:.6f}")
    print(f"  99th percentile: {probabilities[int(n*0.99)]:.6f}")
    print(f"  Max: {probabilities[-1]:.6f}")
    
    return has_header


def filter_predictions(input_file, output_file, threshold=None, top_k_per_protein=None, 
                       target_size_gb=None, analyze_only=False):
    """
    Filter predictions based on threshold or top-k per protein.
    
    Args:
        input_file: Path to input TSV file
        output_file: Path to output TSV file
        threshold: Minimum probability threshold
        top_k_per_protein: Keep only top K predictions per protein
        target_size_gb: Target output file size in GB (auto-calculate threshold)
        analyze_only: Only analyze the file, don't filter
    """
    
    # First, analyze the file
    has_header = estimate_file_stats(input_file)
    
    if analyze_only:
        return
    
    # Auto-calculate threshold if target size is specified
    if target_size_gb and not threshold:
        input_size = Path(input_file).stat().st_size / (1024**3)  # GB
        reduction_ratio = target_size_gb / input_size
        print(f"\nTarget size: {target_size_gb} GB (reduction ratio: {reduction_ratio:.2%})")
        
        # Estimate threshold based on percentile
        percentile = max(0, min(100, (1 - reduction_ratio) * 100))
        print(f"Estimated percentile to keep: {100-percentile:.1f}%")
        print("Calculating exact threshold...")
        
        # Sample more data to find the right threshold
        probabilities = []
        with open(input_file, 'r') as f:
            if has_header:
                next(f)
            for i, line in enumerate(f):
                if i >= 1000000:  # Sample 1M lines
                    break
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    try:
                        probabilities.append(float(parts[2]))
                    except ValueError:
                        continue
        
        probabilities.sort()
        threshold_idx = int(len(probabilities) * (percentile / 100))
        threshold = probabilities[threshold_idx]
        print(f"Auto-calculated threshold: {threshold:.6f}")
    
    if not threshold and not top_k_per_protein:
        print("\nError: Must specify either --threshold, --top-k, or --target-size")
        sys.exit(1)
    
    # Process the file
    print(f"\nFiltering predictions...")
    if threshold:
        print(f"  Threshold: {threshold}")
    if top_k_per_protein:
        print(f"  Top-K per protein: {top_k_per_protein}")
    
    lines_read = 0
    lines_written = 0
    current_protein = None
    protein_predictions = []
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        # Handle header
        if has_header:
            header = fin.readline()
            fout.write(header)
        
        for line in fin:
            lines_read += 1
            if lines_read % 1000000 == 0:
                print(f"  Processed {lines_read:,} lines, kept {lines_written:,} ({lines_written/lines_read*100:.1f}%)")
            
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            
            protein_id, go_term, prob_str = parts[0], parts[1], parts[2]
            
            try:
                probability = float(prob_str)
            except ValueError:
                continue
            
            # Simple threshold filtering
            if threshold and not top_k_per_protein:
                if probability >= threshold:
                    fout.write(line)
                    lines_written += 1
            
            # Top-K per protein filtering
            elif top_k_per_protein:
                if protein_id != current_protein:
                    # Write out previous protein's top predictions
                    if protein_predictions:
                        protein_predictions.sort(key=lambda x: x[1], reverse=True)
                        for pred_line, _ in protein_predictions[:top_k_per_protein]:
                            fout.write(pred_line)
                            lines_written += 1
                    
                    current_protein = protein_id
                    protein_predictions = []
                
                # Apply threshold if specified along with top-k
                if not threshold or probability >= threshold:
                    protein_predictions.append((line, probability))
        
        # Don't forget last protein's predictions
        if top_k_per_protein and protein_predictions:
            protein_predictions.sort(key=lambda x: x[1], reverse=True)
            for pred_line, _ in protein_predictions[:top_k_per_protein]:
                fout.write(pred_line)
                lines_written += 1
    
    output_size = Path(output_file).stat().st_size / (1024**3)  # GB
    input_size = Path(input_file).stat().st_size / (1024**3)  # GB
    
    print(f"\nFiltering complete!")
    print(f"  Lines read: {lines_read:,}")
    print(f"  Lines written: {lines_written:,} ({lines_written/lines_read*100:.1f}%)")
    print(f"  Input size: {input_size:.2f} GB")
    print(f"  Output size: {output_size:.2f} GB")
    print(f"  Size reduction: {(1-output_size/input_size)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Filter low-probability predictions from CAFA 6 submission file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze file distribution
  python filter_predictions.py input.tsv --analyze
  
  # Filter with probability threshold
  python filter_predictions.py input.tsv output.tsv --threshold 0.01
  
  # Keep top 100 predictions per protein
  python filter_predictions.py input.tsv output.tsv --top-k 100
  
  # Auto-calculate threshold to reach target file size
  python filter_predictions.py input.tsv output.tsv --target-size 2.0
  
  # Combine: top 50 per protein with minimum threshold
  python filter_predictions.py input.tsv output.tsv --top-k 50 --threshold 0.005
        """
    )
    
    parser.add_argument('input_file', help='Input TSV file with predictions')
    parser.add_argument('output_file', nargs='?', help='Output TSV file (filtered)')
    parser.add_argument('--threshold', type=float, help='Minimum probability threshold')
    parser.add_argument('--top-k', type=int, help='Keep only top K predictions per protein')
    parser.add_argument('--target-size', type=float, help='Target output file size in GB')
    parser.add_argument('--analyze', action='store_true', help='Only analyze distribution, do not filter')
    
    args = parser.parse_args()
    
    if not args.analyze and not args.output_file:
        parser.error("output_file is required unless --analyze is specified")
    
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    filter_predictions(
        args.input_file,
        args.output_file,
        threshold=args.threshold,
        top_k_per_protein=args.top_k,
        target_size_gb=args.target_size,
        analyze_only=args.analyze
    )


if __name__ == '__main__':
    main()
