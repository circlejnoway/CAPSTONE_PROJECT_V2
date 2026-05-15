"""
Generate character dictionary from training data
"""
import os
from pathlib import Path
from collections import OrderedDict

def generate_char_dict(label_file, output_path):
    """Extract all unique characters from training labels"""
    characters = OrderedDict()
    characters[''] = 0  # Reserve 0 for blank

    char_index = 1

    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                text = parts[1]
                for char in text:
                    if char not in characters:
                        characters[char] = char_index
                        char_index += 1

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for char, idx in characters.items():
            f.write(char + '\n')

    print(f"Generated character dictionary with {len(characters)} characters")
    print(f"Saved to: {output_path}")

    # Print sample
    print("\nFirst 20 characters:")
    for i, (char, idx) in enumerate(list(characters.items())[:20]):
        print(f"  {idx}: '{char}' (ascii: {ord(char) if char else 'BLANK'})")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent

    # Path to training labels
    train_label = project_root / "models/receipt_ocr/paddleocr_receipt/data/train_gt.txt"

    # Output path for character dictionary
    output_dict = project_root / "models/receipt_ocr/paddleocr_receipt/data/ppocr_keys_v1.txt"

    if not train_label.exists():
        print(f"Error: Training label file not found: {train_label}")
        exit(1)

    generate_char_dict(str(train_label), str(output_dict))
