import json
import os
from typing import Dict, List, Tuple

def save_bpe_results(
    vocab: Dict[int, bytes],
    merges: List[Tuple[bytes, bytes]],
    output_dir: str,
    vocab_filename: str = "vocab.json",
    merges_filename: str = "merges.txt"
):
    os.makedirs(output_dir, exist_ok=True)

    vocab_serializable = {}
    for token_id, token_bytes in vocab.items():
        try:
            decoded = token_bytes.decode('utf-8')
            vocab_serializable[str(token_id)] = decoded
        except UnicodeDecodeError:
            decoded = token_bytes.decode('latin-1')
            vocab_serializable[str(token_id)] = decoded
    
    vocab_path = os.path.join(output_dir, vocab_filename)
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)
    

    print(f"Vocabulary saved to {vocab_path}")
    print(f"Vocabulary size: {len(vocab)}")
    
    merges_path = os.path.join(output_dir, merges_filename)
    with open(merges_path, 'w', encoding='utf-8') as f:
        for pair in merges:
            try:
                token1 = pair[0].decode('utf-8')
            except UnicodeDecodeError:
                token1 = pair[0].decode('latin-1')
            
            try:
                token2 = pair[1].decode('utf-8')
            except UnicodeDecodeError:
                token2 = pair[1].decode('latin-1')
            f.write(f"{token1} {token2}\n")
    
    print(f"Merges saved to {merges_path}")
    print(f"Number of merges: {len(merges)}")


def usage():
    from cs336_basics.bpe import bpe

    input_path = "/mnt/e/bpe_data/lfs-data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 10000

    print("Training BPE tokenizer...")
    vocab, merges = bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    output_dir = "./cs336_basics/my_tokenizer"
    save_bpe_results(vocab, merges, output_dir)

def main():
    usage()

if __name__ == '__main__':
    main()

