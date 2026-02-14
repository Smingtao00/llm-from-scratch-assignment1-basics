from cs336_basics.bpe import Tokenizer
from cs336_basics.transformer import *
import torch


tokenizer = Tokenizer.from_files(
    vocab_filepath="/home/smingtao01/Download/CS336/llm-from-scratch-assignment1-basics/cs336_basics/my_tokenizer/vocab.json",
    merges_filepath="/home/smingtao01/Download/CS336/llm-from-scratch-assignment1-basics/cs336_basics/my_tokenizer/merges.txt",
    special_tokens=['<|endoftext|>']
)

def tokenize_file(input_file, output_file, tokenizer):
    from numpy import np
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    token_ids = tokenizer.encode(text)

    arr = np.array(token_ids, dtype=np.uint16)
    arr.tofile(output_file)
    print(f"Saved {len(token_ids)} tokens to {output_file}")

#tokenize_file("/mnt/e/bpe_data/lfs-data/TinyStoriesV2-GPT4-valid.txt", "/home/smingtao01/Download/CS336/llm-from-scratch-assignment1-basics/cs336_basics/my_tokenizer/train.bin", tokenizer)

prompt = "Hello world"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = transformer_lm(
    vocab_size=len(tokenizer.vocab),
    context_length=256,
    d_model=384,
    num_layers=6,
    num_heads=6,
    d_ff=1536,
    rope_theta=10000.0
).to(device)

optimizer = AdamW(
    model.parameters(),
    lr=6e-4,
    weight_decay=0.1
)

target_dir = os.path.join(os.getcwd(), "cs336_basics", "out")
load_checkpoint(os.path.join(target_dir, "best_model.pt"), model, optimizer)

generated_text = model.generate(
    tokenizer.encode(prompt),
    max_new_tokens=200,
    eos_id=None,
    temperature=1,
    top_p=1,
    context_length=10000,
    rng=None,
)

print(generated_text)
print(tokenizer.decode(generated_text.cpu().tolist()))