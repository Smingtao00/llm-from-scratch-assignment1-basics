import os
import heapq
import multiprocessing as mp
from collections import Counter, defaultdict
from typing import BinaryIO
import regex as re
import json
from typing import Iterator, Iterable

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def word2bytes(word):
    return tuple(bytes([i]) for i in list(word.encode("utf-8")))


def merge_results(results: list[dict[tuple[bytes], int]]) -> dict[tuple[bytes], int]:
    total_counts = Counter()

    for result in results:
        total_counts.update(result)

    return dict(total_counts)


def process_chunk(
    file_path: str,
    start: int,
    end: int,
    chunk_id: int,
    special_tokens: list[str],
) -> dict[tuple[bytes], int]:
    token_counts = defaultdict(int)

    with open(file_path, 'rb') as f:
        f.seek(start)
        text = f.read(end - start).decode('utf-8')
        
    division = "|".join(re.escape(tok) for tok in special_tokens)
    slices = re.split(division, text)
    

    for each in slices:
        if each:
            for match in re.finditer(PAT, each):
                word = match.group()
                word_bytes = word2bytes(word)
                if len(word_bytes) >= 2:
                    token_counts[word_bytes] += 1

    return token_counts


def get_pairs(pre_tokens: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
    pair_cnt = defaultdict(int)
    pair2word = defaultdict(set) 
    for word_bytes, cnt in pre_tokens.items():
        for i in range(len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i + 1])
            pair_cnt[pair] += cnt
            pair2word[pair].add(word_bytes)
    return pair_cnt, pair2word


def get_max_pair(pair_cnt: dict[tuple[bytes], int]) -> tuple[bytes]:
    max_pair, _ = max(pair_cnt.items(), key = lambda x: (x[1], x[0]))
    return max_pair


def update_cnt(word_cnt, pair_cnt, pair2word, merge_pair):
    new_word_cnt = defaultdict(int, word_cnt)
    new_pair_cnt = defaultdict(int, pair_cnt)

    for word_bytes in list(pair2word[merge_pair]):
        cnt = word_cnt[word_bytes]
        del new_word_cnt[word_bytes]
        for i in range(len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i + 1])
            try: 
                pair2word[pair].remove(word_bytes)
            except:
                pass

        merged = merge_pair[0] + merge_pair[1]
        new_word_bytes = []
        i = 0
        while i < len(word_bytes):
            if i < len(word_bytes) - 1 and word_bytes[i] == merge_pair[0] and word_bytes[i + 1] == merge_pair[1]:
                new_word_bytes.append(merged)
                if i > 0:
                    new_pair_cnt[tuple([word_bytes[i - 1], merged])] += cnt
                    pair = tuple([word_bytes[i - 1], word_bytes[i]])
                    new_pair_cnt[pair] -= cnt
                    if new_pair_cnt[pair] == 0:
                        del new_pair_cnt[pair]
                if i < len(word_bytes) - 2:
                    new_pair_cnt[tuple([merged, word_bytes[i + 2]])] += cnt
                    pair = tuple([word_bytes[i + 1], word_bytes[i + 2]])
                    new_pair_cnt[pair] -= cnt
                    if new_pair_cnt[pair] == 0:
                        del new_pair_cnt[pair]
                pair = tuple([word_bytes[i], word_bytes[i + 1]])
                new_pair_cnt[pair] -= cnt
                if new_pair_cnt[pair] == 0:
                    del new_pair_cnt[pair]
                i += 2
            else:
                new_word_bytes.append(word_bytes[i])
                i += 1
        
        for i in range(len(new_word_bytes) - 1):
            pair2word[(new_word_bytes[i], new_word_bytes[i + 1])].add(tuple(new_word_bytes))

        new_word_cnt[tuple(new_word_bytes)] += cnt
    return new_word_cnt, new_pair_cnt

def bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i : bytes([i]) for i in range(256)}
    merges = []

    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # 如果已经设置过启动模式，会抛出 RuntimeError，可以忽略
        pass

    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))

        with mp.Pool(processes = num_processes) as pool:
            tasks = []
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                tasks.append((input_path, start, end, i, special_tokens))

            results = pool.starmap(process_chunk, tasks)
        
        pre_tokens = merge_results(results)
        #print(pre_tokens)
    
    pair_cnt, pair2word = get_pairs(pre_tokens)
    
    for i in range(next_id, vocab_size):
        max_pair = get_max_pair(pair_cnt)
        vocab[i] = max_pair[0] + max_pair[1]
        merges.append(max_pair)
        pre_tokens, pair_cnt = update_cnt(pre_tokens, pair_cnt, pair2word, max_pair)
        
    return vocab, merges


def split_by_special(text, special_tokens):
    if not special_tokens:
        return [text]

    special_tokens = sorted(special_tokens, key=len, reverse=True)

    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    pattern = f"({pattern})"

    pattern = re.compile(pattern)
    chunks = pattern.split(text)
    return [c for c in chunks if c]


def apply_merges(word_bytes, merges, vocab2id):
    word_bytes = list(word_bytes)

    while True:
        min_token_id = float('inf')
        best_pair_idx = -1
        merged = None

        for i in range(len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i + 1])
            if pair in merges:
                combined = pair[0] + pair[1]
                token_id = vocab2id.get(combined)
                if token_id is not None and token_id < min_token_id:
                    min_token_id = token_id
                    best_pair_idx = i
                    merged = combined
        
        if best_pair_idx == -1:
            break
            
        word_bytes = word_bytes[:best_pair_idx] + [merged] + word_bytes[best_pair_idx + 2:]
    return tuple(word_bytes)


def encode_merged(text, merges, vocab2id):
    word_list = re.finditer(PAT, text)
    tokens = []
    for match in word_list:
        word_bytes = word2bytes(match.group())
        merged_word_bytes = apply_merges(word_bytes, merges, vocab2id)
        tokens.extend(vocab2id[i] for i in merged_word_bytes)
    return tokens


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = set(merges)
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_bytes = [i.encode('utf-8') for i in self.special_tokens]
        self.vocab2id = {v:k for k, v in vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r', encoding = 'utf-8') as vf:
            vocab_data = json.load(vf)
            vocab = {int(k): bytes(v, 'latin1') if isinstance(v, str) else bytes(v) for k, v in vocab_data.items()}

        # 假设都是 ("a b")
        with open(merges_filepath, 'r', encoding = 'utf-8') as mf:
            lines = mf.readlines()
            merge_pairs = [tuple(line.strip().split()) for line in lines if not line.startswith('#') and line.strip()]
            merges = [(a.encode('utf-8'), b.encode('utf-8')) for a, b in merge_pairs]

        return cls(vocab = vocab, merges = merges, special_tokens = special_tokens)

    def encode(self, text: str) -> list[int]:
        chunks = split_by_special(text, self.special_tokens)
        tokens = []
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(self.vocab2id[chunk.encode('utf-8')])
            else:
                tokens.extend(encode_merged(chunk, self.merges, self.vocab2id))
        return tokens
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        return b''.join([self.vocab[t] for t in ids]).decode('utf-8', errors = 'replace')
    
