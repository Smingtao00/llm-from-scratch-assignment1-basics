import os
import multiprocessing as mp
from collections import Counter, defaultdict
from typing import BinaryIO
import regex as re

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
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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
    for word_bytes, cnt in pre_tokens.items():
        for pair in zip(word_bytes[:-1], word_bytes[1:]):
            pair_cnt[pair] += cnt
    return pair_cnt


def get_max_pair(pair_cnt: dict[tuple[bytes], int]) -> tuple[bytes]:
    max_pair, _ = max(pair_cnt.items(), key = lambda x: (x[1], x[0]))
    return max_pair


def update_cnt(word_cnt, pair_cnt, merge_pair):
    new_word_cnt = defaultdict(int)
    new_pair_cnt = defaultdict(int, pair_cnt)

    for word_bytes, cnt in word_cnt.items():

        old_pairs = list(zip(word_bytes[:-1], word_bytes[1:]))
        if merge_pair not in old_pairs:
            new_word_cnt[word_bytes] += cnt
            continue

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
    
    pair_cnt = get_pairs(pre_tokens)
    
    for i in range(next_id, vocab_size):
        max_pair = get_max_pair(pair_cnt)
        vocab[i] = max_pair[0] + max_pair[1]
        merges.append(max_pair)
        pre_tokens, pair_cnt = update_cnt(pre_tokens, pair_cnt, max_pair)
        
    return vocab, merges





    