import random
from typing import List, Dict

# --- Configuration Constants ---
SEED = 42
MAX_ATTEMPTS_FACTOR = 200
BLOCK_USAGE_LIMIT_FACTOR = 1/3
POSITION_USAGE_LIMIT_FACTOR = 1/10
EXTRA_USAGE = 5

# Diversity thresholds by progress
THRESHOLD_LOW = 8
THRESHOLD_MID = 5
THRESHOLD_HIGH = 4

# --- Helper Functions ---

def sample_strictly_increasing_valid_bounds() -> List[int]:
    """Samples strictly increasing first_params with valid ranges."""
    a = random.randint(1, 4)
    b = random.randint(a + 1, 8)
    c = random.randint(b + 1, 12)
    d = random.randint(c + 1, 16)
    return [a, b, c, d]

def sample_second_params(blocks: List[int]) -> List[int]:
    """Samples valid second_params based on block types, ensuring non-decreasing for 1s and 7s."""
    result = []
    prev = {1: 0, 7: 0}
    for b in blocks:
        if b == 1:
            choices = [v for v in [1, 2, 4, 8] if v >= prev[1]]
            val = random.choice(choices)
            result.append(val)
            prev[1] = val
        elif b == 7:
            choices = [v for v in [4, 8, 16] if v >= prev[7]]
            val = random.choice(choices)
            result.append(val)
            prev[7] = val
        else:
            result.append(0)
    return result

def create_strict_valid_encoding(blocks: List[int]) -> List[int]:
    """Generates a valid 12-length encoding following all constraints."""
    first_params = sample_strictly_increasing_valid_bounds()
    second_params = sample_second_params(blocks)
    encoding = []
    for i in range(4):
        encoding.append(blocks[i])
    for i in range(4):
        encoding.append(first_params[i])
        encoding.append(second_params[i])
    return encoding

def write_encoding_to_file(encoding: List[int], filename: str = "encodings.txt") -> None:
    with open(filename, "a") as f:
        f.write(str(encoding) + "\n")

def all_same() -> List[List[int]]:
    """Creates 7 encodings where all blocks are the same type (1 through 7)."""
    encs: List[List[int]] = []
    for btype in range(1, 8):
        blocks = [btype] * 4
        first_params = [4, 8, 12, 16]  # fixed valid increasing
        second_params = []
        prev = {1: 0, 7: 0}
        for _ in blocks:
            if btype == 1:
                choices = [v for v in [1, 2, 4, 8] if v >= prev[1]]
                val = max(choices)
                second_params.append(val)
                prev[1] = val
            elif btype == 7:
                choices = [v for v in [4, 8, 16] if v >= prev[7]]
                val = max(choices)
                second_params.append(val)
                prev[7] = val
            else:
                second_params.append(0)

        encoding = []
        for i in range(4):
            encoding.append(btype)
        for i in range(4):
            encoding.append(first_params[i])
            encoding.append(second_params[i])
        encs.append(encoding)
        write_encoding_to_file(encoding)
        print(f"All-same encoding for block {btype}: {encoding}")
    return encs

def generate_encodings_multi_objective(n: int) -> List[List[int]]:
    """Generates n diverse, valid encodings according to all constraints."""
    block_usage = {i: 0 for i in range(1, 8)}
    position_usage = [{i: 0 for i in range(1, 8)} for _ in range(4)]
    generated_seqs = set()
    encodings = []

    block_limit = int(n * BLOCK_USAGE_LIMIT_FACTOR) + EXTRA_USAGE
    pos_limit = int(n * POSITION_USAGE_LIMIT_FACTOR) + EXTRA_USAGE
    max_attempts = n * MAX_ATTEMPTS_FACTOR
    attempts = 0

    while len(encodings) < n and attempts < max_attempts:
        attempts += 1
        blocks = [random.randint(1, 7) for _ in range(4)]
        block_tuple = tuple(blocks)

        if len(set(blocks)) == 1 or block_tuple in generated_seqs:
            continue

        encoding = create_strict_valid_encoding(blocks)

        # Diversity score check
        diversity_score = sum(block_usage[b] < block_limit for b in blocks)
        diversity_score += sum(position_usage[i][b] < pos_limit for i, b in enumerate(blocks))

        progress = len(encodings) / n
        if progress < 0.6:
            required_score = THRESHOLD_LOW
        elif progress < 0.7:
            required_score = THRESHOLD_MID
        else:
            required_score = THRESHOLD_HIGH

        if diversity_score >= required_score or progress > 0.9:
            encodings.append(encoding)
            generated_seqs.add(block_tuple)
            for b in blocks:
                block_usage[b] += 1
            for i, b in enumerate(blocks):
                position_usage[i][b] += 1
            write_encoding_to_file(encoding)
            print(f"? {len(encodings)}/{n}: {encoding}")

    # Final fill if needed
    while len(encodings) < n:
        blocks = [random.randint(1, 7) for _ in range(4)]
        block_tuple = tuple(blocks)
        if len(set(blocks)) == 1 or block_tuple in generated_seqs:
            continue
        encoding = create_strict_valid_encoding(blocks)
        encodings.append(encoding)
        generated_seqs.add(block_tuple)
        write_encoding_to_file(encoding)
        print(f"?? Filled {len(encodings)}/{n}: {encoding}")

    print("\n? Generation complete.")
    print(f"Total attempts: {attempts}, Generated: {len(encodings)}")
    return encodings

def main(n: int = 100) -> None:
    random.seed(SEED)
    open("encodings.txt", "w").close()

    print("Generating all-same encodings...")
    encs = all_same()

    remaining = n - len(encs)
    if remaining > 0:
        print(f"Generating {remaining} diverse encodings...")
        encs.extend(generate_encodings_multi_objective(remaining))

    print("\nFinal Encodings:")
    for i, e in enumerate(encs, 1):
        print(f"{i:3d}: {e}")

if __name__ == "__main__":
    main(100)
