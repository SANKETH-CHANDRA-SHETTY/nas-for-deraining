import random
from typing import List, Dict, Tuple

# --- Configuration Constants ---
SEED = 42
MAX_ATTEMPTS_FACTOR = 200
FIRST_PARAMS_BOUNDS = (1, 4, 8, 12, 16)  # a_min, a_max, b_max, c_max, d_max
BLOCK_USAGE_LIMIT_FACTOR = 1/3  # block_usage_limit = int(n * BLOCK_USAGE_LIMIT_FACTOR) + EXTRA_USAGE
POSITION_USAGE_LIMIT_FACTOR = 1/10  # position_usage_limit = int(n * POSITION_USAGE_LIMIT_FACTOR) + EXTRA_USAGE
EXTRA_USAGE = 5

# Diversity score thresholds by progress
THRESHOLD_LOW = 8
THRESHOLD_MID = 5
THRESHOLD_HIGH = 4

# --- Helper Functions ---
def sample_strictly_increasing(a_min: int, a_max: int, b_max: int, c_max: int, d_max: int) -> List[int]:
    """
    Samples four strictly increasing integers:
       a in [a_min, a_max]
       b in [a+1, b_max]
       c in [b+1, c_max]
       d in [c+1, d_max]
    """
    a = random.randint(a_min, a_max)
    b = random.randint(a + 1, b_max)
    c = random.randint(b + 1, c_max)
    d = random.randint(c + 1, d_max)
    return [a, b, c, d]


def sample_second_param(block_type: int, prev: int) -> int:
    """
    Samples the second parameter based on block_type (only types 1 and 7 have params).
    Ensures non-decreasing sequence relative to prev.
    """
    if block_type == 1:
        choices = [1, 2, 4, 8]
    elif block_type == 7:
        choices = [4, 8, 16]
    else:
        return 0

    valid = [c for c in choices if c >= prev]
    return random.choice(valid) if valid else max(choices)


def create_encoding(blocks: List[int]) -> List[int]:
    """Creates a 12-length encoding list from 4 block types."""
    # First params (strictly increasing)
    first_params = sample_strictly_increasing(*FIRST_PARAMS_BOUNDS)

    # Second params per block
    second_params = []
    prev_vals = {1: 0, 7: 0}
    for b in blocks:
        sp = sample_second_param(b, prev_vals.get(b, 0))
        second_params.append(sp)
        if b in prev_vals:
            prev_vals[b] = sp

    # Combine into full encoding: [blocks[0..3], first_params[0..3], second_params[0..3]]
    return blocks + first_params + second_params


def write_encoding_to_file(encoding: List[int], filename: str = "encodings.txt") -> None:
    """Appends an encoding to the specified file."""
    with open(filename, "a") as f:
        f.write(str(encoding) + "\n")


def generate_encodings_multi_objective(n: int) -> List[List[int]]:
    """
    Generates up to n diverse encodings, balancing block and positional usage.
    """
    block_usage: Dict[int, int] = {i: 0 for i in range(1, 8)}
    position_usage: List[Dict[int, int]] = [{i: 0 for i in range(1, 8)} for _ in range(4)]
    generated_seqs: set = set()
    encodings: List[List[int]] = []

    max_attempts = n * MAX_ATTEMPTS_FACTOR
    attempts = 0

    # Precompute usage limits
    block_limit = int(n * BLOCK_USAGE_LIMIT_FACTOR) + EXTRA_USAGE
    pos_limit = int(n * POSITION_USAGE_LIMIT_FACTOR) + EXTRA_USAGE

    while len(encodings) < n and attempts < max_attempts:
        attempts += 1
        # Sample 4 random block types
        blocks = [random.randint(1, 7) for _ in range(4)]
        seq_tuple = tuple(blocks)

        # Skip trivial or duplicate sequences
        if len(set(blocks)) == 1 or seq_tuple in generated_seqs:
            continue

        encoding = create_encoding(blocks)

        # Compute diversity score
        diversity_score = 0
        for b in blocks:
            if block_usage[b] < block_limit:
                diversity_score += 1
        for pos, b in enumerate(blocks):
            if position_usage[pos][b] < pos_limit:
                diversity_score += 1

        progress = len(encodings) / n
        if progress < 0.6:
            required_score = THRESHOLD_LOW
        elif progress < 0.7:
            required_score = THRESHOLD_MID
        else:
            required_score = THRESHOLD_HIGH

        if diversity_score >= required_score or progress > 0.9:
            encodings.append(encoding)
            generated_seqs.add(seq_tuple)
            # Update usage stats
            for b in blocks:
                block_usage[b] += 1
            for pos, b in enumerate(blocks):
                position_usage[pos][b] += 1
            write_encoding_to_file(encoding)
            print(f"Accepted {len(encodings)}/{n}, score={diversity_score}, req={required_score}")

    # Fill any remaining slots without diversity check
    while len(encodings) < n:
        blocks = [random.randint(1, 7) for _ in range(4)]
        seq_tuple = tuple(blocks)
        if len(set(blocks)) == 1 or seq_tuple in generated_seqs:
            continue
        encoding = create_encoding(blocks)
        encodings.append(encoding)
        generated_seqs.add(seq_tuple)
        write_encoding_to_file(encoding)
        print(f"Filled final {len(encodings)}/{n}")

    # Summary
    print("\nGeneration complete")
    print(f"Total attempts: {attempts}, Generated: {len(encodings)}")
    print(f"Block usage: {block_usage}")
    return encodings


def all_same() -> List[List[int]]:
    """Creates 7 encodings where all blocks are the same type (1 through 7)."""
    encs: List[List[int]] = []
    for btype in range(1, 8):
        blocks = [btype] * 4
        first_params = list(FIRST_PARAMS_BOUNDS[1:])  # [4,8,12,16]
        # Compute second params deterministically (max choices)
        second_params = []
        prev_vals = {1: 0, 7: 0}
        for _ in blocks:
            sp = sample_second_param(btype, prev_vals.get(btype, 0))
            second_params.append(sp)
            if btype in prev_vals:
                prev_vals[btype] = sp

        encoding = blocks + first_params + second_params
        encs.append(encoding)
        write_encoding_to_file(encoding)
        print(f"All-same encoding for block {btype}: {encoding}")
    return encs


def main(n: int = 100) -> None:
    """Main entry point: clears file, sets seed, creates encodings."""
    # Reproducibility
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
