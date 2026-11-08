"""Linear-time string matching and prefix-array algorithms."""


def prefix_function(text: str) -> list[int]:
    """Return the KMP prefix function at every string position."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    prefix = [0] * len(text)
    border = 0
    for index in range(1, len(text)):
        while border and text[index] != text[border]:
            border = prefix[border - 1]
        if text[index] == text[border]:
            border += 1
        prefix[index] = border
    return prefix


def kmp_search(text: str, pattern: str) -> list[int]:
    """Return every starting index of pattern in text in O(n + m) time."""
    if not isinstance(text, str) or not isinstance(pattern, str):
        raise TypeError("text and pattern must be strings")
    if pattern == "":
        return list(range(len(text) + 1))
    prefix = prefix_function(pattern)
    matches = []
    matched = 0
    for index, character in enumerate(text):
        while matched and character != pattern[matched]:
            matched = prefix[matched - 1]
        if character == pattern[matched]:
            matched += 1
        if matched == len(pattern):
            matches.append(index - len(pattern) + 1)
            matched = prefix[matched - 1]
    return matches


def z_function(text: str) -> list[int]:
    """Return Z values, where each value is a longest common prefix length."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text:
        return []
    z = [0] * len(text)
    z[0] = len(text)
    left = right = 0
    for index in range(1, len(text)):
        if index <= right:
            z[index] = min(right - index + 1, z[index - left])
        while index + z[index] < len(text) and text[z[index]] == text[index + z[index]]:
            z[index] += 1
        if index + z[index] - 1 > right:
            left, right = index, index + z[index] - 1
    return z


def rabin_karp_search(text: str, pattern: str) -> list[int]:
    """Return every pattern occurrence using a rolling polynomial hash.

    Candidate hash matches are verified against the original substring, so
    hash collisions never create false-positive positions. Empty patterns
    match every boundary, consistent with kmp_search().
    """
    if not isinstance(text, str) or not isinstance(pattern, str):
        raise TypeError("text and pattern must be strings")
    if pattern == "":
        return list(range(len(text) + 1))
    pattern_length = len(pattern)
    if pattern_length > len(text):
        return []

    base = 257
    modulus = 1_000_000_007
    highest_power = pow(base, pattern_length - 1, modulus)
    pattern_hash = 0
    window_hash = 0
    for index in range(pattern_length):
        pattern_hash = (pattern_hash * base + ord(pattern[index]) + 1) % modulus
        window_hash = (window_hash * base + ord(text[index]) + 1) % modulus

    matches = []
    last_start = len(text) - pattern_length
    for start in range(last_start + 1):
        if window_hash == pattern_hash and text[start:start + pattern_length] == pattern:
            matches.append(start)
        if start < last_start:
            outgoing = (ord(text[start]) + 1) * highest_power
            incoming = ord(text[start + pattern_length]) + 1
            window_hash = ((window_hash - outgoing) * base + incoming) % modulus
    return matches
