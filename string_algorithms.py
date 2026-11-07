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
