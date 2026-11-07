"""Disjoint Set Union with path compression and union by size."""


class DisjointSet:
    """Maintain disjoint integer sets with near-constant amortized operations."""

    def __init__(self, size):
        if type(size) is not int or size < 0:
            raise ValueError("size must be a non-negative integer")
        self.parent = list(range(size))
        self.sizes = [1] * size
        self.components = size

    def find(self, element):
        """Return the representative of element, compressing its path."""
        if type(element) is not int or not 0 <= element < len(self.parent):
            raise IndexError("element must be an integer within the set range")
        if self.parent[element] != element:
            self.parent[element] = self.find(self.parent[element])
        return self.parent[element]

    def union(self, first, second):
        """Join two sets and return whether their components were distinct."""
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return False

        if self.sizes[first_root] < self.sizes[second_root]:
            first_root, second_root = second_root, first_root

        self.parent[second_root] = first_root
        self.sizes[first_root] += self.sizes[second_root]
        self.components -= 1
        return True

    def connected(self, first, second):
        """Return whether two elements currently belong to the same set."""
        return self.find(first) == self.find(second)
