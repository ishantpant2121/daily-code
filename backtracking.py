"""Subsets and permutations using backtracking."""
def subsets(values):
    out=[]; path=[]
    def visit(i):
        if i==len(values):out.append(path.copy()); return
        visit(i+1); path.append(values[i]); visit(i+1); path.pop()
    visit(0); return out
def permutations(values):
    out=[]; a=values.copy()
    def visit(i):
        if i==len(a):out.append(a.copy()); return
        for j in range(i,len(a)):
            a[i],a[j]=a[j],a[i]; visit(i+1); a[i],a[j]=a[j],a[i]
    visit(0); return out



def solve_n_queens(size):
    """Return solutions as tuples of column positions, one per row.

    A bit-mask backtracking search tracks occupied columns and diagonals.
    The empty board has one empty solution.
    """
    if type(size) is not int or size < 0:
        raise ValueError("board size must be a non-negative integer")

    mask = (1 << size) - 1
    solutions = []
    placement = []

    def search(columns, descending, ascending):
        if len(placement) == size:
            solutions.append(tuple(placement))
            return

        available = mask & ~(columns | descending | ascending)
        while available:
            position = available & -available
            available -= position
            placement.append(position.bit_length() - 1)
            search(
                columns | position,
                ((descending | position) << 1) & mask,
                (ascending | position) >> 1,
            )
            placement.pop()

    search(0, 0, 0)
    return solutions



def combination_sum(candidates, target):
    """Return unique combinations that sum to target using unlimited reuse.

    Candidate values must be positive integers. Results are returned in
    non-decreasing order, and duplicate input candidates do not duplicate
    output combinations. A target of zero returns one empty combination.
    """
    if type(target) is not int or target < 0:
        raise ValueError("target must be a non-negative integer")
    values = list(candidates)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("candidates must be positive integers")

    values = sorted(set(values))
    combinations = []
    path = []

    def search(start, remaining):
        if remaining == 0:
            combinations.append(path.copy())
            return
        for index in range(start, len(values)):
            value = values[index]
            if value > remaining:
                break
            path.append(value)
            search(index, remaining - value)
            path.pop()

    search(0, target)
    return combinations
