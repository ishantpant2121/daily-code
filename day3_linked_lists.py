"""Linked-list fundamentals for DSA practice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True, eq=False)
class ListNode(Generic[T]):
    """A singly linked node; identity equality keeps cycle checks O(1)."""

    value: T
    next: ListNode[T] | None = None


def reverse_list(head: ListNode[T] | None) -> ListNode[T] | None:
    """Reverse a singly linked list in O(n) time and O(1) extra space."""
    previous: ListNode[T] | None = None
    current = head
    while current is not None:
        next_node = current.next
        current.next = previous
        previous = current
        current = next_node
    return previous


def has_cycle(head: ListNode[T] | None) -> bool:
    """Detect a cycle with Floyd's slow/fast pointer algorithm."""
    slow = fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next if slow is not None else None
        fast = fast.next.next
        if slow is fast:
            return True
    return False


def to_list(head: ListNode[T] | None) -> list[T]:
    """Convert an acyclic linked list; raise ValueError rather than looping on cycles."""
    values: list[T] = []
    visited: set[ListNode[T]] = set()
    current = head
    while current is not None:
        if current in visited:
            raise ValueError("cannot convert a cyclic linked list to a Python list")
        visited.add(current)
        values.append(current.value)
        current = current.next
    return values


if __name__ == "__main__":
    first = ListNode(1, ListNode(2, ListNode(3)))
    reversed_head = reverse_list(first)
    assert to_list(reversed_head) == [3, 2, 1]
    assert has_cycle(reversed_head) is False

    four, five, six = ListNode(4), ListNode(5), ListNode(6)
    four.next, five.next, six.next = five, six, five
    assert has_cycle(four) is True
    print("All Day 3 linked-list examples passed.")
