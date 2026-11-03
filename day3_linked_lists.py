"""Day 3: Linked-list fundamentals for DSA practice.

Includes iterative reversal and cycle detection using Floyd's algorithm.
"""


class ListNode:
    """A node in a singly linked list."""

    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node


def reverse_list(head):
    """Reverse a singly linked list in O(n) time and O(1) extra space."""
    previous = None
    current = head

    while current is not None:
        next_node = current.next
        current.next = previous
        previous = current
        current = next_node

    return previous


def has_cycle(head):
    """Return True if the linked list contains a cycle.

    Floyd's slow/fast pointer algorithm:
    O(n) time and O(1) extra space.
    """
    slow = fast = head

    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True

    return False


def to_list(head):
    """Convert an acyclic linked list to a Python list."""
    values = []
    current = head

    while current is not None:
        values.append(current.value)
        current = current.next

    return values


if __name__ == "__main__":
    # Build: 1 -> 2 -> 3
    first = ListNode(1, ListNode(2, ListNode(3)))
    reversed_head = reverse_list(first)
    assert to_list(reversed_head) == [3, 2, 1]
    assert has_cycle(reversed_head) is False

    # Build a cycle: 4 -> 5 -> 6 -> back to 5
    four = ListNode(4)
    five = ListNode(5)
    six = ListNode(6)
    four.next = five
    five.next = six
    six.next = five
    assert has_cycle(four) is True

    print("All Day 3 linked-list examples passed.")
