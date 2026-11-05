"""Binary tree traversals."""
from collections import deque
from dataclasses import dataclass
@dataclass(slots=True)
class TreeNode:
    value:int
    left:"TreeNode | None"=None
    right:"TreeNode | None"=None
def preorder(root):
    if root is None:return []
    out=[]; stack=[root]
    while stack:
        n=stack.pop(); out.append(n.value)
        if n.right:stack.append(n.right)
        if n.left:stack.append(n.left)
    return out
def inorder(root):
    out=[]; stack=[]; cur=root
    while cur or stack:
        while cur:stack.append(cur); cur=cur.left
        cur=stack.pop(); out.append(cur.value); cur=cur.right
    return out
def level_order(root):
    if root is None:return []
    q=deque([root]); levels=[]
    while q:
        row=[]
        for _ in range(len(q)):
            n=q.popleft(); row.append(n.value)
            if n.left:q.append(n.left)
            if n.right:q.append(n.right)
        levels.append(row)
    return levels
