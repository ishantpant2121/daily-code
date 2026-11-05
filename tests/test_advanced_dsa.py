"""Regression tests for additional DSA modules."""
import unittest
from sorting import merge_sort,quick_sort,counting_sort
from searching import linear_search,lower_bound,rotated_search
from recursion import factorial,fibonacci,sum_recursive
from greedy import activity_selection,fractional_knapsack
from dynamic_programming import climbing_stairs,coin_change,longest_increasing_subsequence
from graphs import bfs,dfs,dijkstra
from trees import TreeNode,preorder,inorder,level_order
from heaps import kth_largest,merge_sorted_lists
from backtracking import subsets,permutations
from union_find import DisjointSet
class TestAdvancedDSA(unittest.TestCase):
 def test_sorting(self):
  for f in (merge_sort,quick_sort,counting_sort):self.assertEqual(f([4,-2,4,0]),[-2,0,4,4])
  with self.assertRaises(TypeError):counting_sort([1,2.5])
 def test_searching(self):
  self.assertEqual(linear_search([3,1,3],3),0); self.assertEqual(lower_bound([1,2,2,8],2),1)
  self.assertEqual(rotated_search([5,6,7,1,2,3],2),4)
 def test_recursion_greedy(self):
  self.assertEqual(factorial(5),120); self.assertEqual(fibonacci(8),21); self.assertEqual(sum_recursive([1,2,3]),6)
  self.assertEqual(activity_selection([(1,3),(2,4),(3,5)]),[(1,3),(3,5)])
  self.assertAlmostEqual(fractional_knapsack([(60,10),(100,20),(120,30)],50),240)
 def test_dp(self):
  self.assertEqual(climbing_stairs(5),8); self.assertEqual(coin_change([1,2,5],11),3)
  self.assertEqual(coin_change([2],3),-1); self.assertEqual(longest_increasing_subsequence([10,9,2,5,3,7,101,18]),4)
 def test_graphs(self):
  g={1:[2,3],2:[4],3:[4],4:[]}; self.assertEqual(bfs(g,1),[1,2,3,4]); self.assertEqual(dfs(g,1),[1,2,4,3])
  self.assertEqual(dijkstra({1:[(2,4),(3,1)],3:[(2,2)]},1),{1:0.0,2:3.0,3:1.0})
  with self.assertRaises(ValueError):dijkstra({1:[(2,-1)]},1)
 def test_trees_heaps(self):
  r=TreeNode(2,TreeNode(1),TreeNode(3)); self.assertEqual(preorder(r),[2,1,3]); self.assertEqual(inorder(r),[1,2,3])
  self.assertEqual(level_order(r),[[2],[1,3]]); self.assertEqual(kth_largest([3,2,1,5,6,4],2),5)
  self.assertEqual(merge_sorted_lists([[1,4],[],[2,3]]),[1,2,3,4])
 def test_backtracking_union_find(self):
  self.assertEqual(len(subsets([1,2,3])),8); self.assertEqual(len(permutations([1,2,3])),6)
  d=DisjointSet(4); self.assertTrue(d.union(0,1)); self.assertFalse(d.union(1,0)); self.assertTrue(d.connected(0,1)); self.assertEqual(d.components,3)
if __name__=="__main__":unittest.main()
