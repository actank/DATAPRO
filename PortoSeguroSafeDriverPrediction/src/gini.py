def gini(actual, pred, cmpcol = 0, sortcol = 1):
     assert( len(actual) == len(pred) )
     all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
     all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
     totalLosses = all[:,0].sum()
     giniSum = all[:,0].cumsum().sum() / totalLosses
 
     giniSum -= (len(actual) + 1) / 2.
     return giniSum / len(actual)
 
 def gini_normalized(a, p):
     return gini(a, p) / gini(a, a)
 
 def test_gini():
     def fequ(a,b):
         return abs( a -b) < 1e-6
     def T(a, p, g, n):
         assert( fequ(gini(a,p), g) )
         assert( fequ(gini_normalized(a,p), n) )
     T([1, 2, 3], [10, 20, 30], 0.111111, 1)
     T([1, 2, 3], [30, 20, 10], -0.111111, -1)
     T([1, 2, 3], [0, 0, 0], -0.111111, -1)
     T([3, 2, 1], [0, 0, 0], 0.111111, 1)
     T([1, 2, 4, 3], [0, 0, 0, 0], -0.1, -0.8)
     T([2, 1, 4, 3], [0, 0, 2, 1], 0.125, 1)
     T([0, 20, 40, 0, 10], [40, 40, 10, 5, 5], 0, 0)
     T([40, 0, 20, 0, 10], [1000000, 40, 40, 5, 5], 0.171428,
       0.6)
     T([40, 20, 10, 0, 0], [40, 20, 10, 0, 0], 0.285714, 1)
     T([1, 1, 0, 1], [0.86, 0.26, 0.52, 0.32], -0.041666,
       -0.333333)
