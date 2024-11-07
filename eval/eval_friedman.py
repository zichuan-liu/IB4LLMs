
import numpy as np  
from scipy.stats import friedmanchisquare  
  
p2 = [5,3,5, 4,3,2, 4,4,2, 3,2,1, 1,2]
p1 = [4,5,4, 4,5,5, 3,3,1, 4,4,1, 2,3]
p3 = [3,4,2, 1,4,1, 2,1,4, 4,3,4, 3,4]
p4 = [1,2,1, 1,1,4, 1,2,5, 1,1,5, 5,5]
p5 = [2,1,3, 1,2,3, 5,5,3, 1,5,1, 3,1]
print(np.mean(p2), np.mean(p1),np.mean(p3),np.mean(p4),np.mean(p5))

stat, p = friedmanchisquare(p2, p1, p3, p4, p5)  
print(f'Friedman Test Statistics={stat:.5f}, p={p:.5f}')  
alpha = 0.05   
if p > alpha:  
    print('No significant difference between groups (fail to reject H0)')  
else:  
    print('Significant difference between groups (reject H0)')  

"""
2.9285714285714284 3.4285714285714284 2.857142857142857 2.5 2.5714285714285716
Friedman Test Statistics=3.08955, p=0.54295
No significant difference between groups (fail to reject H0)
"""

#time
o=[4.962, 5.067, 4.235, 4.095]
print(np.mean(o))
f=[4.850, 4.726, 4.107, 3.873]
print(np.mean(f))
u=[5.014, 5.128, 4.233, 4.042]
print(np.mean(u))
s=[9.551, 8.413, 8.780, 9.208]
print(np.mean(s))
m=[5.297, 5.015, 4.284, 4.319]
print(np.mean(m))
i=[5.509,5.370,4.426,4.251]
print(np.mean(i))
i=[5.664,5.351,4.269,4.528]
print(np.mean(i))
i=[9.605,6.024,10.185,11.322]
print(np.mean(i))

# extractors

p2 = [2,2,1, 3,2,1, 4]
p1 = [4,4,2, 1,3,1, 4]
p3 = [1,1,2, 4,4,1, 1]
p4 = [3,3,4, 1,1,1, 2]
print(np.mean(p2), np.mean(p1),np.mean(p3),np.mean(p4))

stat, p = friedmanchisquare(p2, p1, p3, p4)  
print(f'Friedman Test Statistics={stat:.5f}, p={p:.5f}')  
alpha = 0.05   
if p > alpha:  
    print('No significant difference between groups (fail to reject H0)')  
else:  
    print('Significant difference between groups (reject H0)') 