import numpy as np
import math


f = 2610
x = np.array([[2132, 125, f], [2034, 408, f], [1350, 700, f], [1238, 1275, f], [530, 1639, f], [435, 1755, f],
     [2149, 1275, f], [2234, 554, f]]).astype(np.float64)
p = np.array([[2673, 173, f], [2564, 459, f], [1874, 782, f], [1779, 1326, f], [1146, 1666, f], [1064, 1765, f],
              [2700, 1306, f], [2782, 602, f]]).astype(np.float64)

print("original x:", x)
print("original p:", p)
for i in range(8):
    x[i, :] = x[i, :] / np.linalg.norm(x[i, :])
    p[i, :] = p[i, :] / np.linalg.norm(p[i, :])

W = np.zeros((3, 3))

# print(np.outer(x[1, :].T, p[1, :]))
x = x - np.mean(x, axis=0)
p = p - np.mean(p, axis=0)

print("normalized and mean x:", x)
print("normalized and mean p:", p)
for i in range(8):
    W += np.outer(x[i, :].T, p[i, :])

u, sigma, vh = np.linalg.svd(W)
print(sigma)
print("angle matrix R:", np.dot(u, vh))

R = np.dot(u, vh)
omg = math.acos((np.trace(R) - 1) / 2)

print('angle: ', omg * 180 / math.pi)
