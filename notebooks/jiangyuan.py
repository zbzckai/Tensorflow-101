import numpy as np
A = np.array([[2,0,0],[0,0,1],[0,1,0]])
print('打印A：\n{}'.format(A))
a, b = np.linalg.eig(A)
print('打印特征值a：\n{}'.format(a))
print('打印特征向量b：\n{}'.format(b))