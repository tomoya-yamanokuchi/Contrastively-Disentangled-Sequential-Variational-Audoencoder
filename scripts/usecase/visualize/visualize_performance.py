from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

data = [
    [30, 25, 50, 20],
    [40, 23, 51, 17],
    [35, 22, 45, 19]
]

data = [
    [100.00, 8.9352, 0.0072, 2.1972],
    [100.00, 8.9398, 0.0067, 2.1972],
    [100.00, 8.9378, 0.0069, 2.1972],
]

X   = np.arange(4)
fig, ax = plt.subplots()
# ax  = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, data[0], color = 'b', width = 0.25, label="A")
ax.bar(X + 0.25, data[1], color = 'g', width = 0.25, label="V")
ax.bar(X + 0.50, data[2], color = 'r', width = 0.25, label="C")
ax.legend()

plt.savefig("./performance.png")