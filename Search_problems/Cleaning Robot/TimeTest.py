from time import time
from Problem import *

times = 0
maxTime = 0
minTime = 100000
N = 1000
for i in range(0, N):
    print("Iteration: " + str(i+1))
    vp = VacuumProblem(5, 1/5)
    s = vp.initial_state
    start_time = time()
    solver = Solver(vp, "A*")
    node = solver.solve()
    t = time() - start_time
    if t > maxTime:
        maxTime = t
    if t < minTime:
        minTime = t
    times = times + t
times = times / N
print("Average time: " + str(times))
print("Max time: " + str(maxTime))
print("Min time: " + str(minTime))
