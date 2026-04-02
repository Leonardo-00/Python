from Problem import *

vp = VacuumProblem(8, 1/5)
s = vp.initial_state
solver = Solver(vp, "UCS")
node = solver.solve()


print("Initial state:")
print(s.__str__())
if node == None:
    print("No solution found")
else:
    print("\nSolution path:")
    print(node.path)