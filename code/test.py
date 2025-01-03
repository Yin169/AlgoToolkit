import fastsolver

# Example usage of power_iter function
A = [[4, 1], [2, 3]]  # Example matrix
b = [1, 0]            # Example vector
max_iter = 1000

# Call the power_iter function
eigenvalue, eigenvector = fastsolver.power_iter(A, b, max_iter)
print(f"Eigenvalue: {eigenvalue}")
print(f"Eigenvector: {eigenvector}")

# Example usage of ConjugateGrad class
A = fastsolver.SparseMatrixCSC(2, 2)
A.addValue(0, 0, 4)
A.addValue(0, 1, 1)
A.addValue(1, 0, 2)
A.addValue(1, 1, 3)
A.finalize()

b = fastsolver.VectorObj(2, 1.0)
x = fastsolver.VectorObj(2, 0.0)

cg_solver = fastsolver.ConjugateGrad(A, b, 1000, 1e-6)
cg_solver.solve(x)
print(f"Solution: {x}")

# Example usage of AlgebraicMultiGrid class
amg_solver = fastsolver.AlgebraicMultiGrid()
x = fastsolver.VectorObj(2, 0.0)
amg_solver.amgVCycle(A, b, x, levels=2, smoothingSteps=3, theta=0.25)
print(f"AMG Solution: {x}")