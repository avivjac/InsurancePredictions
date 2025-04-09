import numpy as np

# q.4 
# a
print("Section a:")
A = np.asarray([
        [2,1,2],
        [1,-2,1],
        [1,2,3],
        [1,1,1]])

b = np.asarray([6,1,5,2])

# calculationg least squares solution using numpy's lstsq function
x_star, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)

print("The best approximation to the least square: " + str(x_star))
print()
# b
print("Section b:")
print("The solution x* is unique if and only if A has full column rank.")

# chack if the solution is unique
is_unique = rank == A.shape[1]

# calculate the minimal value of the least squares function ||Ax* - b||_2^2
loss_value = np.linalg.norm(A @ x_star - b) ** 2

print("The rank of A is: " + str(rank))
print("Therfore, the solution x* is unique: " + str(is_unique))
print("The minimal value of the least squares function ||Ax* - b||_2^2 is: " + str(loss_value))
print()

# c
print("Section c:")

r = A @ x_star - b

print("The residual vector r is: " + str(r))

print("A.T @ r = " + str(A.T @ r))
print("Note that due to the computer restrictions, we recive a very small numbers, but in reality we'll get: A.T @ r = 0.")
print("It's not surprising since we know that the least squares solution is the one that minimizes the distance between Ax and b.")
print()

# d
print("Section d:")
W = np.diag([1, 1000, 1, 1])

A_weighted = W @ A
b_weighted = W @ b

# calculationg weighted least squares solution using numpy's lstsq function
x_star_weighted, residuals_weighted, rank_weighted, singular_values_weighted = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)

print("The best approximation to the least square with weights is: " + str(x_star_weighted))
print("The residual vector r is: " + str(A_weighted @ x_star_weighted - b_weighted))
print()

# e
print("Section e:")

lam = 0.5

# calculationg regularized least squares solution
ATA = A.T @ A
ATb = A.T @ b
I = np.identity(ATA.shape[1])

x_star_ridge = np.linalg.inv(ATA + lam * I) @ ATb
print("The best approximation to the least square with ridge regression is: " + str(x_star_ridge))