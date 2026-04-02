import random
import math

def monte_carlo_pi(n):
    inside_circle = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if math.hypot(x, y) <= 1:
            inside_circle += 1
    return (inside_circle / n) * 4

if __name__ == "__main__":
    N = 1000000
    delta = 0.01
    pi_estimate = monte_carlo_pi(N)
    print(f"Estimated value of Pi after {N} iterations: {pi_estimate}")
    print(f"Difference from math.pi: {abs(math.pi - pi_estimate)}")
    
    p = 16 / (N * delta * delta)
    print(f"Probability of finding a point in the circle: {p}")
    print(f"Estimated area of the circle: {math.pi * (1 ** 2)}")
    print(f"Estimated area of the square: {4}")
