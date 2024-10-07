import numpy as np
from scipy import stats

# Sample data: Measurements before and after treatment
before = np.array([7, 6, 8, 5, 7, 6, 9, 8])
after = np.array([9, 7, 8, 6, 10, 8, 10, 9])

# Calculate differences
differences = after - before


# Function to calculate W statistic
def calculate_w(differences):
    # Remove zero differences
    non_zero_diff = differences[differences != 0]

    # Calculate ranks
    ranks = stats.rankdata(np.abs(non_zero_diff))

    # Calculate W+ and W-
    w_plus = np.sum(ranks[non_zero_diff > 0])
    w_minus = np.sum(ranks[non_zero_diff < 0])

    return min(w_plus, w_minus)


# Calculate W statistic
W = calculate_w(differences)

# Calculate n (number of non-zero differences)
n = np.sum(differences != 0)

# Calculate mean and standard deviation of W
mean_w = n * (n + 1) / 4
std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

# Calculate z-score
z = (W - mean_w) / std_w

# Two-sided test
p_value_two_sided = 2 * min(stats.norm.cdf(z), 1 - stats.norm.cdf(z))

# One-sided greater test
p_value_greater = 1 - stats.norm.cdf(z)

# One-sided less test
p_value_less = stats.norm.cdf(z)

print(f"W statistic: {W}")
print(f"Z-score: {z}")
print(f"Two-sided p-value: {p_value_two_sided}")
print(f"One-sided (greater) p-value: {p_value_greater}")
print(f"One-sided (less) p-value: {p_value_less}")


'''
Paired observations: The test is used for matched or paired samples, where two measurements are taken from the same individual 
or object under different conditions.
Calculation of differences: The differences are calculated by subtracting one measurement from the other for each pair. 
For example, if you have "before" and "after" measurements, you would calculate "after" minus "before" for each pair.
Median of differences: The median difference is the middle value of these calculated differences when they are arranged in order.
Null hypothesis: In the Wilcoxon signed-rank test, the null hypothesis typically states that the median difference between pairs of observations is zero.
Alternative hypothesis: Depending on whether it's a two-sided or one-sided test, the alternative hypothesis states that the median difference is not zero, 
greater than zero, or less than zero.
Interpretation: A significant result in the Wilcoxon signed-rank test suggests that there is a consistent difference 
between the paired observations across the sample, as reflected by the median difference.

A p-value doesn't tell you if the null hypothesis is true or false. 
It just indicates the probability of observing such data under the null hypothesis.

cumulative distribution function of the test statistic's distribution
'''