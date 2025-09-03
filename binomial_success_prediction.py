import numpy as np 
import math as m
from scipy.stats import f
from scipy.stats import norm

# Wilson method:
def Wilson_Confidence_Band(p_hat, z, n):

    min = (p_hat  +z / (2 * n)) / (1 + z ** 2 / n) - z / (1 + z ** 2 / n) *  m.sqrt(p_hat *
          (1 - p_hat) / n + z ** 2 / (4 * n ** 2))
    
    max = (p_hat +  z / (2 * n)) / (1 + z ** 2 / n) + z / (1 + z ** 2 / n) * m.sqrt(p_hat * 
          (1 - p_hat) / n + z ** 2 / (4 * n ** 2))
    
    return min, max

def F(a, min, max): return f.ppf(a, min, max) # F distribution percentiles

#Clopper-Pearson method (x: successes, n: total tests, F: Percentiles):
def CP_Confidence_Band(x, n):

    min = x * F((1 - CL) / 2, 2 * x, 2 * (n - x + 1)) / (n - x + 1 + x * 
          F((1 - CL) / 2, 2 * x, 2 * (n - x + 1)))
    
    max = (x + 1) * F(1 - (1 - CL) / 2, 2 * (x + 1), 2 * (n - x)) / (n - x + (x + 1) *
          F(1 - (1 - CL) / 2, 2 * (x + 1), 2 * (n - x)))

    return min, max

# Confidence Level:
CL = 0.68

#==================================================================================
# Class A:
total_students_A = 40
success_A        = 30
p_hat_A          = success_A / total_students_A

minW_class_A,  maxW_class_A  = Wilson_Confidence_Band(p_hat_A, norm.ppf(1-(1-CL)/2), total_students_A)
minCP_class_A, maxCP_class_A = CP_Confidence_Band(success_A, total_students_A)

#==================================================================================
# Class B (1st year):
total_students_B = 200
success_B        = 130
p_hat_B          = success_B / total_students_B

minW_class_B,  maxW_class_B  = Wilson_Confidence_Band(p_hat_B, norm.ppf(1-(1-CL)/2), total_students_B)
minCP_class_B, maxCP_class_B = CP_Confidence_Band(success_B, total_students_B)

#==================================================================================
print("=======================================================================")
print("Confidence bands with Wilson method:")
print(f"FOR CLASS A: [{minW_class_A:.4f}, {maxW_class_A:.4f}]")
print(f"FOR CLASS B: [{minW_class_B:.4f}, {maxW_class_B:.4f}]")
print("=======================================================================")
print("Confidence bands with Clopper-Pearson method:")
print(f"FOR CLASS A: [{minCP_class_A:.4f}, {maxCP_class_A:.4f}]")
print(f"FOR CLASS B: [{minCP_class_B:.4f}, {maxCP_class_B:.4f}]")
#==================================================================================

#==================================================================================
# For the second year in Class B:
total_students_B = 180
success_B        = p_hat_B * total_students_B 

minW_class_B,  maxW_class_B  = Wilson_Confidence_Band(p_hat_B, norm.ppf(1-(1-CL)/2), total_students_B)
minCP_class_B, maxCP_class_B = CP_Confidence_Band(success_B, total_students_B)

print("=======================================================================")
print(f"For Class B in the second year with {total_students_B} total students:")
print("-----------------------------------------------------------------------")
print(f"Confidence bands with Wilson method: [{minW_class_B:.4f}, {maxW_class_B:.4f}]")
print(f"Confidence bands with Clopper-Pearson method: [{minCP_class_B:.4f}, {maxCP_class_B:.4f}]")
print("-----------------------------------------------------------------------")
print("The central value of possibility for success is:")
print(f"With Wilson method: {((maxW_class_B+minW_class_B)/2*100):.4f}%")
print(f"With Clopper-Pearson method: {((maxCP_class_B+minCP_class_B)/2*100):.4f}%")

successes_B_Wilson = (maxW_class_B+minW_class_B)  /2*total_students_B
successes_B_CP     = (maxCP_class_B+minCP_class_B)/2*total_students_B

succ_error_B_Wilson = (maxW_class_B-minW_class_B)  /2*total_students_B
succ_error_B_CP     = (maxCP_class_B-minCP_class_B)/2*total_students_B

print("-----------------------------------------------------------------------")
print(f"The successes are: ")
print(f"With Wilson method: {successes_B_Wilson:.0f}±{succ_error_B_Wilson:.0f}")
print(f"With Clopper-Pearson method: {successes_B_CP:.0f}±{succ_error_B_CP:.0f}")
