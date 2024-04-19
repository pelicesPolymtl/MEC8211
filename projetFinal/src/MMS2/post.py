import numpy as np


def L2(array):
    error_L2 = 0
    for ind in array:
        error_L2 = error_L2 + ind**2
    error_L2 = np.sqrt(1/len(array)*error_L2)
    return error_L2

with open("lvl3_deltap.txt") as f:
    content = f.readlines()

content = [float(x.strip()) for x in content]

print('level 3:',min(content), L2(content))

with open("lvl4_deltap.txt") as f:
    content = f.readlines()

content = [float(x.strip()) for x in content]

print('level 4:',min(content), L2(content))

# with open("delta_u_1024.txt") as f:
#     content = f.readlines()

# content = [float(x.strip()) for x in content]

# print('1k:',min(content), L2(content))


# with open("delta_u_4096.txt") as f:
#     content = f.readlines()

# content = [float(x.strip()) for x in content]

# print('4k:',min(content), L2(content))


# with open("delta_u_16k.txt") as f:
#     content = f.readlines()

# content = [float(x.strip()) for x in content]

# print('16k:',min(content), L2(content))

# with open("delta_u_66k.txt") as f:
#     content = f.readlines()

# content = [float(x.strip()) for x in content]

# print('66k:',min(content), L2(content))