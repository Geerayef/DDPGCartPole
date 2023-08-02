import matplotlib.pyplot as plt
import numpy as np
import os
from subprocess import check_output
from sys import argv


line_data_location = "/home/tibor/Programming/bachelors/tibor-novakovic-diploma/LineData/"
line_data_dir = os.listdir(line_data_location)

# Boolean: 0, 1
plot_normalized = False
normalize = int(argv[1])
if normalize != 0 and normalize != 1:
    print(f"~~~~~ Wrong argument to plot_normalized: {normalize}. Must be 0 or 1.")
    raise SystemExit()
else:
    if normalize == 0:
        plot_normalized = False
    else:
        plot_normalized = True

lines = []
normalized_lines = []

# -------------------------------------------------------------------------------- #

# ~  Function definitions

def count_file_lines(filename):
    return int(check_output(["wc", "-l", filename]).split()[0])

def line_data_to_array():
    tmp = []
    file_count = 0
    for file in line_data_dir:
        f = os.path.join(line_data_location, file)
        if os.path.isfile(f):
            file_lines = count_file_lines(f)
            data_file = open(f, 'r')
            for datapoint in data_file:
                tmp.append(float(datapoint))

            if len(tmp) == file_lines:
                lines.append(np.array(tmp))
                file_count += 1
                tmp = []
            else:
                print(f"~~~~~ Failed to convert {f} to list ( float conversion )")

    print(f"~~~~~ {file_count} files converted to list.")


def normalize_array(array):
    max_lmnt = np.max(np.abs(array))
    return ( array / max_lmnt )


# -------------------------------------------------------------------------------- #

# ~  Main function calls

line_data_to_array()

# -------------------------------------------------------------------------------- #

# ~  Normalize

if plot_normalized:
    for line in lines:
        tmp = normalize_array(line)
        normalized_lines.append(tmp)

# -------------------------------------------------------------------------------- #

# ~  Plot

# Mean of lines
if plot_normalized:
    lines_mean = np.mean(normalized_lines, axis=0)
else:
    lines_mean = np.mean(lines, axis=0)
plt.plot(lines_mean)

# Individual lines
# if plot_normalized:
#     for line in normalized_lines:
#         plt.plot(line)
# else:
#     for line in lines:
#         plt.plot(line)

# Standard deviation
if plot_normalized:
    lines_stddev = np.std(normalized_lines)
else:
    lines_stddev = np.std(lines)
print(f"~~~~~ Standard deviation {lines_stddev}")


plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Reward growth during training')
plt.show()
