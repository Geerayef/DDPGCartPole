import matplotlib.pyplot as plt
import numpy as np
import os


line_data_location = "/home/tibor/Programming/bachelors/tibor-novakovic-diploma/LineData/"
line_data_dir = os.listdir(line_data_location)
lines = []
normalized_arrays = []


# Organize plotting data
def line_data_to_list():
    tmp = []
    file_count = 0
    for file in line_data_dir:
        f = os.path.join(line_data_location, file)
        if os.path.isfile(f):
            line_data = open(f, 'r')
            for txt_line in line_data:
                tmp.append(float(txt_line))

            if len(tmp) == 5000:
                # print(f"~~~~~ {f} converted to list.")
                file_count += 1
                lines.append(tmp)
                tmp = []
            else:
                print(f"~~~~~ Failed to convert {f} to list ( float conversion )")

    print(f"~~~~~ {file_count} files converted to list.")


def normalize_array(array):
    max_lmnt = np.max(np.abs(array))
    return ( array / max_lmnt )


line_data_to_list()

lines_arrays = [np.array(x) for x in lines]
for line_arr in lines_arrays:
    temp = normalize_array(line_arr)
    normalized_arrays.append(temp)

lines_mean = np.mean(normalized_arrays, axis=0)

plt.plot(lines_mean)
plt.xlabel('Index')
plt.ylabel('Mean Value')
plt.title('Mean: Normalized average reward lines')

lines_stddev = np.std(normalized_arrays)
print(f"~~~~~ Standard deviation {lines_stddev}")

# for line_data in lines:
#     ypoints = np.array(line_data)
#     plt.plot(ypoints)

plt.show()
