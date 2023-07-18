import matplotlib.pyplot as plt
import numpy as np
import os


line_data_location = "/home/tibor/Programming/bachelors/tibor-novakovic-diploma/LineData/"
line_data_dir = os.listdir(line_data_location)
lines = []


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

            if len(tmp) == 100:
                # print(f"~~~~~ {f} converted to list.")
                file_count += 1
                lines.append(tmp)
                tmp = []
            else:
                print(f"~~~~~ Failed to convert {f} to list ( float conversion )")

    print(f"~~~~~ {file_count} files converted to list.")


line_data_to_list()

lines_arrays = [np.array(x) for x in lines]
lines_mean = [np.mean(k) for k in zip(*lines_arrays)]

plt.plot(lines_mean)

lines_stddev = np.std(lines_arrays)
print(f"~~~~~ Standard deviation {lines_stddev}")

# plt.errorbar(x=list(range(0, 101)), y=lines_mean, xerr=lines_stddev)

# for line_data in lines:
#     ypoints = np.array(line_data)
#     plt.plot(ypoints)

plt.show()
