import matplotlib.pyplot as plt
import numpy as np
import os
from subprocess import check_output
from sys import argv


lines            = []
normalized_lines = []
position         = []
angle            = []
reward_step      = []

line_data_location = ""
line_data_dir      = None

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

plot_metrics = False
metrics = int(argv[2])
if metrics is not None and metrics != 0 and metrics != 1:
    print(f"~~~~~ Wrong argument to plot_metrics: {plot_metrics}. Must be 0 or 1.")
    raise SystemExit()
else:
    if metrics == 0:
        plot_metrics = False
        line_data_location = "/home/tibor/Programming/bachelors/tibor-novakovic-diploma/LineData/"
        line_data_dir      = os.listdir(line_data_location)
    else:
        plot_metrics = True
        metrics_data_location = "/home/tibor/Programming/bachelors/tibor-novakovic-diploma/MetricsData/"
        metrics_data_dir      = os.listdir(metrics_data_location)

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


def metrics_data_to_array():
    global position, angle, reward_step
    for file in metrics_data_dir:
        f = os.path.join(metrics_data_location, file)
        if os.path.isfile(f):
            base_name = os.path.basename(f)
            if base_name == "pos_avg":
                position = np.load(f)
            elif base_name == "angle_avg":
                angle = np.load(f)
            elif base_name == "rewards_avg_step":
                reward_step = np.load(f)
            else:
                print("~~~~~ No more relevant files.")


def normalize_array(array):
    max_lmnt = np.max(np.abs(array))
    return ( array / max_lmnt )


# -------------------------------------------------------------------------------- #

# ~  Main function calls

if plot_metrics:
    metrics_data_to_array()
else:
    line_data_to_array()

# -------------------------------------------------------------------------------- #

# ~  Normalize rewards

if plot_metrics:
    print("Normalize metrics - not implemented.")
else:
    if plot_normalized:
        for line in lines:
            tmp = normalize_array(line)
            normalized_lines.append(tmp)

# -------------------------------------------------------------------------------- #

# ~  Plot metrics

if plot_metrics:
    print(f"~~~~~ Position array: {position}. Length: {np.size(position)}")
    print(f"~~~~~ Angle array: {angle}. Length: {np.size(angle)}")
    print(f"~~~~~ Rewards/step array: {reward_step}. Length: {np.size(reward_step)}")
    # plt.plot(position)
    # plt.plot(angle)
    plt.plot(reward_step)

# -------------------------------------------------------------------------------- #

# ~  Plot rewards

else:

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
    plt.plot(lines_stddev)

    threshold_line_norm = 195.0 / np.max(np.max(lines, axis=0))
    if plot_normalized:
        plt.plot([0, 2000], [threshold_line_norm, threshold_line_norm], 'k-', lw=1)
    else:
        plt.plot([0, 2000], [195, 195], 'k-', lw=1)

    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Mean reward growth during training')


plt.show()
