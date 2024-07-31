import matplotlib.pyplot as plt
import numpy as np
import os
from subprocess import check_output
from sys import argv

# ~ Parse CL args ----------------------------------------------------------- ~ #


def parse_clargs():
    n = False
    m = False
    c = ""
    for arg in argv:
        if arg is not None:
            if arg == "--normalise=1":
                n = True
            if arg == "--metrics=1":
                m = True
            if arg.find("--custom-dir=", 0) == 0:
                c = arg.split("=")[1]
        else:
            print(f"~~~~~ [ERROR] Invalid argument: `{arg}`.")
            print("~~~~~ Normalise: 0 | 1. Default: 0.")
            print("~~~~~ Metrics: 0 | 1. Default: 0.")
            print("~~~~~ Custom data directory: string 'dir_name'. Default: ''.")
            raise SystemExit()
    return n, m, c


# ~ Util -------------------------------------------------------------------- ~ #


def count_file_lines(filename):
    return int(check_output(["wc", "-l", filename]).split()[0])


def line_data_to_array(path_data_lines, content_dir_data_lines, lines):
    tmp = []
    file_count = 0
    for file in content_dir_data_lines:
        f = os.path.join(path_data_lines, file)
        if os.path.isfile(f):
            file_lines = count_file_lines(f)
            data_file = open(f, "r")
            for datapoint in data_file:
                tmp.append(float(datapoint))
            if len(tmp) == file_lines:
                lines.append(np.array(tmp))
                file_count += 1
                tmp = []
            else:
                print(f"~~~~~ [ERROR] Failed to convert {f} to list (float conversion)")
    print(f"~~~~~ [INFO] {file_count} line-data files converted to list.")
    return lines


def metrics_data_to_array(
    angle, position, reward_step, path_data_metrics, content_dir_data_metrics
):
    for file in content_dir_data_metrics:
        f = os.path.join(path_data_metrics, file)
        if os.path.isfile(f):
            base_name = os.path.basename(f)
            if base_name == "angle_avg":
                angle = np.load(f)
            if base_name == "pos_avg":
                position = np.load(f)
            if base_name == "reward_step":
                reward_step = np.load(f)
    return (angle, position, reward_step)


def normalise_arr(rewards):
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    diff = max_reward - min_reward
    normalized_rewards = [(r - min_reward) / diff for r in rewards]
    return np.array(normalized_rewards)


def normalise_array(array):
    max_lmnt = np.max(np.abs(array))
    return array / max_lmnt


def prepare_plot(
    angle,
    position,
    reward_step,
    lines,
    normalised_lines,
    plot_angle,
    plot_position,
    plot_reward_step,
):
    if plot_angle:
        plt.plot(angle)
    if plot_position:
        plt.plot(position)
    if plot_reward_step:
        plt.plot(reward_step)
    if plot_metrics:
        print(f"~~~~~ Angle array: {angle}. Length: {np.size(angle)}")
        print(f"~~~~~ Position array: {position}. Length: {np.size(position)}")
        print(
            f"~~~~~ Rewards/step array: {reward_step}. Length: {np.size(reward_step)}"
        )
    else:
        # Mean of lines
        if normalise_plot:
            lines_mean = np.mean(normalised_lines, axis=0)
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
        if normalise_plot:
            lines_stddev = np.std(normalised_lines)
        else:
            lines_stddev = np.std(lines)
        print(f"~~~~~ [INFO] Standard deviation: {lines_stddev}")
        plt.plot(lines_stddev)
        # threshold_line_norm = 195.0 / np.max(np.max(lines, axis=0))
        # if normalise_plot:
        # plt.plot([0, 2000], [threshold_line_norm, threshold_line_norm], "k-", lw=1)
        # else:
        # plt.plot([0, 2000], [195, 195], "k-", lw=1)
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Mean reward growth per episode")


# ~ Main -------------------------------------------------------------------- ~ #

if __name__ == "__main__":
    path_root_proj = os.getenv("HOME") + "/dev/tibor-novakovic-diploma"
    path_data_lines = path_root_proj + "/LineData"
    path_data_metrics = path_root_proj + "/MetricsData"
    path_custom_data_lines = ""
    plot_metrics = False
    normalise_plot = False
    normalise_plot, plot_metrics, path_custom_data_lines = parse_clargs()

    lines = []
    normalised_lines = []
    angle = []
    position = []
    reward_step = []
    if plot_metrics:
        print("~~~~~ Normalize metrics - not implemented.")
        content_dir_data_metrics = os.listdir(path_data_metrics)
        angle, position, reward_step = metrics_data_to_array(
            angle, position, reward_step, path_data_metrics, content_dir_data_metrics
        )
    else:
        if path_custom_data_lines != "":
            path_data_lines = path_data_lines + "/" + path_custom_data_lines
        content_dir_data_lines = os.listdir(path_data_lines)
        lines = line_data_to_array(path_data_lines, content_dir_data_lines, lines)
        if normalise_plot:
            for line in lines:
                # tmp = normalise_array(line)
                tmp = normalise_arr(line)
                normalised_lines.append(tmp)
    prepare_plot(
        angle,
        position,
        reward_step,
        lines,
        normalised_lines,
        plot_angle=True,
        plot_position=False,
        plot_reward_step=False,
    )
    plt.show()
