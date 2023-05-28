from tkinter import filedialog, Tk
import os
import struct


def read_float(f):
    data = f.read(4)
    if data == bytes("", "utf-8"):
        return float(0), True
    (value,) = struct.unpack("f", data)
    return value, False


def extract(filename="", time_step = 0.5):
    if filename == "":
        filename = filedialog.askopenfilename()
    if filename == "" or not os.path.isfile(filename):
        return
    filename_csv = filename.split("/")[-1].split(".")[0] + ".csv"
    filename_txt = filename.split("/")[-1].split(".")[0] + ".txt"

    fin = open(filename, "rb")
    data = fin.read(4)
    if data != bytes("cart", "utf-8"):
        fin.close()
        return

    fout_csv = open(filename_csv, "w")

    for i in range(7):
        read_float(fin)

    fout_csv.write("time, x, theta")
    points_x = "# cart position\n"
    points_theta = "# pole position\n"
    eof = False
    frame_time = 0.0
    while not eof:
        time, eof = read_float(fin)
        x, eof = read_float(fin)
        theta, eof = read_float(fin)
        if time >= frame_time and not eof:
            fout_csv.write("\n%.3f,%.2f,%.2f" % (time, x, theta))
            if frame_time > 0:
                points_x += " "
                points_theta += " "
            points_x += "(%.3f, %.2f)" % (time, x)
            points_theta += "(%.3f, %.2f)" % (time, theta)
            frame_time += time_step

    fin.close()
    fout_csv.close()

    fout_txt = open(filename_txt, "w")
    fout_txt.write(points_x + "\n\n" + points_theta + "\n")
    fout_txt.close()


if __name__ == '__main__':
    Tk().withdraw()
    extract()
