from multiprocessing import Process
import os
import time
import glob


def task():
    pwd = os.getcwd()
    collect_plotting_data = glob.glob(pwd + "/collect_plotting_data.sh")

    print("~~~~~ Run collect_plotting_data")
    os.system(collect_plotting_data[0])

    print("~~~~~ Task is done.\n")
    time.sleep(10)


tasks = []
num_processes = 2
all_processes = []

# Run this task with max times
for i in range(0, num_processes):
    tasks.append(task)

for func in tasks:
    p = Process(target=func)
    all_processes.append(p)
    p.start()

for p in all_processes:
    p.join()

print("~~~~~ All threads finished.\n")
