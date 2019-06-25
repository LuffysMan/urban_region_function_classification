import time
import numpy as np
import sys
import datetime
import pandas as pd
import os
import threading
import queue
import multiprocessing
import argparse

from config import basic

parser = argparse.ArgumentParser(description="this program is for transforming visit data")
parser.add_argument("-t","--threadnum", type=int, default=4, help="specify the threads num with an interger. eg. '-t=4'")
args = parser.parse_args()  

g_thread_count = args.threadnum

# 用字典查询代替类型转换，可以减少一部分计算时间
date2position = {}
datestr2dateint = {}
str2int = {}
for i in range(24):
    str2int[str(i).zfill(2)] = i

# 访问记录内的时间从2018年10月1日起，共182天
# 将日期按日历排列
for i in range(182):
    date = datetime.date(day=1, month=10, year=2018)+datetime.timedelta(days=i)
    date_int = int(date.__str__().replace("-", ""))
    date2position[date_int] = [i%7, i//7]
    datestr2dateint[str(date_int)] = date_int


def visit2array(table):
    strings = table[1]
    init = np.zeros((26, 24, 7))
    for string in strings:
        temp = []
        for item in string.split(','):
            temp.append([item[0:8], item[9:].split("|")])
        for date, visit_lst in temp:
            day, week = date2position[datestr2dateint[date]]
            for visit in visit_lst: # 统计到访的总人数
                init[week][str2int[visit]][day] += 1
    return init


def visit2array_test():
    table = pd.read_csv(f'{basic.data_path}/test.csv', dtype=str)
    # filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
    filenames = [a[0] for a in table.values]
    length = len(filenames)
    # for filename in filenames:
    #     # filename = str(i).zfill(6)
    #     # table = pd.read_table("../data/test_visit/test/"+filename+".txt", header=None)
    #     table = pd.read_table(f'{basic.test_path}/visit/{filename}.txt', header=None)
    #     array = visit2array(table)
    #     np.save(f'{basic.npy_path}/test_visit/{filename}.npy', array)
    #     sys.stdout.write('\r>> Processing visit data %d/%d'%(i+1, 10000))
    #     sys.stdout.flush()
    # sys.stdout.write('\n')
    # print("using time:%.2fs"%(time.time()-start_time))

    read_path = f'{basic.test_path}/visit/'
    save_path = f'{basic.test_path}/npy/'
    divide_conquer(filenames, read_path, save_path, length)


def visit2array_train():
    """处理训练集visit数据
    """
    table = pd.read_csv(f'{basic.data_path}/train.csv')
    filenames = [a[0] for a in table.values]
    length = len(filenames)
    # start_time = time.time()
    read_path = f'{basic.train_path}/visit/'
    save_path = f'{basic.train_path}/npy/'
    divide_conquer(filenames, read_path, save_path, length)

def thread_function_visit2array(filenames, read_path, save_path):
    """线程函数
    """
    start_time = time.time()
    for index, filename in enumerate(filenames):
        table = pd.read_table(read_path + filename+".txt", header=None)
        array = visit2array(table)
        np.save(save_path + filename+".npy", array)
        if index % 100 == 0:
            tName = threading.currentThread().getName()
            sys.stdout.write(f'\r>> {tName}- Processing visit data {index+1}')
            sys.stdout.flush()
            # print(f'{threading.Thread.getName(threading.currentThread())}: {index}')
    sys.stdout.write('\n')
    print(f"exit thread: {tName}. time consumption: {time.time()-start_time}")

def divide_conquer(filenames, read_path, save_path, length):
    """划分数据集为g_thread_count份, 和开辟的线程数量一致
    """
    fractions = []
    fraction_length = (length+19)//g_thread_count
    for i in range(g_thread_count):
        fractions.append(filenames[i*fraction_length: (i+1)*fraction_length])

    threadNames = ['visit2array{}'.format(i) for i in range(g_thread_count)]
    threads = []

    print("start {} threads".format(g_thread_count))
    for i, tName in enumerate(threadNames):
        thread = threading.Thread(target=thread_function_visit2array , name=tName, args=(fractions[i], read_path, save_path))
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()


if __name__ == '__main__':
    if not os.path.exists(f'{basic.train_path}/npy'):
        os.makedirs(f'{basic.train_path}/npy')
    if not os.path.exists(f'{basic.test_path}/npy'):
        os.makedirs(f'{basic.test_path}/npy')
    visit2array_train()
    visit2array_test()

