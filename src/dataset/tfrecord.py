import os
import sys
import random
import pandas as pd
import numpy as np
import cv2
import argparse
import tensorflow as tf

import threading

from config import basic

tf.enable_eager_execution()

parser = argparse.ArgumentParser(description="this program is for generating tfrecord")
parser.add_argument("-m","--mode", type=int, help="the operation mode; mode:0, generate train record; \
          mode:1 generate test record; mode: 2 generate train and test record")
args = parser.parse_args()  

    
# def get_data(dataset):
#     print("Loading training set...")
#     table = pd.read_csv("../data/"+dataset, header=None)
#     filenames = [item[0] for item in table.values]
#     class_ids = [int(item[0].split("/")[-1].split("_")[-1].split(".")[0])-1 for item in table.values]
#     data = []
#     for index, filename in enumerate(filenames):
#         image = cv2.imread(filename, cv2.IMREAD_COLOR)
#         visit = np.load("../data/npy/train_visit/"+filename.split('/')[-1].split('.')[0]+".npy")
#         label = class_ids[index]
#         data.append([image, visit, label])
#     random.seed(0)
#     random.shuffle(data)
#     print("Loading completed...")
#     return data


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def generate_example(data, visit, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'data': bytes_feature(data),
        'visit': bytes_feature(visit),
        'label': int64_feature(label),
    }))

def parser(record):
    features = tf.parse_single_example(record,
                                       features={
                                           'data': tf.FixedLenFeature([], tf.string),
                                           'visit': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })  
    decoded_image = tf.decode_raw(features['data'], tf.uint8)
    decoded_image = tf.reshape(decoded_image, [basic.img_height, basic.img_width, basic.img_channel])

    decoded_visit = tf.decode_raw(features['visit'], tf.float64)
    decoded_visit = tf.reshape(decoded_visit, [basic.visit_height, basic.visit_width, basic.visit_channel])

    label = features['label']
    return decoded_image, decoded_visit, label

# def _convert_dataset(data, tfrecord_path, dataset):
#     """ Convert data to TFRecord format. """
#     output_filename = os.path.join(tfrecord_path, dataset+".tfrecord")
#     tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)
#     length = len(data)
#     for index, item in enumerate(data):
#         data_ = item[0].tobytes()
#         visit = item[1].tobytes()
#         label = item[2]
#         example = generate_example(data_, visit, label)
#         tfrecord_writer.write(example.SerializeToString())
#         sys.stdout.write('\r>> Converting image %d/%d' % (index + 1, length))
#         sys.stdout.flush()
#     sys.stdout.write('\n')
#     sys.stdout.flush()

def train_valid_split(files, train_size=0.8):
    total = len(files)
    num_train = round(train_size * total)

    perm = [i for i in range(total)]
    np.random.shuffle(perm)

    return files[0: num_train], files[num_train: ]

def make_one_hot(num_classes, label):
    one_hot = np.zeros((num_classes), np.int32)
    one_hot[label-1] = 1
    return one_hot


def partition_task(all_tasks, num_threads):
    """划分任务, 并行生成record
    Args:
        all_tasks: 任务列表
        num_thread: 线程数
    """
    fraction_size = (len(all_tasks) + num_threads - 1) // num_threads
    tasks = [all_tasks[i*fraction_size: (i+1)*fraction_size] for i in range(num_threads)]
    return tasks

def t_generate_record(task, record_path):
    """线程函数, 产生tfrecord
    """
    # output_filename = os.path.join(tfrecord_path, dataset+".tfrecord")
    tfrecord_writer = tf.python_io.TFRecordWriter(record_path)
    length = len(task)
    for index, (img_file, visit_file, label) in enumerate(task):
        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        visit = np.load(visit_file)
        example = generate_example(image.tobytes(), visit.tobytes(), label)
        tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\r>> generating tfrecord %d/%d' % (index + 1, length))
        sys.stdout.flush()

    sys.stdout.write('\n')
    sys.stdout.flush()

def generate_train_file_groups(image_file_name, train_path):
    """生成测试集任务列表
    Args: 
        image_file_name: 不带后缀的文件名称, 对应一张图片和一个visit文件
        train_path: 训练集路径
    Returns:
        image_visit_label: 任务列表, 每个元素为一个元组(图片全路径, npy文件全路径, 类别), 测试集类别统统为0
    """
    image_visit_label = []
    for filename in image_file_name:
        classId = filename.split('_')[1]
        image_filepath = os.path.join(train_path, 'image', classId, f'{filename}.jpg')
        # visit_filepath = os.path.join(npy_path, 'train_visit', f'{filename}.npy')
        visit_filepath = os.path.join(train_path, 'npy', f'{filename}.npy')
        label = int(classId)
        image_visit_label.append([image_filepath, visit_filepath, label])
    return image_visit_label

def generate_test_file_groups(image_file_name, test_path):
    """生成测试集任务列表
    Args: 
        image_file_name: 不带后缀的文件名称, 对应一张图片和一个visit文件
        test_path: 测试集路径
    Returns:
        image_visit_label: 任务列表, 每个元素为一个元组(图片全路径, npy文件全路径, 类别), 测试集类别统统为0
    """
    image_visit_label = []
    for filename in image_file_name:
        image_filepath = os.path.join(test_path, 'image', f'{filename}.jpg')
        # visit_filepath = os.path.join(npy_path, 'test_visit', f'{filename}.npy')
        visit_filepath = os.path.join(test_path, 'npy', f'{filename}.npy')
        image_visit_label.append([image_filepath, visit_filepath, 0])
    return image_visit_label

def test_generate_record():
    if not os.path.exists(config.record_path):
        os.makedirs(config.record_path)

    df_train_files = pd.read_csv(os.path.join(config.data_path, 'train.csv'))
    train_files = np.array(df_train_files)[:, 0]
    df_test_files = pd.read_csv(os.path.join(config.data_path, 'test.csv'))
    test_files = np.array(df_test_files)[:, 0]
    
    image_visit_label = generate_train_file_groups(train_files, config.train_path, config.npy_path)
    task1 = image_visit_label[0:10]
    task2 = image_visit_label[10:20]
    t_generate_record(task1, os.path.join(config.record_path, f'test_record_1'))
    t_generate_record(task2, os.path.join(config.record_path, f'test_record_2'))

def test_read_tfrecord():
    input_files = ['../data/record/test_record_1', '../data/record/test_record_2']
    dataset = tf.data.TFRecordDataset(input_files)
    dataset = dataset.map(parser)
    iterator = dataset.make_one_shot_iterator()
    image, visit, label = iterator.get_next()

    print(image, visit, label)

def test_generate_simple_record():
    images = np.random.randint(0, 256, size=(10, 7,7,3), dtype=np.uint8)
    labels = np.random.randint(0, 9, size=(10, 1), dtype=np.uint8)
    num_examples = len(labels)
    output_path = os.path.join(basic.record_path, f'test_simple_record_1')
    writer = tf.python_io.TFRecordWriter(output_path)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        label = np.asscalar(labels[index])
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': bytes_feature(image_raw),
            'label': int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    writer.close()

def test_read_simple_record():
    input_files = ['../data/record/test_simple_record_1']
    dataset = tf.data.TFRecordDataset(input_files)
    def parser(serialized_example):
        features = tf.parse_single_example(serialized_example, features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
        return features['image'], features['label']

    dataset = dataset.map(parser)
    iterator = dataset.make_one_shot_iterator()
    image, label = iterator.get_next()
    # with tf.Session() as sess:
    #     img, lb = sess.run([image, label])
    print(image, label)

def generate_records_with_multi_threads(tasks, record_path):
    """生成tfrecord
    Args:
        tasks: 划分好的文件名
        record_path: 保存路径
    Outs:
        生成tfrecord并保存到指定文件夹
    """
    #创建10个线程
    threads = []
    for i, task in enumerate(tasks):
        tName = f't-train-{i}'
        thread = threading.Thread(target=t_generate_record, name=tName, args=(task, os.path.join(record_path, f'record_{i}')))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

def generate_train_record():
    """生成训练集record
    """
    if not os.path.exists(basic.train_record_path):
        os.makedirs(basic.train_record_path)

    df_train_files = pd.read_csv(os.path.join(basic.data_path, 'train.csv'))
    train_files = np.array(df_train_files)[:, 0]
   
    image_visit_label = generate_train_file_groups(train_files, basic.train_path)
    
    train_tasks = partition_task(image_visit_label, 10)
    generate_records_with_multi_threads(train_tasks, basic.train_record_path)

def generate_test_record():
    """生成测试集record
    """
    if not os.path.exists(basic.test_record_path):
        os.makedirs(basic.test_record_path)

    df_test_files = pd.read_csv(os.path.join(basic.data_path, 'test.csv'))
    test_files = np.array(df_test_files)[:, 0]
    test_files = np.array(list(map(lambda x: str(x).zfill(6), test_files)))

    image_visit_label = generate_test_file_groups(test_files, basic.test_path)
    test_tasks = partition_task(image_visit_label, 1)

    generate_records_with_multi_threads(test_tasks, basic.test_record_path)

if __name__ == '__main__':
    if args.mode == 0:
        print("generating train record")
        generate_train_record()
    elif args.mode == 1:
        print("generating test record")
        generate_test_record()
    elif args.mode == 2:
        print("generating train and test record")
        generate_train_record()
        generate_test_record()
    else:
        print("please select an operating mode by specify an argument, using -h or --help to see the info")
