import numpy as np
import tensorflow as tf 

from config import basic, hyper_param

def parse(record):
    """解析tfrecord
    """
    features = tf.parse_single_example(record,
                                       features={
                                           'data': tf.FixedLenFeature([], tf.string),
                                           'visit': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })  # return image and label
    decoded_image = tf.decode_raw(features['data'], tf.uint8)
    decoded_image = tf.reshape(decoded_image, [basic.img_height, basic.img_width, basic.img_channel])

    decoded_visit = tf.decode_raw(features['visit'], tf.float64)
    decoded_visit = tf.reshape(decoded_visit, [basic.visit_height, basic.visit_width, basic.visit_channel])

    label = tf.add(features['label'], -1)   #the original label range from 1 to 9, add -1 to suit for one_hot code
    label_one_hot = tf.one_hot(label, 9, dtype=tf.int32)
    # return decoded_image, decoded_visit, label
    return decoded_image, decoded_visit, label



def preprocess_for_train(image):
    """数据预处理, 裁剪, 翻转, 打乱; 太简单, 需要增加样本
    """
    original_shape = [basic.img_height, basic.img_width]
    distorted_shape =np.array([0.9*basic.img_height, 0.9*basic.img_width, basic.img_channel], dtype=np.int32)

    distorted_image = tf.image.random_flip_up_down(image)     # 随机上下翻转
    distorted_image = tf.image.random_flip_left_right(distorted_image)  # 随机左右翻转
    distorted_image = tf.random_crop(distorted_image, distorted_shape)            # 随机裁剪
    # distorted_image = tf.image.resize_images(distorted_image, original_shape, method=np.random.randint(4))
    return distorted_image
  

def get_tfdataset(tfrecord_files, mode='train'):
    """
    Args:
        tfrecord_files: tfrecord文件全路径名称
        mode: str. 'train':训练模式, 'valid': 验证模式, 'test':测试模式
    Returns:
        dataset: tf.Dataset对象, 训练模式重复无限次, 验证和测试模式不重复也不shuffle
    """
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse)
    dataset = dataset.map(lambda image, visit, label: (preprocess_for_train(image), label))
    if mode == 'train':
        dataset = dataset.shuffle(hyper_param.shuffle_buffer).batch(hyper_param.batch_size)
        # dataset = dataset.repeat(hyper_param.num_epochs)
        dataset = dataset.repeat()
    elif mode == 'valid':
        dataset = dataset.batch(10000)
        dataset = dataset.repeat()
    elif mode == 'test':
        dataset = dataset.batch(10000)
        
    return dataset
