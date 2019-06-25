import os
import numpy as np
import pandas as pd 
import tensorflow as tf 
from tensorflow.keras import layers

from config import basic, hyper_param
# from inference import inference, cal_acc, cal_loss, cal_loss_acc
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from dataset.dataset import get_tfdataset
from networks.resnet import ResnetBuilder
from networks.plainnet import PlainNetBuilder
# tf.enable_eager_execution()
        
# def parse(record):
#     """解析tfrecord
#     """
#     features = tf.parse_single_example(record,
#                                        features={
#                                            'data': tf.FixedLenFeature([], tf.string),
#                                            'visit': tf.FixedLenFeature([], tf.string),
#                                            'label': tf.FixedLenFeature([], tf.int64),
#                                        })  # return image and label
#     decoded_image = tf.decode_raw(features['data'], tf.uint8)
#     decoded_image = tf.reshape(decoded_image, [config.img_height, config.img_width, config.img_channel])

#     decoded_visit = tf.decode_raw(features['visit'], tf.float64)
#     decoded_visit = tf.reshape(decoded_visit, [config.visit_height, config.visit_width, config.visit_channel])

#     label = tf.add(features['label'], -1)   #the original label range from 1 to 9, add -1 to suit for one_hot code
#     label_one_hot = tf.one_hot(label, 9, dtype=tf.int32)
#     # return decoded_image, decoded_visit, label
#     return decoded_image, label



# def preprocess_for_train(image):
#     """数据预处理, 裁剪, 翻转, 打乱; 太简单, 需要增加样本
#     """
#     original_shape = [config.img_height, config.img_width]
#     distorted_shape =np.array([0.9*config.img_height, 0.9*config.img_width, config.img_channel], dtype=np.int32)

#     distorted_image = tf.image.random_flip_up_down(image)     # 随机上下翻转
#     distorted_image = tf.image.random_flip_left_right(distorted_image)  # 随机左右翻转
#     distorted_image = tf.random_crop(distorted_image, distorted_shape)            # 随机裁剪
#     # distorted_image = tf.image.resize_images(distorted_image, original_shape, method=np.random.randint(4))
#     return distorted_image
      

# def cal_acc(pred_one_hot, label_one_hot):
#     pred = np.argmax(pred_one_hot, axis=-1)
#     label = np.argmax(label_one_hot, axis=-1)
#     match = np.equal(pred, label)
#     accuracy = np.sum(match)/len(match)
#     return accuracy

# def validator(valid_files):
#     """验证操作
#     """
#     dataset = tf.data.TFRecordDataset(valid_files)
#     dataset = dataset.map(parse)
#     dataset = dataset.batch(config.batch_size)
#     dataset = dataset.shuffle(config.shuffle_buffer).batch(config.batch_size)
#     dataset = dataset.repeat(config.num_epochs)

#     iterator = dataset.make_one_shot_iterator()
#     image_batch, label_batch = iterator.get_next()


#     def validate(sess, is_training):
#         logit = inference(image_batch, is_training)
#         predictions = tf.argmax(logit, axis=-1, output_type=tf.int32)
#         pred, label = sess.run([predictions, label_batch], feed_dict={is_training: False})
#         return pred, label

#     return iterator, validate

# def train_with_rawop(train_dataset, val_dataset):
#     # is_training = tf.placeholder(tf.bool)
#     is_training = tf.constant(True, dtype=tf.bool, name='is_training')

#     # valid_iterator, validate_func = validator(valid_files)

#     # train_iterator = dataset.make_initializable_iterator()
#     train_iterator = train_dataset.make_one_shot_iterator()
#     image_batch, label_batch = train_iterator.get_next()
#     #test{
#     # input_shape = tf.shape(image_batch)
#     # sess = tf.Session()
#     # sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
#     # sess.run(train_iterator.initializer)
#     # sess.run(valid_iterator.initializer)
#     # print(f'input_shape: {sess.run(input_shape)}')
#     # sess.close
#     # }
    
#     logits = inference(image_batch, is_training)
#     '''   
#     loss, acc = cal_loss_acc(logits, label_batch)

#     global_step = tf.Variable(0, trainable=False, name='global_step')
#     train_step = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(loss, global_step = global_step)
#     '''
#     # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#     #     sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
#     #     sess.run(train_iterator.initializer)
#     #     sess.run(valid_iterator.initializer)
        
#     #     last_ckpt = tf.train.latest_checkpoint(config.ckpt_path)
#     #     varlist = tf.trainable_variables()
#     #     saver = tf.train.Saver(varlist,  max_to_keep=10)
#     #     if last_ckpt:
#     #         saver.restore(sess, last_ckpt)

#     #     while True:
#     #         try:
#     #             _, loss, acc, step = sess.run([train_step,loss, acc, global_step], feed_dict={is_training:True})
#     #             print(f'step:{step} loss:{loss} acc:{acc}')

#     #             if step % 10 == 0:
#     #                 pred, label = validate_func(sess, is_training)
#     #                 acc_valid = cal_acc(pred, label)
#     #                 print(f'step:{step} acc:{acc} acc_valid:{acc_valid}')
#     #             if step % 100 == 0:
#     #                 saver.save(sess, os.path.join(config.ckpt_path, "checkpoint"), global_step=step)
                
#     #         except tf.errors.OutOfRangeError:
#     #             break

# def get_tfdataset(tfrecord_files):
#     dataset = tf.data.TFRecordDataset(tfrecord_files)
#     dataset = dataset.map(parse)
#     dataset = dataset.map(lambda image, label: (preprocess_for_train(image), label))
#     dataset = dataset.shuffle(config.shuffle_buffer).batch(config.batch_size)
#     # dataset = dataset.repeat(config.num_epochs)

    # return dataset

def save_weights(model, model_id, path=basic.ckpt_path):
    """ Save weights to a TensorFlow Checkpoint file
    """
    save_path = f'{path}/{model_id}/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    model.save_weights(save_path)

def load_weights(model, model_id, path=basic.ckpt_path):
    """ 
    """
    ckpt_path = f'{path}/{model_id}/'
    model.load_weights(tf.train.latest_checkpoint(ckpt_path))
    
    return model
def save_model(model, model_id, path=basic.ckpt_path):
    """ Save weights to a TensorFlow Checkpoint file
    """
    save_path = f'{path}/{model_id}'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    model.save(f'{save_path}/{model_id}.h5')

def load_model(model_id, path=basic.ckpt_path):
    """ 加载整个模型
    """
    ckpt_path = f'{path}/{model_id}/{model_id}.h5'
    model = tf.keras.models.load_model(ckpt_path)
    
    return model

def scheduler(epoch):
    """用于学习率递减
    """
    if epoch < 10:
        return 0.001
    else:
        return 0.001*tf.math.exp(0.1*(10-epoch))



def foward(input_shape, num_classes):
    """前向传播过程
    Args:
        input_shape: 输入数据shape, 不包括batch
        num_classes: 输出类别数
    Returns:
        inputs: placeholder
        outputs: 最后一层dense层, 与类别数量对应
    """
    inputs = tf.keras.Input(input_shape)

    conv1 = tf.keras.layers.Conv2D(64, 3, strides=(1,1), padding='SAME', activation='relu',
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))(inputs)
    pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv1)

    conv2 = tf.keras.layers.Conv2D(128, 3, strides=(1,1), padding='SAME', activation='relu',
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))(pool1)
    pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv2)

    conv3 = tf.keras.layers.Conv2D(256, 3, strides=(1,1), padding='SAME', activation='relu',
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))(pool2)
    pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv3)
    pool3_flat = tf.keras.layers.Flatten()(pool3)

    dense1 = tf.keras.layers.Dense(256, activation='relu')(pool3_flat)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(dense1)

    return inputs, outputs

def simple_forward(input_shape, num_classes):
    """前向传播过程
    Args:
        input_shape: 输入数据shape, 不包括batch
        num_classes: 输出类别数
    Returns:
        inputs: placeholder
        outputs: 最后一层dense层, 与类别数量对应
    """
    inputs = tf.keras.Input(input_shape)

    pool3_flat = tf.keras.layers.Flatten()(inputs)

    dense1 = tf.keras.layers.Dense(256, activation='relu')(pool3_flat)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(dense1)

    return inputs, outputs

def flexible_model(input_shape, num_classes):
    """ 根据指定输入数据shape和输出类别, 构建模型
    Args:
        input_shape: 输入数据shape, 不包括batch
        num_classes: 输出类别数
    Returns:
        model: keras Model对象
    """
    if 1: 
        model = PlainNetBuilder.build(input_shape, num_classes)
        # inputs, outputs = foward(input_shape, num_classes)
        # # inputs, outputs = simple_forward(input_shape, num_classes)

        # model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(tf.keras.optimizers.Adam(lr=0.001), 
                loss='sparse_categorical_crossentropy',
                metrics=['acc'])
    if 0:
        model = ResnetBuilder.build_resnet_50(input_shape, num_classes)
        model.compile(tf.keras.optimizers.Adam(lr=0.001), 
                loss='sparse_categorical_crossentropy',
                metrics=['acc'])

    return model

def train(train_set, val_set):
    """训练
    Args:
        train_set: tf.Dataset对象. 训练集, 9个tfrecord组成; tf.Dataset对象, 经过batch, shuffle和repeat
        val_set: tf.Dataset对象. 经过batch, shuffle和repeat
    Outs:
        模型参数会保存到指定文件夹 
    """
    device_id = input("device id: ")
    os.environ["CUDA_VISIBLE_DEVICES"]=device_id
    model_id = input("model id: ")
    checkpoint_prefix = os.path.join(basic.ckpt_path, model_id, 'ckpt_{epoch}')

    model = flexible_model(input_shape=(90,90,3), num_classes=9)

    # 如果路径下已经有之前训练时保存的模型, 则加载后继续训练
    pass

    # Callback for printing the LR at the end of each epoch.
    class PrintLR(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print ('\nLearning rate for epoch {} is {}'.format(
                epoch + 1, tf.keras.backend.get_value(model.optimizer.lr)))

    callbacks = [
    # Interrupt training if `val_loss` stops improving for over 2 epochs
    tf.keras.callbacks.LearningRateScheduler(scheduler),
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
    PrintLR(),
    ]

    model.fit(train_set,
              epochs=10, 
              steps_per_epoch=100, 
              callbacks=callbacks,
              validation_data=val_set, 
              validation_steps=1)

    # save_model(model, model_id, basic.ckpt_path)

def main():
    record_files = list(map(lambda record_name: os.path.join(basic.train_record_path, record_name), os.listdir(basic.train_record_path)))
    train_files = record_files[0: len(record_files) - 1]
    val_files = record_files[len(record_files) - 1]

    train_dataset = get_tfdataset(train_files, mode='train')
    val_dataset = get_tfdataset(val_files, mode='valid')

    train(train_dataset, val_dataset)

if __name__ == "__main__":
    main()