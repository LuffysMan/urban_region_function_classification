import numpy as np 
import pandas as pd 
import os  
import argparse 

from config import basic
from dataset.dataset import get_tfdataset
from train import load_model, flexible_model, load_weights

parser = argparse.ArgumentParser(description='this program is for predicting or evaluating respect to different mode')
parser.add_argument('-m', '--mode', type=str, help="mode: 'pred', run on the test set; mode: 'eval', run on the validation set" )
args = parser.parse_args()

def save_predictions(predictions, save_dir, ckpt_dir):
    """将概率分布转化为预测结果, 并保存到csv文件用于提交
    Args:
        predictions: 最后一个隐藏层经过softmax后的概率分布
    """
    path = f'{save_dir}/result_{ckpt_dir}.txt'


    categories = np.argmax(predictions, axis=1)

    frmt3 = lambda x: str(x).zfill(3)
    frmt6 = lambda x: str(x).zfill(6)
    result = []
    for index, category_id in enumerate(categories):
        result.append([frmt6(index), frmt3(category_id)])

    df = pd.DataFrame(result)
    df.to_csv(path, header=False, index=False)

def test_predict():
    """测试集上进行预测, 输出分类结果
    """
    ckpt_dir = input("specify the dir to load ckpt: ")

    dataset = read_data_sets('../data/MNIST_data', one_hot=True)
    
    images = dataset.test.images
    images = images.reshape((-1, 28,28,1))

    data = tf.data.Dataset.from_tensor_slices(images)
    data = data.batch(100)

    model = flexible_model(input_shape=(28,28,1), num_classes=10)
    model = load_weights(model, ckpt_dir)

    predictions = model.predict(data, steps=100)

    save_predictions(predictions, config.result_path, ckpt_dir)

def evaluate():
    """验证模型性能
    """
    model_id = input("model id: ")
   
    record_files = list(map(lambda record_name: os.path.join(basic.train_record_path, record_name), os.listdir(basic.train_record_path)))
    val_files = record_files[len(record_files) - 1]

    val_dataset = get_tfdataset(val_files, mode='valid')

    model = flexible_model(input_shape=(90,90,3), num_classes=9)
    model = load_weights(model, model_id)

    eval_loss, eval_accuracy = model.evaluate(val_dataset, steps=1)

    print(f'Eval loss: {eval_loss}, Eval accuracy: {eval_accuracy}')


def predict():
    """测试集上进行预测, 输出分类结果
    """
    model_id = input("model id: ")
   
    record_files = list(map(lambda record_name: os.path.join(basic.test_record_path, record_name), os.listdir(basic.test_record_path)))
    dataset = get_tfdataset(record_files, mode='test')

    model = flexible_model(input_shape=(90,90,3), num_classes=9)
    model = load_weights(model, model_id)
    # model = load_model(ckpt_dir)
    predictions = model.predict(dataset, steps=1)

    save_predictions(predictions, basic.result_path, model_id)

if __name__ == "__main__":
    if args.mode == 'pred':
        predict()
    elif args.mode == 'eval':
        evaluate()
    else:
        print('please specify a mode, see -h --help')