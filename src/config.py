
class BasicConfig(object):
    data_path = "../data"
    train_path = "../data/train"
    test_path = "../data/test"
    train_record_path = f"{train_path}/record"
    test_record_path = f"{test_path}/record"

    visualize_path = "../visualize"
    ckpt_path = "../ckpt"

    # npy_path = "../data/npy"

    result_path = "../result/"

    img_height = 100
    img_width = 100
    img_channel = 3

    visit_height = 26
    visit_width = 24
    visit_channel = 7

class HyperParamConfig(object):
    """超参数配置
    """
    num_examples = 40000
    num_classes = 9
    batch_size = 128
    shuffle_buffer = 10000

    num_epochs = 30
    learning_rate = 1e-3
    decay_interval = 164
    decay_rate = 0.96
    # regularization_rate=1e-4

    weight_decay =0e-5

basic = BasicConfig()
hyper_param = HyperParamConfig()