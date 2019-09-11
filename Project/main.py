from network import Network
import tensorflow as tf

base_directory = 'data/'

network = Network(base_directory)
network.create_network()


gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=config)
network.train_network(sess=sess)
network.test_network(sess=sess)