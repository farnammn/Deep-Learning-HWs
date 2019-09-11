import h5py
import numpy as np
import json
import tensorflow as tf
from scipy import misc

class CNN_Loader:

    def __init__(self, base_directory, pooling_file='data_img_pool5.h5', order_file = 'img_ordering.json'):
        self.pooling_file = base_directory  + pooling_file
        self.order_file = base_directory + order_file
        self.load_pool_layer()
        self.load_ordering()

    def load_ordering(self):
        '''
        loading the file that sorts the vector of different images
        '''
        file = open(self.order_file)
        d = file.readlines()
        json_data = json.loads(d[0])
        json_data = json_data["img_train_ordering"]
        j = []
        for js in json_data:
            j.append(int(js[-10:-4]))
        self.img_order = j

    def load_pool_layer(self):
        '''
        loading pool5 layer
        '''
        f = h5py.File(self.pooling_file, 'r')
        self.train_pools = (f['images_train'])
        self.test_pools = (f['images_test'])
        self.pool_shape = self.train_pools[0].shape
        self.pool_length = self.pool_shape[1]
        self.pool_depth = self.pool_shape[0]

    def img2vec(self, img_file_name):
        img = misc.imread(img_file_name)
        img = misc.imresize(img, size=(240, 240))
        IMAGE_SHAPE = (1, 240, 240, 3)

        input_img = tf.placeholder(tf.float32, IMAGE_SHAPE, name='input_img')
        vgg19 = tf.keras.applications.VGG19(weights='imagenet', include_top=False)
        output = vgg19(input_img)

        # img = ... # load and preprocess image
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output_val = sess.run(output, {input_img: img[np.newaxis, :]})

        return output_val

    def batch_img2vec(self, image_ids):
        '''
        :param image_ids: all of a batch image_id
        :return: the vectors corresponding to them with shape [batch_size, 7, 7, 512]
        '''
        vecs = []
        for image_id in image_ids:
            try:
                index = self.img_order.index(image_id)
                vecs.append(self.train_img2vec(index))
            except:
                vecs.append(np.zeros((self.pool_length,self.pool_length,self.pool_depth)))
        return np.array(vecs)


    def train_img2vec(self, index):
        '''
        :param index: the index of vector corresponding to an image
        :return: a vector corresponding to an image with shape [7 , 7, 512]
        '''
        return np.transpose(self.train_pools[index], (1,2,0))

