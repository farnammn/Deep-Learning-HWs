import tensorflow as tf
import numpy as np
import datetime
from cnn_loader import CNN_Loader
from word2vec_loader import Word2Vec_Loader
from sentence2vec_loader import Sentence2Vec_Loader
from data_loader import Data_Loader


class Network:
    def __init__(self,
                 base_directory,
                 conv1_layers = 512,
                 fc1_layers = 1024,
                 num_classes = 1001,
                 sen2vec_hidden = 4900,
                 learning_rate = 0.001,
                 n_epochs = 10,
                 batch_size = 32,
                 num_ans = 10,
                 max_questions_words=23,
                 ):

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.conv1_layers = conv1_layers
        self.fc1_layers = fc1_layers
        self.num_classes = num_classes
        self.sen2vec_hidden = sen2vec_hidden
        self.num_ans = num_ans
        self.learning_rate = learning_rate
        self.max_question_words = max_questions_words

        self.img2vec = CNN_Loader(base_directory)
        self.pool_length = self.img2vec.pool_length
        self.pool_depth = self.img2vec.pool_depth
        self.F_input_dim = sen2vec_hidden // (self.pool_length * self.pool_length) + self.pool_depth

        self.word2vec = Word2Vec_Loader(base_directory)
        self.word_vec_size = self.word2vec.embedding_size

        vec_shape = [self.pool_length, self.pool_length, sen2vec_hidden // (self.pool_length * self.pool_length)]
        self.sen2vec = Sentence2Vec_Loader(num_hidden= sen2vec_hidden, vec_shape = vec_shape)

        self.data_loader = Data_Loader(base_directory)
        self.num_train_questions = self.data_loader.num_train_questions
        self.num_test_questions = self.data_loader.num_test_questions
        self.eps = 1e-10


    def F_layer(self,sen_embedding, image_embedding):
        input = tf.concat([sen_embedding, image_embedding], axis=-1)

        conv1 = tf.nn.conv2d(input,
                             self.F_conv_filter1,
                             strides=[1, 1, 1, 1],
                             padding="SAME",
                             name="conv1") + self.F_conv_bias1
        conv1_relu = tf.nn.relu(conv1)

        conv2 = tf.nn.conv2d(conv1_relu,
                             self.F_conv_filter2,
                             strides=[1, 1, 1, 1],
                             padding="SAME",
                             name="conv1") + self.F_conv_bias2
        conv2_shape = conv2[:,:,:,0].shape
        p1 = tf.reshape(conv2[:,:,:,0] , [conv2_shape[0], -1])
        p1 = tf.nn.softmax(p1)
        p1 = tf.reshape(p1 , conv2_shape)

        p2 = tf.reshape(conv2[:, :, :, 1], [conv2_shape[0], -1])
        p2 = tf.nn.softmax(p2)
        p2 = tf.reshape(p2, conv2_shape)

        im1 = image_embedding * tf.expand_dims(p1, -1)
        im2 = image_embedding * tf.expand_dims(p2, -1)
        return im1, im2

    def G_layers(self, im1, im2 , sen_embedding):
        input = tf.concat([sen_embedding, im1, im2], axis=-1)
        flatten = tf.layers.flatten(input)
        fc1 = tf.layers.dense(flatten, self.fc1_layers)
        fc1_relu = tf.nn.relu(fc1)
        fc2 = tf.layers.dense(fc1_relu, self.num_classes)
        return fc2

    def create_network(self):
        self.img_embedding = tf.placeholder(dtype=tf.float32, shape = [self.batch_size, self.pool_length, self.pool_length, self.pool_depth])
        self.word_embedding = tf.placeholder(dtype=tf.float32, shape = [self.batch_size, self.max_question_words, self.word_vec_size])
        self.y = tf.placeholder(dtype=tf.float32, shape = [self.batch_size, self.num_ans, self.num_classes])
        self.mask = tf.placeholder(dtype=tf.int32, shape =[self.batch_size, self.max_question_words])

        self.sen_embedding = self.sen2vec.create_lstm(inputs=self.word_embedding , mask = self.mask)

        self.F_conv_filter1 = tf.Variable(tf.random_normal([1, 1, self.F_input_dim, self.conv1_layers], name="conv1_filter"))
        self.F_conv_bias1 = tf.Variable(tf.fill([self.conv1_layers], 0.01, name="bias_conv1"))

        self.F_conv_filter2 = tf.Variable(tf.random_normal([1, 1, self.conv1_layers, 2], name="conv2_filter"))
        self.F_conv_bias2 = tf.Variable(tf.fill([2], 0.01, name="bias_conv2"))

        img1 , img2 = self.F_layer(sen_embedding=self.sen_embedding, image_embedding=self.img_embedding)
        result = self.G_layers(im1=img1, im2=img2, sen_embedding=self.sen_embedding)

        self.loss = 0
        max_result = tf.reduce_max(result, axis=-1)
        self.accuracy = tf.zeros([self.batch_size], dtype=tf.int32)
        for i in range(self.num_ans):
            entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=result, labels=self.y[:,i,:])
            ## computes the mean over all the examples in the batch
            self.loss += tf.log(tf.reduce_mean(entropy) + self.eps)
            max_y = tf.reduce_max(self.y[:,i,:], axis = -1)
            self.accuracy += tf.cast(tf.equal(max_y, max_result), tf.int32)

        self.accuracy = self.accuracy / 3
        self.accuracy = tf.minimum(self.accuracy , 1)
        self.accuracy = tf.reduce_mean(self.accuracy)

        self.loss = self.loss / self.num_ans

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()
        print("network created")


    def train_network(self, sess):
        sess.run(tf.global_variables_initializer())
        for i in range(self.n_epochs):
            n_batches = self.num_train_questions // self.batch_size
            total_loss = 0
            total_acc = 0
            for j in range(n_batches):
                questions_batch, answers_batch, img_ids_batch, mask_batch = self.data_loader.load_questions(j * self.batch_size, (j+1)* self.batch_size, "train")
                img_embedding_data = self.img2vec.batch_img2vec(img_ids_batch)
                q_size = questions_batch.shape[1]
                word_embedding_data = self.word2vec.return_word2vec(questions_batch.reshape(self.batch_size * q_size), mask_batch.reshape(self.batch_size * q_size))
                word_embedding_data = word_embedding_data.reshape(self.batch_size, q_size, self.word_vec_size)


                fead = {self.img_embedding : img_embedding_data, self.word_embedding:word_embedding_data, self.y:answers_batch, self.mask:mask_batch}
                train_loss, train_acc, _ = sess.run([self.loss, self.accuracy, self.optimizer], fead)
                total_loss += train_loss
                total_acc += train_acc

                if j % 100 == 99:
                    print("train_loss = {}, train_acc = {}".format(total_loss / 100, total_acc / 100))
                    total_loss = 0
                    total_acc = 0
                    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
                    file = open("res/" + date + ".txt", "w")
                    file.write("epoch_num{}, iteration {}, train_loss = {}, train_acc = {}".format(i, j,total_loss / 100, total_acc / 100))
                    file.close()


            date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            self.saver.save(sess, 'Models/Model-{}'.format(date))


    def test_network(self, sess):

        n_batches = self.num_test_questions // self.batch_size
        total_loss = 0
        total_acc = 0
        for j in range(n_batches):
            questions_batch, answers_batch, img_ids_batch, mask_batch = self.data_loader.load_questions(j * self.batch_size, (j+1)* self.batch_size, "val")
            img_embedding_data = self.img2vec.batch_img2vec(img_ids_batch)
            q_size = questions_batch.shape[1]
            word_embedding_data = self.word2vec.return_word2vec(questions_batch.reshape(self.batch_size * q_size), mask_batch.reshape(self.batch_size * q_size))
            word_embedding_data = word_embedding_data.reshape(self.batch_size, q_size, self.word_vec_size)


            fead = {self.img_embedding : img_embedding_data, self.word_embedding:word_embedding_data, self.y:answers_batch, self.mask:mask_batch}
            test_loss, test_acc, _ = sess.run([self.loss, self.accuracy, self.optimizer], fead)
            total_loss+=test_loss
            total_acc += test_acc

        print("test_loss = {}, test_acc = {}".format(total_loss / n_batches, total_acc /n_batches))
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        file = open("res/" + date + ".txt", "w")
        file.write("test_loss = {}, test_acc = {}".format(total_loss / n_batches, total_acc /n_batches))
        file.close()




