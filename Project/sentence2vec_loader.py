import tensorflow as tf
class Sentence2Vec_Loader:
    def __init__(self, num_hidden, vec_shape):
        self.num_hidden = num_hidden
        self.vec_shape = vec_shape
    def create_lstm(self , inputs, mask):
        '''
        :param input: a placeholder with shape [batch_size, sentence_legth, word_vec_size]
        :return: the last hidden_layer of lstm
        '''
        #### the lstm cell
        length = tf.reduce_sum(mask, axis = 1)
        lstm = tf.nn.rnn_cell.BasicLSTMCell(self.num_hidden)
        initial_state = lstm.zero_state(inputs.shape[0], dtype=tf.float32)
        #### run the cell
        hidden, state = tf.nn.dynamic_rnn(lstm, inputs, initial_state=initial_state, dtype=tf.float32)
        choose = tf.one_hot(length - 1, inputs.shape[1])

        return tf.reshape( tf.reduce_sum(hidden * tf.expand_dims(choose, axis = -1),axis = 1), [-1, self.vec_shape[0], self.vec_shape[1],self.vec_shape[2]])


