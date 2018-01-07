import tensorflow as tf 
from tensorflow.contrib import learn
from cnn_model import CNN
import yaml


file_config = ""
with open(file_config, 'rb') as f:
    params = yaml.load(f)


def process(learning_rate, batch_size, epochs):
    '''
    tf.summary'll be used to see the histogram 
    Args:
        learning_rate (float):
        batch_size (int):
        epochs (int):
    '''
    with tf.Graph().as_default():
        '''
        '''
        cnn = CNN(
            self,
            num_classes=params["num_classes"],
            num_filters=params["num_filters"],
            filter_sizes=params["filter_sizes"],
            embedding_size=params["embedding_size"],
            vocab_size=params[""],
            sequence_length=params["sequence_length"],
            l2_reg_lambda=params["l2_reg_params"],            
        )
        
        loss = cnn.loss
        accuracy = cnn.accuracy
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        
        inti = tf.initialize_all_variables()
        with tf.Session() as sess:
            '''
            run the code 
            '''
            def train_op():
                _, score, loss_ = sess.run([train_op, accuracy, loss], feed_dict=None)                
            
            def dev_op():
                score, loss_ = sess.run([accuracy, loss], feed_dict=None)                

            def test_op():
                score, loss_ = sess.run([accuracy, loss], feed_dict=None)            
            
            sess.run(init)            
                    

if __name__ == "__main__":
    process()
    pass