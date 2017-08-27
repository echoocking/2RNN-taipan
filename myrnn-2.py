import numpy as np
import tensorflow as tf
import mydata
import random

from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple

tf.reset_default_graph()
sess = tf.InteractiveSession()

PAD = 0
EOS = 1

#data
data_size = len(mydata.r[0])
print ("Data size = %d" % data_size)
test_size = 178
print ("Test size = %d" % test_size)
train_size =  data_size - test_size

#embedding settings
vocab_size = len(mydata.dict)
embedding_size = 100
max_sentence_length = 10

n_hidden_1 = 512
n_hidden_2 = 512
n_classes = 2


# Parameters
learning_rate = 0.0001
training_epochs = 200
batch_size = 40
display_step = 1


encoder_hidden_units = 256

#Variables
embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, embedding_size],-1,1),name="W")

x_shrt_term = tf.placeholder(tf.int32, [None,  7, max_sentence_length])
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')

stock_prices = tf.placeholder(tf.float32, [None,  7, 1])

lookup_shrt = tf.nn.embedding_lookup(embedding_matrix, x_shrt_term)
ff_input = tf.reduce_mean(lookup_shrt, 2) 
#ff_input = tf.reshape(avg, [-1 , embedding_size * max_sentence_length])
print (ff_input)
#embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)



#encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
#print (encoder_inputs_embedded)

encoder_cell = LSTMCell(encoder_hidden_units)

((encoder_fw_outputs,
  encoder_bw_outputs),
 (encoder_fw_final_state,
  encoder_bw_final_state)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=ff_input,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32, time_major=False, scope="RNN1")
    )
    

encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

encoder_final_state_c = tf.concat(
    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

encoder_final_state_h = tf.concat(
    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
    


encoder_cell2 = LSTMCell(encoder_hidden_units)

((encoder_fw_outputs2,
  encoder_bw_outputs2),
 (encoder_fw_final_state2,
  encoder_bw_final_state2)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell2,
                                    cell_bw=encoder_cell2,
                                    inputs=stock_prices,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32, time_major=False, scope="RNN2")
    )
    

encoder_outputs2 = tf.concat((encoder_fw_outputs2, encoder_bw_outputs2), 2)

encoder_final_state_c2 = tf.concat(
    (encoder_fw_final_state2.c, encoder_bw_final_state2.c), 1)

encoder_final_state_h2 = tf.concat(
    (encoder_fw_final_state2.h, encoder_bw_final_state2.h), 1)    
    

y = tf.placeholder(tf.float32, [None, n_classes])


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_uniform([encoder_hidden_units*4, n_hidden_1],-1,1)),
    'h2': tf.Variable(tf.random_uniform([n_hidden_1, n_hidden_2],-1,1)),
    'out': tf.Variable(tf.random_uniform([n_hidden_2, n_classes],-1,1))
}
biases = {
    'b1': tf.Variable(tf.random_uniform([n_hidden_1],-1,1)),
    'b2': tf.Variable(tf.random_uniform([n_hidden_2],-1,1)),
    'out': tf.Variable(tf.random_uniform([n_classes],-1,1))
}
    
#model
def multilayer_perceptron(x, weights, biases):

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	
    return out_layer    

concat = tf.concat([encoder_final_state_c, encoder_final_state_c2], 1)
    
pred = multilayer_perceptron(concat, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # sample_data = [ [[1,2,3,4,5],[2,7,10,15,17]], [[1,2,3,4,5],[2,7,10,15,17]]]
    # res = sess.run(lookup_shrt, feed_dict={x_shrt_term: sample_data})
    # print (res)
    # print ("---------")
    # res = sess.run(avg, feed_dict={x_shrt_term: sample_data})
    # print (res)
    # print ("---------")
    # res = sess.run(ff_input, feed_dict={x_shrt_term: sample_data})
    # print (res)
    
    
    train_x = mydata.r[0][:train_size]
    train_p = mydata.r[1][:train_size]
    train_p = np.reshape(train_p, (-1, 7, 1))
    train_y = mydata.r[2][:train_size]



    test_x = mydata.r[0][-test_size:]
    test_p = mydata.r[1][-test_size:]
    test_p = np.reshape(test_p, (-1, 7, 1))
    test_y = mydata.r[2][-test_size:]
    #training_epochs = 1
    total_batch = train_size//batch_size
    for epoch in range(training_epochs):
        
        avg_cost = 0.
        seq = [ k for k in range (total_batch)]
        random.shuffle(seq)
        for i in range(total_batch):
            batch_x = train_x[seq[i]:seq[i]+batch_size]
            batch_p = train_p[seq[i]:seq[i]+batch_size]
            batch_y = train_y[seq[i]:seq[i]+batch_size]
            len_x   = [7]*batch_size
            #print (sess.run(concat, feed_dict={x_shrt_term: batch_x,stock_prices: batch_p,                                             encoder_inputs_length: len_x,
            #                                            y: batch_y}))
            
            #print (len(sess.run(ff_input, {x_shrt_term: batch_x})[0][0]))
            #break
            
            _, c = sess.run([optimizer, cost], feed_dict={x_shrt_term: batch_x,stock_prices: batch_p,                                             encoder_inputs_length: len_x,
                                                          y: batch_y})
            avg_cost += c / total_batch
        print (avg_cost)
        len_test = [7]*test_size
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Epoch %d  -- Accuracy: %f", (epoch,accuracy.eval({x_shrt_term: test_x,stock_prices: test_p, encoder_inputs_length:len_test, y: test_y})))