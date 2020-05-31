import pandas as pd
import tensorflow as tf
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def load_dataset():
	dataset = pd.read_csv('pulsar_stars.csv').dropna()



	data = dataset[['Mean of the integrated profile','Standard deviation of the integrated profile','Excess kurtosis of the integrated profile','Skewness of the integrated profile','Mean of the DM-SNR curve','Standard deviation of the DM-SNR curve','Excess kurtosis of the DM-SNR curve','Skewness of the DM-SNR curve','target_class']]
	label = dataset[['target_class']]

	return data, label

def predict():
    wx1 = tf.matmul(x, weight['hidden'])
    wx1_bias = wx1 + bias['hidden']
    y1 = tf.nn.sigmoid(wx1_bias)
    
    wx2 = tf.matmul(y1, weight['output'])
    wx2_bias = wx2 + bias['output']
    y2 = tf.nn.sigmoid(wx2_bias)

    return y2




data, label = load_dataset()

scaler = MinMaxScaler()
scaler = scaler.fit(data)
data = scaler.transform(data)

data_train, data_test, label_train, label_test = train_test_split(data,label,test_size=0.2)


layer = {
	'node_input' : 8,
	'node_hidden' : 4,
	'node_output' : 2

}

weight = {
	'hidden' : tf.Variable(tf.random_normal([layer['node_input'],layer['node_hidden']])),
	'output' : tf.Variable(tf.random_normal([layer['node_hidden'],layer['node_output']]))
}

bias = {
	"hidden" : tf.Variable(tf.random_normal([layer['node_hidden']])),
    "output" : tf.Variable(tf.random_normal([layer['node_output']]))
}

x = tf.placeholder(tf.float32, [None, layer['node_input']])
y_true = tf.placeholder(tf.float32, [None, layer['node_output']])
y_predict = predict()

learning_rate = 0.5
number_epoch = 5000

loss = tf.reduce_mean((0.5 * (y_true - y_predict) ) **2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    arg = {x: data_train, y_true:label_train}

    for i in range(1,number_epoch+1):
        sess.run(training, feed_dict=arg)

        if i%200 ==0 :
            matches = tf.equal(tf.argmax(y_true,axis = 1),tf.argmax(y_predict,axis = 1))

            acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            print('Iteration : {} || loss : {}'.format(i,sess.run(loss, feed_dict={x: data_test ,y_true: label_test})))


    print ('Accuracy :{}' .format(sess.run(acc, feed_dict={y_true : label_test, x: data_test})))