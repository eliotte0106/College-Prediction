import tensorflow as tf

#Linear Regression
'''
shoe size expectation by height
'''
height = 170
size = 260
a = tf.Variable(0.1)
b = tf.Variable(0.2)

def loss_function():
    expect = height * a + b
    return tf.square(260 - expect)

opt = tf.keras.optimizers.Adam(learning_rate = 0.1) # helper
for i in range(300):
    opt.minimize(loss_function, var_list=[a,b]) #lossfuntion, var_list
    print(a.numpy(),b.numpy())
#y = ax + b
#size = height * a + b
