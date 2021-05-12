import tensorflow as tf

tensor1 = tf.constant([3,4,5])
tensor2= tf.constant([6,7,8])
tensor3= tf.constant([[1,2],
                       [3,4]])

                   

'''
tf.add()
tf.subtract()
tf.divide()
tf.multiply()
'''

'''
matrix
tf.matmul()
'''

tensor4 = tf.zeros(10)
tensor5 = tf.zeros([2,2]) #shape 2 x 2
tensor6 = tf.zeros([2,2,3]) #shape
print(tensor5)
print(tensor6)

#variable
w = tf.Variable(1.0)
print(w)
print(w.numpy())
w.assign(2)
