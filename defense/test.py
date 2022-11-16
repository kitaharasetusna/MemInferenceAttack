import tensorflow as tf

x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)
  y = x * x
  print(y)
dy_dx = g.gradient(y, x)
print(dy_dx)

