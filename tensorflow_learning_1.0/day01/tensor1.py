import tensorflow as tf
# 去除CPU警告
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 实现一个加法运算
a = tf.constant(10.0)   # 为tensor类型，是OP操作 Tensor("Const:0", shape=(), dtype=float32)
b = tf.constant(20.0)



c = tf.add(a, b) # 为tensor类型，是OP操作
print(a)

# 开启一个会话
with tf.Session() as sess:
    c_res = sess.run(c)
    # 增加一个TensorBoard的可视化，将图运算可视化出来。
    filewriter = tf.summary.FileWriter("./tmp/summary", graph=sess.graph)
    print(c_res)

# 在运行的时候填充数据
plt_a = tf.placeholder(dtype=tf.float32)
plt_b = tf.placeholder(dtype=tf.float32)

with tf.Session() as sess:
    c_res = sess.run(c,feed_dict={plt_a:10,plt_b:20})
    # 增加一个TensorBoard的可视化，将图运算可视化出来。
    # filewriter = tf.summary.FileWriter("./tmp/summary", graph=sess.graph)
    print(c_res)