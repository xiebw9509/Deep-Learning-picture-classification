import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
from PIL import Image
import os


def read_and_decode_tfrecord(filename):
    filename_deque = tf.train.string_input_producer(filename)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_deque)
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string)})
    label = tf.cast(features['label'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32) / 255.0        #将矩阵归一化0-1之间
    return img, label


x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y_ = tf.placeholder(tf.float32, [None])
train_list = []
batch_size_ = 50
class_path = "tf_file/"
for dir_name in os.listdir(class_path):
    if "train" in dir_name:
        train_list.append(os.path.join(class_path, dir_name))
print(train_list)
# 随机打乱顺序
img, label = read_and_decode_tfrecord(train_list)
img_batch, label_batch = tf.train.shuffle_batch([img, label], num_threads=2, batch_size=batch_size_, capacity=10000,
                                                min_after_dequeue=9900)
model_dir = "model-50/model-1000"    # 模型地址
result_dir = 'result/result.txt'     # 生成输出结果
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
classes = []
for i in range(50):
    classes.append(i)
print(classes)    # 标签顺序


# 将label值进行onehot编码
one_hot_labels = tf.one_hot(indices=tf.cast(y_, tf.int32), depth=50)
pred, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=50, is_training=True)
pred = tf.reshape(pred, shape=[-1, 50])

a = tf.argmax(pred, 1)
b = tf.argmax(one_hot_labels, 1)
correct_pred = tf.equal(a, b)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_dir)
    # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    # 启动QueueRunner,此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    while True:
        i += 1
        b_image, b_label = sess.run([img_batch, label_batch])
        acc_train = sess.run(accuracy, feed_dict={x: b_image, y_: b_label})
        print(acc_train)
    coord.request_stop()
    # 其他所有线程关闭之后，这一函数才能返回
    coord.join(threads)
