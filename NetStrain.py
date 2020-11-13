import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
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


save_dir = "model/model"
batch_size_ = 2
lr = tf.Variable(0.0001, dtype=tf.float32)
x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y_ = tf.placeholder(tf.float32, [None])

train_list = []
class_path = "tf_file_small/"
for dir_name in os.listdir(class_path):
    if "train" in dir_name:
        train_list.append(os.path.join(class_path, dir_name))
# 随机打乱顺序
img, label = read_and_decode_tfrecord(train_list)
img_batch, label_batch = tf.train.shuffle_batch([img, label], num_threads=2, batch_size=batch_size_, capacity=10000,
                                                min_after_dequeue=9950)

# 将label值进行onehot编码
one_hot_labels = tf.one_hot(indices=tf.cast(y_, tf.int32), depth=50)
pred, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=50, is_training=True)
pred = tf.reshape(pred, shape=[-1, 50])

# 定义损失函数和优化器
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=one_hot_labels))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# 准确度
a = tf.argmax(pred, 1)
b = tf.argmax(one_hot_labels, 1)
correct_pred = tf.equal(a, b)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    # 启动QueueRunner,此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    while True:
        i += 1
        b_image, b_label = sess.run([img_batch, label_batch])
        if len(b_label) == 50:
            _, loss_, y_t, y_p, a_, b_ = sess.run([optimizer, loss, one_hot_labels, pred, a, b], feed_dict={x: b_image,
                                                                                                            y_: b_label})
            print('step: {}, train_loss: {}'.format(i, loss_))
            _loss, acc_train = sess.run([loss, accuracy], feed_dict={x: b_image, y_: b_label})
            print('--------------------------------------------------------')
            print('step: {}  train_acc: {}  loss: {}'.format(i, acc_train, _loss))
            print('--------------------------------------------------------')
            if i % 20 == 0:
                _loss, acc_train = sess.run([loss, accuracy], feed_dict={x: b_image, y_: b_label})
                print('--------------------------------------------------------')
                print('step: {}  train_acc: {}  loss: {}'.format(i, acc_train, _loss))
                print('--------------------------------------------------------')
                if i == 100:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 200:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 300:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 400:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 500:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 600:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 700:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 800:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 900:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 1000:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 1500:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 2000:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 2500:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 3000:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 4000:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 5000:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 6000:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 7000:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 8000:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 9000:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 10000:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 20000:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 30000:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 50000:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 60000:
                    saver.save(sess, save_dir, global_step=i)
                elif i == 100000:
                    saver.save(sess, save_dir, global_step=i)
                    break
    coord.request_stop()
    # 其他所有线程关闭之后，这一函数才能返回
    coord.join(threads)
