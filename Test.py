import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
from PIL import Image
import os
img_dir_path = "data/resize_train_data/"
classes = []
for i in os.listdir(img_dir_path):
    classes.append(i)

test_dir = r'./data/id_data'     # 原始的test文件夹，含带预测的图片
model_dir = r'./model-50/model-1000'     # 模型地址
result_dir = r'./result.txt'     # 生成输出结果
x = tf.placeholder(tf.float32, [None, 224, 224, 3])
pred, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=50, is_training=False)
pred = tf.reshape(pred, shape=[-1, 50])
a = tf.argmax(pred, 1)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_dir)
    for pic in os.listdir(test_dir):
        class_path = os.path.join(test_dir, pic)
        for img_file_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file_name)
            print(img_path)
            img = Image.open(img_path)
            img = img.resize((224, 224))
            img = tf.reshape(img, [1, 224, 224, 3])
            img1 = tf.reshape(img, [1, 224, 224, 3])
            img = tf.cast(img, tf.float32) / 255.0
            b_image, b_image_raw = sess.run([img, img1])
            t_label = sess.run(a, feed_dict={x: b_image})
            index_ = t_label[0]
            print(t_label)
            predict = classes[index_]
            with open(result_dir, 'a') as f1:
                if pic == predict:
                    print(pic, img_file_name, predict, file=f1)
            break
