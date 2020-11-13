import os
from PIL import Image
import tensorflow as tf
import shutil
# 图片路径，两组标签都在该目录下
img_dir_path = "data/resize_train_data/"
# 验证集
id_img_dir_path = "data/id_data/"
# tfrecord文件保存路径
file_path = "tf_file"

# 每个tfrecord存放图片个数
bestnum = 500

# 第几个图片
num = 0

# 第几个TFRecord文件
recordfilenum = 0

# 将labels放入到classes中
classes = []
for i in os.listdir(img_dir_path):
    classes.append(i)
print(classes)

# tfrecords格式文件名
ftrecordfilename = ("train_data_game.tfrecords-%.3d" % recordfilenum)
if not os.path.exists(file_path):
    os.makedirs(file_path)
writer = tf.python_io.TFRecordWriter(os.path.join(file_path, ftrecordfilename))
for index, name in enumerate(classes):
    class_path = os.path.join(img_dir_path, name)
    for img_name in os.listdir(class_path):
        num = num + 1
        if num > bestnum:  # 超过500，写入下一个tfrecord
            num = 1
            recordfilenum += 1
            ftrecordfilename = ("train_data_game.tfrecords-%.3d" % recordfilenum)
            writer = tf.python_io.TFRecordWriter(os.path.join(file_path, ftrecordfilename))

        print(num)
        img_path = os.path.join(class_path, img_name)  # 每一个图片的地址
        id_path = os.path.join(id_img_dir_path, name)
        if not os.path.exists(id_path):
            os.makedirs(id_path)
        if num % 8 == 0:
            print(num)
            shutil.move(img_path, os.path.join(id_path, img_name))
        else:
            img = Image.open(img_path, 'r')
            img_raw = img.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                }))
            writer.write(example.SerializeToString())  # 序列化为字符串
writer.close()