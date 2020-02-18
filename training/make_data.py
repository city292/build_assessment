import os

import gdal
import numpy
from PIL import ImageEnhance
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import array_to_img, img_to_array
from keras.utils.np_utils import to_categorical


def load_data(
        data_dir=None,
        classes_num=None,
        bands_num=3,
        class_label=None,
        regularizer=True
):
    if class_label is None:
        class_label = [0, 1, 2, 3]
    if classes_num is None:
        classes_num = [0, 1, 2, 3]
    pwd = os.getcwd()

    if data_dir is None:
        data_dir = pwd + '/data/'
        if not os.path.exists(data_dir):
            data_dir = pwd + '/../../../data/'
        if not os.path.exists(data_dir):
            data_dir = pwd + '/../data/'

    # print(data_dir)

    width = 88
    heigh = 88
    gdal.AllRegister()
    cnt_class = numpy.ndarray([len(classes_num)], dtype=int)
    cnt_class[:] = 0

    val_cnt_class = numpy.ndarray([len(classes_num)], dtype=int)
    val_cnt_class[:] = 0

    for j in range(len(classes_num)):
        tem_j = '%d' % classes_num[j]
        l_dir = data_dir + 'train/' + tem_j
        cnt_class[j] = len(os.listdir(l_dir))
        val_l_dir = data_dir + 'val/' + tem_j
        val_cnt_class[j] = len(os.listdir(val_l_dir))

    cnt = sum(cnt_class)
    val_cnt = sum(val_cnt_class)
    print(cnt_class, val_cnt_class)

    x_test = numpy.ndarray([cnt, bands_num, width, heigh], dtype=int)
    y_test = numpy.ndarray([cnt], dtype=int)

    x_val = numpy.ndarray([val_cnt, bands_num, width, heigh], dtype=int)
    y_val = numpy.ndarray([val_cnt], dtype=int)
    index = 0
    val_index = 0
    for j in range(len(classes_num)):
        tem_j = '%d' % classes_num[j]
        l_dir = data_dir + 'train/' + tem_j

        val_l_dir = data_dir + 'val/' + tem_j
        for fname in os.listdir(l_dir):
            if not fname.endswith('.tif'):
                print(l_dir, fname)
                continue
            #       print(os.path.join(l_dir,fname))
            p = gdal.Open(os.path.join(l_dir, fname), gdal.GA_ReadOnly)
            oo = p.ReadAsArray()
            x_test[index][:] = oo[:]
            y_test[index] = int(class_label[classes_num[j]])

            index += 1
        for fname in os.listdir(val_l_dir):
            if not fname.endswith('.tif'):
                print(val_l_dir, fname)
                continue
            #       print(os.path.join(l_dir,fname))
            p = gdal.Open(os.path.join(val_l_dir, fname), gdal.GA_ReadOnly)
            oo = p.ReadAsArray()
            x_val[val_index][:] = oo[:]
            y_val[val_index] = int(class_label[classes_num[j]])

            val_index += 1
            #   return

    # test_x /255

    y_test = to_categorical(y_test, num_classes=max(class_label) + 1)
    y_val = to_categorical(y_val, num_classes=max(class_label) + 1)
    if regularizer:
        x_test = x_test.astype('float32')
        x_test = (x_test) / 256
        x_val = x_val.astype('float32')
        x_val = (x_val) / 256
    val = (x_val, y_val)
    # print(x_test.shape,y_test.shape,y_val.shape,x_val.shape)
    return (x_test, y_test, val)


class myMultiOutByGen(Iterator):
    def __init__(self, n, shuffle, seed, gen=None, batch_size=32):
        super().__init__(n, batch_size, shuffle, seed)
        self.gen = gen

    def next(self):
        x, y = next(self.gen)
        #     y2 = y.argmax(axis=-1).reshape([-1,1])
        return x, [y, x]


class myBrightOutByGen():
    def __init__(self, gen=None, batch_size=32, brightness_range=None):
        self.gen = gen
        self.brightness_range = brightness_range

    def __iter__(self):
        return self

    def __next__(self):
        x, y = next(self.gen)
        if self.brightness_range is not None:
            for i in range(x.shape[0]):
                tx = x[i]

                #     print(tx.shape)
                #    print(tx.shape)
                tx = tx * 256
                #     print(tx[0,0],tx[1,1])
                #    tx = tx.astype('uint8')
                #   tx = random_brightness(tx, self.brightness_range) / 256

                tx = array_to_img(tx, scale=False)
                # help(x)
                imgenhancer_Brightness = ImageEnhance.Brightness(tx)
                u = numpy.random.uniform(self.brightness_range[0], self.brightness_range[1])

                tx = imgenhancer_Brightness.enhance(u)
                tx = img_to_array(tx) / 256

                x[i][:] = tx[:]

        return x, y


def categorical2ord(y):
    ty = y.argmax(axis=-1)
    classes = numpy.max(ty)

    res = numpy.zeros([y.shape[0], (classes) * 2], numpy.int8)

    yn = numpy.zeros([y.shape[0]], numpy.int8)
    for i in range(classes):
        yn[:] = 1
        yn[numpy.where(ty <= i)] = 0
        res[:, 2 * i:2 * i + 2] = to_categorical(yn)

    return res


if __name__ == '__main__':
    test_x, test_y, val = load_data(
        bands_num=3,
        classes_num=[0, 1, 2, 3],
        class_label=[0, 1, 2, 3]
    )
    categorical2ord(to_categorical([0, 1, 2, 3]))
    categorical2ord(test_y)
    exit()
    x, y = val
    print(y)

    test_x = test_x.transpose([0, 2, 3, 1])
    genor = ImageDataGenerator(horizontal_flip=True,
                               vertical_flip=True,
                               #     rotation_range=5,
                               #               brightness_range=[1, 1]

                               )
    next(myBrightOutByGen(genor.flow(test_x, test_y, batch_size=32), brightness_range=[0.75, 1.25]))
    next(myBrightOutByGen(genor.flow(test_x, test_y, batch_size=32), brightness_range=[0.75, 1.25]))
