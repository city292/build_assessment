import keras.applications.resnet50
import shutil
from keras import backend as K
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.engine import Layer
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, UpSampling2D, Deconv2D
from keras.layers import Dense, Activation, Flatten, Conv2D, Dropout
from keras.layers import Input, Concatenate, GlobalAveragePooling2D, GlobalMaxPool2D
from keras.layers.merge import Add
from keras.models import Model
from keras.models import model_from_json
from keras.regularizers import L1L2, l2, l1
from keras.utils.generic_utils import get_custom_objects
from keras.utils.vis_utils import plot_model


class Swish(Layer):

    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        return K.sigmoid(inputs) * inputs

    def get_config(self):
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()))


get_custom_objects().update({'Swish': Swish})

# K.set_image_dim_ordering('th')
filters = ([16, 16, 32], [16, 16, 32], [16, 16, 32], [16, 16, 32], [64, 64, 128])
import os


# s = os.getenv('path')
# os.environ['path'] = s + r';C:\Program Files (x86)\Graphviz2.38\bin'


def abs_loss(y_true, y_pred):
    return K.sum(K.not_equal(y_true, y_pred))


def get_reg(w1=0.001, w2=0.001):
    if w1 is None and w2 is None:
        return None
    if w1 is None:
        return l2(l=w2)
    if w2 is None:
        return l1(l=w1)
    return L1L2(l1=w1, l2=w2)


def identity_block(x, nb_filter, kernel_size=3, stage=0, num=0, reg=None):
    k1, k2, k3 = nb_filter
    out = Convolution2D(k1, (1, 1), name="conv_s%d_n%d_b1" % (stage, num), kernel_regularizer=reg)(x)
    out = BatchNormalization(name="batch_s%d_n%d_b1" % (stage, num))(out)
    out = Swish(name='active_s%d_n%d-b1' % (stage, num))(out)

    out = Convolution2D(k2, (kernel_size, kernel_size), padding='same', name="conv_s%d_n%d_b2" % (stage, num),
                        kernel_regularizer=reg)(out)
    out = BatchNormalization(name="batch_s%d_n%d_b2" % (stage, num))(out)
    out = Swish(name='active_s%d_n%d-b2' % (stage, num))(out)

    out = Convolution2D(k3, (1, 1), name="conv_s%d_n%d_b3" % (stage, num), kernel_regularizer=reg)(out)
    out = BatchNormalization(name="batch_s%d_n%d_b3" % (stage, num))(out)

    out = Add(name='add_s%d_n%d' % (stage, num))([out, x])
    out = Swish(name='active_s%d_n%d-c' % (stage, num))(out)
    return out


def conv_block(x, nb_filter, kernel_size=3, stage=0, num=0, reg=None):
    k1, k2, k3 = nb_filter

    out = Convolution2D(k1, (1, 1), name="conv_s%d_n%d_b1" % (stage, num), kernel_regularizer=reg)(x)
    out = BatchNormalization(name="batch_s%d_n%d_b1" % (stage, num))(out)
    out = Swish(name='active_s%d_n%d-b1' % (stage, num))(out)

    out = Convolution2D(k2, (kernel_size, kernel_size), padding="same", name="conv_s%d_n%d_b2" % (stage, num),
                        kernel_regularizer=reg)(out)
    out = BatchNormalization(name="batch_s%d_n%d_b2" % (stage, num))(out)
    out = Swish(name='active_s%d_n%d-b2' % (stage, num))(out)

    out = Convolution2D(k3, (1, 1), name="conv_s%d_n%d_b3" % (stage, num), kernel_regularizer=reg)(out)
    out = BatchNormalization(name="batch_s%d_n%d_b3" % (stage, num))(out)

    x = Convolution2D(k3, (1, 1), name="conv_s%d_n%d_a" % (stage, num), kernel_regularizer=reg)(x)
    x = BatchNormalization(name="batch_s%d_n%d_a" % (stage, num))(x)

    out = Add(name='add_s%d_n%d' % (stage, num))([x, out])
    out = Swish(name='active_s%d_n%d-c' % (stage, num))(out)
    return out


def block(x, n, stage, reg=None):
    x = conv_block(x, filters[stage - 1], stage=stage, num=1, reg=reg)
    for i in range(n):
        x = identity_block(x, filters[stage - 1], stage=stage, num=2 + i, reg=reg)
    return x


def get_ae_model(channels=3, classes=4,
                 optimizer=optimizers.SGD(lr=0.1, decay=0.1, momentum=0.9, nesterov=True),
                 active='relu', used_dropout=None,
                 L1=None, L2=0.01):
    if K.image_dim_ordering() == 'tf':
        image_shape = (88, 88, channels)
    else:
        image_shape = (channels, 88, 88)

    input_img = Input(shape=image_shape)

    x = Convolution2D(12, (5, 5), kernel_regularizer=get_reg(L1, L2), name='conv1')(input_img)
    x = BatchNormalization()(x)
    x = Swish()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Convolution2D(32, (3, 3), kernel_regularizer=get_reg(L1, L2), name='conv2')(x)
    x = BatchNormalization()(x)
    x = Swish()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Convolution2D(32, (3, 3), kernel_regularizer=get_reg(L1, L2), name='conv3')(x)
    x = BatchNormalization()(x)
    x = Swish()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Convolution2D(32, (3, 3), kernel_regularizer=get_reg(L1, L2), name='conv4')(x)
    x = BatchNormalization()(x)
    x = Swish()(x)

    xd = x
    xd = Deconv2D(32, (3, 3), kernel_regularizer=get_reg(L1, L2), name='conv_d1')(xd)
    xd = BatchNormalization()(xd)
    xd = Swish()(xd)
    xd = UpSampling2D((2, 2))(xd)

    xd = Deconv2D(32, (3, 3), kernel_regularizer=get_reg(L1, L2), name='conv_d2')(xd)
    xd = BatchNormalization()(xd)
    xd = Swish()(xd)
    xd = UpSampling2D((2, 2))(xd)

    xd = Deconv2D(12, (3, 3), kernel_regularizer=get_reg(L1, L2), name='conv_d3')(xd)
    xd = BatchNormalization()(xd)
    xd = Swish()(xd)
    xd = UpSampling2D((2, 2))(xd)

    xd = Deconv2D(3, (5, 5), activation='sigmoid', kernel_regularizer=get_reg(L1, L2), name='ae_output')(xd)

    x = GlobalAveragePooling2D()(x)

    class_output = Dense(classes, activation='softmax', name='class_output')(x)

    res = Model(inputs=input_img, outputs=[class_output, xd])
    res.compile(loss={"class_output": "categorical_crossentropy", "ae_output": "mean_squared_error"},
                optimizer=optimizer,
                metrics={'class_output': 'accuracy'}
                #       , sample_weight_mode='temporal'
                )
    return res


def conv_fac(x, nb_filters, w1=None, w2=0.1, dropout_rate=None):
    x = Convolution2D(nb_filters, (3, 3), kernel_regularizer=get_reg(w1, w2), padding='same')(x)
    x = BatchNormalization(beta_regularizer=get_reg(w1, w2), gamma_regularizer=get_reg(w1, w2))(x)
    x = Swish()(x)

    return x


def get_densnet_model(channels=3, classes=4, str='',
                      optimizer=optimizers.SGD(lr=0.1, decay=0.1, momentum=0.9, nesterov=True),
                      active='relu', used_dropout=None,
                      L1=None, L2=0.001):
    if K.image_dim_ordering() == 'tf':
        image_shape = (88, 88, channels)
    else:
        image_shape = (channels, 88, 88)
    input_img = Input(shape=image_shape)
    nb_filter = 8

    x = input_img
    list_x = [x]
    growth_rate = 8
    for i in range(2):
        x = Convolution2D(4 * growth_rate, (1, 1), kernel_regularizer=get_reg(L1, L2), padding='same',
                          name='conv1-%da' % i)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(growth_rate, (3, 3), kernel_regularizer=get_reg(L1, L2), padding='same',
                          name='conv1-%db' % i)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        list_x.append(x)
        x = Concatenate()(list_x)

        nb_filter += growth_rate

    x = Convolution2D(16, (3, 3), kernel_regularizer=get_reg(L1, L2), padding='same', name='conv1-x')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((2, 2))(x)

    nb_filter = 8
    list_x = [x]
    growth_rate = 8
    for i in range(2):
        x = Convolution2D(4 * growth_rate, (1, 1), kernel_regularizer=get_reg(L1, L2), padding='same',
                          name='conv2-%da' % i)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(growth_rate, (3, 3), kernel_regularizer=get_reg(L1, L2), padding='same',
                          name='conv2-%db' % i)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        list_x.append(x)
        x = Concatenate()(list_x)
        nb_filter += growth_rate

    x = Convolution2D(32, (3, 3), kernel_regularizer=get_reg(L1, L2), padding='same', name='conv2-x')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((2, 2))(x)

    nb_filter = 8
    list_x = [x]
    growth_rate = 8
    for i in range(2):
        x = Convolution2D(4 * growth_rate, (1, 1), kernel_regularizer=get_reg(L1, L2), padding='same',
                          name='conv3-%da' % i)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(growth_rate, (3, 3), kernel_regularizer=get_reg(L1, L2), padding='same',
                          name='conv3-%db' % i)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        list_x.append(x)
        x = Concatenate()(list_x)
        nb_filter += growth_rate

    x = Convolution2D(32, (3, 3), kernel_regularizer=get_reg(L1, L2), padding='same', name='conv3-x')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((2, 2))(x)

    nb_filter = 8
    list_x = [x]
    growth_rate = 8
    for i in range(2):
        x = Convolution2D(4 * growth_rate, (1, 1), kernel_regularizer=get_reg(L1, L2), padding='same',
                          name='conv4-%da' % i)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(growth_rate, (3, 3), kernel_regularizer=get_reg(L1, L2), padding='same',
                          name='conv4-%db' % i)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        list_x.append(x)
        x = Concatenate()(list_x)
        nb_filter += growth_rate

    x = Convolution2D(32, (3, 3), kernel_regularizer=get_reg(L1, L2), padding='same', name='conv4-x')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)

    # x = MaxPooling2D((2, 2))(x)

    # x = Flatten()(x)
    x = Dense(256, kernel_regularizer=get_reg(L1, L2))(x)
    x = BatchNormalization()(x)

    y = Activation('relu')(x)

    if str.find('-ord-') >= 0:

        y0 = Dense(2, activation='softmax', name='class_output_0', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
        y1 = Dense(2, activation='softmax', name='class_output_01', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
        y2 = Dense(2, activation='softmax', name='class_output_012', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
        y = Concatenate()([y0, y1, y2])
        # y = Dense(classes, activation='softmax', name='class_output')(y)
        model = Model(outputs=y, inputs=input_img)
    else:

        y = Dense(classes, activation='softmax', name='class_output', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)

        model = Model(outputs=y, inputs=input_img)

    return model


def get_res_model(channels=3, classes=4,
                  optimizer=optimizers.SGD(lr=10, decay=0.01, momentum=0.9, nesterov=True),
                  active='relu',
                  L1=None, L2=0.001):
    # optimizer = optimizers.rmsprop(lr=0.1)

    #  print(K.image_dim_ordering())
    if K.image_dim_ordering() == 'tf':
        image_shape = (88, 88, channels)
    else:
        image_shape = (channels, 88, 88)
    input_img = Input(shape=image_shape)
    x = block(input_img, 1, stage=1, reg=get_reg(L1, L2))
    # x = Convolution2D(8, (3, 3), kernel_regularizer=get_reg(w1, w2))(input)
    x = MaxPooling2D((2, 2))(x)

    x = block(x, 1, stage=2, reg=get_reg(L1, L2))

    x = MaxPooling2D((2, 2))(x)
    # x = Convolution2D(64, (3, 3), strides=2, padding="valid")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    # x = LeakyReLU()(x)
    x = block(x, 1, stage=3, reg=get_reg(L1, L2))

    x = MaxPooling2D((2, 2))(x)
    # x = Convolution2D(128, (3, 3), strides=2, padding="valid")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    # x = LeakyReLU()(x)

    x = block(x, 2, stage=4, reg=get_reg(L1, L2))

    x = MaxPooling2D((2, 2))(x)
    # x = Convolution2D(32, (3, 3), strides=2, padding="valid")(x)

    # x = block(x, 1, stage=5, reg=get_reg(L1, L2))

    # x = MaxPooling2D((2, 2))(x)
    # x = Convolution2D(256, (3, 3), strides=2, padding="valid")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    # x = LeakyReLU()(x)
    x = Flatten()(x)
    x = Dense(64, kernel_regularizer=get_reg(L1, L2))(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    y = Swish()(x)
    y0 = Dense(2, activation='softmax', name='class_output_0', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
    y1 = Dense(2, activation='softmax', name='class_output_01', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
    y2 = Dense(2, activation='softmax', name='class_output_012', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
    y = Concatenate()([y0, y1, y2])

    # value_output = Dense(1, name='value_output')(x)

    class_output = Dense(classes, activation='softmax', name='class_output')(x)
    res = Model(outputs=y, inputs=input_img)

    return res


def get_model(channels=3, classes=3,
              optimizer=optimizers.SGD(lr=1, decay=0.001),
              active='relu',
              used_dropout=-1,
              str='',
              init='he_normal', L1=None, L2=0.01):
    # optimizer = optimizers.Adadelta()

    print("creating model...")

    if K.image_dim_ordering() == 'tf':
        image_shape = (88, 88, channels)
    else:
        image_shape = (channels, 88, 88)

    print('channels-%s' % channels)
    input_x = Input(shape=image_shape)
    y = Conv2D(16, (3, 3), padding='same')(input_x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = MaxPooling2D((2, 2))(y)

    y = Conv2D(32, (3, 3), padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = MaxPooling2D((2, 2))(y)

    y = Conv2D(32, (3, 3), padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = MaxPooling2D((2, 2))(y)
    y = Conv2D(32, (3, 3))(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = MaxPooling2D((2, 2))(y)
    y = Flatten()(y)
    y = Dense(64)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    if str.find('-ord-') >= 0:

        y0 = Dense(2, activation='softmax', name='class_output_0', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
        y1 = Dense(2, activation='softmax', name='class_output_01', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
        y2 = Dense(2, activation='softmax', name='class_output_012', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
        y = Concatenate()([y0, y1, y2])
        # y = Dense(classes, activation='softmax', name='class_output')(y)
        model = Model(input_x, y)
    else:

        y = Dense(classes, activation='softmax', name='class_output', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)

        model = Model(input_x, y)

    return model


def get_vgg4_model(channels=3, classes=3,
                   optimizer=optimizers.SGD(lr=1, decay=0.001),
                   active='relu',
                   used_dropout=-1,
                   str='',
                   init='he_normal', L1=0.001, L2=0.001):
    if K.image_dim_ordering() == 'tf':
        image_shape = (88, 88, channels)
    else:
        image_shape = (channels, 88, 88)

    model = VGG16(classes=classes, input_shape=image_shape, include_top=False)

    for lay in model.layers:
        lay.trainable = False
    y = Conv2D(128, (3, 3), name='block5_conv1', activation='relu', kernel_regularizer=get_reg(w1=L1, w2=L2))(
        model.get_layer(name='block3_pool').output)

    y = Conv2D(128, (3, 3), name='block5_conv2', activation='relu', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)

    y = Conv2D(128, (3, 3), name='block5_conv3', activation='relu', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
    y = Conv2D(128, (3, 3), name='block5_conv4', activation='relu', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)

    y = Flatten()(y)
    y = Dense(128, activation='relu', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)

    if str.find('-ord-') >= 0:

        y0 = Dense(2, activation='softmax', name='class_output_0', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
        y1 = Dense(2, activation='softmax', name='class_output_01', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
        y2 = Dense(2, activation='softmax', name='class_output_012', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
        y = Concatenate()([y0, y1, y2])
        # y = Dense(classes, activation='softmax', name='class_output')(y)
        model = Model(model.input, y)
    else:

        y = Dense(classes, activation='softmax', name='class_output', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)

        model = Model(model.input, y)
    for lay in model.layers:
        lay.trainable = True

    # y = (model.get_layer(name='dropout_2').output)
    #
    #
    # y0 = Dense(2, activation='softmax', name='class_output_0', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
    # y1 = Dense(2, activation='softmax', name='class_output_01', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
    # y2 = Dense(2, activation='softmax', name='class_output_012', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
    # y = Concatenate()([y0, y1, y2])
    # # y = Dense(classes, activation='softmax', name='class_output')(y)
    # model = Model(model.input, y)
    return model


def get_vgg_model(channels=3, classes=3,
                  optimizer=optimizers.SGD(lr=1, decay=0.001),
                  active='relu',
                  used_dropout=-1,
                  str='',
                  init='he_normal', L1=0.001, L2=0.001):
    model = load_model_from_json(
        '/home/city/log/builds/19-02-16-7833926-sgd.01-ord-vggtune-L2e2-b0.15-d0d1d2d3-0-ok-good-0.7765/model.json')
    model.load_weights(
        '/home/city/log/builds/19-02-16-7833926-sgd.01-ord-vggtune-L2e2-b0.15-d0d1d2d3-0-ok-good-0.7765/val_ord_acc/val_ord_acc-049-0.7765.hdf5')

    for lay in model.layers:
        lay.trainable = True

    # y = (model.get_layer(name='dropout_2').output)
    #
    #
    # y0 = Dense(2, activation='softmax', name='class_output_0', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
    # y1 = Dense(2, activation='softmax', name='class_output_01', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
    # y2 = Dense(2, activation='softmax', name='class_output_012', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
    # y = Concatenate()([y0, y1, y2])
    # # y = Dense(classes, activation='softmax', name='class_output')(y)
    # model = Model(model.input, y)
    return model


def get_ntune_model(channels=3, classes=3,
                    optimizer=optimizers.SGD(lr=1, decay=0.001),
                    active='relu',
                    used_dropout=-1,
                    str='',
                    init='he_normal', L1=0.001, L2=0.001):
    model = load_model_from_json(
        '/home/city/log/builds/19-02-16-7833926-sgd.01-ord-vggtune-L2e2-b0.15-d0d1d2d3-0-ok-good-0.7765/model.json')
    model.load_weights(
        '/home/city/log/builds/19-02-16-7833926-sgd.01-ord-vggtune-L2e2-b0.15-d0d1d2d3-0-ok-good-0.7765/val_ord_acc/val_ord_acc-049-0.7765.hdf5')

    for lay in model.layers:
        lay.trainable = False
    y = (model.get_layer(name='global_max_pooling2d_1').output)
    y = Dense(256, name='dense_1', kernel_regularizer=get_reg(w1=L1, w2=L2), activation='relu')(y)
    # y = Dropout(0.5)(y)
    y = Dense(256, name='dense_2', kernel_regularizer=get_reg(w1=L1, w2=L2), activation='relu')(y)

    y0 = Dense(2, activation='softmax', name='class_output_0', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
    y1 = Dense(2, activation='softmax', name='class_output_01', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
    y2 = Dense(2, activation='softmax', name='class_output_012', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
    y = Concatenate()([y0, y1, y2])
    # y = Dense(classes, activation='softmax', name='class_output')(y)
    model = Model(model.input, y)
    return model


def get_good_vgg_tune_model(channels=3, classes=3,
                            optimizer=optimizers.SGD(lr=1, decay=0.001),
                            active='relu',
                            used_dropout=-1,
                            str='',
                            init='he_normal', L1=0.001, L2=0.001):
    if K.image_dim_ordering() == 'tf':
        image_shape = (88, 88, channels)
    else:
        image_shape = (channels, 88, 88)
    model = VGG16(classes=classes, input_shape=image_shape, include_top=False)

    for lay in model.layers:
        lay.trainable = False
    y = GlobalMaxPool2D()(model.get_layer(name='block4_pool').output)
    y = Dense(256, name='dense_1', kernel_regularizer=get_reg(w1=L1, w2=L2), activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(256, name='dense_2', kernel_regularizer=get_reg(w1=L1, w2=L2), activation='relu')(y)
    y = Dropout(0.5)(y)

    if str.find('-ord-') >= 0:

        y0 = Dense(2, activation='softmax', name='class_output_0', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
        y1 = Dense(2, activation='softmax', name='class_output_01', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
        y2 = Dense(2, activation='softmax', name='class_output_012', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
        y = Concatenate()([y0, y1, y2])
        # y = Dense(classes, activation='softmax', name='class_output')(y)
        model = Model(model.input, y)
    else:

        y = Dense(classes, activation='softmax', name='class_output', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)

        model = Model(model.input, y)

    return model


def get_tune_model(channels=3, classes=3,
                   optimizer=optimizers.SGD(lr=1, decay=0.001),
                   active='relu',
                   used_dropout=-1,
                   str='',
                   init='he_normal', L1=0.001, L2=0.001):
    if K.image_dim_ordering() == 'tf':
        image_shape = (88, 88, channels)
    else:
        image_shape = (channels, 88, 88)
    model = VGG16(classes=classes, input_shape=image_shape, include_top=False)

    for lay in model.layers:
        lay.trainable = False
    y = Flatten()(model.get_layer(name='block5_pool').output)
    y = Dense(256, name='dense_1', kernel_regularizer=get_reg(w1=L1, w2=L2), activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(256, name='dense_2', kernel_regularizer=get_reg(w1=L1, w2=L2), activation='relu')(y)
    y = Dropout(0.5)(y)

    if str.find('-ord-') >= 0:

        y0 = Dense(2, activation='softmax', name='class_output_0', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
        y1 = Dense(2, activation='softmax', name='class_output_01', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
        y2 = Dense(2, activation='softmax', name='class_output_012', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
        y = Concatenate()([y0, y1, y2])
        # y = Dense(classes, activation='softmax', name='class_output')(y)
        model = Model(model.input, y)
    else:

        y = Dense(classes, activation='softmax', name='class_output', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)

        model = Model(model.input, y)

    return model


def get_res50_net(channels=3, classes=3,
                  optimizer=optimizers.SGD(lr=1, decay=0.001),
                  active='relu',
                  used_dropout=-1,
                  init='he_normal', L1=0.001, L2=0.001):
    if K.image_dim_ordering() == 'tf':
        image_shape = (88, 88, channels)
    else:
        image_shape = (channels, 88, 88)

    model = keras.applications.resnet50.ResNet50(include_top=False, input_shape=(88, 88, 3))
    model.summary()
    for lay in model.layers:
        lay.trainable = False
    y = Flatten()(model.get_layer(name='activation_49').output)
    y = Dense(128)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y0 = Dense(2, activation='softmax', name='class_output_0', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
    y1 = Dense(2, activation='softmax', name='class_output_01', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
    y2 = Dense(2, activation='softmax', name='class_output_012', kernel_regularizer=get_reg(w1=L1, w2=L2))(y)
    y = Concatenate()([y0, y1, y2])
    # y = Dense(classes, activation='softmax', name='class_output')(y)
    model = Model(model.input, y)
    return model


def load_model_from_json(filename):
    json_file = open(filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)

    # loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return loaded_model


def add_layer_after(model, stage, num):
    for layer in model.layers:
        layer.trainable = False
        if layer.name == 'active_%d_%d-c' % (stage, num):
            print(layer)
            x = identity_block(layer.output, filters[1], stage=2, num=1)
            tmpm = Model(layer.input, model.layers[-1].output)
            end = tmpm(x)
            return Model(model.input, end)


if __name__ == '__main__':
    # model=get_model(channels=3, classes=4)
    # model=get_res50_net(channels=3, classes=4)
    model = get_vgg_model(channels=3, classes=4, str='-ord-')
    model.summary()
    exit()
    # model = get_densnet_model(channels=3, classes=4)
    # model = get_ae_model(channels=3,classes=3)
    # model = get_model(channels=3, classes=2)
    # model = get_tune_model(channels=3, classes=4)
    # model.summary()
    plot_model(model, '../png/%d.png' % model.count_params(), show_layer_names=True, show_shapes=True)

    try:
        plot_model(model, '../png/%d.png' % model.count_params(), show_layer_names=True, show_shapes=True)
        shutil.copyfile(os.getcwd() + '/build_model.py',
                        os.getcwd() + '/../png/build_model-%d.py' % model.count_params())
    except:
        print('plot fail')
