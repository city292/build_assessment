import os

# sys.path.append(r'C:\\Users\\citia\\PycharmProjects\\build_assessment')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = 1

from keras.preprocessing.image import ImageDataGenerator
from training.myCallback import MyModelCheckpoint, MyCSVLogger, CheckTrainOverFit, MyEarlyStopping
from training.build_model import *

from training.make_data import load_data, myBrightOutByGen, categorical2ord
import datetime
import shutil
import keras.utils
from training.data_parameter import *
from keras.utils.multi_gpu_utils import multi_gpu_model
import tensorflow
from keras.backend.tensorflow_backend import set_session
from training.myloss import *
import numpy as np

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tensorflow.Session(config=config))

K.set_image_dim_ordering('tf')
# pwd = os.getcwd()


pwd = '/home/city/disk/log'
# pwd = 'c:\log'
brightrange = [1, 1]
print(__file__)


def get_dir(n_params=None, opt=None, classes=None, words=None, times=0):
    now = datetime.datetime.now()
    day = now.strftime('%y-%m-%d')
    log_dir = '3'

    if n_params is None:
        s_params = ''
    else:
        s_params = '-%d' % n_params
    if classes is None:
        n_class = ''
    else:
        n_class = '-c%d' % classes
    if words is None:
        words = ''
    else:
        words = '-%s' % words
    if opt is None:
        opt = ''
    else:
        opt = '-%s' % opt
    if not os.path.exists(pwd + '/builds-ld'):
        os.mkdir(pwd + '/builds-ld')
    cnt = 0
    for name in os.listdir(pwd + '/builds-ld/'):
        if name.find(s_params + opt + n_class + words) >= 0 and name.find('-ok') >= 0:
            cnt += 1
        if cnt > times:
            return None
    for i in range(cnt, 1000):
        tem = '%d' % i
        log_dir = pwd + '/builds-ld/' + day + s_params + opt + n_class + words + '-' + tem + '/'

        if os.path.exists(log_dir):
            continue
        os.mkdir(log_dir)
        os.mkdir(log_dir + '/py')
        os.mkdir(log_dir + '/epoch')
        os.mkdir(log_dir + '/graph')
        break
    return log_dir


def back_py(filedir):
    if not os.path.exists(os.path.join(filedir, 'py')):
        os.mkdir(os.path.join(filedir, 'py'))
    shutil.copyfile(os.path.dirname(__file__) + '/build_model.py',
                    filedir + '/py/build_model.py')
    shutil.copyfile(os.path.dirname(__file__) + '/make_data.py',
                    filedir + '/py/make_data.py')
    shutil.copyfile(os.path.dirname(__file__) + '/training_model.py',
                    filedir + '/py/training_model.py')
    shutil.copyfile(os.path.dirname(__file__) + '/myCallback.py',
                    filedir + '/py/myCallback.py')
    shutil.copyfile(os.path.dirname(__file__) + '/data_parameter.py',
                    filedir + '/py/data_parameter.py')
    shutil.copyfile(os.path.dirname(__file__) + '/myloss.py',
                    filedir + '/py/myloss.py')
    shutil.copyfile(os.path.dirname(__file__) + '/transfer_train.py',
                    filedir + '/py/transfer_train.py')


def train_with_generator(opt='adamax', classes_type=0, op=None, n_epoch=100, dsize=-1, iters=0
                         ):
    for dir in os.listdir(os.path.join(pwd, 'builds-ld')):
        if dir.find('-ok') < 0:
            print(os.path.join(os.path.join(pwd, 'builds-ld'), dir))
            shutil.rmtree(os.path.join(os.path.join(pwd, 'builds-ld'), dir))

    # model = get_good_vgg_tune_model(channels=3, classes=max(classes_labels_dict[classes_type]) + 1, optimizer=op, L2=L2,
    #                                 str=opt)
    model = get_vgg4_model(channels=3, classes=max(classes_labels_dict[classes_type]) + 1, optimizer=op, L2=L2,
                           str=opt)
    filedir = None
    # model = load_model_from_json(modelfile)
    flag = False
    for times in range(iters):
        for i in range(8400, 99, -100):

            for ds in ['lu', ]:

                dsize = i
                op_str = '%s%s-ds%04d-d0d1d2d3' % (opt, ds, dsize)
                if i > 2500 and op_str.find('-yu-') >= 0:
                    continue
                filedir = get_dir(n_params=model.count_params(), opt=op_str, times=times)
                if filedir is None:
                    continue
                else:
                    flag = True
                    opt = op_str
                    break
            if flag:
                break
        if flag:
            break

    if filedir is None:
        return
    print(filedir, dsize)
    back_py(filedir)
    if gpus > 1:
        model = multi_gpu_model(model=model, gpus=gpus)
    if opt.find('-ord-') >= 0:
        loss_f = ordinal_regression_loss
        myMetrics = [ord_acc]
    else:
        loss_f = 'categorical_crossentropy'
        myMetrics = ['accuracy']
    model.compile(loss=loss_f,
                  optimizer=op,
                  metrics=myMetrics
                  # ,sample_weight_mode='temporal'
                  )
    # for i in range(len(classes_nums_dict[classes_type])):
    #     print('class %d is labeled by %d' % (classes_nums_dict[classes_type][i],classes_labels_dict[classes_type][classes_nums_dict[classes_type][i]]))
    if opt.find('-lu-') >= 0:
        test_x, test_y, val = load_data(
            bands_num=3,
            classes_num=classes_nums_dict[classes_type],
            class_label=classes_labels_dict[classes_type]
        )
    else:
        test_x, test_y, val = load_data(data_dir='/home/city/PycharmProjects/build_assessment/data/yushu-val/',
                                        bands_num=3,
                                        classes_num=classes_nums_dict[classes_type],
                                        class_label=classes_labels_dict[classes_type]
                                        )

    test_x = test_x.transpose([0, 2, 3, 1])

    arr = np.arange(test_x.shape[0])
    np.random.shuffle(arr)
    new_x = np.zeros([dsize, test_x.shape[1], test_x.shape[2], test_x.shape[3]])
    new_y = np.zeros([dsize, test_y.shape[1]])
    if dsize > test_x.shape[0]:
        new_x = None
    elif dsize > 500 * 4:

        for i in range(dsize):
            new_x[i, :] = test_x[arr[i], :]
            new_y[i, :] = test_y[arr[i], :]

        test_y = new_y
        test_x = new_x
    else:
        d_cnt = np.zeros([4], np.int32)
        d_cnt[:] = dsize / 4
        ind_sample = 0
        for j in range(4):
            ind = 0
            for i in range(d_cnt[j]):
                while True:
                    if test_y[arr[ind], j] == 1:

                        break
                    else:
                        ind += 1
                new_x[ind_sample, :] = test_x[arr[ind], :]
                new_y[ind_sample, :] = test_y[arr[ind], :]
                ind += 1
                ind_sample += 1
        test_y = new_y
        test_x = new_x

    print(test_x.shape, test_y.shape)
    val = (val[0].transpose([0, 2, 3, 1]), val[1])
    genor = ImageDataGenerator(horizontal_flip=True,
                               vertical_flip=True
                               )

    try:
        keras.utils.plot_model(model, filedir + 'model.png', show_layer_names=True, show_shapes=True)
    except:
        pass
    model_json = model.to_json()
    with open(filedir + 'model.json', "w") as json_file:
        json_file.write(model_json)

    nb_worker = 1

    csv_logger = MyCSVLogger(filedir + 'log_file.csv')

    callbacks = [csv_logger]

    callbacks.append(MyEarlyStopping(monitor='val_loss', patience=100, min_delta=0.00001, mode='min', baseline=2.))

    if opt.find('-ord-') >= 0:
        callbacks.append(
            MyEarlyStopping(monitor='val_ord_acc', patience=200, min_delta=0.00001, mode='max', verbose=1))

        callbacks.append(CheckTrainOverFit(monitor1='ord_acc', monitor2='val_ord_acc', patience=80))

        metrics = ['loss', 'ord_acc', 'val_loss'
                   ]
        metric_call = MyModelCheckpoint(
            filedir + 'val_ord_acc/val_ord_acc-{epoch:03d}-{val_ord_acc:.4f}.hdf5',
            monitor='val_ord_acc',
            verbose=1, save_best_only=True)

    # else:
    #     callbacks.append(
    #         MyEarlyStopping(monitor='val_acc', patience=100, min_delta=0.0001, baseline=0.6, mode='max', verbose=1))
    #
    #     callbacks.append(CheckTrainOverFit(monitor1='acc', monitor2='val_acc', patience=100))
    #     callbacks.append(CheckTrainOverFit(monitor1='acc', monitor2='val_acc', patience=100, gap=0.3))
    #     metrics = ['loss', 'acc', 'val_loss'
    #                ]
    #     metric_call = MyModelCheckpoint(
    #         filedir + 'val_acc/val_acc-{epoch:03d}-{val_acc:.4f}.hdf5',
    #         monitor='val_acc',
    #         verbose=1, save_best_only=True)
    callbacks.append(metric_call)

    for metric in metrics:
        callbacks.append(
            MyModelCheckpoint(
                filedir + '%s/%s-{epoch:03d}-{%s:.3f}.hdf5' % (metric, metric, metric),
                monitor=metric,
                verbose_only=True,
                verbose=1, save_best_only=True)
        )

    if opt.find('-ord-') >= 0:
        test_y = categorical2ord(test_y)
        val = (val[0], categorical2ord(val[1]))
    print('start training')
    # val = (val[0], [val[1], val[0]])
    #  print(val[0].shape, val[1].shape)
    # myMultiOutByGen(genor.flow(test_x, test_y, batch_size=64))
    # genor.flow(test_x, test_y, batch_size=16),

    model.fit_generator(
        myBrightOutByGen(genor.flow(test_x, test_y, batch_size=32 * gpus), brightness_range=brightrange),
        # myMultiOutByGen(genor.flow(test_x, test_y, batch_size=64)),
        #    genor.flow(test_x, test_y, batch_size=32),
        steps_per_epoch=int(test_x.shape[0] / (32 * gpus) + 1),
        validation_data=val,
        workers=nb_worker,
        max_queue_size=10,
        #    class_weight=class_weight,
        #        nb_val_samples=600,
        epochs=n_epoch, verbose=2,
        use_multiprocessing=True,
        callbacks=callbacks
    )
    if metric_call.best > 0.7:
        os.rename(filedir[:-1], filedir[:-1] + '-ok-good-%.4f-e-%04d' % (metric_call.best, metric_call.epoch_n))
    else:
        os.rename(filedir[:-1], filedir[:-1] + '-ok-bad-%.4f-e-%04d' % (metric_call.best, metric_call.epoch_n))

    exit()


if __name__ == '__main__':

    n_epoch = 10000000
    opts_dict = [
        # (optimizers.sgd(lr=0.00001, momentum=0.9, decay=0.0001, nesterov=True), 'sgd.00001'),
        (optimizers.sgd(lr=0.01, momentum=0.9, decay=0.0001, nesterov=True), 'sgd.01'),
        #          (optimizers.sgd(lr=0.001, momentum=0.9, decay=0.0001, nesterov=True), 'sgd.001'),
        #          (optimizers.sgd(lr=0.1, momentum=0.9, decay=0.0001, nesterov=True), 'sgd.1'),

        # (optimizers.sgd(lr=0.001, momentum=0.9, decay=0.0001, nesterov=True), 'sgd.001'),
        # (optimizers.adadelta(), 'adad'),
        # (optimizers.adagrad(), 'adag'),
    ]
    # '18-12-25-7833412-sgd.0001-vgg3-L2e3-b0.00-d0d1d2d3-0-ok-good-0.7750'
    for j in range(4):
        if j == 1 or j == 2 or j == 3:
            continue

        brightrange = [1 - j * 0.05, 1 + j * 0.05]
        L2 = 0.1
        for k in range(4):

            L2 = L2 / 10
            if k != 2:
                continue

            for iters in range(100):
                for op, ops in opts_dict:
                    for i in range(len(classes_type_dic)):
                        if classes_type_dic[i] != 'd0d1d2d3':
                            continue
                        train_with_generator(opt=ops + '-ord-vgg-' + 'L2e%d-b%.2f-' % (k, j * 0.05),
                                             iters=iters,
                                             classes_type=i, op=op, n_epoch=n_epoch)

    print('end')

    #
