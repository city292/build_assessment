import csv
import os
import random
import sys
import warnings
from collections import OrderedDict, Iterable

import datetime
import numpy as np
import sklearn.metrics as sklm
import time
from keras.callbacks import Callback


class TrainableAfter(Callback):
    def __init__(self, monitor='val_acc', trainafter='0.7', cooldown=20):
        super(TrainableAfter, self).__init__()
        self.monitor = monitor
        self.cooldown = cooldown
        self.best = 0
        self.n_epoch = 0
        self.trainafter = trainafter
        self.trainable = False

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if not self.trainable:
            if current < self.best:
                if self.cooldown < self.n_epoch:
                    for layer in self.model.layers:
                        layer.trainable = True
                    print('start finetuning')
                    self.model.compile(loss='categorical_crossentropy', optimizer=self.model.optimizer,
                                       metrics=['accuracy'])
                    self.trainable = True
                self.n_epoch += 1
            else:
                self.best = current

                self.cooldown = 2 * self.n_epoch
                self.n_epoch = 0


class MyModelCheckpoint(Callback):
    '''Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then multiple files will be save with the epoch number and
    the validation loss.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).

    '''

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 verbose_only=False,
                 mode='auto'):
        super(MyModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.time = time.time()
        self.pre_time = time.time()
        self.epoch_n = 0
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose_only = verbose_only
        if not os.path.exists(os.path.dirname(filepath)):
            if not self.verbose_only:
                os.mkdir(os.path.dirname(filepath))
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        filepath = self.filepath.format(epoch=epoch, **logs)
        self.epoch_n = epoch
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        timeused = time.time() - self.time
                        timestr = '%ds' % timeused
                        e_time = time.time() - self.pre_time

                        if timeused > 60:
                            timestr = '%dm%ds' % (timeused / 60, int(timeused) % 60)
                        if timeused > 3600:
                            timestr = '%dh%dm%ds' % (timeused / 3600, (int(timeused) % 3600) / 60, int(timeused) % 60)

                        print('Epoch %05d: %s improved from %0.4f to %0.4f,'
                              ' time used: %s ,epoch time: %ds'
                              % (epoch, self.monitor, self.best, current,
                                 timestr, e_time))
                        if 'pred' in self.monitor:
                            print('Epoch %05d: %s improved from %0.3f to %0.3f,'
                                  ' time used: %s ,epoch time: %ds'
                                  % (epoch, self.monitor, self.best, current,
                                     timestr, e_time))

                    self.best = current
                    if not self.verbose_only:
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)



                else:
                    if self.verbose > 1:
                        print('Epoch %05d: %s did not improve' %
                              (epoch, self.monitor))
        else:
            if self.verbose > 1:
                print('Epoch %05d: saving model to %s' % (epoch, filepath))
            if self.save_weights_only:
                self.model.save_weights(filepath, overwrite=True)

            else:
                self.model.save(filepath, overwrite=True)
        self.pre_time = time.time()


class CheckTrainable(Callback):
    """Callback that records events into a `History` object.

    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    """

    def switchblock(self, stage, num, trainable):
        res = False
        for layer in self.model.layers:
            subname = 's%d_n%d_' % (stage, num)
            if layer.name.find(subname) >= 0:
                layer.trainable = trainable
                res = True
        return res

    def switchdense(self, num, trainable):
        res = False
        for layer in self.model.layers:
            subname = 'dense_%d' % num
            if layer.name.find(subname) >= 0:
                layer.trainable = trainable
                res = True
        return res

    def switchall(self, trainable):
        for layer in self.model.layers:
            layer.trainable = trainable

    def on_epoch_end(self, epoch, logs=None):
        startstage = True
        if epoch == 30:
            self.switchall(True)
            print('switch all')
            return

        if epoch == 60:
            self.switchall(False)
            startstage = False
            self.switchdense(1, True)
            print('finetuning start')
            return
        if startstage:
            return
        if epoch % 5 == 0:
            print('switch trainable')
            self.switchall(False)
            self.switchdense(1, True)
            while True:
                stage = random.randint(1, 5)
                num = random.randint(1, 5)
                if self.switchblock(stage, num, False):
                    break


class CheckTrainOverFit(Callback):
    def __init__(self, monitor1='acc', monitor2='val_acc', init_num=0, gap=0.1, mode='auto', patience=500):
        super().__init__()
        self.val_m1 = init_num
        self.val_m2 = init_num
        self.gap = gap
        self.monitor1 = monitor1
        self.monitor2 = monitor2
        self.patience = patience

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor1:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        m1 = logs.get(self.monitor1)
        m2 = logs.get(self.monitor2)
        if self.monitor_op(m1, self.val_m1):
            self.val_m1 = m1
        if self.monitor_op(m2, self.val_m2):
            self.val_m2 = m2
        if epoch > self.patience:
            if self.val_m1 - self.val_m2 > self.gap:
                print('stop when %s: %.4f %s: %.4f' % (self.monitor1, self.val_m1, self.monitor2, self.val_m2))
                self.model.stop_training = True


class MyCSVLogger(Callback):
    '''Callback that streams epoch results to a csv file.
    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    # Example
        ```python
            csv_logger = CSVLogger('training.log')
            model.fit(X_train, Y_train, callbacks=[csv_logger])
        ```

    Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    '''

    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = os.path.dirname(filename)
        self.append = append
        self.writer = None
        self.keys = None
        self.time = time.time()
        self.pre_time = time.time()
        self.stone_writer = None
        self.stone = 0.01
        self.val_stone = 0.01
        self.loss_stone = 100
        self.val_loss_stone = 100
        super(MyCSVLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        if self.append:
            self.csv_file = open(self.filename + '/log_file.csv', 'a', newline='')
        else:
            self.csv_file = open(self.filename + '/log_file.csv', 'w', newline='')

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(lambda x: str(x), k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(logs.keys())

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch'] + self.keys + ['time'] + ['all'] + ['day'] + ['h'] + [
                                             'm'] + ['s'])
            self.writer.writeheader()
        #    self.stone_writer = csv.DictWriter(self.csv_stone, fieldnames=['epoch'] + ['type'] + ['stone'] + self.keys + ['time']+['all']+['day']+['h']+['m']+['s'])
        #    self.stone_writer.writeheader()
        #   if time.time() - self.time %
        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        row_dict.update({'time': int(time.time() - self.pre_time)})
        row_dict.update({'all': int(time.time() - self.time)})
        row_dict.update({'day': datetime.datetime.now().strftime('%d')})
        row_dict.update({'h': datetime.datetime.now().strftime('%H')})
        row_dict.update({'m': datetime.datetime.now().strftime('%M')})
        row_dict.update({'s': datetime.datetime.now().strftime('%S')})
        #    self.writer.writer()
        self.writer.writerow(row_dict)
        #    self.writer.writer()
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        if logs is None:
            logs = {}
        self.csv_file.close()


class Progbar(object):
    def __init__(self, target, width=30, Epoch=0, verbose=1, interval=0.01):
        '''
            @param target: total number of steps expected
            @param interval: minimum visual progress update interval (in seconds)
        '''
        self.Epoch = Epoch
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.last_update = 0
        self.interval = interval
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, force=False):
        '''
            @param current: index of current step
            @param values: list of tuples (name, value_for_last_step).
            The progress bar will display averages for these values.
            @param force: force visual progress update
        '''
        if values is None:
            values = []
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            #    if  (now - self.last_update) < self.interval:
            #        return

            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")
            if force:
                sys.stdout.flush()
                return
            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = 'Epoch %05d: %%%dd/%%%dd [' % (self.Epoch, numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                info += ' - %s:' % k
                if type(self.sum_values[k]) is list:
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self.sum_values[k]

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s:' % k
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                sys.stdout.write(info + "\n")

        self.last_update = now

    def add(self, n, values=None):
        if values is None:
            values = []
        self.update(self.seen_so_far + n, values)


class MyProgbarLogger(Callback):
    '''Callback that prints metrics to stdout.
    '''

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.verbose = 1
        print(self.params)

    #    self.nb_epoch = self.params['epoch']

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if self.verbose:
            self.progbar = Progbar(target=self.params['steps'],
                                   Epoch=epoch,
                                   verbose=self.verbose)
        self.seen = 0

    def on_batch_begin(self, batch, logs=None):
        if logs is None:
            logs = {}
        if self.seen < self.params['steps']:
            self.log_values = []

    def on_batch_end(self, batch, logs=None):
        #    batch_size = logs.get('size', 0)
        if logs is None:
            logs = {}
        self.seen += 1

        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        # skip progbar update for the last batch;
        # will be handled by on_epoch_end
        if self.verbose and self.seen < self.params['steps']:
            self.progbar.update(self.seen, self.log_values)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        if self.verbose:
            self.progbar.update(self.seen, self.log_values, force=True)


class Metrics(Callback):
    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.confusion = []
        self.precision = []
        self.recall = []
        self.f1s = []
        self.kappa = []
        self.auc = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        score = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))
        targ = self.validation_data[1]

        self.auc.append(sklm.roc_auc_score(targ, score))
        self.confusion.append(sklm.confusion_matrix(targ, predict))
        self.precision.append(sklm.precision_score(targ, predict))
        self.recall.append(sklm.recall_score(targ, predict))
        self.f1s.append(sklm.f1_score(targ, predict))
        self.kappa.append(sklm.cohen_kappa_score(targ, predict))

        return


class MyEarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """

    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(MyEarlyStopping, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            self.patience = max(self.patience, 3 * self.wait)
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print('Epoch %05d: early stopping when %s is %.4f' % (self.stopped_epoch + 1, self.monitor, self.best))
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping when %s is %.4f' % (self.stopped_epoch + 1, self.monitor, self.best))

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value
