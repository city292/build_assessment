from keras import backend as K


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred) + 1


def ordinal_regression_loss(y_true, y_pred):
    return - K.mean(y_true * K.log(y_pred + K.epsilon()))


def ord_acc(y_true, y_pred):
    id_t = K.cast(K.argmax(K.slice(y_true, [0, 0], [-1, 2])), dtype='int8') + \
           K.cast(K.argmax(K.slice(y_true, [0, 2], [-1, 2]), axis=-1), dtype='int8') + \
           K.cast(K.argmax(K.slice(y_true, [0, 4], [-1, 2]), axis=-1), dtype='int8')

    id_p = K.cast(K.argmax(K.slice(y_pred, [0, 0], [-1, 2]), axis=-1), dtype='int8') + \
           K.cast(K.argmax(K.slice(y_pred, [0, 2], [-1, 2]), axis=-1), dtype='int8') + \
           K.cast(K.argmax(K.slice(y_pred, [0, 4], [-1, 2]), axis=-1), dtype='int8')
    return K.cast(K.equal(id_t, id_p), K.floatx())


def ord_acc0(y_true, y_pred):
    id_t = K.cast(K.argmax(K.slice(y_true, [0, 0], [-1, 2]), axis=-1), dtype='int8')
    id_p = K.cast(K.argmax(K.slice(y_pred, [0, 0], [-1, 2]), axis=-1), dtype='int8')
    return K.mean(K.equal(id_t, id_p))


def ord_acc1(y_true, y_pred):
    id_t = K.cast(K.argmax(K.slice(y_true, [0, 2], [-1, 2]), axis=-1), dtype='int8')
    id_p = K.cast(K.argmax(K.slice(y_pred, [0, 2], [-1, 2]), axis=-1), dtype='int8')
    return K.mean(K.equal(id_t, id_p))


def ord_acc2(y_true, y_pred):
    id_t = K.cast(K.argmax(K.slice(y_true, [0, 4], [-1, 2]), axis=-1), dtype='int8')
    id_p = K.cast(K.argmax(K.slice(y_pred, [0, 4], [-1, 2]), axis=-1), dtype='int8')
    return K.mean(K.equal(id_t, id_p))
