from keras import optimizers

op = None
classes_type = 0
n_epoch = 10000
classes_nums_dict = [[0, 1, 2, 3],

                     [0, 1], [0, 2], [0, 3],
                     [1, 2], [1, 3],
                     [2, 3],

                     [0, 1, 2],
                     [1, 2, 3],

                     [0, 1, 2, 3],
                     [0, 1, 2, 3],

                     [0, 1, 2, 3],
                     [0, 1, 2, 3]]

classes_labels_dict = [[0, 1, 2, 3],

                       [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1],
                       [0, 0, 1, 0], [0, 0, 1, 1],
                       [0, 0, 0, 1],

                       [0, 1, 2, 0], [0, 0, 1, 2],

                       [0, 0, 1, 1], [0, 0, 0, 1],

                       [0, 0, 1, 2], [0, 1, 1, 2]
                       ]
classes_type_dic = ['d0d1d2d3',
                    'd0d1', 'd0d2', 'd0d3',
                    'd1d2', 'd1d3',
                    'd2d3',
                    'd0d1d2', 'd1d2d3',
                    'd01d23', 'd012d3',
                    'd01d2d3', 'd0d12d3']

opts_dict = [
    (optimizers.sgd(lr=1, momentum=0.9, decay=0.0001, nesterov=True), 'sgd1'),
    (optimizers.sgd(lr=0.1, momentum=0.9, decay=0.0001, nesterov=True), 'sgd.1'),

    (optimizers.sgd(lr=0.01, momentum=0.9, decay=0.0001, nesterov=True), 'sgd.01'),
    (optimizers.sgd(lr=0.001, momentum=0.9, decay=0.0001, nesterov=True), 'sgd.001'),
    (optimizers.sgd(lr=0.0001, momentum=0.9, decay=0.0001, nesterov=True), 'sgd.0001'),
    (optimizers.sgd(lr=0.00001, momentum=0.9, decay=0.0001, nesterov=True), 'sgd.00001'),
    (optimizers.adadelta(), 'adad'),

    (optimizers.adagrad(), 'adag'),
    (optimizers.sgd(lr=1, momentum=0.8, decay=0.01, nesterov=True), 'sgd1'),
    (optimizers.adam(), 'adam'),

    (optimizers.rmsprop(), 'rmsp'),
    (optimizers.nadam(), 'nadam')
]

if __name__ == '__main__':
    import os

    env_name = os.getenv("COMPUTERNAME")
    print(env_name)
    exit()

strs = ['d0d1d2d3',
        'd0d1', 'd0d2', 'd0d3',
        'd1d2', 'd1d3',
        'd2d3',
        'd0d1d2', 'd1d2d3',
        'd01d23', 'd012d3',
        'd01d2d3', 'd0d12d3']


def classes_list(model_index='s'):
    for d_str in model_index.split('-'):

        for i in range(len(strs)):
            if strs[i] == d_str:
                return classes_nums[i], class_labels[i], d_str.count('d')
    return None, None, None
