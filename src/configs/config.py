mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'cub200': (0.4707, 0.4601, 0.4549),
    'car196': (0.4460, 0.4311, 0.4319),
    'inshop': (0.7575, 0.7162, 0.7072),
    'sop': (0.5461, 0.4972, 0.4565),
    'fmnist': (0.5, 0.5, 0.5)
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'cub200': (0.2767, 0.2760, 0.2850),
    'car196': (0.2903, 0.2884, 0.2956),
    'inshop': (0.2810, 0.2955, 0.2961),
    'sop': (0.2867, 0.2894, 0.2994),
    'fmnist': (0.5,0.5,0.5)
}

num_classes = {
    'cifar10': 10,
    'cifar100': 100,
    'cub200': 200,
    'car196': 196,
    'inshop': 2173,
    'sop': 5036,
    'mnist':10,
    'fmnist':10
}

imsize = {
    'resnet': 224,
    'vgg': 224,
    'densenet': 224,
    'inception': 299
}

imresize = {
    'resnet': 256,
    'vgg': 256,
    'densenet': 256,
    'inception': 330
}

momentum = {
    'cifar10': 0.90,
    'cub200': 0.9,
    'mnist': 0.9,
    'fmnist': 0.9

}

train_batch = {
    'mnist': 10,
    'fmnist':10,
    'cifar10': 50,
    'cub200': 16

}

lr = {
    'mnist':0.01,
    'fmnist':0.01,
    'cifar10': 0.1,
    'cub200': 0.01
}

weight_decay = {
    'mnist': 5e-4,
    'cifar10': 5e-4,
    'cub200': 5e-4,
    'fmnist': 5e-4
}