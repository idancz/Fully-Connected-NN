import matplotlib.pyplot as plt
from torch import optim
from train_and_eval import *

epochs = 110
mini_batch_size = 100
learning_rate = 0.1
num_hidden_cells = 50

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True)


def dataset_sample():
    global x_train, t_train, x_test, t_test
    print('the training data set contains ' + str(x_train.shape[0]) + ' samples')

    img = x_train[0]
    label = t_train[0]

    img = img.reshape(28, 28)
    print('each sample image from the training data set is a column-stacked grayscale image of ' + str(
        x_train.shape[1]) + ' pixels'
          + '\n this vectorized arrangement of the data is suitable for a Fully-Connected NN (as apposed to a Convolutional NN)')
    print('these column-stacked images can be reshaped to an image of ' + str(img.shape) + ' pixels')

    # printing a sample from the dataset

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('The ground truth label of this image is ' + str(label))
    plt.show()


def dataset_visualize():
    # Visualize some examples from the dataset.
    # We'll show a few examples of training images from each class.
    global x_train
    num_classes = 10
    samples_per_class = 7
    for cls in range(num_classes):
        idxs = np.argwhere(t_train == cls)
        sample = np.random.choice(idxs.shape[0], samples_per_class, replace=False)  # randomly picks 7 from the appearences
        idxs = idxs[sample]

        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + cls + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            img = x_train[idx].reshape(28, 28)

            plt.imshow(img, cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()


# Visualize some weights. features of digits should be somehow present.
def show_net_weights(params):
    W1 = params['W1']
    print(W1.shape)
    for i in range(5):
        W = W1[:, i * 5].reshape(28, 28)
        plt.imshow(W, cmap='gray')
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    train_acc, test_acc, train_loss, net_params = train(epochs, mini_batch_size, learning_rate, num_hidden_cells)
    print(f"################### Starts Numpy Part ###############################")
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc))
    plt.plot(x, train_acc, label='train acc')
    plt.plot(x, test_acc, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend(loc='lower right')
    plt.show()

    markers = {'train': 'o'}
    x = np.arange(len(train_loss))
    plt.plot(x, train_loss, label='train loss')
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend(loc='lower right')
    plt.show()

    show_net_weights(net_params)


    # learning_rate = 0.1
    # epochs = 110
    # mini_batch_size = 100
    print(f"################### Starts PyTorch Part ###############################")
    model = TwoLayerFC()  # default input_size=784, hidden_size=50, output_size=10
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    t = Trainer(model, optimizer, criterion, epochs, mini_batch_size)
    train_acc, test_acc, train_loss, network_params = t.train()

    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc))
    plt.plot(x, train_acc, label='train acc')
    plt.plot(x, test_acc, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend(loc='lower right')
    plt.show()

    markers = {'train': 'o'}
    x = np.arange(len(train_loss))
    plt.plot(x, train_loss, label='train loss')
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend(loc='lower right')
    plt.show()
    show_net_weights(network_params)
