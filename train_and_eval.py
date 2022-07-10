from fully_connected_model import *
from data_handeling import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def accuracy(y, t):
    """
    Computes the accuracy of the NN's predictions.
    Inputs:
    - t:  A numpy array of shape (N,C) containing training labels, it is a one-hot array,
      with t[GT]=1 and t=0 elsewhere, where GT is the ground truth label ;
    - y: the output probabilities for the minibatch (at the end of the forward pass) of shape (N,C)
    Returns:
    - accuracy: a single float of the average accuracy.
    """
    shape = y.shape[0]
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)
    accuracy = np.sum(y == t) / float(shape)
    return accuracy


def train(epochs_num, batch_size, lr, H):
    #  Dividing a dataset into training data and test data

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    C = 10
    D = x_train.shape[1]
    network_params = TwoLayerNet(input_size=D, hidden_size=H,
                                 output_size=C)  # hidden_size is the only hyperparameter here

    train_size = x_train.shape[0]
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    iter_per_epoch = round(train_size / batch_size)

    print('training of ' + str(epochs_num) + ' epochs, each epoch will have ' + str(iter_per_epoch) + ' iterations')
    for i in range(epochs_num):

        train_loss_iter = []
        train_acc_iter = []

        for k in range(iter_per_epoch):
            # Select part of training data (mini-batch) randomly
            batch_pick = np.random.choice(train_size, batch_size)
            x_batch, t_batch = x_train[batch_pick], t_train[batch_pick]
            # Calculate the predictions and the gradients to reduce the value of the loss function
            grad, y_batch = Model(network_params, x_batch, t_batch)

            # Update weights and biases with the gradients
            for key in grad:
                network_params[key] -= lr * grad[key]

                # Calculate the loss and accuracy for visalizaton
            error = cross_entropy_error(y_batch, t_batch)
            train_loss_iter.append(error)
            acc_iter = accuracy(y_batch, t_batch)
            train_acc_iter.append(acc_iter)
            if k == iter_per_epoch - 1:
                train_acc = np.mean(train_acc_iter)
                train_acc_list.append(train_acc)
                train_loss_list.append(np.mean(train_loss_iter))

                _, y_test = Model(network_params, x_test, t_test)
                test_acc = accuracy(y_test, t_test)
                test_acc_list.append(test_acc)
                print("train acc: " + str(train_acc)[:5] + "% |  test acc: " + str(
                    test_acc) + "% |  loss for epoch " + str(i) + ": " + str(np.mean(train_loss_iter)))
    return train_acc_list, test_acc_list, train_loss_list, network_params


 ######################################################################################################################################
 ##################################################### PyTorch Implementation #########################################################
 ######################################################################################################################################

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim,
                 criterion,
                 epochs_num: int,
                 batch_size: int):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs_num = epochs_num
        self.batch_size = batch_size

    def train(self):
        #  Dividing a dataset into training data and test data
        (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

        train_size = x_train.shape[0]
        train_loss_list = []
        train_acc_list = []
        test_acc_list = []
        iter_per_epoch = round(train_size / self.batch_size)

        print('training of ' + str(self.epochs_num) + ' epochs, each epoch will have ' + str(
            iter_per_epoch) + ' iterations')
        for i in range(self.epochs_num):
            self.model = self.model.train()
            train_loss_iter = []
            train_acc_iter = []

            for k in range(iter_per_epoch):
                batch_pick = np.random.choice(train_size, self.batch_size)
                x_batch, t_batch = torch.tensor(x_train[batch_pick]), torch.tensor(t_train[batch_pick])

                prediction = self.model(x_batch)
                loss = self.criterion(prediction, t_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                correct_labeled_samples = (t_batch.argmax(dim=1) == prediction.argmax(dim=1)).sum().item()
                accuracy = (correct_labeled_samples / len(t_batch))

                train_loss_iter.append(loss.item())
                train_acc_iter.append(accuracy)
                if k == iter_per_epoch - 1:
                    train_acc = np.mean(train_acc_iter)
                    train_acc_list.append(train_acc)
                    train_loss_list.append(np.mean(train_loss_iter))
                    test_acc = self.evaluate(x_test, t_test)
                    test_acc_list.append(test_acc)
                    print("train acc: " + str(train_acc)[:5] + "% |  test acc: " + str(
                        test_acc) + "% |  loss for epoch " + str(i) + ": " + str(np.mean(train_loss_iter)))
        return train_acc_list, test_acc_list, train_loss_list, self.get_params()

    def evaluate(self, x_test, t_test):
        self.model.eval()
        x_test = torch.tensor(x_test)
        t_test = torch.tensor(t_test)
        with torch.no_grad():
            y_test = self.model(x_test)
            test_correct_labeled_samples = (t_test.argmax(dim=1) == y_test.argmax(dim=1)).sum().item()
            test_acc = (test_correct_labeled_samples / len(t_test))
        return test_acc

    def get_params(self):
        params = {'W1': self.model.fullyconnected1.state_dict()['weight'].T,
                  'W2': self.model.fullyconnected2.state_dict()['weight'].T,
                  'b1': self.model.fullyconnected2.state_dict()['bias'].T}
        return params
