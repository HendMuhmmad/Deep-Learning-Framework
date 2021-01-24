import numpy as np
from keras.datasets import cifar10

# filter 3x3 padding = 1 keep size of image unchanged
# maxpool divide by 2
# batch-normalization doesn't affect image size 

net = Net(layers=[Conv2D(3, 4, 3, padding=1), MaxPool2D(kernel_size=2), ReLU(), BatchNorm2D(4),
                  Conv2D(4, 8, 3, padding=1), MaxPool2D(kernel_size=2), ReLU(), BatchNorm2D(8),
                  Flatten(), Linear(8*8*8, 10)],
          loss=CrossEntropyLoss())

(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# reshaping
X_train, X_test = X_train.reshape(-1, 1, 32, 32), X_test.reshape(-1, 1, 32, 32)
y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

# # normalizing and scaling data
X_train, X_test = X_train.astype('float32')/255, X_test.astype('float32')/255

# reshaping
X_train, X_test = X_train.reshape(-1, 3, 32, 32), X_test.reshape(-1, 3, 32, 32)



n_epochs = 1
n_batch = 20
for epoch_idx in range(n_epochs):
    batch_idx = np.random.choice(range(len(X_train)), size=n_batch, replace=False)
    print("X_train.shape = ",X_train[batch_idx].shape)
    out = net(X_train[batch_idx])
    preds = np.argmax(out, axis=1).reshape(-1, 1)
    accuracy = 100*(preds == y_train[batch_idx]).sum() / n_batch
    loss = net.loss(out, y_train[batch_idx])
    net.backward()
    net.update_weights(lr=0.01)
    print("Epoch no. %d loss =  %2f4 \t accuracy = %d %%" % (epoch_idx + 1, loss, accuracy))