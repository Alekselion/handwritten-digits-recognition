# Handwritten digits recognition
- [Feedforward Neural Network](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/model_ffnn.ipynb)
   - Create model (using [PyTorch](https://pytorch.org/get-started/locally/) with CUDA)
   - Train model (using dataset [MNIST](https://deepai.org/dataset/mnist))
   - Evaluate model
      - Loss
      - Accuracy
   - Test model
      - Testing on random data
      - Build confusion matrix
- [Convolutional Neural Network](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/model_ffnn.ipynb)
   - Create model (using [PyTorch](https://pytorch.org/get-started/locally/) with CUDA)
   - Train model (using dataset [MNIST](https://deepai.org/dataset/mnist))
   - Evaluate model
      - Loss
      - Accuracy
   - Test model
      - Testing on random data
      - Build confusion matrix
- Graphical User Interface (using [OpenCV](https://pypi.org/project/opencv-python/))
    - Pre-setting
    - Select model
    - Load model
    - Select color
    - Set colors
    - Run test 

## Feedforward Neural Network
Create model
```Python
class Net(nn.Module):
    def __init__(self, hid1, hid2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, hid1)
        self.fc2 = nn.Linear(hid1, hid2)
        self.fc3 = nn.Linear(hid2, 10)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.softmax(self.fc3(x), dim=1)
        return x

    def fit(self, x, y, epochs=1, batch_size=1, valid_size=1):
        history = {"train_loss":[], "valid_loss":[], "train_acc":[], "valid_acc":[]}
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        # create a validation dataset 
        order = np.random.permutation(int(len(x) * valid_size))
        x_valid = x[order].to(device)
        y_valid = y[order].to(device)
        
        # delete the validation data from the training dataset 
        x = x[~order].to(device)
        y = y[~order].to(device)
        
        start_training = time.time()
        
        for epoch in range(1, epochs + 1):
            print(f"Epoch: {epoch}/{epochs}")
            print("-" * 12)
            
            # shuffle dataset
            order = np.random.permutation(len(x))

            for start_index in range(0, len(x), batch_size):
                model.train()
                
                # create a batches
                batch_indexes = order[start_index:start_index+batch_size]
                x_batch = x[batch_indexes]
                y_batch = y[batch_indexes]
                
                optimizer.zero_grad()
                
                # forward propagate
                pred = model.forward(x_batch)
                loss = criterion(pred, y_batch)
                
                # gradient descent
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                # loss and accuracy of the training data
                train_preds = model.forward(x)
                train_loss = criterion(train_preds, y).float().mean().data.cpu()
                train_acc = (train_preds.argmax(dim=1) == y).float().mean().data.cpu()
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                
                # loss and accuracy of the validation dataorward(x_valid)
                valid_preds = model.forward(x_valid)
                valid_loss = criterion(valid_preds, y_valid).float().mean().data.cpu()
                valid_acc = (valid_preds.argmax(dim=1) == y_valid).float().mean().data.cpu()
                history["valid_loss"].append(valid_loss)
                history["valid_acc"].append(valid_acc)
                
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            
            print(f"Train => Loss: {train_loss :.5f}\tAccuracy: {train_acc :.5f}")
            print(f"Valid => Loss: {valid_loss :.5f}\tAccuracy: {valid_acc :.5f}\n")

        finish_training = time.time() - start_training
        
        print("Training completed".center(10, "*"))
        print(f"Time: {int(finish_training // 60)}m {int(finish_training % 60)}s")
        print(f"Best accuracy: {best_acc :.5f}")
        model.load_state_dict(best_model_wts)
        
        return model, history

    def evaluate(self, x, y):
        model.eval()
        with torch.no_grad():
            test_preds = model.forward(x)
            test_loss = criterion(test_preds, y).float().mean().data.cpu()
            test_acc = (test_preds.argmax(dim=1) == y).float().mean().data.cpu()

        return test_loss, test_acc
```
Train model
```Python
model = Net(100, 80)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
model, his = model.fit(x=x_train, y=y_train, epochs=80, batch_size=10, valid_size=0.2)
```
Result
```
Epoch: 80/80
------------
Train => Loss: 1.47430	Accuracy: 0.98667
Valid => Loss: 1.50944	Accuracy: 0.95133

Training completed
Time: 1m 50s
Best accuracy: 0.95842
```
Evaluate loss
```Python
model.eval()
with torch.no_grad():
    evaluate = model.evaluate(x_test, y_test)

print(f"Test loss: {evaluate[0] :.3f}")
plt.plot(his["train_loss"], label="Train loss")
plt.plot(his["valid_loss"], label="Valid loss")
plt.legend()
plt.show()
```
Result

![ffnn_loss](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/illustrations/ffnn_loss.jpg)

Evaluate accuracy
```Python
model.eval()
with torch.no_grad():
    evaluate = model.evaluate(x_test, y_test)

print(f"Test accuracy: {(evaluate[1] * 100) :.2f}%")    
plt.plot(his["train_acc"], label="Train accuracy")
plt.plot(his["valid_acc"], label="Valid accuracy")
plt.legend()
plt.show()
```
Result

![ffnn_acc](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/illustrations/ffnn_accuracy.jpg)

Testing on random data
```Python
img = x_test.reshape(-1, 28, 28).cpu()
plt.figure(figsize=(8, 8))
for i in range(50):
    model.eval()
    x = x_test[i]
    y = (model.forward(x)).argmax(dim=1)
    color = "red" if y != y_test[i] else "green"

    plt.subplot(5, 10, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[i], cmap=plt.cm.binary)
    plt.xlabel(labels[int(y)], color=color)
plt.show()
```
Result

![ffnn_test](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/illustrations/ffnn_test.jpg)

Build confusion matrix
```Python
# compute confusion matrix
tp, fp, fn, tn = 0, 0, 0, 0
for i in range(len(x_test)):
    if model.forward(x_test[i]).argmax(dim=1) and y_test[i]:
        tp += 1
    elif model.forward(x_test[i]).argmax(dim=1) and not y_test[i]:
        fp += 1
    elif not model.forward(x_test[i]).argmax(dim=1) and y_test[i]:
        fn += 1
    elif not model.forward(x_test[i]).argmax(dim=1) and not y_test[i]:
        tn += 1
        
confusion_matrix = np.array([[tp, fp], [fn, tn]])

# show confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        ax.text(x=j, y=i, s=confusion_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
```
Result

![ffnn_matrix](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/illustrations/ffnn_matrix.jpg)

## Convalutional Neural Network
Create model
```Python
class Net(nn.Module):
    def __init__(self, hid1, hid2):
        super(Net, self).__init__()
        
        # image 1*28*28
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # image 32*14*14
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # image 64*7*7
        self.fc_block = nn.Sequential(
            nn.Linear(64*7*7, hid1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hid1, hid2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hid2, 10),
            nn.Softmax(dim=1)  
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        
        if device != "cpu":
            torch.cuda.empty_cache()
            
        return x
    
    def fit(self, x, y, epochs=1, batch_size=1, valid_size=1):
        history = {"train_loss":[], "valid_loss":[], "train_acc":[], "valid_acc":[]}
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        # create a validation dataset 
        order = np.random.permutation(int(len(x) * valid_size))
        x_valid = x[order].to(device)
        y_valid = y[order].to(device)
        
        # delete the validation data from the training dataset 
        x = x[~order].to(device)
        y = y[~order].to(device)
        
        start_training = time.time()
        
        for epoch in range(1, epochs + 1):
            print(f"Epoch: {epoch}/{epochs}")
            print("-" * 12)
            
            # shuffle dataset
            order = np.random.permutation(len(x))

            for start_index in range(0, len(x), batch_size):
                model.train()
                
                # create a batches
                batch_indexes = order[start_index:start_index+batch_size]
                x_batch = x[batch_indexes]
                y_batch = y[batch_indexes]
                
                optimizer.zero_grad()
                
                # forward propagate
                pred = model.forward(x_batch)
                loss = criterion(pred, y_batch)
                
                # gradient descent
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                # loss and accuracy of the training data
                train_preds = model.forward(x)
                train_loss = criterion(train_preds, y).float().mean().data.cpu()
                train_acc = (train_preds.argmax(dim=1) == y).float().mean().data.cpu()
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                
                # loss and accuracy of the validation dataorward(x_valid)
                valid_preds = model.forward(x_valid)
                valid_loss = criterion(valid_preds, y_valid).float().mean().data.cpu()
                valid_acc = (valid_preds.argmax(dim=1) == y_valid).float().mean().data.cpu()
                history["valid_loss"].append(valid_loss)
                history["valid_acc"].append(valid_acc)
                
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            
            print(f"Train => Loss: {train_loss :.5f}\tAccuracy: {train_acc :.5f}")
            print(f"Valid => Loss: {valid_loss :.5f}\tAccuracy: {valid_acc :.5f}\n")

        finish_training = time.time() - start_training
        
        print("Training completed".center(10, "*"))
        print(f"Time: {int(finish_training // 60)}m {int(finish_training % 60)}s")
        print(f"Best accuracy: {best_acc :.5f}")
        model.load_state_dict(best_model_wts)
        
        if device != "cpu":
            torch.cuda.empty_cache()
            
        return model, history

    def evaluate(self, x, y):
        model.eval()
        with torch.no_grad():
            test_preds = model.forward(x)
            test_loss = criterion(test_preds, y).float().mean().data.cpu()
            test_acc = (test_preds.argmax(dim=1) == y).float().mean().data.cpu()
        
        if device != "cpu":
            torch.cuda.empty_cache()
            
        return test_loss, test_acc
```
Train model
```Python
model = Net(100, 80)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
model, his = model.fit(x=x_train, y=y_train, epochs=80, batch_size=10, valid_size=0.2)
```
Result
```
None
```
Evaluate loss
```Python
model.eval()
with torch.no_grad():
    evaluate = model.evaluate(x_test, y_test)

print(f"Test loss: {evaluate[0] :.3f}")
plt.plot(his["train_loss"], label="Train loss")
plt.plot(his["valid_loss"], label="Valid loss")
plt.legend()
plt.show()
```
Result

![cnn_loss](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/illustrations/cnn_loss.jpg)

Evaluate accuracy
```Python
model.eval()
with torch.no_grad():
    evaluate = model.evaluate(x_test, y_test)

print(f"Test accuracy: {(evaluate[1] * 100) :.2f}%")    
plt.plot(his["train_acc"], label="Train accuracy")
plt.plot(his["valid_acc"], label="Valid accuracy")
plt.legend()
plt.show()
```
Result

![cnn_acc](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/illustrations/cnn_accuracy.jpg)

Testing on random data
```Python
x_test = x_test.unsqueeze(1)
img = x_test.reshape(-1, 28, 28).cpu()
plt.figure(figsize=(8, 8))
for i in range(50):
    model.eval()
    x = x_test[i]
    y = (model.forward(x)).argmax(dim=1)
    color = "red" if y != y_test[i] else "green"

    plt.subplot(5, 10, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[i], cmap=plt.cm.binary)
    plt.xlabel(labels[int(y)], color=color)
plt.show()
```
Result

![cnn_test](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/illustrations/cnn_test.jpg)

Build confusion matrix
```Python
# compute confusion matrix
tp, fp, fn, tn = 0, 0, 0, 0
for i in range(len(x_test)):
    if model.forward(x_test[i]).argmax(dim=1) and y_test[i]:
        tp += 1
    elif model.forward(x_test[i]).argmax(dim=1) and not y_test[i]:
        fp += 1
    elif not model.forward(x_test[i]).argmax(dim=1) and y_test[i]:
        fn += 1
    elif not model.forward(x_test[i]).argmax(dim=1) and not y_test[i]:
        tn += 1
        
confusion_matrix = np.array([[tp, fp], [fn, tn]])

# show confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        ax.text(x=j, y=i, s=confusion_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
```
Result

![cnn_matrix](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/illustrations/cnn_matrix.jpg)

-----

## Graphical User Interface (GUI)
- Pre-setting
- Select model
- Load model
- Select color
- Set colors
- Run test 

- Feedforward Neural Network
    - [black canvas](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/ffnn_black.py)
    - [white canvas](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/ffnn_white.py)
- Convolutional Neural Network
    - [black canvas](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/cnn_black.py)
    - [white canvas](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/cnn_white.py)
    - [Common](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/main.py)


## Result
![res](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/illustrations/draw_digit.gif)
