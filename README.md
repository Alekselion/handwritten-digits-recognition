# Handwritten digits recognition
Creating 2 neural networks models (feedforward and convolutional) using [PyTorch](https://pytorch.org/get-started/locally/) with CUDA and dataset [MNIST](https://deepai.org/dataset/mnist). Building GUI using [OpenCV](https://pypi.org/project/opencv-python/).

-----

- Feedforward Neural Network ([full code](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/model_ffnn.ipynb))
   - Create model
   - Train model
   - Evaluate model
   - Test model
   - Build confusion matrix
- Convolutional Neural Network ([full code](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/model_ffnn.ipynb))
   - Create model
   - Train model
   - Evaluate model
   - Test model
   - Build confusion matrix
- Graphical User Interface ([full code](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/main.py))
    - Create canvas
    - Load model
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

Output
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

Output

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

Output

![ffnn_acc](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/illustrations/ffnn_accuracy.jpg)

Test model
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

Output

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
Output

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

Output
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

Output

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

Output

![cnn_acc](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/illustrations/cnn_accuracy.jpg)

Test model
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

Output

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

Output

![cnn_matrix](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/illustrations/cnn_matrix.jpg)

## Graphical User Interface
Create canvas
```Python
# color
textColor = (255, 0, 255)  # text color (purple)
contentColor = (0, 0, 0)  # canvas color (black)
fieldsColor = (50, 50, 50)  # header and footer color (gray)

# size
widthCanvas, heightCanvas = 300, 400  # canvas size
widthFields, heightFields = widthCanvas, 50  # header and footer size
widthContent, heightContent = widthCanvas, heightCanvas - heightFields * 2  # content size

# canvas
canvas = np.zeros((heightCanvas, widthCanvas, 3), np.uint8)  # create

# header
header = np.zeros((heightFields, widthFields, 3), np.uint8)  # create
header[:, :, :] = fieldsColor  # change color
canvas[:heightFields, :, :] = header  # add to canvas

# content
content = np.zeros((heightContent, widthFields, 3), np.uint8)  # create
content[:, :, :] = contentColor  # change color
canvas[heightFields:heightCanvas-heightFields, :, :] = content  # add to canvas

# footer
footer = np.zeros((heightFields, widthFields, 3), np.uint8)  # create
footer[:, :, :] = fieldsColor  # change color
canvas[heightCanvas-heightFields:, :, :] = footer  # add to canvas

# show canvas
cv2.imshow("Sample canvas", canvas)
cv2.waitKey(0)
cv2.destroyWindow("Sample canvas")
```

Output


Load model
```Python
if modelName == "ffnn":
    class Net(nn.Module):
        def __init__(self, hid1, hid2):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(28*28, hid1)
            self.fc2 = nn.Linear(hid1, hid2)
            self.fc3 = nn.Linear(hid2, 10)
            self.drop = nn.Dropout(0.2)

        def forward(self, x):
            x = x.view(-1, 28*28)  # flatten
            x = F.relu(self.fc1(x))
            x = self.drop(x)
            x = F.relu(self.fc2(x))
            x = self.drop(x)
            x = F.softmax(self.fc3(x), dim=1)
            return x


    model = Net(100, 80)
    pth = os.path.join(os.getcwd(), "models")
    model.load_state_dict(torch.load(pth + "/model_ffnn.pth"))

elif modelName == "cnn":
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
            x = x.view(x.size(0), -1)  # flatten
            x = self.fc_block(x)
                        
            return x


    model = Net(100, 80)
    pth = os.path.join(os.getcwd(), "models")
    model.load_state_dict(torch.load(pth + "/model_cnn.pth"))
```

Run test
```Python
# change footer
cv2.putText(footer, "Left mouse button to draw", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
cv2.putText(footer, "Right mouse button to clear", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
canvas[heightCanvas-heightFields:, :, :] = footer  # add to canvas
footer[:, :, :] = fieldsColor  # clear footer

flag = False


def drawCircle(event, x, y, *otherParams):
    """ left mouse button to draw
        right mouse button to clear
    """
    global flag
    if event == cv2.EVENT_LBUTTONDOWN:
        flag = True
        cv2.circle(content, (x, y-heightFields), 15, penColor, -1)
        canvas[heightFields:heightCanvas-heightFields, :, :] = content  # add to canvas
    elif event == cv2.EVENT_LBUTTONUP:
        flag = False

    # draw
    if flag and event == cv2.EVENT_MOUSEMOVE:
        cv2.circle(content, (x, y-heightFields), 15, penColor, -1)
        canvas[heightFields:heightCanvas-heightFields, :, :] = content  # add to canvas

    # clear
    if not flag and event == cv2.EVENT_RBUTTONDOWN:
        content[:, :, :] = contentColor  # clear content
        canvas[heightFields:heightCanvas-heightFields, :, :] = content  # add to canvas


emptyContent = content.copy()
cv2.namedWindow("canvas")  
cv2.setMouseCallback("canvas", drawCircle)
while True: 
    img = content.copy()  # copy content
    isEmpty = img.all() == emptyContent.all() if colorName == "white" else img.any() == emptyContent.any()
    if isEmpty:  # check content is empty
        # change header
        cv2.putText(header, "Canvas is empty", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
        canvas[:heightFields, :, :] = header  # add header to canvas
        header[:, :, :] = fieldsColor  # clear header
    else:
        # image processing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change image color to gray
        img = cv2.resize(img, (28, 28))  # image resize to 28*28
        img = torch.from_numpy(img)  # image to tensor
        if colorName == "white":
            img = 255.0 - img  # image inversion
        
        img = img / 255.0  # image normalize
        if modelName == "cnn":
            img = img.unsqueeze(0)  # add 'batch size'  img.shape = (B, W, H)
            img = img.unsqueeze(1)  # add 'channel'  img.shape = (B, C, W, H)
        
        # predict
        model.eval()
        pred = (model.forward(img))
        predNumber = int(pred.argmax(dim=1))
        predAccuracy = float(pred[0][predNumber]) * 100
        result = f"Number: {predNumber} ({predAccuracy :.2f}%)"
        
        # change header
        cv2.putText(header, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
        canvas[:heightFields, :, :] = header  # add header to canvas
        header[:, :, :] = fieldsColor  # clear header
    
    cv2.imshow("canvas", canvas)
    if cv2.waitKey(1) & 0xFF == ord("q"):  
        break

cv2.destroyAllWindows()
```

Result

![res](https://github.com/Alekselion/handwritten-digits-recognition/blob/master/illustrations/draw_digit.gif)
