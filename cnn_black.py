import os
import cv2
import numpy as np
import torch
from torch import nn

# ###########
# PRE-SETTING 
# ###########

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


# ############
# CREATE MODEL
# ############

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


model = Net(100, 80)  # create model
pth = os.path.join(os.getcwd(), "models")
model.load_state_dict(torch.load(pth + "/model_cnn.pth"))  # load model weights

# ########
# RUN TEST
# ########

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
        cv2.circle(content, (x, y-heightFields), 15, (255, 255, 255), -1)
        canvas[heightFields:heightCanvas-heightFields, :, :] = content  # add to canvas
    elif event == cv2.EVENT_LBUTTONUP:
        flag = False

    # draw
    if flag and event == cv2.EVENT_MOUSEMOVE:
        cv2.circle(content, (x, y-heightFields), 15, (255, 255, 255), -1)
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
    if img.any() == emptyContent.any():  # check content is empty
        # change header
        cv2.putText(header, "Canvas is empty", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
        canvas[:heightFields, :, :] = header  # add header to canvas
        header[:, :, :] = fieldsColor  # clear header
    else:
        # image processing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change image color to gray
        img = cv2.resize(img, (28, 28))  # image resize to 28*28
        img = torch.from_numpy(img)  # image to tensor      
        img = img / 255.0  # image normalize
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