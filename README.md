# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Transfer Learning is a technique where a pre-trained model (trained on a large dataset such as ImageNet) is used as a starting point for a different but related task. It leverages learned features from the original task to improve learning efficiency and performance on the new task.

VGG19 is a convolutional neural network with 19 layers. It consists of multiple convolutional layers for feature extraction, followed by fully connected layers for classification. In transfer learning, we typically freeze the convolutional layers and retrain the final fully connected layers to match our dataset.

<img width="407" height="109" alt="image" src="https://github.com/user-attachments/assets/007e567b-5ada-41b9-8caf-2170188befff" />s

## DESIGN STEPS
### STEP 1: 
Import required libraries and define image transforms.

### STEP 2: 
Load training and testing datasets using ImageFolder.

### STEP 3: 
Visualize sample images from the dataset.


### STEP 4: 
Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.


### STEP 5: 
Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6: 
Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.

## PROGRAM

### Name:Ananda Rakshan K V

### Register Number: 212223230014

```python
# Load Pretrained Model and Modify for Transfer Learning
model=models.vgg19(weights=models.VGG19_Weights.DEFAULT)
# Modify the final fully connected layer to match the dataset classes
model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)
# Include the Loss function and optimizer
criterion =nn.BCEWithLogitsLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)
# Train the model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses=[]
    val_losses=[]
    model.train()
    for epoch in range(num_epochs):
        running_loss=0.0
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels.unsqueeze(1).float())

            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        train_losses.append(running_loss/len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss=0.0
        with torch.no_grad():
          for images,labels in test_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            loss=criterion(outputs,labels.unsqueeze(1).float())
            val_loss+=loss.item()
        val_losses.append(val_loss/len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')
```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="736" height="707" alt="image" src="https://github.com/user-attachments/assets/d1307689-6661-4ac2-bb38-787195d492c8" />


## Confusion Matrix

<img width="865" height="673" alt="image" src="https://github.com/user-attachments/assets/957d17ef-5e0c-40f3-9ca7-1fd75ecce02a" />


## Classification Report
<img width="532" height="247" alt="image" src="https://github.com/user-attachments/assets/556d7b92-3c0e-45d0-9507-ac6267cc3ea0" />



### New Sample Data Prediction
<img width="532" height="446" alt="image" src="https://github.com/user-attachments/assets/55d1eb06-5786-4a59-b024-ea28edce0f50" />
<img width="458" height="437" alt="image" src="https://github.com/user-attachments/assets/481f533b-7e42-4a94-a661-5d5b0d5a3397" />


## RESULT
The image classification model using transfer learning with VGG19 architecture for the given dataset has been executed successfully.
