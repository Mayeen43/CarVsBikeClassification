import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from model import *



def predict_image(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
    class_names = ['bike', 'car']
    prediction = class_names[predicted.item()]
    # displaying the title
    #plt.title(prediction,
    #         fontsize='20',
    #          backgroundcolor='red',
    #         color='white')
    #plt.imshow(image)
    return prediction

model = CarBikeClassifier(num_classes=2)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))

# image_file = "/content/drive/MyDrive/Colab Notebooks/carvsbike_classification/data/test/bike/Bike (261).jpeg"
# image_file = '/content/drive/MyDrive/Colab Notebooks/carvsbike_classification/data/test/car/Car (229).jpeg'
image_file="../data/Test/Bike/Bike (12345).jpg"
print(predict_image(model, image_file, device='cpu'))