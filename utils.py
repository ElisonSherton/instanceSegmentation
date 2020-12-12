import cv2
import torch
import torchvision

class utils:
    @staticmethod
    def load_model():
        # Load the DeepLab v3 model to system
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
        model.to(device).eval()
        return model
    
    @staticmethod
    def grab_frame(cap):
        # Given a video capture object, read frames from the same and convert it to RGB
        _, frame = cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def get_pred(img, model):
        # See if GPU is available and if yes, use it
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Define the standard transforms that need to be done at inference time
        imagenet_stats = [[0.485, 0.456, 0.406], [0.485, 0.456, 0.406]]
        preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(mean = imagenet_stats[0],
                                                                                      std  = imagenet_stats[1])])
        input_tensor = preprocess(img).unsqueeze(0)
        input_tensor = input_tensor.to(device)

        # Make the predictions for labels across the image
        with torch.no_grad():
            output = model(input_tensor)["out"][0]
            output = output.argmax(0)
        
        # Return the predictions
        return output.cpu().numpy()