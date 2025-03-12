import torch
from torchvision import transforms, models, datasets
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def load_model(model_path, num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_image(model, image_path, transform, classes, device):
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0)  # Adiciona uma dimensão extra para o batch
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs.data, 1)
    
    predicted_class = classes[predicted.item()]
    print(f'Classe prevista: {predicted_class}')
    
    return img_tensor, predicted_class

def save_attention_map(model, img_tensor, image_path, device):
    model.eval()
    img_tensor = img_tensor.to(device)
    
    def hook_function(module, input, output):
        global activation
        activation = output
    
    hook = model.layer4[1].conv2.register_forward_hook(hook_function)
    
    with torch.no_grad():
        model(img_tensor)
    
    hook.remove()
    
    activation = activation.squeeze().cpu().numpy()
    attention_map = np.mean(activation, axis=0)
    
    img = Image.open(image_path)
    plt.imshow(img)
    plt.imshow(attention_map, cmap='jet', alpha=0.5)
    plt.colorbar()
    plt.title('Mapa de Atenção')
    plt.savefig(f"{image_path}_attention_map.png")
    plt.show()
    print(f"Mapa de atenção salvo como '{image_path}_attention_map.png'")