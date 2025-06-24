import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Defining Model Architecture
class CurrencyCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CurrencyCNN, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 32 * 32, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Loading Trained Model
model = CurrencyCNN(num_classes=7)  # Change to actual number of classes
model.load_state_dict(torch.load("D:/AI/currency_detector.pth", map_location=torch.device('cpu')))
model.eval()

# Class Names
class_names = ['10', '100', '1000', '20', '50', '500', '5000']  

# Images Transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Prediction Function
def predict_notes(images):
    counts = {name: 0 for name in class_names}
    
    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            label = class_names[predicted.item()]
            counts[label] += 1

    total = 0
    rows = ""
    for note, count in counts.items():
        if count > 0:
            value = int(note) * count
            rows += f"<tr><td>{note}</td><td>{count}</td><td>{value}</td></tr>"
            total += value

    rows += f"<tr style='font-weight:bold;'><td>Total</td><td></td><td>{total}</td></tr>"

    html_table = f"""
    <table style='font-size:18px; border-collapse: collapse; width: 100%; text-align: center;'>
        <thead>
            <tr style='background-color: #f0f0f0; font-weight: bold;'>
                <th>Note</th>
                <th>Count</th>
                <th>Value</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """
    return html_table

# Gradio Interface
iface = gr.Interface(
    fn=predict_notes,
    inputs=gr.File(type="filepath", label="Upload Note Images", file_types=[".png", ".jpg", ".jpeg"], file_count = "multiple"),
    outputs=gr.HTML(label="💰 Prediction Table"),
    title="""
    <center><p style='font-size:35px; font-family:Times New Roman; font-weight:bold; color:#1a1a1a;'>💸 Currency Note Detector</p></center>
    """,
    description="""
    <center>
        <p style='font-size:16px; font-family:Verdana; color:#333;'>
        Upload clear images of Pakistani currency notes.<br>
        This AI model will detect, count, and calculate the total amount.
        </p>
    </center>
    """
)

iface.launch()