import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image

# Model Definition
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

# Load model
model = CurrencyCNN(num_classes=7)
model.load_state_dict(torch.load("D:/AI/currency_detector.pth", map_location=torch.device('cpu'))) # Adjust path according to your own
model.eval()

# Class labels
class_names = ['10', '100', '1000', '20', '50', '500', '5000']

# Image transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Prediction function
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

    return [images, html_table]

# Gradio UI Layout
with gr.Blocks() as app:
    # Title and Description
    gr.Markdown("""
    <center><p style='font-size:35px; font-family:Times New Roman; font-weight:bold; color:#1a1a1a;'>💸 Currency Note Detector</p></center>
    <center>
        <p style='font-size:16px; font-family:Verdana; color:#333;'>
        Upload clear images of Pakistani currency notes.<br>
        This AI model will detect, count, and calculate the total amount.
        </p>
    </center>
    """)

    # Custom CSS for Detect Button
    gr.HTML("""
    <style>
    #detect-button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #fff;
        font-weight: bold;
        font-size: 18px;
        padding: 12px 30px;
        border: none;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transition: all 0.3s ease-in-out;
        width: 250px;
        margin: 10px auto;
        display: block;
    }
    #detect-button:hover {
        transform: scale(1.05);
        cursor: pointer;
        background: linear-gradient(90deg, #36d1dc 0%, #5b86e5 100%);
    }
    </style>
    """)

    # Upload Box
    with gr.Row():
        image_input = gr.File(
            type="filepath",
            label="📤 Upload Note Images",
            file_types=[".png", ".jpg", ".jpeg"],
            file_count="multiple"
        )

    # Detect Button
    with gr.Row():
        detect_btn = gr.Button("🔍 Detect Currency Notes", elem_id="detect-button")

    # Output Section: Gallery + Table Side by Side
    with gr.Row():
        gallery_output = gr.Gallery(label="📷 Uploaded Images")
        table_output = gr.HTML(label="💰 Prediction Table")

    # Connect button to function
    detect_btn.click(fn=predict_notes, inputs=image_input, outputs=[gallery_output, table_output])

# Launch App
app.launch()
