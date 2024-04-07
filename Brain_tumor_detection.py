from roboflow import Roboflow
rf = Roboflow(api_key="input your roboflow api key")
project = rf.workspace().project("brain-tumor-detection-u7ysj")
model = project.version(1).model

# infer on a local image
print(model.predict("b.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("b.jpg", confidence=40, overlap=30).save("JK.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
import torch
torch.save(model, "model.pt")
