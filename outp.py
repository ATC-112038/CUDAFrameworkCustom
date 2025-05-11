import torch    
import cv2
import numpy as np
import requests
from torchvision import models, transforms
import time
from PIL import Image

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def get_imagenet_labels():
    url = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
    try:
        response = requests.get(url, timeout=5)
        return eval(response.text)
    except Exception as e:
        print(f"Label download failed: {e}, using fallback")
        return {i: f"Class {i}" for i in range(1000)}

def load_model():
    try:
        weights = models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        model = models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
        return model, weights.transforms()
    except Exception as e:
        print(f"Model loading warning: {e}")
        model = models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
        return model, transforms.Compose([transforms.ToTensor()])

def init_camera():
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap
    raise RuntimeError("Could not open camera")

def prepare_input(frame, transform, device):
    # Convert to PIL Image and apply transforms
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img).to(device)
    return [img_tensor]  # Return as list of tensors

def main():
    # Initialize
    model, transform = load_model()
    model.eval()
    imagenet_classes = get_imagenet_labels()
    
    try:
        cap = init_camera()
    except Exception as e:
        print(f"Camera failed: {e}")
        return

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")

    # Warm up with correct input format
    print("Warming up...")
    with torch.no_grad():
        dummy_input = [torch.randn(3, 320, 320).to(device)]  # As list of tensors
        _ = model(dummy_input)

    # FPS tracking
    frame_count = 0
    start_time = time.time()
    fps = 0
    CONFIDENCE_THRESHOLD = 0.65

    print("Running - Press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera error")
                break
            
            # Process frame
            try:
                # Prepare input correctly
                input_tensors = prepare_input(frame, transform, device)
                
                # Detect
                with torch.no_grad():
                    predictions = model(input_tensors)
                
                # Get results
                boxes = predictions[0]['boxes'].cpu().numpy()
                scores = predictions[0]['scores'].cpu().numpy()
                labels = predictions[0]['labels'].cpu().numpy()
                
                # Filter and draw
                for box, score, label in zip(boxes, scores, labels):
                    if score >= CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        class_name = imagenet_classes[int(label)].split(',')[0]
                        cv2.putText(frame, f"{class_name}: {score:.2f}", 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # FPS calculation
                frame_count += 1
                if frame_count % 10 == 0:
                    fps = 10 / (time.time() - start_time)
                    start_time = time.time()
                
                # Display info
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Detections: {sum(scores >= CONFIDENCE_THRESHOLD)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Object Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"Processing error: {e}")
                continue

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped")

if __name__ == "__main__":
    main()