from ultralytics import YOLO

def run_test(source_path):
    model = YOLO('yolov8n.pt')
    print(f"Processing: {source_path}")
    
    # YOLOv8 can handle URLs directly if they point to an image
    results = model(source_path)
    
    results[0].save(filename='yolo_result.jpg')
    print("[SUCCESS] Result saved as 'yolo_result.jpg'")

if __name__ == "__main__":
    # This must be a DIRECT link to an image file
    test_image = "room.jpg" 
    run_test(test_image)
