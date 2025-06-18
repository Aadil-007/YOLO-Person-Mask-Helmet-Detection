# YOLOv8 Person, Mask, and Helmet Detection
A real-time computer vision project using YOLOv8 to detect people, masks, and helmets.

## Features
- Detects people (green boxes).
- Identifies mask presence (blue/red boxes).
- Detects helmets (yellow boxes).

## Setup
1. Clone the repository: `git clone https://github.com/Aadil-007/YOLO-Person-Mask-Helmet-Detection.git`
2. Install dependencies: `pip install ultralytics torch torchvision torchaudio opencv-python`
3. Download trained weights (not included due to size): 
   - Mask: `runs/detect/mask_detection_improved/weights/best.pt`
   - Helmet: `runs/detect/helmet_detection_improved/weights/best.pt`
4. Run: `python mask_detection.py`

## Credits
Developed by [Aadil-007](https://github.com/Aadil-007) with assistance from Grok 3 (xAI).

## License
[MIT License](LICENSE) (add a `LICENSE` file if desired).