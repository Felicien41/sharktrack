from flask import Flask, request, jsonify
from utils.path_resolver import generate_output_path, convert_to_abs_path
from utils.species_classifier import SpeciesClassifier
from ultralytics import YOLO
import os
import torch

# Initialize Flask app
app = Flask(__name__)

# Load model and global settings
class Model:
    def __init__(self, model_path="models/sharktrack.pt"):
        self.model_path = model_path
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = YOLO(self.model_path)

    def process_video(self, input_path, output_path, conf=0.25, imgsz=640):
        # Example inference implementation
        self.model.track(
            source=input_path,
            conf=conf,
            imgsz=imgsz,
            save=True,
            save_dir=output_path
        )
        return output_path

# Global model instance
sharktrack_model = Model()

@app.route('/process', methods=['GET'])
def process_videos():
    # Retrieve input path from request arguments
    input_path = request.args.get('path')
    if not input_path:
        return jsonify({"error": "Missing 'path' parameter"}), 400

    input_path = convert_to_abs_path(input_path)
    output_path = generate_output_path(None, input_path, annotation_folder="flask_results", resume=False)
    os.makedirs(output_path, exist_ok=True)

    try:
        result_path = sharktrack_model.process_video(input_path, output_path)
        return jsonify({"output_path": result_path}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load model at startup
    print("Loading SharkTrack model...")
    app.run(host='0.0.0.0', port=5000)
