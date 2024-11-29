from flask import Flask, request, jsonify
from utils.sharktrack_annotations import save_analyst_output, save_peek_output, extract_sightings, resume_previous_run
from utils.path_resolver import generate_output_path, convert_to_abs_path, sort_files
from utils.time_processor import ms_to_string
from utils.video_iterators import stride_iterator, keyframe_iterator
from utils.species_classifier import SpeciesClassifier
from utils.reformat_gopro import valid_video
from ultralytics import YOLO
from tqdm import tqdm
import pandas as pd
import shutil
import torch
import cv2
import os

app = Flask(__name__)

class Model():
    def __init__(self, input_path, output_path, **kwargs):
        self.input_path = input_path
        self.max_video_cnt = kwargs.get("limit", 1000)
        self.stereo_prefix = kwargs.get("stereo_prefix")
        self.is_chapters = kwargs.get("chapters", False)
        self.species_classifier = SpeciesClassifier.build_species_classifier(kwargs.get("species_classifier"))
        self.output_path = output_path

        self.model_path = "models/sharktrack.pt"

        self.model_args = {
            "conf": kwargs.get("conf", 0.25),
            "iou": 0.5,
            "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
            "verbose": False,
        }

        if kwargs.get("peek", False):
            print("NOTE: You are using the model 'peek' mode. Please be aware of the following:")
            print("  ✅ Runs significantly faster")
            print("  ⛔️ You won't be able to compute MaxN, but only find interesting frames")
            print("  💡 You can safely ignore the ffmpeg warnings below")
            print("-" * 20)
            print("")
            self.inference_type = self.keyframe_detection
            self.model_args["imgsz"] = kwargs.get("imgsz", 640)
            self.save_output = save_peek_output
        else:
            self.inference_type = self.track_video
            self.fps = 3
            self.model_args["tracker"] = "trackers/tracker_3fps.yaml"
            self.model_args["persist"] = True
            self.model_args["imgsz"] = kwargs.get("imgsz", 640)
            self.save_output = save_analyst_output

        self.next_track_index, self.processed_videos = resume_previous_run(output_path)

    def track_video(self, video_path):
        """
        Uses ultralytics built-in tracker to automatically track a video with OpenCV.
        This is faster but it fails with GoPro Audio format, requiring reformatting.
        """
        print(f"Processing video: {video_path} on device {self.model_args['device']}.")
        model = YOLO(self.model_path)

        vid_stride, tot_frames_to_process = self._get_frame_skip(video_path)

        video_iterator = stride_iterator(video_path, vid_stride)

        sightings = []

        for frame, time, frame_idx in tqdm(video_iterator, total=tot_frames_to_process):
            results = model.track(
                frame,
                **self.model_args,
                stream=True
            )
            sightings += extract_sightings(video_path, self.input_path, next(results), frame_idx, ms_to_string(time), **{"tracking": True})

        sightings_df = pd.DataFrame(sightings)
        self.save_results(video_path, sightings_df, **{"fps": self.fps, "input": self.input_path})

    def _get_frame_skip(self, video_path):
        cap = cv2.VideoCapture(video_path)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = round(actual_fps / self.fps)
        tot_frames_to_process = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / frame_skip)
        cap.release()  # Ensure the video capture is released
        return frame_skip, tot_frames_to_process

    def save_results(self, video_path, yolo_results, **kwargs):
        """
        Save the results of the YOLO model inference.
        """
        kwargs["input"] = self.input_path
        kwargs["species_classifier"] = self.species_classifier
        kwargs["is_chapters"] = self.is_chapters
        next_track_index = self.save_output(video_path, yolo_results, self.output_path, self.next_track_index, **kwargs)
        assert next_track_index is not None, f"Error saving results for {video_path}"
        self.next_track_index = next_track_index

    def keyframe_detection(self, video_path):
        """
        Tracks keyframes using PyAv for SharkTrack peek version
        """
        print(f"Processing video: {video_path} on device {self.model_args['device']}")

        model = YOLO(self.model_path)

        video_iterator = keyframe_iterator(video_path)
        for frame, time, frame_idx in video_iterator:
            print(f"  processing frame {frame_idx}...", end="\r")
            frame_results = model(
                source=frame,
                **self.model_args
            )
            self.save_results(video_path, frame_results, **{"time": time, "frame_id": frame_idx})

    def run(self):
        self.input_path = self.input_path
        videos_already_processed = len(self.processed_videos)

        if valid_video(self.input_path):
            self.inference_type(self.input_path)
            self.processed_videos.add(self.input_path)
        else:
            for (root, _, files) in os.walk(self.input_path):
                files = sort_files(files)
                for file in files:
                    if len(self.processed_videos) == self.max_video_cnt:
                        print(f"Reached Maximum BRUVS count you specified of {self.max_video_cnt}. This includes the videos processed in previous run for the same output folder")
                        break
                    stereo_filter = self.stereo_prefix is None or file.startswith(self.stereo_prefix)
                    video_path = os.path.join(root, file)
                    if valid_video(video_path) and stereo_filter:
                        if video_path in self.processed_videos:
                            print(f"Video already processed in previous run {video_path}")
                            continue
                        self.inference_type(video_path)
                        self.processed_videos.add(video_path)

        if len(self.processed_videos) == videos_already_processed:
            print("No BRUVS videos found in the given folder")
            return

        return self.processed_videos

    def live_track(self, video_path, output_folder='./output'):
        """
        Plot the live tracking video for debugging purposes
        """
        print("Live tracking video...")
        if not valid_video(video_path):
            print(f"⚠️ Live Tracking is heavy so it can be done on only one video at a time. Please provide the full path of a single video with the --input argument")
            shutil.rmtree(output_folder)
            return
        cap = cv2.VideoCapture(video_path)

        filename = video_path.split('/')[-1]
        FILE_OUTPUT = os.path.join(output_folder, f"{filename.split('.')[0]}_tracked.avi")
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter(FILE_OUTPUT, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

        model = YOLO(self.model_path)

        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                frame_idx += 1
                print(f"  processing frame {frame_idx}...", end="\r")
                results = model.track(frame, **self.model_args)
                annotated_frame = results[0].plot()

                out.write(annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

@app.route('/process', methods=['GET'])
def process_videos():
    # Récupérer les paramètres depuis la requête GET
    input_path = request.args.get('path', './input_videos')
    stereo_prefix = request.args.get('stereo_prefix')
    limit = int(request.args.get('limit', 1000))
    imgsz = int(request.args.get('imgsz', 640))
    conf = float(request.args.get('conf', 0.25))
    output = request.args.get('output', None)
    species_classifier = request.args.get('species_classifier', None)
    peek = request.args.get('peek', 'false').lower() == 'true'
    chapters = request.args.get('chapters', 'false').lower() == 'true'
    live = request.args.get('live', 'false').lower() == 'true'
    resume = request.args.get('resume', 'false').lower() == 'true'

    input_path = os.path.normpath(input_path)
    input_path = convert_to_abs_path(input_path)
    output_path = generate_output_path(output, input_path, annotation_folder="internal_results", resume=resume)
    if output_path is None:
        return jsonify({"error": "Failed to generate output path."}), 400

    os.makedirs(output_path, exist_ok=True)

    model = Model(
        input_path,
        output_path,
        stereo_prefix=stereo_prefix,
        limit=limit,
        imgsz=imgsz,
        conf=conf,
        species_classifier=species_classifier,
        peek=peek,
        chapters=chapters
    )

    if live:
        model.live_track(input_path, output_path)
    else:
        processed_videos = model.run()

    return jsonify({"processed_videos": list(processed_videos)})

if __name__ == "__main__":
    # avoid duplicate libraries exception caused by numpy installation
    os.environ["KMP_DUPLICATE_LIB_OK"]="True"
    app.run(host='0.0.0.0', port=5001, debug=True)
