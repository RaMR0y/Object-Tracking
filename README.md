## Object Tracking with Adaptive Tracker Switching (YOLOv8 + DeepSORT)

This project implements an object tracking pipeline that combines YOLOv8 object detection with adaptive tracker switching between three trackers:

- Simple tracker (ID assignment only)
- DeepSORT (deep features + Kalman filter, robust re-ID)
- Kalman-based particle-style tracker (prototype)

The system monitors tracking quality and automatically switches to a more robust tracker when quality drops, with basic hysteresis to avoid rapid toggling. A demo notebook is provided in `Parallelizaton.ipynb` and is designed to run on Google Colab (with Google Drive for input video).


### Features
- **YOLOv8 detection** using `yolov8n.pt` (pretrained) and example training on `coco8.yaml` in-notebook
- **Adaptive tracker switching**: Simple ➜ DeepSORT ➜ Particle-style Kalman when quality degrades
- **Visualization**: Draws bounding boxes and track IDs on each frame
- **Colab-friendly**: Minimal setup using pip installs inside the notebook


### Repository Contents
- `Parallelizaton.ipynb`: End-to-end demo notebook (Colab-ready)
- `README.md`: Project overview and usage
- `Project Delivery Report.pdf`: Project report (reference)


## Quick Start (Colab)
1. Open `Parallelizaton.ipynb` in Google Colab.
2. Run the first cell to mount Google Drive:
   - Your input video should be in your Drive, e.g. `My Drive/Yolov8/test.mp4`.
3. Run the install cell (installs `ultralytics`, `deep-sort-realtime`, `filterpy`, `tensorflow`).
4. Execute the remaining cells to:
   - Load YOLOv8
   - Optionally train/validate on the toy dataset (`coco8.yaml`)
   - Run the tracker on your video


## Local Setup (Optional)
While the notebook is optimized for Colab, you can run locally if you have a compatible environment.

Prerequisites:
- Python 3.9+
- GPU recommended (CUDA) for real-time performance, but CPU will work slower

Install dependencies:

```bash
pip install ultralytics deep-sort-realtime filterpy tensorflow opencv-python numpy
```

Notes:
- `tensorflow` is used in the notebook; depending on your platform, you may prefer `tensorflow==2.15.*` or a GPU build.
- If you only need DeepSORT without TensorFlow, consider adapting the feature extraction pipeline accordingly.


## How It Works
1. YOLOv8 detects objects per frame, yielding bounding boxes and confidences.
2. A feature hook extracts appearance embeddings for DeepSORT re-identification.
3. The current tracker updates tracks for the frame.
4. A simple tracking-quality heuristic evaluates if quality is low for several consecutive frames. If so, the system upgrades the tracker (Simple ➜ DeepSORT ➜ Particle-style Kalman).
5. The notebook visualizes results inline.

Key notebook classes/functions:
- `SimpleTracker`: Naïve ID assignment per detection
- `DeepSORTTracker`: Wraps `deep_sort_realtime.DeepSort`
- `ParticleFilterTracker`: Prototype Kalman-based tracker
- `tracking_quality_low(...)`: Heuristic to trigger switching
- `track_video(video_path)`: Orchestrates detection, tracking, quality checks, and visualization


## Usage (Notebook)
Inside `Parallelizaton.ipynb`, adjust the input video path and run:

```python
video_path = '/content/gdrive/My Drive/Yolov8/test.mp4'
track_video(video_path)
```

Tips:
- Make sure the video exists at the specified Drive path after mounting.
- To use your own video, upload it to Drive and update `video_path` accordingly.


## Configuration
- **Model**: `yolov8n.pt` (changeable if you prefer another YOLOv8 variant)
- **Quality heuristic**: Thresholds inside `tracking_quality_low` and consecutive low-quality frame count (`quality_threshold_frames`) control switching sensitivity
- **DeepSORT settings**: `DeepSort(max_age=30, n_init=3)` used by default; tune for your data


## Results & Expectations
- On modest hardware, YOLOv8n with DeepSORT typically achieves smooth multi-object tracking on simple videos.
- Real-time speed depends on your GPU/CPU and video resolution.
- The particle-style tracker is a prototype and primarily included as a fallback/experimental component.


## Limitations
- The particle-style tracker is not a full particle filter and may need refinement.
- The tracking-quality metric is simplistic and can be improved.
- Notebook visualization uses `cv2_imshow` (Colab). For local runs, adapt display logic as needed.


## Future Enhancements
- Implement a true particle filter for non-linear/non-Gaussian motion
- Improve tracking-quality metrics and switching strategy
- Add quantitative metrics (MOTA, MOTP, IDF1) and clearer benchmarking
- Provide a standalone Python script/CLI outside the notebook


## Acknowledgements
- YOLOv8 by Ultralytics
- DeepSORT and `deep-sort-realtime`
- `filterpy` Kalman utilities


## License
If you plan to open-source, add a license (e.g., MIT). Otherwise, keep usage internal.
