# InteliVid

Inteli(gent)Vid(eo), based on DeepSeek-V3 and DeepSeek-Janus-Pro.

Supported:

- Video Summarization
- Video Q&A
- Video Classification
- Video Semantic Search

### Installation

```bash
conda create -n InteliVid python=3.10
```

### Quickstart

```bash
python keyframe_extraction.py -m scenedetect '.\assets\scenario_01\Humanoid Robots Showcase Folk Dance Skills on Spring Festival Gala Stage.mp4'

python multimodal_understanding.py

python video_summarization.py
```
