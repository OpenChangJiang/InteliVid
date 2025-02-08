<div align="center">

# InteliVid
Inteli(gent)Vid(eo), based on DeepSeek-V3 and DeepSeek-Janus-Pro.

![GitHub license](https://img.shields.io/github/license/OpenChangJiang/InteliVid)
![GitHub stars](https://img.shields.io/github/stars/OpenChangJiang/InteliVid)

**English** | [**简体中文**](docs/cn/README.md)

</div>

### Supported

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
