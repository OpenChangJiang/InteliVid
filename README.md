<div align="center">

# InteliVid

Video Agent

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

#### CLI

```bash
python keyframe_extraction.py -m scenedetect '.\assets\scenario_01\Humanoid Robots Showcase Folk Dance Skills on Spring Festival Gala Stage.mp4'

python multimodal_understanding.py

python video_summarization.py
```

#### GUI

```
python server.py
```

### FAQ

#### If there is no local GPU available, how should it be run?

Since DeepSeek has not officially provided Docker images for Janus Pro or Ollama models, and according to https://github.com/ollama/ollama/issues/8618 , the community is still in the process of adaptation, I am currently utilizing my local GPU for inference. Major cloud computing providers either have not yet deployed this model or have categorized it under Text-to-Image. If a GPU is not available, one might consider using APIs from other multimodal models.
