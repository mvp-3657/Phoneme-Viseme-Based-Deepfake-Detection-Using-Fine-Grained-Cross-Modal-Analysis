# Phoneme Viseme Mismatch: Fine Grained Cross Modal Deepfake Detection

> A research driven multimodal deepfake detection framework that explicitly models phoneme viseme temporal alignment using cross modal attention.

---

## 1. Introduction

Recent advances in deepfake generation have significantly improved visual realism and audio quality. Despite these improvements, maintaining precise synchronization between spoken audio (phonemes) and corresponding mouth movements (visemes) remains challenging. Subtle temporal inconsistencies often persist, even in high quality deepfakes.

Most existing deepfake detection approaches either:
- Focus on visual artifacts,
- Rely on unimodal cues (audio only or video only), or
- Apply coarse multimodal fusion assuming implicit alignment between modalities.

Such assumptions limit robustness against modern deepfakes.

This project addresses this limitation by explicitly modeling **phoneme viseme temporal alignment** as a forensic signal for deepfake detection.

---

## 2. Base Paper and Motivation

### Base Paper
**MultimodalTrace**  
*A Multimodal Audio Visual Deepfake Detection Framework with Late Fusion and Feature Level Integration* (SpringerLink)

**Key characteristics of the base paper:**
- Combines audio and visual features using late fusion
- Treats modalities as parallel inputs
- Achieves 92.9% accuracy on multimodal datasets

### Limitation
Late fusion strategies do not explicitly model *temporal correspondence* between speech and facial motion, which is critical for detecting subtle synchronization errors.

### Motivation
This project extends the base paper by removing the assumption of implicit alignment and instead **learning temporal correspondence directly** through cross modal attention.

---

## 3. Related Work

- **Agarwal et al., CVPRW 2020** demonstrated that phoneme viseme mismatch is an effective deepfake cue, but relied on coarse alignment.
- **MultimodalTrace** showed the benefits of multimodal learning but did not model fine grained temporal relationships.

This work bridges these approaches by introducing **fine grained, attention based temporal modeling**.

---

## 4. Key Novelty

> Instead of asking *“Do audio and video look real?”*,  
> this framework asks *“Are audio and video temporally aligned correctly?”*

### Novel Contributions
- Explicit phoneme viseme temporal alignment modeling
- Transformer based cross modal attention mechanism
- Detection of subtle synchronization errors missed by late fusion
- Improved accuracy over the base paper on the same dataset

---

## 5. Methodology Overview

### High-Level Pipeline
1. Extract phoneme aware audio embeddings
2. Extract viseme representative visual embeddings
3. Temporally align audio and video sequences
4. Apply cross modal attention
5. Perform binary classification (Real / Fake)

---

## 6. Dataset Information

### Dataset Used
**FakeAVCeleb v1.2**

FakeAVCeleb is a large scale audio visual deepfake dataset specifically designed to study audio video inconsistencies. It contains controlled combinations of real and manipulated audio and video streams.

### Manipulation Categories
- RealVideo–RealAudio
- FakeVideo–RealAudio
- RealVideo–FakeAudio
- FakeVideo–FakeAudio

For this project:
- RealVideo–RealAudio samples are labeled **Real**
- Any manipulated audio or video sample is labeled **Fake**

### Data Handling
- Subject wise train/validation/test split
- Prevents identity leakage
- Ensures fair generalization

⚠️ The dataset is not redistributed due to size and licensing constraints.

Official dataset source:  
https://github.com/DASH-Lab/FakeAVCeleb

---

## 7. Engineering Challenges and Solutions

### Challenges
- Large scale video processing
- Long training times under limited compute
- Runtime interruptions
- Severe class imbalance

### Solutions
- Cached audio and video embeddings
- Segment based training with checkpointing
- Class weighted loss functions
- Optimized decision threshold instead of fixed 0.5

These choices reflect practical engineering constraints rather than idealized assumptions.

---

## 8. Code and Project Walkthrough

### Metadata Construction
The dataset directory structure is parsed into structured metadata files containing:
- Video paths
- Subject IDs
- Manipulation labels

This metadata governs all downstream processing and ensures reproducibility.

### Audio Feature Extraction
- Model: Wav2Vec2
- Produces phoneme aware embeddings
- Cached to disk to avoid recomputation

### Visual Feature Extraction
- Model: ResNet 18
- Frame level embeddings representing visemes
- Lightweight and CPU compatible

### Temporal Alignment
Audio and video embeddings are synchronized across time to enable fine grained correspondence.

### Cross-Modal Attention (Core Component)
- Audio embeddings act as queries
- Visual embeddings act as keys and values
- Learns temporal alignment and mismatch patterns

### Classification
- Temporal pooling
- Binary classification head
- AdamW optimizer
- Binary Cross Entropy loss with class weighting

---

## 9. Training Strategy

- Training performed in controlled batch segments
- Frequent checkpointing for fault tolerance
- CPU compatible execution
- Stable optimization prioritizing reproducibility

---

## 10. Evaluation Metrics

- Accuracy
- F1-score (primary forensic metric)
- ROC-AUC (reported with careful interpretation)

An optimized decision threshold is selected using validation data to reflect real world forensic deployment.

---

## 11. Results

### Performance on FakeAVCeleb

| Method | Accuracy |
|------|----------|
| MultimodalTrace | 92.9% |
| **Proposed Framework** | **~97%** |

Additional results:
- F1 score ≈ 0.98
- ROC AUC ≈ 0.80

### Visual Diagnostics
- Confusion matrix
- ROC curve
- Accuracy comparison with base paper

---

## 12. Live Website (Hugging Face Deployment)

A live interactive demo of the proposed phoneme viseme deepfake detection system is publicly available via Hugging Face Spaces:

 **Live Demo:**  
https://huggingface.co/spaces/tilak-kateghar/deepfake-detector

This web interface allows users to upload a video and obtain a deepfake prediction based on phoneme viseme temporal consistency.

---

### 12.1 System Description

**Phoneme Viseme Deepfake Detection**

This system detects deepfake videos by analyzing the consistency between spoken audio (**phonemes**) and facial lip movements (**visemes**).  
Instead of only judging visual realism or audio quality independently, the system evaluates whether the timing and correspondence between speech and mouth motion are naturally aligned.

---

### 12.2 Recommended Usage Guidelines

For reliable and meaningful predictions, users are advised to follow these guidelines when uploading videos:

• Upload videos where a **human face is clearly visible**.  
• The person should be **speaking for most of the video duration**.  
• Avoid videos with **long off screen narration**.  
• Best results are obtained with a **single speaker facing the camera**.  
• Heavy motion blur, occlusion, or low resolution may affect predictions.  

Videos that do not meet these conditions may produce uncertain or unstable results due to insufficient phoneme viseme correspondence cues.

---

### 12.3 How to Use the Live Website

1. Open the live demo link:  
   https://huggingface.co/spaces/tilak-kateghar/deepfake-detector

2. Upload a video file using the **“Upload a video file”** button.

3. Click **Submit** to start analysis.

4. The system outputs:
   - A predicted label (**REAL** or **FAKE**) with confidence score  
   - A textual explanation describing the detected level of phoneme–viseme consistency

The output interpretation includes qualitative categories such as:
- Strong REAL  
- Probably REAL  
- Uncertain  
- Probably FAKE  
- Strong FAKE  

These categories help users understand the reliability of the prediction rather than relying only on a raw probability score.

---

### 12.4 How It Works (Deployment Perspective)

The deployed system implements the same trained architecture described in the research framework:

1. **Audio Processing:**  
   Speech audio is extracted from the uploaded video and encoded using **Wav2Vec2**, producing phoneme aware embeddings.

2. **Visual Processing:**  
   Video frames are sampled and passed through **ResNet-18** to extract visual motion features representing visemes.

3. **Cross-Modal Alignment:**  
   Audio embeddings and visual embeddings are aligned and passed into a **cross attention fusion network** that explicitly models phoneme–viseme correspondence.

4. **Classification:**  
   The fused representation is passed through a binary classifier producing a real/fake probability score.

---

### 12.5 Use of the Trained Model (`baseline_pvm_model.pt`)

The live web demo directly loads the trained checkpoint: **baseline_pvm_model.pt**

This model represents the trained baseline phoneme–viseme mismatch detector learned from the FakeAVCeleb dataset.  
It is loaded at runtime in the Hugging Face environment and used purely for inference (evaluation mode).

Key characteristics:
- No retraining occurs during deployment  
- Model weights remain fixed  
- Only forward inference is performed  
- CPU compatible execution  

This ensures that the live demo faithfully reflects the behavior of the trained research model rather than a simplified proxy.

---

### 12.6 Intended Use and Limitations

The live demo is intended for:
- Research demonstration
- Educational purposes
- Qualitative exploration of phoneme viseme mismatch behavior

It is **not intended as a production grade forensic system**.  
Predictions may be unreliable when:
- Faces are not visible
- Audio is missing or heavily compressed
- Multiple speakers appear
- Extreme camera angles or occlusions are present

The system should be interpreted as a proof-of-concept for phoneme viseme alignment based detection rather than a universal deepfake detector.

---

## 13. Reproducibility and Resources

Google Drive (view-only):  
https://drive.google.com/drive/folders/16UAnPF5pEzwA_TbdwQsyTpxC9f8CsdRi

Includes:
- Complete Jupyter Notebook implementation
- Trained model checkpoints
- Metadata splits
- Sample cached embeddings
- Result visualizations

---

## 14. References

1. Khalid et al., *MultimodalTrace: A Multimodal Audio-Visual Deepfake Detection Framework with Late Fusion and Feature-Level Integration*, SpringerLink.
2. Agarwal et al., *Detecting Deep-Fake Videos from Phoneme–Viseme Mismatches*, CVPR Workshops, 2020.
3. FakeAVCeleb Dataset: A Large-Scale Audio-Visual Deepfake Dataset for Studying Audio–Video Inconsistencies.  
   https://github.com/DASH-Lab/FakeAVCeleb
4. Baevski et al., *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*, NeurIPS 2020.
5. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
6. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.

---

## 15. Disclaimer

This repository is intended for **academic and research purposes only**.  
Commercial use of the dataset or trained models must comply with the respective dataset and license terms.

## Citation

If you use this work, codebase, or experimental setup in your research or projects, please cite the following resources accordingly.

### Proposed Framework
Phoneme Viseme Mismatch: A Fine Grained Cross Modal Attention Framework for Robust Deepfake Detection  
GitHub Repository: https://github.com/Tilak-Kateghar/phoneme-viseme-deepfake-detection

### Base Paper
Khalid et al.,  
*MultimodalTrace: A Multimodal Audio Visual Deepfake Detection Framework with Late Fusion and Feature Level Integration*,  
SpringerLink.

### Phoneme–Viseme Mismatch Reference
Agarwal, S., Chowdary, A., and Venkatesh, R.,  
*Detecting Deep Fake Videos from Phoneme Viseme Mismatches*,  
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2020.  
https://openaccess.thecvf.com/content_CVPRW_2020/papers/w39/Agarwal_Detecting_Deep-Fake_Videos_From_Phoneme-Viseme_Mismatches_CVPRW_2020_paper.pdf

### Dataset
Khalid, H. et al.,  
*FakeAVCeleb: A Large Scale Audio Visual Deepfake Dataset for Studying Audio Video Inconsistencies*.  
Official Repository: https://github.com/DASH-Lab/FakeAVCeleb

### Core Technologies
- Baevski et al., *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*, NeurIPS 2020.
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.