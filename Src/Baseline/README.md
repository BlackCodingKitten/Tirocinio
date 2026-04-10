### 🔹 **4 — Full Multimodal (Video + Transcription)**

**Setup:**
- ✅ Video input  
- ✅ Transcription provided  
- ✅ Prompt + options  

**Objective:**
Evaluate the model in a **fully informed multimodal setting**, where both visual and textual/audio-derived information are available.

**Key Aspects:**
- Joint reasoning across:
  - Visual dynamics (objects, actions, temporal evolution)
  - Linguistic content (dialogue, narration, semantic cues)
- Implicit modality fusion:
  - Alignment between what is seen and what is said
  - Resolution of ambiguities present in single modalities

**Interpretation:**
- Performance ≥ max(Experiment 2, Experiment 3):
  - Indicates **effective multimodal fusion**
- Performance ≈ one single modality:
  - Suggests **modality dominance** (unimodal collapse)
- Performance < both modalities:
  - Indicates **interference or fusion failure**

**Advanced Analysis:**
- Synergy gain:
  - `Accuracy(4) - max(Accuracy(2), Accuracy(3))`
- Redundancy check:
  - If transcription duplicates visual info, marginal gain expected
- Conflict resolution:
  - Evaluate cases where video and transcript suggest different answers

---

## 🔗 Final Comparison Overview

| Experiment | Video | Transcription | Role |
|----------|------|--------------|------|
| **1A** | ❌ | ❌ | Bias baseline |
| **1B** | ❌ | ❌ | Confidence calibration |
| **2** | ❌ | ✅ | Text-only reasoning |
| **3** | ✅ | ❌ | Vision-only reasoning |
| **4** | ✅ | ✅ | Multimodal fusion |

---

## 🧠 Core Insight

Experiment **4** is the **reference upper bound**:  
it reveals whether the model is truly **multimodal** or simply **selecting the easiest modality available**.