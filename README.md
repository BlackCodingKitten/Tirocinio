<div align="center">

# 🎓 <span style="color:#8A2BE2;">Internship Project</span>

### <span style="color:#8A2BE2;">Automatic Video Audio Transcription and Multimodal Benchmark Enrichment</span>

<p align="center">
<a href="https://www.fbk.eu/en/" target="_blank">
  <img src="https://img.shields.io/badge/Research-FBK-8A2BE2?style=for-the-badge" alt="FBK Research Badge" />
</a>
  <img src="https://img.shields.io/badge/Domain-Multimodal%20AI-7B1FA2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Task-Automatic%20Transcription-6A1B9A?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Models-Whisper--1%20%7C%20GPT--4.0-9C27B0?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Focus-Benchmark%20Enrichment-AB47BC?style=for-the-badge" />
</p>

<p align="center">
  <em>A research internship project developed in collaboration with <a href="https://www.fbk.eu/en/">FBK – Fondazione Bruno Kessler</a> to explore automatic transcription, transcript merging, multimodal benchmark enrichment, and new evaluation baselines.</em>
</p>

</div>

---

## 💼 <span style="color:#8A2BE2;">Project Overview</span>

This internship project is carried out in collaboration with **[FBK – Fondazione Bruno Kessler](https://www.fbk.eu/en/)** and is supervised by **Dr. Bernardo Magnini** and **Davide Testa**.

The project focuses on the **automatic transcription of audio extracted from videos**, with the broader aim of supporting research on **multimodal AI systems** and their performance under different informational conditions.

---

## 🎧 <span style="color:#8A2BE2;">What the Project Does</span>

The core activity of the project consists of generating **automatic transcriptions of video audio** using two models:

- **Whisper-1**
- **GPT-4.0**

The outputs produced by these models are then **merged and combined** in order to obtain a more reliable final transcription of the dialogues.

This merging process is based on the **confidence with which the models predict the text**, allowing the system to select, combine, and refine transcription segments according to the reliability of each prediction.

---

## ⚙️ <span style="color:#8A2BE2;">Workflow</span>

The project pipeline can be summarized as follows:

1. **Audio extraction** from video sources  
2. **Automatic transcription** using **Whisper-1** and **GPT-4.0**  
3. **Comparison of transcription outputs**  
4. **Merge of the results** based on the confidence associated with the predicted text  
5. **Combination of dialogue transcriptions** into a more accurate final version  
6. **Integration of textual information** into the benchmark  
7. **Creation of new baselines** for evaluation  

This approach makes it possible to leverage the complementary strengths of different transcription models and improve the overall quality of the generated text while supporting a more robust evaluation framework.

---

## 🔬 <span style="color:#8A2BE2;">Next Phases</span>

In the following stages of the project, the automatic transcription is used to **enrich the MAIA Benchmark** by adding the **textual content spoken in the video**.

The goal is to use this enriched benchmark to evaluate:

- **unimodal collapse**
- how the **performance of multimodal models changes** when they are provided with **more or less information** about the video
- the effectiveness of **new baselines** designed for a more structured and informative evaluation process

More specifically, the project investigates how multimodal systems behave when textual information extracted from audio is added to the available input, and whether this additional information affects model robustness, balance across modalities, and overall task performance.

---

## 🧠 <span style="color:#8A2BE2;">Research Motivation</span>

A video is not only a sequence of visual frames: it also contains spoken information that can reveal intentions, actions, relations, goals, and event dynamics that may not be fully captured by vision alone. For this reason, video understanding should be treated as a genuinely multimodal problem, especially when the goal is to evaluate how models reason over events unfolding in time. This idea is fully consistent with the motivation behind MAIA, which was introduced as a competence-oriented benchmark for the fine-grained analysis of multimodal reasoning on videos. :contentReference[oaicite:0]{index=0}

This perspective is explicitly grounded in [*MAIA: a Benchmark for Multimodal AI Assessment*](https://aclanthology.org/2025.clicit-1.106.pdf) by **Testa, Bonetta, Bernardi, Bondielli, Lenci, Miaschi, Passaro, and Magnini (2025)** and further developed in [*All-in-one: Understanding and Generation in Multimodal Reasoning with the MAIA Benchmark*](https://arxiv.org/abs/2502.16989) by **Testa, Bonetta, Bernardi, Bondielli, Lenci, Miaschi, Passaro, and Magnini (2025)**. These works frame MAIA as a benchmark designed to disentangle the contribution of language and vision across multiple reasoning categories, while also exposing situations in which multimodal models rely too heavily on a single modality rather than achieving genuinely grounded multimodal understanding. 

Starting from this perspective, this project explores the idea that automatically transcribed audio can enrich the **MAIA Benchmark** by adding a textual layer directly grounded in the video. The key assumption is not that audio and transcription are always beneficial, but rather that they become valuable when the spoken content is actually related to the events taking place in the scene. When speech is semantically aligned with the actions, entities, or temporal structure of the video, transcription can provide complementary evidence and improve understanding. When this correlation is weak or absent, the additional text may contribute little or may even introduce noise. This is coherent with the MAIA framework, whose design is meant precisely to analyze how different modalities contribute to reasoning under controlled conditions. 

This point is especially relevant for the study of **unimodal collapse**, namely the tendency of multimodal models to over-rely on one modality instead of integrating all available evidence in a balanced way. In the MAIA line of work, this issue emerges through the need for more robust and fine-grained evaluation settings, capable of distinguishing genuine multimodal reasoning from performance driven by linguistic priors, shallow shortcuts, or modality-specific biases. Within this context, adding speech transcription is not just a data enrichment step: it becomes a principled way to test whether extra textual information truly supports video understanding or whether it simply reinforces language-dominant behavior. 

More broadly, the project is based on the conviction that **audio and transcription do influence the final outcome of multimodal evaluation, but only when they are meaningfully connected to what is happening in the video**. In this sense, enriching MAIA with spoken content makes it possible to study not only whether more information improves performance, but also whether that improvement reflects better grounded multimodal understanding rather than a stronger dependence on language alone.

---

## 🚀 <span style="color:#8A2BE2;">Main Objectives</span>

- Develop an automatic transcription pipeline for video audio  
- Compare transcription outputs from **Whisper-1** and **GPT-4.0**  
- Merge transcripts according to **prediction confidence**  
- Produce improved dialogue transcriptions  
- Enrich the **MAIA Benchmark** with textual information extracted from videos  
- Create **new baselines** for evaluation  
- Evaluate the impact of additional textual information on **multimodal model performance**  
- Study the phenomenon of **unimodal collapse**

---

## 🧩 <span style="color:#8A2BE2;">Evaluation Perspective</span>

A relevant part of the project is dedicated not only to transcription quality, but also to the design of a stronger evaluation setting.

By enriching the benchmark and introducing **new baselines**, the project aims to provide clearer points of comparison for measuring how multimodal models behave under different input configurations, especially when textual information is added or removed.

---

## 📌 <span style="color:#8A2BE2;">Project Summary</span>

This internship project combines:

- **automatic speech transcription**
- **multi-model transcript merging**
- **confidence-based dialogue combination**
- **benchmark enrichment for multimodal evaluation**
- **creation of new evaluation baselines**
- **analysis of unimodal collapse in multimodal systems**

Overall, the project contributes to the study of **multimodal AI evaluation**, with a focus on understanding how the presence or absence of textual information influences model behavior and performance.

---

## 🤝 <span style="color:#8A2BE2;">Collaboration</span>

**Institution:** [FBK — Fondazione Bruno Kessler](https://www.fbk.eu/en/)  
**Supervisors:** **Dr. Bernardo Magnini** and **Davide Testa**

---

<div align="center">

### ✨ <span style="color:#8A2BE2;">Researching transcription, fusion, benchmark enrichment, and multimodal understanding</span>

</div>