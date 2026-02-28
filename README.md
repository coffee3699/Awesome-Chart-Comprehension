# Awesome Chart Comprehension in the Era of Multimodal LLMs

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

> Now we have a Chinese version available [here](https://lidongyuan.notion.site/chart-survey)!

## Why is Chart Comprehension a Unique Challenge?
Chart images are not just "visual inputs"; they are "visual expressions of structured data."

1.  **Underlying Structured Information**: Charts are visual mappings of data (using position, length, color, etc.). They contain an implicit tabular structure that models must reverse-engineer. Natural images, in contrast, depict unstructured scenes.
2.  **Unique Components**: Charts are a tight coupling of graphical elements (bars, lines, points) and textual elements (titles, axes, legends). Both must be parsed accurately for full comprehension. In natural images, text is often secondary.
3.  **Distinct Low-level Features**: Charts consist of simple, regular features like solid colors, sharp geometric shapes, and clean boundaries. Their complexity is logical and relational, not textural or lighting-based like in natural images.


## Pre-training Tasks for Chart Comprehension
Specialized pre-training tasks are designed to help models better understand the **numerical, structural, and semantic** information in charts.

1.  **Structural Alignment Tasks**: These tasks convert charts into structured formats.
    *   **Chart-to-Table**: Reconstructs the underlying data table, enhancing numerical extraction and alignment.
    *   **Chart-to-JSON/CSV**: Converts chart elements into a hierarchical structure, improving semantic structure understanding.
    *   **Chart-to-Code**: Generates the plotting script (e.g., in Python), capturing both data and rendering logic, which boosts abstract reasoning.
2.  **Language Alignment Tasks**: These tasks align charts with natural language descriptions.
    *   **Chart Captioning/Description**: Generates a sentence or paragraph describing the chart's content.
    *   **Chart Summarization**: Generates a high-level summary or the main conclusion from the chart.

## Downstream Tasks
| Level | Representative Task | Typical Output | Main Challenge |
| :--- | :--- | :--- | :--- |
| **Perception** | Type Recognition, Text Extraction | Chart Category, OCR Text | Visual Structure Modeling |
| **Structural Understanding** | Metadata Extraction, Chart Reconstruction | Table / JSON / Code | OCR & Geometric Alignment |
| **Semantic Understanding**| QA, Summarization, Description | Natural Language Answer | Multimodal Alignment, Numerical Reasoning |
| **Advanced Reasoning** | Complex QA, Chart Generation/Editing | Natural Language Answer, Code  | Symbolic/Casual/Visual Reasoning, Tool Use |

## Table of Contents
* [Models and Methods](#models-and-methods)
	* [Two-Stage Paradigm: First Recognize, Then Understand](#two-stage-paradigm-first-recognize-then-understand)
	* [End-to-End Models: Supervised Fine-Tuning at Scale](#end-to-end-models-supervised-fine-tuning-at-scale)
	* [Innovations in Training and Architecture](#innovations-in-training-and-architecture)
	* [Beyond SFT: Reinforcement Learning and Agentic Systems](#beyond-sft-reinforcement-learning-and-agentic-systems)
	* [OCR Models](#ocr-models)
* [Benchmarks and Datasets](#benchmarks-and-datasets)
	* [Establishing the Task: Synthetic & Template-Based](#establishing-the-task-synthetic--template-based)
	* [Pushing for Complexity and Diversity](#pushing-for-complexity-and-diversity)
	* [The Generative Paradigm](#the-generative-paradigm)
	* [New Frontiers: Evaluating Trust, Safety, and Access](#new-frontiers-evaluating-trust-safety-and-access)
* [Analysis Papers](#analysis-papers)



## Models and Methods

### Two-Stage Paradigm: First Recognize, Then Understand
Early research followed a two-stage approach: an extraction module first parses the chart image into an intermediate representation (like a table), which is then fed to a language model (LM) for reasoning.

- __Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding.__
  _Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu, Fangyu Liu, Julian Martin Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, Kristina Toutanova._ <img src='https://img.shields.io/badge/ICML-2023-yellow'> <a href='https://proceedings.mlr.press/v202/lee23g/lee23g.pdf'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://huggingface.co/google/pix2struct-base'><img src='https://img.shields.io/badge/Model-green'></a>

- __MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering.__
  _Fangyu Liu, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Yasemin Altun, Nigel Collier, Julian Eisenschlos._ <img src='https://img.shields.io/badge/ACL-2023-yellow'> <a href='https://aclanthology.org/2023.acl-long.714/'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://huggingface.co/google/matcha-base'><img src='https://img.shields.io/badge/Model-green'></a>

- __DePlot: One-shot visual language reasoning by plot-to-table translation.__
  _Fangyu Liu, Julian Eisenschlos, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Wenhu Chen, Nigel Collier, Yasemin Altun._ <img src='https://img.shields.io/badge/ACL_Findings-2023-yellow'> <a href='https://aclanthology.org/2023.findings-acl.660/'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://huggingface.co/google/deplot'><img src='https://img.shields.io/badge/Model-green'></a>

### End-to-End Models: Supervised Fine-Tuning at Scale
This paradigm shift was driven by the insight that model capabilities could be significantly enhanced through *large-scale, high-quality, and diverse* training data.

- __UniChart: A Universal Vision-language Pretrained Model for Chart Comprehension and Reasoning.__
  _Ahmed Masry, Parsa Kavehzadeh, Xuan Long Do, Enamul Hoque, Shafiq Joty._ <img src='https://img.shields.io/badge/EMNLP-2023-yellow'> <a href='https://aclanthology.org/2023.emnlp-main.906/'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://github.com/vis-nlp/UniChart'><img src='https://img.shields.io/badge/Code-green'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/vis-nlp/UniChart?style=social)

- __ChartLlama: A Multimodal LLM for Chart Understanding and Generation.__ 
  _Yucheng Han, Chi Zhang, Xin Chen, Xu Yang, Zhibin Wang, Gang Yu, Bin Fu, Hanwang Zhang._ <img src='https://img.shields.io/badge/Arxiv-2023-yellow'> <a href='https://arxiv.org/abs/2311.16483'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://github.com/tingxueronghua/ChartLlama-code'><img src='https://img.shields.io/badge/Code-green'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/tingxueronghua/ChartLlama-code?style=social)

- __MMC: Advancing Multimodal Chart Understanding with Large-scale Instruction Tuning.__ 
  _Fuxiao Liu, Xiaoyang Wang, Wenlin Yao, Jianshu Chen, Kaiqiang Song, Sangwoo Cho, Yaser Yacoob, Dong Yu._ <img src='https://img.shields.io/badge/NAACL-2024-yellow'> <a href='https://arxiv.org/abs/2311.10774'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://github.com/FuxiaoLiu/MMC'><img src='https://img.shields.io/badge/Code-green'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/FuxiaoLiu/MMC?style=social)

- __ChartAssistant: A Universal Chart Multimodal Language Model via Chart-to-Table Pre-training and Multitask Instruction Tuning.__ 
  _Fanqing Meng, Wenqi Shao, Quanfeng Lu, Peng Gao, Kaipeng Zhang, Yu Qiao, Ping Luo._ <img src='https://img.shields.io/badge/ACL_Findings-2024-yellow'> <a href='https://arxiv.org/abs/2401.02384'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://github.com/OpenGVLab/ChartAst'><img src='https://img.shields.io/badge/Code-green'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/OpenGVLab/ChartAst?style=social)

- __ChartX & ChartVLM: A Versatile Benchmark and Foundation Model for Complicated Chart Reasoning.__ 
  _Renqiu Xia, Bo Zhang, Hancheng Ye, Xiangchao Yan, Qi Liu, Hongbin Zhou, Zijun Chen, Min Dou, Botian Shi, Junchi Yan, Yu Qiao._ <img src='https://img.shields.io/badge/Arxiv-2024-yellow'> <a href='https://arxiv.org/abs/2402.12185'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://github.com/UniModal4Reasoning/ChartVLM'><img src='https://img.shields.io/badge/Code-green'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/UniModal4Reasoning/ChartVLM?style=social)

- __ChartInstruct: Instruction Tuning for Chart Comprehension and Reasoning.__ 
  _Ahmed Masry, Mehrad Shahmohammadi, Md Rizwan Parvez, Enamul Hoque, Shafiq Joty._ <img src='https://img.shields.io/badge/ACL_Findings-2024-yellow'> <a href='https://arxiv.org/abs/2403.09028'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://github.com/vis-nlp/ChartInstruct'><img src='https://img.shields.io/badge/Code-green'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/vis-nlp/ChartInstruct?style=social)

- __ChartGemma: Visual Instruction-tuning for Chart Reasoning in the Wild.__ 
  _Ahmed Masry, Megh Thakkar, Aayush Bajaj, Aaryaman Kartha, Enamul Hoque, Shafiq Joty._ <img src='https://img.shields.io/badge/COLING-2025-yellow'> <a href='https://arxiv.org/abs/2407.04172v1'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://github.com/vis-nlp/ChartGemma'><img src='https://img.shields.io/badge/Code-green'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/vis-nlp/ChartGemma?style=social)

### Innovations in Training and Architecture
This section highlights the ongoing search for deeper capabilities and greater efficiency of chart models. It covers the trend of developing specialized training techniques and novel architectural components designed to master the unique logical and structural complexity of chart comprehension.

- __TinyChart: Efficient Chart Understanding with Visual Token Merging and Program-of-Thoughts Learning.__ 
  _Liang Zhang, Anwen Hu, Haiyang Xu, Ming Yan, Yichen Xu, Qin Jin, Ji Zhang, Fei Huang._ <img src='https://img.shields.io/badge/EMNLP-2024-yellow'> <a href='https://arxiv.org/abs/2404.16635'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/TinyChart'><img src='https://img.shields.io/badge/Code-green'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/X-PLUG/mPLUG-DocOwl?style=social)

- __On Pre-training of Multimodal Language Models Customized for Chart Understanding.__
  _Wan-Cyuan Fan, Yen-Chun Chen, Mengchen Liu, Lu Yuan, Leonid Sigal._ <img src='https://img.shields.io/badge/NeurlPS_Workshop-2024-yellow'> <a href='https://arxiv.org/abs/2407.14506'><img src='https://img.shields.io/badge/PDF-blue'></a>

- __ChartMoE: Mixture of Expert Connector for Advanced Chart Understanding.__ 
  _Zhengzhuo Xu⁺, Bowen Qu⁺, Yiyan Qi⁺, Sinan Du, Chengjin Xu, Chun Yuan, Jian Guo._ <img src='https://img.shields.io/badge/ICLR-2025-yellow'> <a href='https://arxiv.org/abs/2409.03277'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://huggingface.co/IDEA-FinAI/chartmoe'> <a href='https://github.com/DataArcTech/ChartMoE'><img src='https://img.shields.io/badge/Code-green'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/DataArcTech/ChartMoE?style=social)

- __EvoChart: A Self-Training Framework for Chart Data Synthesis and Model Co-Evolution.__
  _Muye Huang, Han Lai, Xinyu Zhang, Wenjun Wu, Jie Ma, Lingling Zhang, Jun Liu._ <img src='https://img.shields.io/badge/AAAI-2025-yellow'> <a href='https://arxiv.org/abs/2409.01577'><img src='https://img.shields.io/badge/PDF-blue'></a> <a href='https://github.com/MuyeHuang/EvoChart'><img src='https://img.shields.io/badge/Code-green'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/MuyeHuang/EvoChart?style=social)

- __Distill Visual Chart Reasoning Ability from LLMs to MLLMs.__
  _Wei He, Zhiheng Xi, Wanxu Zhao, Xiaoran Fan, Yiwen Ding, Zifei Shan, Tao Gui, Qi Zhang, Xuanjing Huang._ <img src='https://img.shields.io/badge/EMNLP_Findings-2025-yellow'> <a href='https://arxiv.org/abs/2410.18798'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://github.com/hewei2001/ReachQA'><img src='https://img.shields.io/badge/Code-green'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/hewei2001/ReachQA?style=social)

### Beyond SFT: Reinforcement Learning and Agentic Systems
Inspired by the debut and success of [DeepSeek-R1](https://arxiv.org/abs/2501.12948) and [OpenAI o3](https://openai.com/index/thinking-with-images/), the field turned to reinforcement learning and more dynamic, interactive training methods to unlock new capabilities, like visual reasoning.

- __Bespoke-MiniChart-7B: Pushing The Frontiers Of Open VLMs For Chart Understanding.__
  _Liyan Tang, Shreyas Pimpalgaonkar, Kartik Sharma, Alexandros G. Dimakis, Mahesh Sathiamoorthy, Greg Durrett._ <img src='https://img.shields.io/badge/Blog-2025-yellow'> <a href='https://www.bespokelabs.ai/blog/bespoke-minichart-7b'><img src='https://img.shields.io/badge/Website-blue'></a>
  <a href='https://huggingface.co/bespokelabs/Bespoke-MiniChart-7B'><img src='https://img.shields.io/badge/Model-green'></a>

- __ChartSketcher: Reasoning with Multimodal Feedback and Reflection for Chart Understanding.__
  _Muye Huang, Lingling Zhang, Jie Ma, Han Lai, Fangzhi Xu, Yifei Li, Wenjun Wu, Yaqiang Wu, Jun Liu._ <img src='https://img.shields.io/badge/Arxiv-2025-yellow'> <a href='https://arxiv.org/abs/2505.19076'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://github.com/MuyeHuang/ChartSketcher'><img src='https://img.shields.io/badge/Code-green'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/MuyeHuang/ChartSketcher?style=social)

- __Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation.__
  _Lei Chen, Xuanle Zhao, Zhixiong Zeng, Jing Huang, Liming Zheng, Yufeng Zhong, Lin Ma._ <img src='https://img.shields.io/badge/Arxiv-2025-yellow'> <a href='https://arxiv.org/abs/2508.13587'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://github.com/DocTron-hub/MSRL'><img src='https://img.shields.io/badge/Code-green'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/DocTron-hub/MSRL?style=social)

- __Chart-R1: Chain-of-Thought Supervision and Reinforcement for Advanced Chart Reasoner.__
  _Lei Chen, Xuanle Zhao, Zhixiong Zeng, Jing Huang, Yufeng Zhong, Lin Ma._ <img src='https://img.shields.io/badge/Arxiv-2025-yellow'> <a href='https://arxiv.org/abs/2507.15509'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://github.com/DocTron-hub/Chart-R1'><img src='https://img.shields.io/badge/Code-green'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/DocTron-hub/Chart-R1?style=social)

- __BigCharts-R1: Enhanced Chart Reasoning with Visual Reinforcement Finetuning.__
  _Ahmed Masry, Abhay Puri, Masoud Hashemi, Juan A. Rodriguez, Megh Thakkar, Khyati Mahajan, Vikas Yadav, Sathwik Tejaswi Madhusudhan, Alexandre Piché, Dzmitry Bahdanau, Christopher Pal, David Vazquez, Enamul Hoque, Perouz Taslakian, Sai Rajeswar, Spandana Gella._ <img src='https://img.shields.io/badge/COLM-2025-yellow'> <a href='https://arxiv.org/abs/2508.09804'><img src='https://img.shields.io/badge/PDF-blue'></a>

- __Do MLLMs Really Understand the Charts?__
  _Xiao Zhang, Dongyuan Li, Liuyu Xiang, Yao Zhang, Cheng Zhong, Zhaofeng He._ <img src='https://img.shields.io/badge/Arxiv-2025-yellow'> <a href='https://arxiv.org/abs/2509.04457'><img src='https://img.shields.io/badge/PDF-blue'></a>

- __ChartAgent: A Multimodal Agent for Visually Grounded Reasoning in Complex Chart Question Answering.__
  _Rachneet Kaur, Nishan Srishankar, Zhen Zeng, Sumitra Ganesh, Manuela Veloso._ <img src='https://img.shields.io/badge/Arxiv-2025-yellow'> <a href='https://arxiv.org/abs/2510.04514'><img src='https://img.shields.io/badge/PDF-blue'></a>

### OCR Models
Industry leading lightweight models. Excel at extracting structural information from charts (but not limited to) with impressive speed and accuracy.

- __PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model.__
  _Cheng Cui, Ting Sun, Suyin Liang, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Xueqing Wang, Changda Zhou, Hongen Liu, Manhui Lin, Yue Zhang, Yubo Zhang, Handong Zheng, Jing Zhang, Jun Zhang, Yi Liu, Dianhai Yu, Yanjun Ma._ <img src='https://img.shields.io/badge/Arxiv-2025-yellow'> <a href='https://arxiv.org/abs/2510.14528'><img src='https://img.shields.io/badge/PDF-blue'></a> <a href='https://github.com/PaddlePaddle/PaddleOCR'><img src='https://img.shields.io/badge/Code-green'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR)

- __DeepSeek-OCR: Contexts Optical Compression.__
  _Haoran Wei, Yaofeng Sun, Yukun Li._ <img src='https://img.shields.io/badge/Arxiv-2025-yellow'> <a href='https://arxiv.org/abs/2510.14528'><img src='https://img.shields.io/badge/PDF-blue'></a> <a href='https://github.com/PaddlePaddle/PaddleOCR'><img src='https://img.shields.io/badge/Code-green'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/deepseek-ai/DeepSeek-OCR)

## Benchmarks and Datasets

### Establishing the Task: Synthetic & Template-Based
The initial phase of modern chart understanding research was dedicated to establishing Chart Question Answering (CQA) as a core evaluation task. These foundational benchmarks were instrumental in defining the problem, but were characterized by computer-generated charts, template-based questions, and simple visual queries (e.g., finding a maximum value).

- __DVQA: Understanding Data Visualizations via Question Answering.__ 
  _Kushal Kafle, Brian Price, Scott Cohen, Christopher Kanan._ <img src='https://img.shields.io/badge/CVPR-2018-yellow'> <a href='https://openaccess.thecvf.com/content_cvpr_2018/papers/Kafle_DVQA_Understanding_Data_CVPR_2018_paper.pdf'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://github.com/kushalkafle/DVQA_dataset'><img src='https://img.shields.io/badge/Dataset-red'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/kushalkafle/DVQA_dataset?style=social)

- __FigureQA: An Annotated Figure Dataset for Visual Reasoning.__ 
  _Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Akos Kadar, Adam Trischler, Yoshua Bengio._ <img src='https://img.shields.io/badge/ICLR_Workshop-2018-yellow'> <a href='https://arxiv.org/abs/1710.07300'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://www.microsoft.com/en-us/research/project/figureqa-dataset/'><img src='https://img.shields.io/badge/Dataset-red'></a>

- __PlotQA: Reasoning over Scientific Plots.__ 
  _Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra, Pratyush Kumar._ <img src='https://img.shields.io/badge/WACV-2020-yellow'> <a href='https://arxiv.org/abs/1909.00997'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://github.com/NiteshMethani/PlotQA'><img src='https://img.shields.io/badge/Dataset-red'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/NiteshMethani/PlotQA?style=social)

### The Generative Evaluation
While question answering remains a central task, a more unique demonstration of chart understanding involves generation—creating new artifacts based on the visual information.

- __ChartSumm: A Comprehensive Benchmark for Automatic Chart Summarization of Long and Short Summaries.__
  _Raian Rahman, Rizvi Hasan, Abdullah Al Farhad, Md Tahmid Rahman Laskar, Md. Hamjajul Ashmafee, Abu Raihan Mostofa Kamal._ <img src='https://img.shields.io/badge/CANAI-2023-yellow'> <a href='https://arxiv.org/abs/2304.13620'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://github.com/pranonrahman/ChartSumm'><img src='https://img.shields.io/badge/Dataset-red'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/pranonrahman/ChartSumm?style=social)

- __ChartMimic: Evaluating LMM's Cross-Modal Reasoning Capability via Chart-to-Code Generation.__
  _Chufan Shi, Cheng Yang, Yaxin Liu, Bo Shui, Junjie Wang, Mohan Jing, Linran Xu, Xinyu Zhu, Siheng Li, Yuxiang Zhang, Gongye Liu, Xiaomei Nie, Deng Cai, Yujiu Yang._ <img src='https://img.shields.io/badge/ICLR-2025-yellow'> <a href='https://arxiv.org/abs/2406.09961'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://github.com/ChartMimic/ChartMimic'><img src='https://img.shields.io/badge/Dataset-red'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/ChartMimic/ChartMimic?style=social)

### Pushing for Complexity and Diversity
As the capabilities of MLLMs rapidly advanced, their performance on foundational benchmarks began to approach saturation. In response, the research community  developed a new generation of benchmarks designed to be far more challenging, diverse, and representative of real-world complexity.

As of late 2025, the most complex charts are now effectively infographics, requiring models to reason about design, metaphors, and stylized elements, not just data.

- __InfographicVQA.__
  _Minesh Mathew, Viraj Bagal, Rubèn Pérez Tito, Dimosthenis Karatzas, Ernest Valveny, C.V Jawahar._ <img src='https://img.shields.io/badge/WACV-2022-yellow'> <a href='https://arxiv.org/abs/2104.12756'><img src='https://img.shields.io/badge/PDF-blue'></a> <a href='https://www.docvqa.org/datasets/infographicvqa'><img src='https://img.shields.io/badge/Dataset-red'></a> 

- __ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning.__ 
  _Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, Enamul Hoque._ <img src='https://img.shields.io/badge/ACL_Findings-2022-yellow'> <a href='https://aclanthology.org/2022.findings-acl.177/'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://github.com/vis-nlp/ChartQA'><img src='https://img.shields.io/badge/Dataset-red'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/vis-nlp/ChartQA?style=social)

- __ChartBench: A Benchmark for Complex Visual Reasoning in Charts.__ 
  _Zhengzhuo Xu, Sinan Du, Yiyan Qi, Chengjin Xu, Chun Yuan, Jian Guo._ <img src='https://img.shields.io/badge/Arxiv-2023-yellow'> <a href='https://arxiv.org/abs/2312.15915'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://huggingface.co/datasets/SincereX/ChartBench'><img src='https://img.shields.io/badge/Dataset-red'></a>

- __ChartInsights: Evaluating Multimodal Large Language Models for Low-Level Chart Question Answering.__
  _Yifan Wu, Lutao Yan, Leixian Shen, Yunhai Wang, Nan Tang, Yuyu Luo._ <img src='https://img.shields.io/badge/EMNLP_Findings-2024-yellow'> <a href='https://arxiv.org/abs/2405.07001'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://chartinsight.github.io/'><img src='https://img.shields.io/badge/Dataset-red'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/HKUSTDial/ChartInsights?style=social)

- __ChartX & ChartVLM: A Versatile Benchmark and Foundation Model for Complicated Chart Reasoning.__ 
  _Renqiu Xia, Bo Zhang, Hancheng Ye, Xiangchao Yan, Qi Liu, Hongbin Zhou, Zijun Chen, Min Dou, Botian Shi, Junchi Yan, Yu Qiao._ <img src='https://img.shields.io/badge/Arxiv-2024-yellow'> <a href='https://arxiv.org/abs/2402.12185'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://github.com/UniModal4Reasoning/ChartVLM'><img src='https://img.shields.io/badge/Dataset-red'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/UniModal4Reasoning/ChartVLM?style=social) 

- __CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs.__
  _Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, Alexis Chevalier, Sanjeev Arora, Danqi Chen._ <img src='https://img.shields.io/badge/NeurlPS-2024-yellow'> <a href='https://arxiv.org/abs/2406.18521'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://huggingface.co/datasets/princeton-nlp/CharXiv'><img src='https://img.shields.io/badge/Dataset-red'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/princeton-nlp/CharXiv?style=social)

- __MultiChartQA: Benchmarking Vision-Language Models on Multi-Chart Problems.__
  _Zifeng Zhu, Mengzhao Jia, Zhihan Zhang, Lang Li, Meng Jiang._ <img src='https://img.shields.io/badge/NAACL-2025-yellow'> <a href='https://github.com/Zivenzhu/Multi-chart-QA'><img src='https://img.shields.io/badge/Dataset-red'></a> <a href='https://arxiv.org/abs/2410.14179'><img src='https://img.shields.io/badge/PDF-blue'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/Zivenzhu/Multi-chart-QA?style=social)

- __ChartQAPro: A More Diverse and Challenging Benchmark for Chart Question Answering.__
  _Mehrad Shahmohammadi, Jason Obeid, Monish Singh, Md Rizwan Parvez, Enamul Hoque, Shafiq Joty._ <img src='https://img.shields.io/badge/ACL_Findings-2025-yellow'> <a href='https://arxiv.org/abs/2504.05506v2'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://github.com/vis-nlp/ChartQAPro'><img src='https://img.shields.io/badge/Dataset-red'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/vis-nlp/ChartQAPro?style=social)

- __ChartMuseum: Testing Visual Reasoning Capabilities of Large Vision-Language Models.__
  _Liyan Tang, Grace Kim, Xinyu Zhao, Thom Lake, Wenxuan Ding, Fangcong Yin, Prasann Singhal, Manya Wadhwa, Zeyu Leo Liu, Zayne Sprague, Ramya Namuduri, Bodun Hu, Juan Diego Rodriguez, Puyuan Peng, Greg Durrett._ <img src='https://img.shields.io/badge/NeurIPS-2025-yellow'> <a href='https://huggingface.co/datasets/lytang/ChartMuseum'><img src='https://img.shields.io/badge/Dataset-red'></a> <a href='https://arxiv.org/abs/2505.13444'><img src='https://img.shields.io/badge/PDF-blue'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/Liyan06/ChartMuseum?style=social)

- __InfoChartQA: A Benchmark for Multimodal Question Answering on Infographic Charts__
  _Tianchi Xie, Minzhi Lin, Mengchen Liu, Yilin Ye, Changjian Chen, Shixia Liu._ <img src='https://img.shields.io/badge/Arxiv-2025-yellow'> <a href='https://arxiv.org/abs/2505.19028'><img src='https://img.shields.io/badge/PDF-blue'></a> <a href='https://github.com/CoolDawnAnt/InfoChartQA'><img src='https://img.shields.io/badge/Dataset-red'></a>

- __InfoCausalQA: Can Models Perform Non-explicit Causal Reasoning Based on Infographic?.__
  _Keummin Ka, Junhyeong Park, Jaehyun Jeon, Youngjae Yu._ <img src='https://img.shields.io/badge/Arxiv-2025-yellow'> <a href='https://arxiv.org/abs/2508.06220'><img src='https://img.shields.io/badge/PDF-blue'></a> 

### New Frontiers: Evaluating Trust, Safety, and Access
As chart-understanding models become more capable, several critical new frontiers have emerged, including ensure their outputs are trustworthy and safe, and capabilities are accessible to a global audience.

- __CHOCOLATE: Do LVLMs Understand Charts? Analyzing and Correcting Factual Errors in Chart Captioning.__ 
  _Kung-Hsiang Huang, Mingyang Zhou, Hou Pong Chan, Yi R. Fung, Zhenhailong Wang, Lingyu Zhang, Shih-Fu Chang, Heng Ji._ <img src='https://img.shields.io/badge/ACL_Findings-2024-yellow'> <a href='https://arxiv.org/abs/2312.10160'><img src='https://img.shields.io/badge/PDF-blue'></a>
  <a href='https://huggingface.co/datasets/khhuang/CHOCOLATE'><img src='https://img.shields.io/badge/Dataset-red'></a> ![GitHub Repo stars](https://img.shields.io/github/stars/khuangaf/CHOCOLATE?style=social)

- __Unmasking Deceptive Visuals: Benchmarking Multimodal Large Language Models on Misleading Chart Question Answering.__
  _Zixin Chen, Sicheng Song, Kashun Shum, Yanna Lin, Rui Sheng, Weiqi Wang, Huamin Qu._ <img src='https://img.shields.io/badge/EMNLP-2025-yellow'> <a href='https://arxiv.org/abs/2503.18172'><img src='https://img.shields.io/badge/PDF-blue'></a> 

- __POLYCHARTQA: Benchmarking Large Vision-Language Models with Multilingual Chart Question Answering.__
  _Yichen Xu, Liangyu Chen, Liang Zhang, Wenxuan Wang, Qin Jin._ <img src='https://img.shields.io/badge/Arxiv-2025-yellow'> <a href='https://arxiv.org/abs/2507.11939'><img src='https://img.shields.io/badge/PDF-blue'></a> 

## Analysis Papers
- __Are Large Vision Language Models up to the Challenge of Chart Comprehension and Reasoning? An Extensive Investigation into the Capabilities and Limitations of LVLMs.__
  _Mohammed Saidul Islam, Raian Rahman, Ahmed Masry, Md Tahmid Rahman Laskar, Mir Tafseer Nayeem, Enamul Hoque._ <img src='https://img.shields.io/badge/EMNLP_Findings-2024-yellow'> <a href='https://arxiv.org/abs/2406.00257'><img src='https://img.shields.io/badge/PDF-blue'></a>

- __Unraveling the Truth: Do VLMs really Understand Charts? A Deep Dive into Consistency and Robustness.__
  _Srija Mukhopadhyay, Adnan Qidwai, Aparna Garimella, Pritika Ramu, Vivek Gupta, Dan Roth._ <img src='https://img.shields.io/badge/EMNLP_Findings-2024-yellow'> <a href='https://arxiv.org/abs/2407.11229'><img src='https://img.shields.io/badge/PDF-blue'></a>

- __On the Perception Bottleneck of VLMs for Chart Understanding.__
  _Junteng Liu, Weihao Zeng, Xiwen Zhang, Yijun Wang, Zifei Shan, Junxian He._ <img src='https://img.shields.io/badge/EMNLP_Findings-2025-yellow'> <a href='https://arxiv.org/abs/2503.18435'><img src='https://img.shields.io/badge/PDF-blue'></a>