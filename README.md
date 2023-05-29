# Natural language processing course 2022/23: `Slovenian Language Assistant for Virtual Conversations (SLAVC)`

Team members:
 * `Robert Jutreša`, `63180138`, `rj7149@student.uni-lj.si`
 * `Luka Škodnik`, `63180033`, `ls1906@student.uni-lj.si`
 * `Valter Hudovernik`, `63160134`, `vh0153@student.uni-lj.si`
 
 Group public acronym/name: `SLAB`
 > This value will be used for publishing marks/scores. It will be known only to you and not you colleagues.

 # Installation
 ```bash	
conda env create -f conda.yaml
conda activate nlp
 ```


# Data

OASST Data:

[oasst-ready_for_export](https://huggingface.co/datasets/OpenAssistant/oasst1/blob/main/2023-04-12_oasst_ready.trees.jsonl.gz)


Translated OASST Data from English, Spanish, Russian and German:

[oasst-ready_for_export-translated](https://unilj-my.sharepoint.com/:u:/g/personal/ls1906_student_uni-lj_si/EZTsjBHFsbtPrb2Ur6k1AiwBRZDVcilb3zWtvCto38deFA?e=DHn9az)

[Kaggle Version](https://www.kaggle.com/datasets/valterh/oasst1-sl)


# Trained models

* [ls1906/t5-sl-small-finetuned-assistant](https://huggingface.co/ls1906/t5-sl-small-finetuned-assistant)
* [vh-student/t5-sl-large-oasst-pairs](https://huggingface.co/vh-student/t5-sl-large-oasst-pairs)
* [vh-student/t5-sl-large-oasst-context](https://huggingface.co/vh-student/t5-sl-large-oasst-context)
* [vh-student/gpt-sl-oasst1-pairs](https://huggingface.co/vh-student/gpt-sl-oasst1-pairs)
* [vh-student/gpt-sl-oasst1-context](https://huggingface.co/vh-student/gpt-sl-oasst1-context)
* [vh-student/sloberta-si-rrhf](https://huggingface.co/vh-student/sloberta-si-rrhf)


# Run inference
Evaluation located in [evaluation.ipynb](./notebooks/evaluation/evaluation.ipynb) can be rerun for reproducibility purposes.

# Report
[Final Report](https://github.com/UL-FRI-NLP-Course-2022-23/nlp-course-vlr-hsj/blob/main/final_report.pdf)
