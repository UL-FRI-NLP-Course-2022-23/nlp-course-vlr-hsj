# Natural language processing course 2022/23: `Slovenian Language Assistance Bot`

Team members:
 * `Robert Jutreša`, `63180138`, `rj7149@student.uni-lj.si`
 * `Luka Škodnik`, `63180033`, `ls1906@student.uni-lj.si`
 * `Valter Hudovernik`, `63160134`, `vh0153@student.uni-lj.si`
 
Group public acronym/name: `SLAB`
 > This value will be used for publishing marks/scores. It will be known only to you and not you colleagues.

# First Submission Planning

Developing the idea:
1. Choosing a model - We would use a GPT based model (because of it's popularity), more specifically the RoBERTa implementation, found [here](https://huggingface.co/xlm-roberta-base). Or potentially some other model found on Hugging Face [here](https://huggingface.co/models?language=sl&sort=downloads).
2. Collecting data:
   1. We need to keep privacy redaction and deduplication in mind.
   2. Public Pool of Prompts (P3) - It's a dataset of question prompts and answers, that we would translate into Slovenian and quality control them by averaging the translations of different translators or models. The data is available [here](https://huggingface.co/datasets/bigscience/P3).
   3. [GOS data source](http://ssj.slovenscina.eu/korpusi/gos)
   4. Generated "conversations":
      * FAQ sections of website (Potential problem: context dependancy).
      * Fiction writing (GIGAFIDA in KRES?).
      * Forums - we were thinking each of us finds 2 and we use them, they have to be apropriate, not like r\Slovenia or comments in 24ur posts.
      * Generated questions and answers from SSKJ.
      * Generated questions and answers from factual data sources, example can be found [here](https://podatki.gov.si/data/search?publisher=ministrstvo_za_javno_upravo&all_licence=%2F&page=2)
      * We use ChatGPT/Google to generate $x$ question and answers about some topic $y$.
 3. Follow up to inital ideas:  
    1. [Clarin](http://www.clarin.si/info/about/) could be used as an overall resource for both models and data in Slovene.
    2. Another resource we could potentially translate is [SuperGLUE](https://paperswithcode.com/dataset/superglue).
    3. Yet another resource focused on Q&A is [this one here](https://rajpurkar.github.io/SQuAD-explorer/).
    4. Another idea is to translate annotated TV subtitles found [here](https://github.com/zll17/TV4Dialog) with documentation [here](https://ieeexplore.ieee.org/document/9023129).
    5. **IMPORTANT**: We need to estimate how much data is enough to properly train a model.

# Second Submission Planning

*Next steps*:
+ Add decoder blocks to the end of a frozen model + possible pre training?
+ Model choice: T5 (transformers), GPT2, SloBERTa.
+ Define an evalaution protocol.
+ Choose format to unify the data (P3, Alpaka, ...).
+ Make a translation pipeline.
+ Preliminary training and testing.

**Possible tools**:
- All-In-One NLP pipelines: https://haystack.deepset.ai/overview/intro

**Text-Generation Models** are available on [Hugging Face](https://huggingface.co/models?pipeline_tag=text2text-generation&language=sl&sort=downloads).  
**Model Documentation**:  
*Google*  
+ mT5 variants (multilingual T5, T5 = encoder decoder transformer for text generation, https://arxiv.org/abs/2010.11934)
+ byT5 variants (some adaptation of mT5, https://arxiv.org/abs/2105.13626)

*BLOOM*  
+ mT0 (multilingual 0-shot instruction following, https://arxiv.org/abs/2211.01786)

*CVJT*
+ T5-SL (pretraingin for slovenian language, https://arxiv.org/abs/2207.13988)


# Environment Setup
TODO

# Run inference
TODO

# Trained models

* https://huggingface.co/ls1906/t5-sl-small-finetuned-assistant
* https://huggingface.co/vh-student/t5-sl-large-oasst-pairs
* https://huggingface.co/vh-student/t5-sl-large-oasst-context
* https://huggingface.co/vh-student/gpt-sl-oasst1-pairs
* https://huggingface.co/vh-student/gpt-sl-oasst1-context
* https://huggingface.co/vh-student/sloberta-si-rrhf


# Data

OASST Data:

[oasst-ready_for_export](https://huggingface.co/datasets/OpenAssistant/oasst1/blob/main/2023-04-12_oasst_ready.trees.jsonl.gz)


Translated OASST Data from English, Spanish, Russian and German:

[oasst-ready_for_export-translated](https://unilj-my.sharepoint.com/:u:/g/personal/ls1906_student_uni-lj_si/EZTsjBHFsbtPrb2Ur6k1AiwBRZDVcilb3zWtvCto38deFA?e=DHn9az)

[Kaggle Version](https://www.kaggle.com/datasets/valterh/oasst1-sl)
