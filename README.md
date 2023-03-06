# Natural language processing course 2022/23: `Put name of your project here`

Team members:
 * `Robert Jutreša`, `63180138`, `rj7149@student.uni-lj.si`
 * `Luka Škodnik`, `63180033`, `ls1906@student.uni-lj.si`
 * `Valter Hudovernik`, `63160134`, `vh0153@student.uni-lj.si`
 
Group public acronym/name: `THINK OF PUBLIC STRING FOR YOUR GROUP`
 > This value will be used for publishing marks/scores. It will be known only to you and not you colleagues.

# Preliminary Research

First citations we will look at:
+ [About Cramming](https://arxiv.org/pdf/2212.14034.pdf).
+ [About the instructions on the GPT approach](https://arxiv.org/pdf/2203.02155.pdf).
+ [About the BLOOM the multilignual language model](https://arxiv.org/pdf/2211.05100.pdf).
+ [The opensource ChatGPT](https://github.com/LAION-AI/Open-Assistant).

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
