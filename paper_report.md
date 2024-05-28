# A1. Abstract : XX's Project for News Classifier Analysis
XXX have a need to develop a media intelligence application that allows them to map and monitor the public activies in order to maintain national security as well as preventing unpleasant and undesirable threats and actions as the its main objectives. The aforementioned application is expected to have several functionalities to help the objectives achieved, one of those functionality is to observe the diversity of news categories of from news media on Indonesia demography. To fulfill that specific functionality, a program that has capabilty to distinguish and differentiate news categories is essentially required. In this case, News Classifier, a task in NLP, is proved to tackle the very problem.

XXX can be attained with opinion mining as well. The implementation is similar with Sentiment Analysis areas in Natural Language Processing (NLP) and text mining in recent years. Its popularity is mainly due to two reasons. First, it has a wide range of applications because opinions are central to almost all human activities and are key influencers of our behaviors. Second, it presents many challenging research problems, which had never been attempted before the year 2000. Part of the reason for the lack of study before was that there was little opinionated text in digital forms. It is thus no surprise that the inception and the rapid growth of the field coincide with those of the social media on the Web. In fact, the research has also spread outside of computer science to management sciences and social sciences due to its importance to business and society as a whole.
<br>
<br>


# A2. Abstract : XX's Project for News Classifier Analysis
# Natural Language Processing and Sentiment Analysis Introduction

## Brief Introduction to NLP

Computers are great at working with structured data like spreadsheets and database tables. But us humans usually communicate in words, not in tables. That’s unfortunate for computers 😢

<center><img src='https://s3.us-east-2.amazonaws.com/ardhiraka.com/img/shakes.png' width="20%" /></center>

<!-- <div style="text-align: justify"> -->
According to industry estimates, only 21% of the available data is present in structured form. Data is being generated as we speak, as we tweet, as we send messages on Whatsapp and in various other activities. Majority of this data exists in the textual form, which is highly unstructured in nature.
Few notorious examples include – tweets / posts on social media, user to user chat conversations, news, blogs and articles, product or services reviews and patient records in the healthcare sector. A few more recent ones includes chatbots and other voice driven bots.
Despite having high dimension data, the information present in it is not directly accessible unless it is processed (read and understood) manually or analyzed by an automated system.

Based on information given above, lot of information in the world is unstructured — raw text in English (or bahasa) or another human language. How can we get a computer to understand unstructured text and extract data from it? **Can Computers Understand Language?**

> Natural language processing (NLP) is a subfield of linguistics, computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data. - Wikipedia

So, in short; Natural Language Processing, or NLP, is the sub-field of AI that is **focused on enabling computers to understand and process human languages**.
With the help of a Bunch of Algorithms and rules the computer able to understand and communicate with humans in vast human languages and scales other language-related tasks. With NLP, it is possible to perform certain tasks like Automated Speech and Automated Text Writing in less time. Due to the evolving of large data (text), why not to use the computers which have high computing power, capable of working all day and ability to run several algorithms to perform tasks in no time.

NLP can be divided into 3 categories (Rule-based systems, Classical Machine Learning models and Deep Learning models).
1. Rule-based systems rely heavily on crafting domain-specific rules (e.g: regular expressions), can be used to solve simple problems such as extracting structured data (e.g: emails) from unstructured data (e.g: web-pages), but due to the complexity of human natural languages, rule-based systems fail to build models that can really reason about language.
2. Classical Machine Learning approaches can be used to solve harder problems which rule-based systems can’t solve very well (e.g: Spam Detection), it rely on a more general approach to understanding language, using hand-crafted features (e.g: sentence length, part of speech tags, occurrence of specific words) then providing those features to a statistical machine learning model (e.g: Naive Bayes), which learns different patterns in the training set and then be able to reason about unseen data (inference).
3. Deep Learning models are the hottest part of NLP research and applications now, they generalize even better than the classical machine learning approaches as they don’t need hand-crafted features because they work as feature extractors in an automatic way, which helped a lot in building end-to-end models (little human-interaction). Aside from the feature engineering part, deep learning algorithms learning capabilities are more powerful than the shallow/classical ML ones, which paved its way to achieving the highest scores on different hard NLP tasks (e.g: Machine Translation).
    
## Extracting Meaning from Text is Hard
    
The process of reading and understanding English (or bahasa) is very complex — and that’s not even considering that English (or bahasa) doesn’t follow logical and consistent rules. For example, what does this news headline mean?
    
> "Environmental regulators grill business owner over illegal coal fires."
    
Are the regulators questioning a business owner about burning coal illegally? Or are the regulators literally cooking the business owner? As you can see, parsing English with a computer is going to be complicated. Another challenge are listed below:
    
1. Ambiguity / Crash Blossom <br />
    It is the challenge when a Single word has different meanings or a sentence that has different meanings in the context and even a sentence refers to sarcasm.
    - Lexical Ambiguity is the presence of two or more possible meanings within a single word. (ie. I saw her _duck_)
    - Syntactic Ambiguity is the presence of two or more possible meanings within a single sentence or sequence of words. (ie. The chicken is _ready to eat_)
2. Syntax <br />
    Think of how a sentence is valid, it based on two things called syntax and semantics where syntax refers to the grammatical rules, on the other hand, semantics is the meaning of the vocabulary symbols within that structure. People change the ordering of sentences it is valid in some cases but not all.
3. Co-reference / Anaphora Resolution <br />
    The problem of resolving what a pronoun, or a noun phrase refers to. (ie. Hacktiv8's employee took two trips around France. _They_ were both wonderful.)
4. Slang
5. Sarcasm <br />
    Same words different meaning refers to the Ambiguity topic. Suppose when someone does something wrong you reply as very good or well done. it’s also a challenge for a computer to understand the sarcasm because it’s a way more different than a normal conversation.

## Aplications of NLP

- Machine Translation
- Information Extraction
- **Sentiment Analysis**
- Information Retrieval
- Question Answering
- Summarization



# B. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
BERT, which stands for Bidirectional Encoder Representations from Transformers, is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering, language inference, and sentiment analysis.

BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement). With this reason, BERT is choosed to be implemented as the model, and is expected can help us to distinguish the sentiment of public opinion as well as mapping the,m

 - [Bert Paper](https://arxiv.org/abs/1810.04805)
 - [Transformers Paper](https://arxiv.org/abs/1706.03762)
<br>

![image](https://github.com/daniel-satria/XXX/assets/110766651/6a8a09cc-69ea-4518-837d-2092c3df2d59)
<p align="center"> 
  <i>Bert Model Architecture</i>
</p>

<br>
<br>

# C. Fine-Tuning BERT for Sentiment Classifier of XX Project

To fine-tuned the model for news analysis classifier of XX Project, 195.383 of record is used. It consists of the labels; "ideologi", "politik", "ekonomi", "sosial & budaya", "pertahanan & keamanan" or usually abbreviated as "ipoleksusbudhankam", with below proportion of data points.
<br>
- Ideologi              : 30.294
- Politik               : 44.063
- Ekonomi               : 50.719
- Sosial & Budaya       : 38.169
- Pertahanan & Keamanan : 31.508

<br>
The train and test data are splitted into 0.85 of train and 0.15 test data. It also stratfiedly splitted based on language category which contain 'indonesia', and 'english', with the proportion as follow :

- inggris       : 105.214
- indonesia      : 90169

<br>
Although there are many of the model version of BERT available, for this specific task the model used is Bert Based Uncased; it does not make a difference between ideologi, Ideologi or IDEOLOGI.

## C.1 Training Procedure
The model is trained with AWS Sagemaker platforms in following instance :
- ml.p3.8xlarge

### Training Details
The training is conducted in 1 epoch, with 2e-5 learning rate.
It yields result with more than 80% accuracy.

- [Bert Base Uncased on HuggingFace](https://huggingface.co/bert-base-uncased)

```
> First Epoch => 6044s 340ms/step - loss: 0.3983 - accuracy: 0.8580 - val_loss: 0.2789 - val_accuracy: 0.8975
```

    Tue Jan  9 15:59:52 2024       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 470.57.02    Driver Version: 470.57.02    CUDA Version: 11.4     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla V100-SXM2...  Off  | 00000000:00:1E.0 Off |                    0 |
    | N/A   41C    P0    37W / 300W |      0MiB / 16160MiB |      2%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                  
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+




    Classification Score
    778/778 [==============================] - 301s 382ms/step
                  precision    recall  f1-score   support

              0       0.94      0.93      0.93      6708
              1       0.82      0.89      0.85      3961
              2       0.90      0.86      0.88      3788
              3       0.90      0.88      0.89      5588
              4       0.91      0.91      0.91      4825

        accuracy                           0.90     24870
      macro avg       0.89      0.89      0.89     24870
    weighted avg       0.90      0.90      0.90     24870
<br>
<br>


# D. How to Use The Fined-Tuned Model
1. Install all the package in the requirements.txt from model directory.

2. Download the model artifacts; model itself and the preprocess.
   
3. Load the model using 'load_predictor' method from ktrain module as below.
   <br>
```
>>> reloaded_model = ktrain.load_predictor('model/news_classifier_v1.1')
```

4. Use predict method from the model to predict the sentiment of the sentence with string type data.
```
>>> reloaded_model.predict('Ini berita tentang ekonomi.')
```
5. It will return the label/news categories as a string.
```
'ekonomi'
```
6. Use predict_proba method from the model to predict the sentiment of the sentence with string type data to get the probability of the prediction.
```
>>> reloaded_model.predict_proba('Ini berita tentang ekonomi.' )
```
7. It will return list of arrays from the label probability respectively.
```
array([0.8133873 , 0.00305308, 0.14447974, 0.01021839, 0.02886151],
      dtype=float32)
```
8. To show all the label/news categories names use get_classes() method.
```
>>> reloaded_model.get_classes()
```
9. It will return the list of the classes.
```
['ekonomi', 'hankam', 'ideologi', 'politik', 'sosbud']
```
<br>

| :exclamation:  This is very important for model input   |
|-----------------------------------------|
## D.1 Model Input
The data should be cleaned, before feeded into the model. The specific requirements as follow:

a. All the input must be string or list of strings. <br>
b. The acceptable input for the model are emoji(s) and alphanumerics; text(s) and number(s). <br>
c. Other input than specified on point (b) above should be deleted and replace with ***space***. <br>
d. Please be aware that emoji should ***not*** be deleted. <br>
e. All the characters should be converted to lower case. <br>
f. All URL, "\n", "\t", must be deleted. <br>

<br>
<br>

# E. Costraint, Bias & Improvement

## E.1 Constraint
Bert is the solely model used for the fine-tune. The other alternative models are eliminated because it's proven the best one from previous experimentation for task in Sentimen Analysis also for XX project. When the project's launched and the predicted data are growing over the time, the model can be fine-tuned again later with the additional supervised data, which to be expected yield a better performance.

## E.2 Bias
The model is expected to have fairly good result, especially when dealing with English and Indonesian data. However, the possibility that the model yield a bias result also is not zero.  This may be caused by the feeded data into the fine-tuned model, or because the data that were feeded into initial model.

## E.3 Improvement
Improvement can be done by re-fine tuned the model with additional data. As it explains the data proportion in menu #3, the model should perform fairly good when dealing with english and indonesia data, but may perform not too well when predicting some other Indonesian local language. The data gathered are expected to grow more and more as the application launchs, and the data could be used to train the model in order to get better performance. Other models also can be tested to be fine-tuned as an experiment, whether other models can give a better result than Bert in the future.

<br>
<br>

# F. BibTeX entry and citation info
```bibtex
@article{DBLP:journals/corr/abs-1810-04805,
  author    = {Jacob Devlin and
               Ming{-}Wei Chang and
               Kenton Lee and
               Kristina Toutanova},
  title     = {{BERT:} Pre-training of Deep Bidirectional Transformers for Language
               Understanding},
  journal   = {CoRR},
  volume    = {abs/1810.04805},
  year      = {2018},
  url       = {http://arxiv.org/abs/1810.04805},
  archivePrefix = {arXiv},
  eprint    = {1810.04805},
  timestamp = {Tue, 30 Oct 2018 20:39:56 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1810-04805.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}


@article{maiya2020ktrain,
    title={ktrain: A Low-Code Library for Augmented Machine Learning},
    author={Arun S. Maiya},
    year={2020},
    eprint={2004.10703},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    journal={arXiv preprint arXiv:2004.10703},
}



@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}


@article{scikit-learn,
 title={Scikit-learn: Machine Learning in {P}ython},
 author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
 journal={Journal of Machine Learning Research},
 volume={12},
 pages={2825--2830},
 year={2011}
}


@inproceedings{koto2020indolem,
  title={IndoLEM and IndoBERT: A Benchmark Dataset and Pre-trained Language Model for Indonesian NLP},
  author={Fajri Koto and Afshin Rahimi and Jey Han Lau and Timothy Baldwin},
  booktitle={Proceedings of the 28th COLING},
  year={2020}
}
```
<hr>

**Creator** : Daniel Satria

