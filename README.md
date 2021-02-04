# ModernSlavery
 Working repo for contributions to Project AIMS: Artificial Intelligence against Modern Slavery (see also: https://github.com/the-future-society/modern-slavery-statements-research). This repo specifically addresses the task of the hackathon:

- Corporations doing business in the United Kingdom have submitted statements to the Government explaining what they are doing to ensure that there is no Modern Slavery occurring anywhere in their supply chains
- There is no predefined structure to these documents
- There is lots of valuable information that could be gleaned from these statements. As a first step, the Hackathon focused on one topic: training that corporations are providing to their employees about Modern Slavery.
- The goal of the Hackathon was to classify documents into one of two categories:
  - Class 1: provides evidence that the corporation is currently providing Anti-Modern Slavery training to their employees
  - Class 0: does not provide evidence
- The documents in the Hackathon had been human-labeled
- The documents had been split into a training set and a hidden-label test set



This repo contains 3 threads:

1. Exploratory Data Analysis
2. Convolutional Neural Network Approach
3. Zero-shot Question-Answering Approach



### 1. Exploratory Data Analysis

An investigation into the dataset that revealed issues with the dataset including that many documents appear mislabeled. This finding inspired the zero-shot approach below. See EDA.ipynb

### 2. Convolutional Neural Network

This was the initial approach for the Hackathon. As it was a compressed schedule, I dove into the modeling without taking the time to explore the data, a mistake which I will not repeat. Because the labels were unreliable, this approach proved unsuccessful. (Note: since this approach was unsuccessful, it was left in a fairly hacked-together state and has not been well documented)

The thinking here was that there is likely to be no more than a handful of sections of the document that talk about training, with most of the rest of the text irrelevant to the task at hand. This corresponds well to a 1-D CNN which trains a filter that will return a high value for these relevant sections of text and a low value elsewhere. Using a global max_pooling layer near the top of the network will allow the model to tell us whether the filter flagged a high value somewhere in the document or not, thereby classifying it. This also has the advantage of being length-independent as the document lengths vary greatly with some being significantly long.

This approach could be revisited in the future if the labels are cleaned up.

### 3. Zero-shot Question-Answering

Since the labels have proven unreliable, I wondered if the documents could be classified while ignoring the labels using a pre-trained (zero-shot) model. The motivation behind this approach is to use models pre-trained to extract relevant answers (as a span) from a document (context) in order to automate the identification of which small subsets of the documents might be relevant to modern slavery training. These smaller subsets can then make the job of human-labelling additional documents more efficient or be fed into another model which can only handle a limited number of tokens (perhaps a transformer trained for sequence classification).

The pre-trained models to be used for this task are Question-Answering transformers (trained on SQuAD v2 such that they can return a "no span found" result) which will be used to ask questions of the documents. Since most documents in the dataset are longer than the maximum input length, a sliding window approach is used.

## Repo files:

### 0. Data files common to all approaches

- train (3).csv - the labeled documents for the hackathon
- test (3).csv - the hidden-label documents for the hackathon

### 1. EDA files

- EDA.ipynb - The jupyter notebook containing the exploratory data analysis

### 2. CNN files

- CNN.ipynb - The jupyter notebook used to train the CNN model
- Transformer_Embeddings.ipynb - Conducts preprocessing of the data before it was used to train the CNN
- df_labeled.csv - an updated version of train (3).csv containing information pertaining to where each document is located in the preprocessed data coming out of Transformer_Embeddings.ipynb
- df_hidden.csv - an updated version of test (3).csv containing information pertaining to where each document is located in the preprocessed data coming out of Transformer_Embeddings.ipynb

### 3. QA files

- SlidingWindowTransformersQA.py - Contains the custom classes and functions used to carry out the sliding window QA approach
- QA-sliding window.ipynb - Carries out the sliding window QA approach on the full dataset
- QA results viewer.ipynb - Visualizes the resulting answers given by the sliding window QA model
- df_with_segments.parquet - A Pandas dataframe containing the documents and the spans detected by the sliding window QA model, stored in parquet format.
- df_token_classes.parquet - A Pandas dataframe containing the token ids and token classes for each document, stored in parquet format.

### Generic files

- environment.yml - defines the environment used in Conda
- LICENSE - the license for this repo
- .gitignore - defines what local files and directories are excluded from the repo
- README.md - this file!