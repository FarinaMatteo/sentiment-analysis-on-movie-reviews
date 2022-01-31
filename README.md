# sentiment-analysis-on-movie-reviews
Implementation of a Two Stage Classifier (Naive Bayes Classifier + SVM) for sentiment analysis applied on the Movie Reviews dataset. The work is inspired by: https://aclanthology.org/I13-1114.pdf  

## Installation
Dependencies of this repository are provided as a Conda Environment. In order to install them, run the following at the repository root:
```
conda create -f environment.yml
```

NLTK resources are also used throghout the code. Most should be automatically installed during execution, but the following installs them manually:
```
nltk.download("wordnet")
nltk.download("sentiwordnet")
nltk.download("universal_tagset")
```