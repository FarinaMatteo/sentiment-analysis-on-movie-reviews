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

## Run the code
In order to run the exhaustive evaluation in **src/main.py**, one has to train the subjectivity detector first. You can do so with the following:
```
cd src/subjectivity_detection
python sentence_level.py
cd ..
```
To check the possible parameters instantiations of the subjectivity detector, run `python sentence_level.py --help`. When training is done, you can find your trained model under the `models` folder in the repo root (will be created if needed).

Afterwards, run `python main.py` to execute the exhaustive evaluation as described in the Experimental Section of **report.pdf**. 
In case you want to experiment with a different subjectivity detector (the default one is a Multinomial Naive Bayes classifier trained with Count Vectorization), please run the subjectivity detection training script with the appropriate cmd line arguments and modify the *subj_det_filename* parameter default value in main.py.  
As an example, to run the evaluation using a Bernoulli based subjectivity detector trained on the tfidf BoW document representation, run the following:
```
cd src/subjectivity_detection
python sentence_level.py -r tfidf -clf bernoulli
cd ..
```
Then, modify the *subj_det_filename* argument of the *main* function in *main.py* as such:
```
def main(
    ...
    subj_det_filename="tfidf_bernoulli",
    ...
)
```
Finally, run `python main.py`.