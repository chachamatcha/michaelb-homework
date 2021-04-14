## Replication Info
To set up the python environment I run the main script in, run the following in your terminal (note this may download a lot of packages!!):
    
    `bash setup.sh`

To run the actual pipeline, type the following in your terminal:

1. Activate the environment:

    `source nyt_env/bin/activate`

2. Run main script (takes like 15-20 minutes)

    `python run.py`

3. Exit my environment

    `deactivate`



## Data Cleaning (clean.py)
At a high level, I reduced the raw data to the story level and computed the modal emotional reaction. A few cleaning steps I took before reducing the data:
 - loaded the raw json into a pandas dataframe, as the raw data lived on the reviewer-story level. 
 - removed emotion_9 as it always equaled 1
 - removed review-stories with erroneous values (i.e. not 0 or 1)
 - removed stories with weird headlines/summaries (e.g. `["fnord","-1","cat","-2",""]`)

After reducing the data, I also:
 -  removed repetitive stories (e.g. stories with the summary "Here’s what you need to know to start your day")
 - porter stemmed the text columns (i.e. headline, summary)

## Analysis Design Choices (analysis.py)
As a disclaimer, I had quite a busy week and learned a lot about NLP during the limited time I had to work on this assignment :D.

The two main model building steps are: 1) map the text associated with each observation to a vector space; 2) build a classifier which predicts labels using the vectorized text. 

I started off with a simple model:
 - Embedding Step: concatenate the summary and headline, and convert the corpus of raw words into a matrix of TF-IDF features using sklearn's handy TfidfVectorizer[https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html]. This function does the following:
   - removes stop words, computes the number of words in the corpus and creates a sparse count matrix where entry [i,j] indicates the count of word j in story i.
   - weights the counts by the word's tf-idf score, which indicates how frequently a word is used in the observation, relative to other observations
- Classifier Step: train a Support Vector Machine with a Gaussian Kernel on 80% of the data
   - I chose SVMs because they're fast, handle multiclass prediction, and increase in flexibility easily via the kernel trick. I first tried a linear SVM, but the Gaussian kernel yielded 15% higher accuracy.
   - I found the optimal regularizing scalar using Grid Search Cross Validation

In my main pipeline, I output results from this simple model, which yields a test classification accuracy around 50%.

In an attempt to increase classification accuracy, I built a model that conducted a more complex embedding step. Tf-idf embeddings are great, but they don't incorperate sentence context (i.e. the order of words in the sentence). Further, tf-idf embeddings fail to incorperate prior knowledge of words in the corpus (i.e. if the model only sees the word 'sad' once in the corpus, it will not confidently associate the word with an emotion). So, if I created a better set of embeddings, some vanilla classification algorithm could have an easier time predicting sentiment. 

For this more complex model, I fine tuned a partially pre-trained deep learning model, created by HuggingFace, with three main components:
 - Tokenize the words with HuggingFace's fancy autotokenizer. Deals with stemming, padding, etc.
 - Embedding with a Pretrained BERT Transformer. The BERT transformer incorperates prior knowledge, as it was originally trained on a ton of language data. Transformers are also designed to learn the context of the input data.  
 - Classification using a fully connected layer with a soft max activation function, which essentially functions like a multinomial logistic regression. So, after the transformer returns a set of embeddings, the fully connected layer computes the probability that the embedded vector is associated with each emotion, and predicts the emotion with the highest probability.

 During fine tuning, I beleive the BERT parameters are slightly tweaked, and the fully connected layers are fully learned. 

After 3 arduous hours of training on my wheensy Macbook Air, this model only returned an accuracy of 54%. Further, accuracy slightly decreased further into training, which suggests to me that I'm doing something wrong 0___0.  See `bert_analysis.ipynb`.

## Tradeoffs
While the deep model is potentially more accurate, it takes 3 hours to train instead of the 20 minutes required for the SVM, though I imagine BERT training would significnatly speed up if I trained on a GPU or a more powerful CPU. 

## More Time
I'd want to learn more about NLP! Specifically, I'd want to read more about embedding strategies for sentiment analysis. I'd also think more deeply about how to feed the header text and summary text into the model. I currently concatenate them into two sentences, but perhaps there's a more clever way to pass them in seperatly. I'd also want to debug my transformer! 