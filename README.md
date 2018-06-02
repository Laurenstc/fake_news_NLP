# fake_news_NLP
Fake news classification experiment with small dataset. This was a project for NLP class at IE university.

This was a combined effort of Laurens ten Cate and Annie Pi

# Summary of Results 
Initial Model Results: We played aorund with several NLP techniques, including testing different tokenizers, stemmers, lemmatizers, vectorizers, etc. We had a few ways of dealing with the data, since there was both a title and text field, but we found the best results came from concatenating title and text together and then applying NLP techniques. In terms of vectorizers: for Naive Bayes, Count Vectorizer worked best, but generally Tfidf Vectorizer provided the best results.

Many techniques that we thought would help performance didn't do so great, such as applying a Tfidf Transformer and adding textual features. On the other hand - adding or tuning parameters often helped to boost performance, such as removing stopwords, applying n-grams, and stripping acents (depending on the classifier).

Our best results were found using the Passive Aggressive Classifier, Tfidf Vectorization, and text pre-processing (removing punctuation and special characters, changing to lowercase, and stemming).

Deep Learning Results: The current trend in NLP research is all deep learning, so tried a few different methods.

First, we used various word embeddings: locally-trained, locally-trained word2vec, locally-trained FastText, pre-trained 300d GloVe, and pre-trained 300d word2vec. When we obtained these embeddings we used them in two ways.

We created features from these embeddings and pushed these, together with our other hand-crafted features, in classical machine learning models.
We used the embedding vectors as intitialization weights for embedding layers in more complex neural network architectures.
Performance-wise we noticed that the locally trained word2vec embeddings performed the best with classic models while with the deeper architectures the locally trained embedding layer performed the best.

Next the embeddings we also worked with various neural network architectures. Since text data is inherently sequential we chose to experiment with various forms of recurrent neural networks. From SimpleRNN to LSTMs to GRUs and finally our best performer a combination of 1D CNN layers with LSTM layer.

One final thing to note is that due to computational cost of deep learning we employed Google Compute Cloud to perform most deep learning training on a 4vcpu + 2 NVIDIA tesla 100p GPU VM instance.

Unfortunately, even though we still believe that deep learning will perform best with NLP tasks nowadays, the dataset was too small to really employ the power of deep learning. The highest performance wasn't achieved with deep learning models but with a relatively simple PassiveAggressive classifier.

Further Investigation: Due to time limitations we were unable to implement everything as planned, but in the future, we would like to look at creating different ensembles, hand-coded rules or dictionaries, topic classification for fake/real news, sentiment analysis, POS features, and other deep learning model architectures.
