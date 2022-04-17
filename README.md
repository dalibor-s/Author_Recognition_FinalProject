# BRAINSTER PROJECT – NLP AUTHOR RECOGNITION

## INTRODUCTION
*What if one could determine who wrote a piece of text? Reveal the writers behind the texts? Is online review fake and written by spam account?*
Authorship recognition is an important topic in the field of **Natural Language Processing (NLP)**. It enables us to identify the most likely author of articles, news or messages. Yet sometimes we find unfinished pieces of work or a whole bunch of manuscripts with a wide variety of possible authors. In order to assess the importance of such a manuscript, it is vital to know who wrote it. Authorship identification can be applied to tasks such as identifying anonymous author, detecting plagiarism or finding ghost writer. <br />**The goal of our project is after investigating a collection text from multiple authors, if an unforeseen text was written by a particular individual.**

## DATASET
Dataset contains text from works written by 19th century English language writing authors of the public domain from **[Gutenberg](https://www.gutenberg.org/)**. With these criteria 7 authors have been selected and their books were queried: **Arthur Conan Doyle**, **Charles Dickens**, **Edgar Allan Poe**, **George Eliot**, **Jack London**, **Jane Austin** and **Mark Twain**. We collected 3 books of each author and built up the data set in csv file. For each book, we selected 10% of the paragraphs. 
The objective is to accurately identify the author of the sentences in the test set. 

## TEXT PREPROCESSING
We tried two different preprocessing techniques. The first (text_preprocessed) will be fully preprocessed (low letters, missing punctuation, words represented like tokens and then lemmatization. The second dataset (text_preprocessed_1) will only have low letters and will be tokenized.
Dataset was preprocessed using couple of features, relating to the form of the text, and relating to the substance of the text like:
* *Lower casing* - to prevent multiple versions of the same word, all the words despite their casing are normalized to lowercase form so they can all be counted collectively;
* *Separate punctuation from words* – punctuation marks are normalized and removed;
* *Tokenization* – a task of splitting a sentence into pieces, called tokens and removal of unnecessary words at the same time. Each word presents a token;
* *Removing stop-words* – Stop words are generally words that appear commonly at high frequency in a corpus. They don't actually contribute much to the learning or predictive process as a learning model;
* *Lemmatization* –Lemmatization reduce words based on an actual dictionary and therefore will not chop off words into stemmed forms that do not carry any lexical meaning.


## MODELING – extracting test and training data
We approached this project at different levels, with different machine learning models and classifiers, and natural language processing techniques. 
The first approach was using **CountVectorizer** in understanding the type of text by the frequency of words in it. Results showed that better accuracy is seen from **Logistic Regression** with 82.67%, followed by **Random Forest** with 78%. But it lacks in identifying more important and less important words, but considers abundant words as the most statistically significant. It also doesn’t identify the relationships between words such as linguistic similarity between them.
For the second approach **TF-IDF** was used, because it not only focuses on the frequency of words present in the text but also provides the importance of the words. The best result was shown with the **Voting** classifier with accuracy of 89.66%.
<br />The last, but not the least we tried **Transformer**, which is considered to be the state of the art solution for NLP problems. 
<br />The Transformer NLP model introduced an 'attention' mechanism that takes into account the relationship between all the words in the sentence. It creates differential weightings indicating which other elements in the sentence are most critical to the interpretation of a problem word.
## MODEL EXPLAINABILITY
We used SHAP (SHapley Additive exPlanations) to derive a relevance score for each word in an instance. The relevance scores are then aggregated together to achieve global variable importance of the model. 
<br />The word **“mr”** is the biggest signal word used by our model, contribute most to class George Elliot predictions.
![img_1](https://github.com/Kicho94/Author_Recognition_Brief/blob/main/IMAGES/image_1.png)
<br />The visualization with SHAP for the same sentence is illustrated in Image 2. Here, the features with the strongest impact on the prediction correspond to longer arrows that point in the direction of the predicted class. In the context of the sentiment analysis model here red corresponds to a more positive review and blue a more negative review.
![img](https://github.com/Kicho94/Author_Recognition_Brief/blob/main/IMAGES/image_2.png)
<br />The force plot provides much more quantitative information than the text coloring. Hovering over a chuck of text will underline the portion of the force plot that corresponds to that chunk of text, and hovering over a portion of the force plot will underline the corresponding chunk of text.
![img](https://github.com/Kicho94/Author_Recognition_Brief/blob/main/IMAGES/image_3.png)


 
## FINAL MODEL EVALUATION
Among all the trained models the best model has been selected which was TF-IDF Voting with accuracy of 89.66%, followed by Logistic Regression with 86.67%.
The results show that the logistic regression algorithm achieved the best result in the first scope compared with the other models, with 82.67% performance accuracy. In all models there is only a slight difference between accuracy and f1 score.

Algorithm | Accuracy % | F1 score % 
--- | --- | --- 
**TFIDF Voting** | **89.66** | **88.63**
TFIDF LR | 87.67 | 86.32 
TFIDF RF | 75.67 | 74.08
Transformer | 85.67 | 84.26 
CV LR | 82.67 | 81.83  
CV RF | 78.00 | 76.39 


## SUMMARY OF PROBLEMS ENCOUNTERED
The first challenge was gathering and preprocessing data, as we decided to make our own dataset.
Besides that computing time with SHAP was high, we found that when transformer model has been executed accuracy was decreasing due to randomness.


## CONCLUSION
In this project, four types of writing-style features (lexical, syntactic, structural, and content-specific features) are extracted and inductive learning algorithms are used to build feature-based classification models to recognize authorship of certain text. To examine this framework, we conducted experiments with several classification techniques: Logistic Regression, Random Forest, Multinomial Naïve Bayes, Convolutional Neural Networks, Support Vector Machines, Voting and Transformers. The experimental results showed that the proposed approach was able to identify authors with satisfactory accuracy of 75 to 90%. All four types of features contributed to discriminating authors. Voting outperformed the other two classification techniques in our experiments. Very high performance we achieved with Transformer model too.
Regarding our inexperience in such a complex process as NLP, we managed to build a relevant model with high accuracy.

## FUTURE WORK
There are more models that can be applied and then, the accuracy result can be compared to the result achieved in this project. It is possible in future to expand the scope to more authors and books.
This concept can be expanded and improved for Author profiling (characterization)  -  getting the writer’s profile or characteristics; for example, gender, age, background, and language, or Similarity detection for finding the similarity between the texts to determine the possibility of them having been produced by a single writer, without necessarily finding the real author.

## TEAM
* Kristijan Spasovski
* Katerina Dimevska
* Dalibor Stoilkovski
