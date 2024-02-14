import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import string
from collections import Counter
import re
import nltk
from rake_nltk import Rake, Metric


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# packages for performance metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# import data
df = pd.read_csv("C:\\Users\\alyam\\PycharmProjects\\Principal\\Data\\growth-data-all.csv")

# drop unnecessary columns
X = df
X = X.drop(columns=['sentiment_negative', 'sentiment_neutral ', 'sentiment_positive', 'sentiment_NA',
                    'not_relevant', 'relevant', 'very_relevant', 'extremely_relevant'])

# Split to  train and test data
train, test = train_test_split(X, test_size=0.3, random_state=0, shuffle=True)

X_train = train
X_train = X_train.drop(columns=['target'])
y_train = train.loc[:, 'target'].values

X_test = test
X_test = X_test.drop(columns=['target'])
y_test = test.loc[:, 'target'].values

class FrequentWordsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def fit(self, X, y = None):
        return self

    def remove_stopwords(self, sentence_list, stopwords):
        new_list = []
        for word in sentence_list:
            if word not in stopwords:
                new_list.append(word)
        return new_list

    def new_features(self, current_columns, wordlist):
        new_features = []
        for word in wordlist:
            if word not in current_columns:
                new_features.append(word)
        return new_features

    def new_wordlist(self):
        df = self.train_dataset
        df1 = df[df.target == 1]

        # convert column Sentences to list
        df_all_1 = df['Sentences'].astype(str).values.tolist()
        df_rel_1 = df1['Sentences'].astype(str).values.tolist()

        str1_all = ''.join(df_all_1)
        str1_rel = ''.join(df_rel_1)

        str1_all = str1_all.split()
        str1_rel = str1_rel.split()

        # get stopwords and add a few
        stop_words = stopwords.words('english')
        stop_words.extend(['The', 'also'])

        # remove punctuation in list
        str1_all = [''.join(c for c in s if c not in string.punctuation) for s in str1_all]
        str1_rel = [''.join(c for c in s if c not in string.punctuation) for s in str1_rel]

        # change to all lower case
        str1_all = [x.lower() for x in str1_all]
        str1_rel = [x.lower() for x in str1_rel]

        # create a new list that do not contain stop words
        wordlist_all = self.remove_stopwords(str1_all, stop_words)
        wordlist_rel = self.remove_stopwords(str1_rel, stop_words)

        # list of top 20 most common words
        top20_all = Counter(wordlist_all).most_common(20)
        top20_rel = Counter(wordlist_rel).most_common(20)

        #
        list1, list2 = zip(*top20_all)  # split tuple into two list
        list3, list4 = zip(*top20_rel)  # split relative data tuple into two list

        difference = set(list3) - set(list1)
        new_features_list = list(difference)

        # print('new-features-list:')
        # print(new_features_list)

        return new_features_list

    def co_occur_words(self):
        # print('in co occur words')

        df = self.train_dataset
        df1 = df[df.target == 1]

        # convert column Sentences to list
        df_1 = df1['Sentences'].astype(str).values.tolist()

        # RAKE
        r = Rake(min_length=1, max_length=2, ranking_metric=Metric.WORD_FREQUENCY)
        words = r.extract_keywords_from_sentences(df_1)
        coocwords = r.get_ranked_phrases()

        topcoocwords = Counter(coocwords).most_common(5)

        top_cooccur_words, list4 = zip(*topcoocwords)  # split relative data tuple into two list
        topcoocwords = list(top_cooccur_words)

        return topcoocwords


    def transform(self, X, y = None):
        # print('in transform\n')
        # print(X.shape)

        frequent_words = self.new_wordlist()
        frequent_words = self.new_features(X.columns.values, frequent_words)
        rake_cooccur_words = self.co_occur_words()

        new_features = frequent_words + rake_cooccur_words

        for i in new_features:
            X.loc[:, i] = ""
            for ind, row in X.iterrows():
                if i in X.loc[ind, 'Sentences']:
                    X.loc[ind, i] = 1
                    # print('1')
                else:
                    X.loc[ind, i] = 0
                    # print('0')
        X = X.drop(columns='Sentences')

        print('Complete Frequent words\n')
        return X

class FeatureEngineering(BaseEstimator, TransformerMixin):

    def get_num_words_per_sample(self, sample_texts):
        """Returns the number of words per sample given corpus.
        # Arguments
            sample_texts: list, sample texts.

        # Returns
            int, number of words per sample.
        """
        num_words = len(sample_texts.split())
        return num_words

    def num_of_upper_case(self, sample_texts):
        x = re.findall(r"\b[A-Z]+", sample_texts)
        return len(x)


    def num_punctuation(self, sample_texts):
        x = re.findall("[,.]+", sample_texts)
        return len(x)


    def question(self, sample_texts):
        x = re.findall("r[?]+", sample_texts)
        return len(x)

    def you(self, sample_texts):
        x = re.findall("you ", sample_texts)
        if len(x) > 0:
            return 1
        else:
            return 0

    def speech_tag(self, sample_texts):
        sentences = nltk.sent_tokenize(sample_texts)
        data = []

        for sent in sentences:
            data = data + nltk.pos_tag(nltk.word_tokenize(sent))

        return data

    def count_noun(self, speech_list):
        count = 0
        for x in speech_list:
            if x[1] == 'NNP' or x[1] == 'NN' or x[1] == 'NNS' or x[1] == 'NNPS':
                # print(x[1])
                count = count + 1
                # print(count)

        return count

    def count_verb(self, speech_list):
        count = 0
        for x in speech_list:
            if x[1] == 'VB' or x[1] == 'VBD' or x[1] == 'VBG' or x[1] == 'VBN' or x[1] == 'VBP' or x[1] == 'VBZ':
                # print(x[1])
                count = count + 1
                # print(count)

        return count

    def count_adjective(self, speech_list):
        count = 0
        for x in speech_list:
            if x[1] == 'JJ' or x[1] == 'JJR' or x[1] == 'JJS':
                # print(x[1])
                count = count + 1
                # print(count)

        return count

    def count_adverb(self, speech_list):
        count = 0
        for x in speech_list:
            if x[1] == 'RB' or x[1] == 'RBR' or x[1] == 'RBS':
                count = count + 1

        return count

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        i = 0

        sentence_as_list = X['Sentences'].tolist()

        X['num_words'] = X['Sentences'].apply(self.get_num_words_per_sample)
        # print(i+1)

        X['num_char'] = sentence_as_list
        X['num_char'] = X['Sentences'].apply(len)
        # print(i+2)

        X['num_uppercase'] = sentence_as_list
        X.loc[:, 'num_uppercase'] = X['Sentences'].apply(self.num_of_upper_case)
        # print(i + 3)

        X['num_punctuation'] = sentence_as_list
        X.loc[:, 'num_punctuation'] = X['Sentences'].apply(self.num_punctuation)
        # print(i + 4)

        X['question'] = sentence_as_list
        X.loc[:, 'question'] = X['Sentences'].apply(self.question)
        # print(i + 5)

        X['you'] = sentence_as_list
        X.loc[:, 'you'] = X['Sentences'].apply(self.you)
        # print(i + 6)

        X['speech_tag'] = sentence_as_list
        X.loc[:, 'speech_tag'] = X['speech_tag'].apply(self.speech_tag)
        # print(i + 7)

        X.loc[:, 'num_noun'] = X['speech_tag'].apply(self.count_noun)
        # print(i + 8)
        X.loc[:, 'num_verb'] = X['speech_tag'].apply(self.count_verb)
        # print(i + 9)
        X.loc[:, 'num_adjective'] = X['speech_tag'].apply(self.count_adjective)
        # print(i + 10)
        X.loc[:, 'num_adverb'] = X['speech_tag'].apply(self.count_adverb)
        # print(i + 11)

        X = X.drop(columns=['speech_tag', 'Sentences'])
        # print(i + 12)

        print('Complete Feature extraction and engineering\n')
        return X.values

preprocessor = FeatureUnion( transformer_list = [ ( 'frequent_words', FrequentWordsTransformer(train) ),
                                                  ( 'feature_eng', FeatureEngineering() ) ] )

cv = 5

def report(results, n_top=1):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# """
# ADABOOST
adaboost_grid = { 'n_estimators': np.arange(50, 150, 10),
                  'learning_rate': [0.0001, 0.01, 0.1, 1] }

ada_model = AdaBoostClassifier()

ada_cv = GridSearchCV(ada_model, adaboost_grid, cv=cv)

ab_pipeline = Pipeline(steps = [ ( 'preprocessor', preprocessor),
                                  ( 'std_scaler', StandardScaler() ),
                                  ( 'model', ada_cv ) ])

print("Fitting AdaBoost")
ab_pipeline.fit(X_train, y_train)
ab_y_pred = ab_pipeline.predict(X_test)

print("GridSearchCV took %d candidate parameter settings."
      % (len(ada_cv.cv_results_['params'])))
print("AdaBoost performance:")
print("Best parameter:", ada_cv.best_params_)
report(ada_cv.cv_results_)
print(classification_report(ab_y_pred,y_test))
print("ROC-AUC score: ", roc_auc_score(y_test, ab_y_pred))
# """

# """
# GRADIENT BOOST
gb_param = {'n_estimators':range(50,100,10),
            'max_depth':range(5,20,5),
            'min_samples_split':range(500,1001,100),
            'min_samples_leaf': range(20, 60, 10)}

gb_model = GradientBoostingClassifier(learning_rate=0.1,
                                      max_features='sqrt',
                                      subsample=0.8,
                                      random_state=10)

gb_cv = GridSearchCV(gb_model, gb_param, cv=cv)

gb_pipeline = Pipeline(steps = [ ( 'preprocessor', preprocessor),
                                  ( 'std_scaler', StandardScaler() ),
                                  ( 'model', gb_cv ) ])

print("Fitting Gradient Boost")
gb_pipeline.fit(X_train, y_train)
gb_pred = gb_pipeline.predict(X_test)

print("GridSearchCV took %d candidate parameter settings."
      % (len(gb_cv.cv_results_['params'])))
print("Gradient Boost performance:")
print("Best parameter:", gb_cv.best_params_)
print(classification_report(gb_pred,y_test))
print("ROC-AUC score: ", roc_auc_score(y_test, gb_pred))
# """