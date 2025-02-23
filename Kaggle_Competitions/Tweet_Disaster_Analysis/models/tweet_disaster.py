import pandas as pd
import spacy
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from xgboost import XGBClassifier

nlp = spacy.load("en_core_web_sm")
def preprocess_text_lemma_spacy(text):
    doc = nlp(text.lower())
    lemmatized_words = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    return ' '.join(lemmatized_words)

def stemming(text):
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    return ' '.join([stemmer.stem(word) for word in words])
    # stemmer = SnowballStemmer('english')
    # stop_words = set(stopwords.words('english'))
    # words = word_tokenize(text)
    # text = [stemmer.stem(word) for word in words if word not in stop_words]
    # #text = [i for i in text if len(i) > 2]
    # return ' '.join(text)

def preprocess_text(text):
    # text = re.sub(r"won\'t", " will not", text)
    # text = re.sub(r"won\'t've", " will not have", text)
    # text = re.sub(r"can\'t", " can not", text)
    # text = re.sub(r"don\'t", " do not", text)
    # text = re.sub(r"can\'t've", " can not have", text)
    # text = re.sub(r"ma\'am", " madam", text)
    # text = re.sub(r"let\'s", " let us", text)
    # text = re.sub(r"ain\'t", " am not", text)
    # text = re.sub(r"shan\'t", " shall not", text)
    # text = re.sub(r"sha\n't", " shall not", text)
    # text = re.sub(r"o\'clock", " of the clock", text)
    # text = re.sub(r"y\'all", " you all", text)
    # text = re.sub(r"n\'t", " not", text)
    # text = re.sub(r"n\'t've", " not have", text)
    # text = re.sub(r"\'re", " are", text)
    # text = re.sub(r"\'s", " is", text)
    # text = re.sub(r"\'d", " would", text)
    # text = re.sub(r"\'d've", " would have", text)
    # text = re.sub(r"\'ll", " will", text)
    # text = re.sub(r"\'ll've", " will have", text)
    # text = re.sub(r"\'t", " not", text)
    # text = re.sub(r"\'ve", " have", text)
    # text = re.sub(r"\'m", " am", text)
    # text = re.sub(r"\'re", " are", text)
    text = re.sub(r'bin laden', 'Binladen', text, flags=re.IGNORECASE)
    text = re.sub(r"http\S+|www\S+|https\S+", 'http', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    # text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\b(?<!breaking)news\b|\b(?<!breaking)\w*news\w*\b', 'news', text)
    # text = re.sub(r'[^\x00-\x7F]+', '', text)
    # text = re.sub(r'<.*?>', ' ', text)
    # text = re.sub("["
    #               u"\U0001F600-\U0001F64F"  # removal of emoticons
    #               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    #               u"\U0001F680-\U0001F6FF"  # transport & map symbols
    #               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    #               u"\U00002702-\U000027B0"
    #               u"\U000024C2-\U0001F251"
    #               "]+", ' ', text)
    #
    # text = re.sub(r"\([^()]*\)", "", text)
    return text



train_df = pd.read_csv(r"D:\elggak\kaggle\Tweet Disaster Competition\nlp-getting-started\train.csv")
test_df = pd.read_csv(r"D:\elggak\kaggle\Tweet Disaster Competition\nlp-getting-started\test.csv")
train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)
train_df['text'] = train_df['text'].apply(preprocess_text_lemma_spacy)
test_df['text'] = test_df['text'].apply(preprocess_text_lemma_spacy)
# train_df['text'] = train_df['text'].apply(stemming)
# test_df['text'] = test_df['text'].apply(stemming)
# train_df['text'] = train_df['text'].apply(text_cleaning)
# test_df['text'] = test_df['text'].apply(text_cleaning)

train_df_id = train_df['id']
test_df_id = test_df['id']
X = train_df['text']
y = train_df['target']
X_test = test_df['text']
print(X.head(15))
def tfid_preprocessing(train_data, test_data, train_target):
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    train_data = vectorizer.fit_transform(train_data)
    test_data = vectorizer.transform(test_data)
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_target.values, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val, test_data

def count_preprocessing(train_data, test_data, train_target):
    vectorizer = CountVectorizer()
    train_data = vectorizer.fit_transform(train_data)
    test_data = vectorizer.transform(test_data)
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_target.values, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val, test_data

def naive_bayes(X_train, X_val, y_train, y_val, X_test):
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_val)
    print(f'Accuracy: {accuracy_score(y_val, y_pred)}')
    print(classification_report(y_val, y_pred, target_names=['ham', 'spam'], digits=6))
    y_pred = nb_model.predict(X_test)
    output_df = pd.DataFrame({
        'id': test_df_id,
        'target': y_pred
    })

    output_df.to_csv(r'D:\Kaggle\disaster tweets\nb_normal.csv', index=False)

def logistic_regression(X_train, X_val, y_train, y_val, X_test):
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_val)
    print(f'Accuracy: {accuracy_score(y_val, y_pred)}')
    print(classification_report(y_val, y_pred, target_names=['ham', 'spam'], digits=6))
    y_pred = lr_model.predict(X_test)
    output_df = pd.DataFrame({
        'id': test_df_id,
        'target': y_pred
    })

    output_df.to_csv(r'D:\Kaggle\disaster tweets\lr_normal.csv', index=False)

def svm_classification(X_train, X_val, y_train, y_val, X_test):
    svm_model = SVC(kernel='rbf')
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_val)
    print(f'Accuracy: {accuracy_score(y_val, y_pred)}')
    print(classification_report(y_val, y_pred, target_names=['ham', 'spam'], digits=6))
    y_pred = svm_model.predict(X_test)
    output_df = pd.DataFrame({
        'id': test_df_id,
        'target': y_pred
    })

    output_df.to_csv(r'D:\Kaggle\disaster tweets\svm_normal.csv', index=False)


def run_models(X_train, X_val, y_train, y_val, X_test, model_name):
    print(f"Running models on {model_name} dataset...")

    # Naive Bayes
    nb_model = MultinomialNB(alpha=0.7)
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_val)
    print(f'Naive Bayes Accuracy: {accuracy_score(y_val, y_pred_nb)}')
    print(classification_report(y_val, y_pred_nb, target_names=['ham', 'spam'], digits=6))

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_val)
    print(f'Logistic Regression Accuracy: {accuracy_score(y_val, y_pred_lr)}')
    print(classification_report(y_val, y_pred_lr, target_names=['ham', 'spam'], digits=6))

    # SVM
    svm_model = SVC(kernel='rbf')
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_val)
    print(f'SVM Accuracy: {accuracy_score(y_val, y_pred_svm)}')
    print(classification_report(y_val, y_pred_svm, target_names=['ham', 'spam'], digits=6))

    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_val)
    print(f'XGB Accuracy: {accuracy_score(y_val, y_pred_xgb)}')
    print(classification_report(y_val, y_pred_xgb, target_names=['ham', 'spam'], digits=6))

    ensemble_model = VotingClassifier(estimators=[
        ('nb', nb_model),
        ('lr', lr_model),
        ('svm', svm_model),
        ('xgb', xgb_model)
    ], voting='hard')  # Hard voting for majority vote

    ensemble_model.fit(X_train, y_train)
    y_pred_ensemble = ensemble_model.predict(X_val)
    print(f'Ensemble Model Accuracy: {accuracy_score(y_val, y_pred_ensemble)}')
    print(classification_report(y_val, y_pred_ensemble, target_names=['ham', 'spam'], digits=6))

    # Predict on test data for all models
    y_pred_nb_test = nb_model.predict(X_test)
    y_pred_lr_test = lr_model.predict(X_test)
    y_pred_svm_test = svm_model.predict(X_test)
    y_pred_xgb_test = xgb_model.predict(X_test)
    y_pred_ensemble_test = ensemble_model.predict(X_test)

    # Saving predictions to CSV
    output_nb = pd.DataFrame({'id': test_df['id'], 'target': y_pred_nb_test})
    output_lr = pd.DataFrame({'id': test_df['id'], 'target': y_pred_lr_test})
    output_svm = pd.DataFrame({'id': test_df['id'], 'target': y_pred_svm_test})
    output_xgb = pd.DataFrame({'id': test_df['id'], 'target': y_pred_xgb_test})
    output_ensemble = pd.DataFrame({'id': test_df['id'], 'target': y_pred_ensemble_test})

    output_nb.to_csv(r'D:\Kaggle\disaster tweets\nb_predictions.csv', index=False)
    output_lr.to_csv(r'D:\Kaggle\disaster tweets\lr_predictions.csv', index=False)
    output_svm.to_csv(r'D:\Kaggle\disaster tweets\svm_predictions.csv', index=False)
    output_xgb.to_csv(r'D:\Kaggle\disaster tweets\xgb_predictions.csv', index=False)
    output_ensemble.to_csv(r'D:\Kaggle\disaster tweets\ensemble_predictions.csv', index=False)

if __name__ == '__main__':
    # Run the pipeline for TF-IDF
    X_train_tfidf, X_val_tfidf, y_train_tfidf, y_val_tfidf, X_test_tfidf = tfid_preprocessing(X, X_test, y)
    run_models(X_train_tfidf, X_val_tfidf, y_train_tfidf, y_val_tfidf, X_test_tfidf, "TF-IDF")

    # # Run the pipeline for Count Vectorization
    X_train_count, X_val_count, y_train_count, y_val_count, X_test_count = count_preprocessing(X, X_test, y)
    run_models(X_train_count, X_val_count, y_train_count, y_val_count, X_test_count, "Count Vectorization")