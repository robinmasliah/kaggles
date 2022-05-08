from wordcloud import WordCloud
from textwrap import wrap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.get_backend()
import seaborn as sns
plt.style.use('seaborn')
import nltk
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV

class MyTokenizer:
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        transformed_X = []
        for document in X:
            tokenized_doc = []
            for sent in nltk.sent_tokenize(document):
                tokenized_doc += nltk.word_tokenize(sent)
            transformed_X.append(np.array(tokenized_doc))
        return np.array(transformed_X)
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.wv.syn0[0])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = MyTokenizer().fit_transform(X)
        
        return np.array([
            np.mean([self.word2vec.wv[w] for w in words if w in self.word2vec.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    
    def fit_transform(self, X, y=None):
        return self.transform(X)
    
def score_cross_val(scores):
    
    # Scores sur cross validation
    scores = pd.DataFrame(scores)
    scores.plot()

    plt.xlabel('CV')
    plt.ylabel('Score')
    plt.title("Scores sur le train de chaque cross validation")

    plt.axis()
    plt.legend("scores", loc='upper center')
    plt.show()

def plot_roc_auc_curve(model, X_test, y_test):
    
    # overall accuracy
    acc = model.score(X_test, y_test)

    # get roc/auc info
    # predict_proba is the porbability that X takes a class
    Y_score = model.predict_proba(X_test)[:, 1] 
    fpr = dict()
    tpr = dict()
    fpr, tpr, _ = roc_curve(y_test, Y_score)

    roc_auc = dict()
    roc_auc = auc(fpr, tpr)

    # make the plot
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.plot(fpr, tpr, label='AUC = {0}'.format(roc_auc))
    plt.legend(loc="lower right", shadow=True, fancybox=True)
    plt.title('ROC curve and AUC')
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    
    #Generate the confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    group_names = ['True Neg','False Pos','False Neg','True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()

def plot_corr_matrix(df):
    
    data_correlation = df[list(df.columns)].corr()
    mask = np.array(data_correlation)
    mask[np.tril_indices_from(mask)] = False
    fig = plt.subplots(figsize=(30,20))
    sns.heatmap(data_correlation, mask=mask, vmax=1, square=True, annot=True);
    
# Function for generating word clouds
def generate_wordcloud(data,title):
    wc = WordCloud(width=400, height=330, max_words=150, colormap="Dark2").generate_from_frequencies(data)
    plt.figure(figsize=(10,8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title('\n'.join(wrap(title,60)), fontsize=13)
    plt.show()