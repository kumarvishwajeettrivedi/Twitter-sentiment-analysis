# Twitter Sentiment Analysis

This project employs machine learning to categorize tweets as positive, negative, or neutral, offering insights into public opinion on various topics. The model is trained using Logistic Regression, Decision Tree, and XGBoost, leveraging TF-IDF features and Bag-of-Words for textual data representation.

## Features

- **Sentiment Analysis**: Classifies tweets into positive, negative, or neutral categories.
- **Hashtag Analysis**: Extracts and analyzes the impact of hashtags on tweet sentiments.
- **Word Frequency Analysis**: Identifies and visualizes the most frequent words in positive and negative tweets.
- **Text Feature Extraction**: Utilizes Bag-of-Words and TF-IDF for feature extraction from tweets.
- **Machine Learning Models**: Implements Logistic Regression, Decision Tree, and XGBoost for classification tasks.

## Word Cloud Visualization

```python
Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))
image_colors = ImageColorGenerator(Mask)

wc = WordCloud(background_color='black', height=1500, width=4000, mask=Mask).generate(all_words_negative)
plt.figure(figsize=(10, 20))
plt.imshow(wc.recolor(color_func=image_colors), interpolation="gaussian")
plt.axis('off')
plt.show()
```

## Hashtag Analysis

### Extracting Hashtags

```python
def Hashtags_Extract(x):
    hashtags = []
    for i in x:
        ht = re.findall(r'#(\w+)', i)
        hashtags.append(ht)
    return hashtags

ht_positive = Hashtags_Extract(combine['Tidy_Tweets'][combine['label'] == 0])
ht_positive_unnest = sum(ht_positive, [])
ht_negative = Hashtags_Extract(combine['Tidy_Tweets'][combine['label'] == 1])
ht_negative_unnest = sum(ht_negative, [])
```

### Word Frequency of Positive Hashtags

```python
word_freq_positive = nltk.FreqDist(ht_positive_unnest)
df_positive = pd.DataFrame({'Hashtags': list(word_freq_positive.keys()), 'Count': list(word_freq_positive.values())})
sns.barplot(data=df_positive.nlargest(20, columns='Count'), y='Hashtags', x='Count')
sns.despine()
```

### Word Frequency of Negative Hashtags

```python
word_freq_negative = nltk.FreqDist(ht_negative_unnest)
df_negative = pd.DataFrame({'Hashtags': list(word_freq_negative.keys()), 'Count': list(word_freq_negative.values())})
sns.barplot(data=df_negative.nlargest(20, columns='Count'), y='Hashtags', x='Count')
sns.despine()
```

## Feature Extraction

### Bag of Words

```python
from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(combine['Tidy_Tweets'])
df_bow = pd.DataFrame(bow.todense())
```

### TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(combine['Tidy_Tweets'])
df_tfidf = pd.DataFrame(tfidf_matrix.todense())
```

## Model Implementation

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

Log_Reg = LogisticRegression(random_state=0, solver='lbfgs')

# Using Bag-of-Words Features
Log_Reg.fit(x_train_bow, y_train_bow)
prediction_bow = Log_Reg.predict_proba(x_valid_bow)

# Calculating F1 Score
from sklearn.metrics import f1_score
prediction_int = prediction_bow[:, 1] >= 0.3
prediction_int = prediction_int.astype(np.int64)
log_bow = f1_score(y_valid_bow, prediction_int)

# Using TF-IDF Features
Log_Reg.fit(x_train_tfidf, y_train_tfidf)
prediction_tfidf = Log_Reg.predict_proba(x_valid_tfidf)

# Calculating F1 Score
prediction_int = prediction_tfidf[:, 1] >= 0.3
prediction_int = prediction_int.astype(np.int64)
log_tfidf = f1_score(y_valid_tfidf, prediction_int)
```

### XGBoost

```python
from xgboost import XGBClassifier

model_bow = XGBClassifier(random_state=22, learning_rate=0.9)
model_bow.fit(x_train_bow, y_train_bow)
```
