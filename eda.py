import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1','v2']].rename(columns={'v1':'label','v2':'text'})
df['label_num']     = df['label'].map({'ham':0,'spam':1})
df['num_chars']     = df['text'].apply(len)
df['num_words']     = df['text'].apply(lambda x: len(x.split()))
df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Spam vs Ham — EDA Dashboard', fontsize=16, fontweight='bold')

# 1. Class distribution pie
counts = df['label'].value_counts()
axes[0,0].pie(counts, labels=counts.index, autopct='%1.1f%%',
               colors=['#4fc3f7','#ef5350'])
axes[0,0].set_title('Class Distribution')

# 2. Character count histogram
df[df['label']=='ham']['num_chars'].plot(
    kind='hist', bins=40, ax=axes[0,1], alpha=0.6,
    label='Ham', color='#4fc3f7')
df[df['label']=='spam']['num_chars'].plot(
    kind='hist', bins=40, ax=axes[0,1], alpha=0.6,
    label='Spam', color='#ef5350')
axes[0,1].set_title('Character Count Distribution')
axes[0,1].legend()

# 3. Word count boxplot
sns.boxplot(data=df, x='label', y='num_words', ax=axes[0,2],
            palette={'ham':'#4fc3f7','spam':'#ef5350'})
axes[0,2].set_title('Word Count by Class')

# 4. Spam word cloud
spam_text = ' '.join(df[df['label']=='spam']['text'])
wc1 = WordCloud(width=400, height=200, background_color='white',
                colormap='Reds').generate(spam_text)
axes[1,0].imshow(wc1); axes[1,0].axis('off')
axes[1,0].set_title('Spam Word Cloud')

# 5. Ham word cloud
ham_text = ' '.join(df[df['label']=='ham']['text'])
wc2 = WordCloud(width=400, height=200, background_color='white',
                colormap='Blues').generate(ham_text)
axes[1,1].imshow(wc2); axes[1,1].axis('off')
axes[1,1].set_title('Ham Word Cloud')

# 6. Correlation heatmap
cols = ['num_chars','num_words','num_sentences','label_num']
sns.heatmap(df[cols].corr(), annot=True, fmt='.2f',
            ax=axes[1,2], cmap='coolwarm')
axes[1,2].set_title('Feature Correlation Matrix')

plt.tight_layout()
plt.savefig('eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("EDA plots saved to eda.png")
