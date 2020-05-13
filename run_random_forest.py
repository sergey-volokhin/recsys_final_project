import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


features = ['engaged_with_user_follower_count',
            'engaged_with_user_following_count', 'engaged_with_user_is_verified',
            'enaging_user_follower_count', 'enaging_user_following_count',
            'enaging_user_is_verified',
            'engagee_follows_engager']
label_column = 'label'

df = pd.read_csv('class_dataframe_train.tsv', sep='\t')
df = df.loc[df['label'] != 'no_engagement']
pd.concat([df.loc[df['label'] == 'like_timestamp'][:10000],
           df.loc[df['label'] == 'reply_timestamp'][:10000],
           df.loc[df['label'] == 'retweet_timestamp'][:10000],
           df.loc[df['label'] == 'retweet_with_comment_timestamp'][:10000]], axis=1)

X_train = df[features]
y_train = df[label_column]

print('done reading train')
df = pd.read_csv('class_dataframe_val.tsv', sep='\t')
df = df.loc[df['label'] != 'no_engagement']
df = df[:100000]
X_test = df[features]
y_test = df[label_column]

print('done reading test')

clf = RandomForestClassifier(n_estimators=200, max_depth=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('done training')
results = f1_score(y_test, y_pred, average=None)

print(results)
