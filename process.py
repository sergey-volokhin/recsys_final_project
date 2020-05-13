import pandas as pd


def get_label(row):
    if row['reply_timestamp']:
        return 'reply_timestamp'
    elif row['retweet_with_comment_timestamp']:
        return 'retweet_with_comment_timestamp'
    elif row['retweet_timestamp']:
        return 'retweet_timestamp'
    elif row['like_timestamp']:
        return 'like_timestamp'
    return 'no_engagement'


all_features = ['text_tokens', 'hashtags', 'tweet_id', 'present_media', 'present_links', 'present_domains',\
                'tweet_type', 'language', 'tweet_timestamp', 'engaged_with_user_id', 'engaged_with_user_follower_count',\
                'engaged_with_user_following_count', 'engaged_with_user_is_verified', 'engaged_with_user_account_creation',\
                'enaging_user_id', 'enaging_user_follower_count', 'enaging_user_following_count', 'enaging_user_is_verified',\
                'enaging_user_account_creation', 'engagee_follows_engager']

all_features_to_idx = dict(zip(all_features, range(len(all_features))))
labels_to_idx = {'reply_timestamp': 20, 'retweet_timestamp': 21, 'retweet_with_comment_timestamp': 22, 'like_timestamp': 23}
my_features = all_features[9:] + list(labels_to_idx.keys())
rows = {i: [] for i in my_features}

for file_name in ['train', 'val']:
    with open(f'{file_name}_short.tsv', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            for name, value in zip(my_features, line.split('\x01')[9:]):
                rows[name].append(value)

    df = pd.DataFrame(rows)

    df['label'] = df.apply(lambda row: get_label(row), axis=1)
    df = df.drop(['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp'], axis=1)
    df.to_csv(f'class_dataframe_{file_name}.tsv', sep='\t', index=False)
