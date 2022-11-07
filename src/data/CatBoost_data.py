import numpy as np
import pandas as pd
from catboost import *
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def change_age(x):
	if x < 10: return 0
	elif 10 <= x < 20: return 1
	elif 20 <= x < 30: return 2
	elif 30 <= x < 40: return 3
	elif 40 <= x < 50: return 4
	elif 50 <= x < 60: return 5
	elif 60 <= x < 70: return 6
	else: return 7


def cat_data_load(args):

    ######################## DATA LOAD
    users = pd.read_csv(args.DATA_PATH + 'users2.csv')
    books = pd.read_csv(args.DATA_PATH + 'books3.csv')
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')
    # summary = pd.read_csv(args.DATA_PATH + 'summary_merge_df.csv')

    users['age'] = users['age'].apply(change_age)

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    users_ = users.copy()
    books_ = books.copy()
    books_ = books_.drop(['img_url','img_path'],axis=1)
    # arr_box = []
    # for path in tqdm(books['img_path']):
    #     img_arr = np.asarray(Image.open('data/'+path))
    #     arr_box.append(img_arr.sum())
    # books_['img_arr'] = arr_box                                      # books에 이미지 유무를 판단하는 img_arr 추가

    train = pd.merge(train, users_, on='user_id', how='left')
    sub = pd.merge(sub, users_, on='user_id', how='left')
    test = pd.merge(test, users_, on='user_id', how='left')
    train = pd.merge(train, books_, on='isbn', how='left')
    sub = pd.merge(sub, books_, on='isbn', how='left')
    test = pd.merge(test, books_, on='isbn', how='left')

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)

    train['year_of_publication'] = train['year_of_publication'].astype(int)
    sub['year_of_publication'] = sub['year_of_publication'].astype(int)
    test['year_of_publication'] = test['year_of_publication'].astype(int)

    train = train.fillna('-1')
    sub = sub.fillna('-1')
    test = test.fillna('-1')

    sub_rating = sub['rating']
    sub = sub.drop(columns='rating')
    sub['rating'] = sub_rating

    data = {
            'train':train,
            'test':test.drop(['rating'], axis=1),
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data

def cat_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data