import os
import pandas as pd
import numpy as np
import catboost
from catboost import *
from catboost.utils import create_cd
from sklearn.model_selection import train_test_split

def cat_data_load(args):

    ######################## DATA LOAD
    users = pd.read_csv(args.DATA_PATH + 'users.csv')
    books = pd.read_csv(args.DATA_PATH + 'books.csv')
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')
    # books_lang = pd.read_csv(args.DATA_PATH + 'books_lang.csv')
    # books_cate = pd.read_csv(args.DATA_PATH + 'books_fillna_cate.csv')

    # books['language'] = books_lang['language']
    # books['category'] = books_cate['category']
    # books['high_category'] = books_cate['high_category']

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()
    location = users['location'].unique()
    age = users['age'].unique()
    book_title = books['book_title'].unique()
    book_author = books['book_author'].unique()
    year_of_publication = books['year_of_publication'].unique()
    publisher = books['publisher'].unique()
    language = books['language'].unique()
    category = books['category'].unique()
    # high_category = books['high_category'].unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}
    location2idx = {location:idx for idx, location in enumerate(location)}
    age2idx = {age:idx for idx, age in enumerate(age)}
    title2idx = {title:idx for idx, title in enumerate(book_title)}
    author2idx = {author:idx for idx, author in enumerate(book_author)}
    year2idx = {title:idx for idx, title in enumerate(year_of_publication)}
    publisher2idx = {year:idx for idx, year in enumerate(publisher)}
    language2idx = {language:idx for idx, language in enumerate(language)}
    category2idx = {category:idx for idx, category in enumerate(category)}
    # high_cate2idx = {high_cate:idx for idx, high_cate in enumerate(high_category)}

    users_ = users.copy()
    books_ = books.copy()
    books_ = books_.drop(['img_url','img_path','summary'],axis=1)

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

    train['location'] = train['location'].map(location2idx)
    sub['location'] = sub['location'].map(location2idx)
    test['location'] = test['location'].map(location2idx)

    train['age'] = train['age'].map(age2idx)
    sub['age'] = sub['age'].map(age2idx)
    test['age'] = test['age'].map(age2idx)
    
    train['book_title'] = train['book_title'].map(title2idx)
    sub['book_title'] = sub['book_title'].map(title2idx)
    test['book_title'] = test['book_title'].map(title2idx)

    train['book_author'] = train['book_author'].map(author2idx)
    sub['book_author'] = sub['book_author'].map(author2idx)
    test['book_author'] = test['book_author'].map(author2idx)

    train['year_of_publication'] = train['year_of_publication'].map(year2idx)
    sub['year_of_publication'] = sub['year_of_publication'].map(year2idx)
    test['year_of_publication'] = test['year_of_publication'].map(year2idx)

    train['publisher'] = train['publisher'].map(publisher2idx)
    sub['publisher'] = sub['publisher'].map(publisher2idx)
    test['publisher'] = test['publisher'].map(publisher2idx)

    train['language'] = train['language'].map(language2idx)
    sub['language'] = sub['language'].map(language2idx)
    test['language'] = test['language'].map(language2idx)

    train['category'] = train['category'].map(category2idx)
    sub['category'] = sub['category'].map(category2idx)
    test['category'] = test['category'].map(category2idx)

    # train['high_category'] = train['high_category'].map(high_cate2idx)
    # sub['high_category'] = sub['high_category'].map(high_cate2idx)
    # test['high_category'] = test['high_category'].map(high_cate2idx)

    train = train.fillna('-1')
    sub = sub.fillna('-1')
    test = test.fillna('-1')

    sub_rating = sub['rating']
    sub = sub.drop(columns='rating')
    sub['rating'] = sub_rating
    
    cat_features = list(range(0, train.drop(columns='rating').shape[1]))
    feature_names = dict(list(enumerate(train.drop(columns='rating').keys()[1:])))
    
    pool = Pool(data=train.drop(columns='rating'), label=train['rating'], cat_features=cat_features)


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