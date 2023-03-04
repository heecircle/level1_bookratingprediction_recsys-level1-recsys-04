import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from tqdm import tqdm
import os


class Image_Dataset(Dataset):                               #################### <-------------------- 6 ####################
    def __init__(self, user_isbn_vector, img_vector, label):
        self.user_isbn_vector = user_isbn_vector                                                # user_id, isbn 행렬
        self.img_vector = img_vector                                                            # 이미지 텐서
        self.label = label                                                                      # rating
    def __len__(self):
        return self.user_isbn_vector.shape[0]                                                   # 평가한 횟수
    def __getitem__(self, i):
        return {
                'user_isbn_vector' : torch.tensor(self.user_isbn_vector[i], dtype=torch.long),  # i번째 행의 user_id와 isbn
                'img_vector' : torch.tensor(self.img_vector[i], dtype=torch.float32),           # i번째 행의 이미지 텐서
                'label' : torch.tensor(self.label[i], dtype=torch.float32),                     # i번째 행의 rating
                }


def image_vector(path):                     #################### <-------------------- 3 ####################
    img = Image.open(path)                  # 이미지 파일 열기
    scale = transforms.Resize((32, 32))     # 이미지 크기 변경
    tensor = transforms.ToTensor()          
    img_fe = Variable(tensor(scale(img)))   # torch.autograd.Variable은 더 이상 사용되지 않고 tensor를 return한다.
    return img_fe                           # 이미지 텐서 return


def process_img_data(df, books, user2idx, isbn2idx, train=False):       #################### <-------------------- 2 ####################
    books_ = books.copy()
    books_['isbn'] = books_['isbn'].map(isbn2idx)

    if train == True:
        print('train 작업')
        df_ = df.copy()
    else:
        print('test 작업')
        df_ = df.copy()
        df_['user_id'] = df_['user_id'].map(user2idx)
        df_['isbn'] = df_['isbn'].map(isbn2idx)

    df_ = pd.merge(df_, books_[['isbn', 'img_path']], on='isbn', how='left')
    df_['img_path'] = df_['img_path'].apply(lambda x: 'data/'+x)
    img_vector_df = df_[['img_path']].drop_duplicates().reset_index(drop=True).copy()
    data_box = []
    for idx, path in tqdm(enumerate(sorted(img_vector_df['img_path']))):
        data = image_vector(path)
        if data.size()[0] == 3:
            data_box.append(np.array(data))
        else:
            data_box.append(np.array(data.expand(3, data.size()[1], data.size()[2])))
    img_vector_df['img_vector'] = data_box
    df_ = pd.merge(df_, img_vector_df, on='img_path', how='left')
    return df_


def image_data_load(args):

    users = pd.read_csv(args.DATA_PATH + 'users.csv')
    books = pd.read_csv(args.DATA_PATH + 'books.csv')
    # books = pd.read_csv('/opt/ml/input/code/books_img.csv')
    # train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    train = pd.read_csv('/opt/ml/input/code/train_img.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')


    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)

    img_train = process_img_data(train, books, user2idx, isbn2idx, train=True)
    img_test = process_img_data(test, books, user2idx, isbn2idx, train=False)

    data = {
            'train':train,
            'test':test,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            'img_train':img_train,
            'img_test':img_test,
            }

    return data                         #################### --------------------> Cycle 1 종료 ####################


def image_data_split(args, data):           #################### <-------------------- 4 (Cycle 2) ####################
    X_train, X_valid, y_train, y_valid = train_test_split(                                          # validation set 만들기
                                                        data['img_train'][['user_id', 'isbn', 'img_vector']],
                                                        data['img_train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data                             #################### --------------------> Cycle 2 종료 ####################


def image_data_loader(args, data):                              #################### <-------------------- 5 (Cycle 3) ####################
    train_dataset = Image_Dataset(                              #################### <-------------------- 6 ####################
                                data['X_train'][['user_id', 'isbn']].values,
                                data['X_train']['img_vector'].values,
                                data['y_train'].values
                                )
    valid_dataset = Image_Dataset(
                                data['X_valid'][['user_id', 'isbn']].values,
                                data['X_valid']['img_vector'].values,
                                data['y_valid'].values
                                )
    test_dataset = Image_Dataset(
                                data['img_test'][['user_id', 'isbn']].values,
                                data['img_test']['img_vector'].values,
                                data['img_test']['rating'].values
                                )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=True)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)
    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data                         #################### --------------------> Cycle 3 종료 ####################
