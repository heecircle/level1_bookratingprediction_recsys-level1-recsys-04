import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from tqdm import tqdm

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
        df_ = df.copy()
    else:
        df_ = df.copy()
        df_['user_id'] = df_['user_id'].map(user2idx)
        df_['isbn'] = df_['isbn'].map(isbn2idx)

    df_ = pd.merge(df_, books_[['isbn', 'img_path']], on='isbn', how='left')            # df_에 있는 isbn만 골라서 img_path 합치기
    df_['img_path'] = df_['img_path'].apply(lambda x: 'data/'+x)                        # 파일명 앞에 directory 붙이기
    img_vector_df = df_[['img_path']].drop_duplicates().reset_index(drop=True).copy()   # img_path의 중복 없애고 행 idx 0부터 다시 붙이기
    data_box = []
    for idx, path in tqdm(enumerate(sorted(img_vector_df['img_path']))):
        data = image_vector(path)                                       #################### <-------------------- 3 ####################
        if data.size()[0] == 3:
            data_box.append(np.array(data))
        else:
            data_box.append(np.array(data.expand(3, data.size()[1], data.size()[2])))   # 이미지 텐서가 2차원일 경우 3차원으로 늘려서 append
    img_vector_df['img_vector'] = data_box                                              # 이미지 텐서 column 추가
    df_ = pd.merge(df_, img_vector_df, on='img_path', how='left')                       # df_에 있는 img_path만 골라서 img_vector 합치기
    return df_      # user_id, isbn, rating, img_path, img_vector 를 가진 dataframe return


def image_data_load(args):          #################### <-------------------- 1 (Cycle 1) ####################

    users = pd.read_csv(args.DATA_PATH + 'users.csv')               ###################################################################
    books = pd.read_csv(args.DATA_PATH + 'books.csv')
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')       #                       csv 파일 열기
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')     ###################################################################

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()    # train과 sub에 있는 user_id 전부 모으기
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()        # train과 sub에 있는 isbn 전부 모으기

    idx2user = {idx:id for idx, id in enumerate(ids)}               # key: idx, value: user_id (?)
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}         #

    user2idx = {id:idx for idx, id in idx2user.items()}             # key: user_id, value: idx
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}         #

    train['user_id'] = train['user_id'].map(user2idx)               # user_id를 0부터 번호 다시 매기기
    sub['user_id'] = sub['user_id'].map(user2idx)                   #

    train['isbn'] = train['isbn'].map(isbn2idx)                     # isbn을 0부터 번호 다시 매기기
    sub['isbn'] = sub['isbn'].map(isbn2idx)                         #

    img_train = process_img_data(train, books, user2idx, isbn2idx, train=True)  #################### <-------------------- 2 ####################
    img_test = process_img_data(test, books, user2idx, isbn2idx, train=False)   #

    data = {                            # 이미지 관련 data 총 정리
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
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=False)
    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data                         #################### --------------------> Cycle 3 종료 ####################
