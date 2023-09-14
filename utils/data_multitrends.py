import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


ImageFile.LOAD_TRUNCATED_IMAGES = True


class ZeroShotDataset():
    def __init__(self, data_df, img_root, gtrends, trend_len, reviews):
        self.data_df = data_df
        self.gtrends = gtrends
        self.trend_len = trend_len
        # self.img_root = img_root

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        return self.data_df.iloc[idx, :]

    def preprocess_data(self):
        data = self.data_df

        # Get the Gtrends time series associated with each product
        # Read the images (extracted image features) as well
        gtrends, image_features = [], []
        # img_transforms = Compose([Resize((256, 256)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        for (idx, row) in tqdm(data.iterrows(), total=len(data), ascii=True):
            # cat, col, fab, start_date = row['category'], row['color'], row['fabric'], \
            #     row['release_date']
            keyword = row['keyword']
            # Get the gtrend signal up to the previous year (52 weeks) of the release date
            # gtrend_start = start_date - pd.DateOffset(weeks=52)
            # cat_gtrend = self.gtrends.loc[gtrend_start:start_date][cat][-52:].values[:self.trend_len]
            # col_gtrend = self.gtrends.loc[gtrend_start:start_date][col][-52:].values[:self.trend_len]
            # fab_gtrend = self.gtrends.loc[gtrend_start:start_date][fab][-52:].values[:self.trend_len]

            # cat_gtrend = MinMaxScaler().fit_transform(cat_gtrend.reshape(-1,1)).flatten()
            # col_gtrend = MinMaxScaler().fit_transform(col_gtrend.reshape(-1,1)).flatten()
            # fab_gtrend = MinMaxScaler().fit_transform(fab_gtrend.reshape(-1,1)).flatten()
            item_cat = data[data['keyword'] == keyword]['cat2']
            print('item_cat', item_cat)
            cat_gtrend = self.gtrends.loc[item_cat][1:1+self.trend_len].values
            brand_gtrend = self.gtrends.loc[keyword][1:1+self.trend_len].values

            cat_gtrend = MinMaxScaler().fit_transform(cat_gtrend.reshape(-1, 1)).flatten()
            brand_gtrend = MinMaxScaler().fit_transform(brand_gtrend.reshape(-1, 1)).flatten()
            multitrends = np.vstack([cat_gtrend, brand_gtrend])
            print(brand, 'cat_trend: ', cat_gtrend)
            print(brand, 'multi_Treand', multitrends)

            # Read images
            # img = Image.open(os.path.join(self.img_root, img_path)).convert('RGB')

            # Append them to the lists
            gtrends.append(multitrends)
            # image_features.append(img_transforms(img))

        # Convert to numpy arrays
        gtrends = np.array(gtrends)

        # Remove non-numerical information
        # data.drop(['external_code', 'season', 'release_date', 'image_path'], axis=1, inplace=True)

        # Create tensors for each part of the input/output

        # categories, colors, fabrics = [self.cat_dict[val] for val in data.iloc[:].category.values], \
        #                               [self.col_dict[val] for val in data.iloc[:].color.values], \
        #                               [self.fab_dict[val] for val in data.iloc[:].fabric.values]
        # data['category'] = pd.Series(categories)
        # data['color'] = pd.Series(colors)
        # data['fabric'] = pd.Series(fabrics)
        # print('sdaf', data.iloc[:, :12].values, data.iloc[:, 13:17].values)
        # item_sales, temporal_features = torch.FloatTensor(data.iloc[:, :12].values), torch.FloatTensor(
        #     data.iloc[:, 13:17].values)
        # print('item sale, temporal_feature', item_sales, temporal_features)
        # print('c, c, f', categories, colors, fabrics)
        # categories, colors, fabrics = torch.LongTensor(categories), torch.LongTensor(colors), torch.LongTensor(fabrics)

        item_sale_li = []
        # def extract_target_data(x):
        #     item_sale_li.append(data.loc[:, '20220103':'20221226'].values)
        data.apply(lambda x: item_sale_li.append(data.loc[:, '20220103':'20221226'].values), axis=1)
        item_sales = torch.FloatTensor(item_sale_li)
        gtrends = torch.FloatTensor(gtrends)
        model = SentenceTransformer('beomi/KcELECTRA-base-v2022')
        
        res = 0
        words = []
        for k in range(len(data['keyword'].values)):
            class_doc = reviews[reviews['keyword'] == k]
            if len(class_doc.index) != 0:
                df2 = pd.DataFrame(class_doc.values, columns=['text'])
            
                tfidf_vector = TfidfVectorizer()
                tfidf_matrix = tfidf_vector.fit_transform(df2['text']).toarray()
                tfidf_feature = tfidf_vector.get_feature_names_out()
                result = pd.DataFrame(tfidf_matrix, columns=tfidf_feature)
                tmp = result.iloc[k]
                tmp.sort_values(ascending=False, inplace=True)
                words.append(' '.join(list(tmp[:5].index)+[k, data[data['keyword'] == k]['cat2']]))
            else:
                words.append([k, data[data['keyword'] == k]['cat2'], data[data['keyword'] == k]['cat3']])
        word_embeddings = model.encode(words)
        text = torch.FloatTensor(word_embeddings)
        # images = torch.stack(image_features)

        return TensorDataset(item_sales, text, gtrends)

    def get_loader(self, batch_size, train=True):
        print('Starting dataset creation process...')
        data_with_gtrends = self.preprocess_data()
        data_loader = None
        if train:
            data_loader = DataLoader(data_with_gtrends, batch_size=batch_size, shuffle=True, num_workers=4)
        else:
            data_loader = DataLoader(data_with_gtrends, batch_size=1, shuffle=False, num_workers=4)
        print('Done.')

        return data_loader

