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
    def __init__(self, data_df, img_root, gtrends, trend_len, reviews, do_data_aug):
        self.data_df = data_df
        self.gtrends = gtrends
        self.trend_len = trend_len
        self.reviews = reviews
        # self.img_root = img_root
        self.do_data_aug = do_data_aug

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        return self.data_df.iloc[idx, :]

    def preprocess_data(self):
        data = self.data_df
        reviews = self.reviews
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

            # data augmentation test
            if self.do_data_aug:
                for delta in [0, 9, 22, 35]:
                    item_cat = data[data['keyword'] == keyword]['cat2'].values[0]
                    cat_gtrend = self.gtrends.loc[self.gtrends['keyword'] == item_cat.replace('/', '')].values[0][1:1+self.trend_len]
                    brand_gtrend = self.gtrends.loc[self.gtrends['keyword'] == keyword].values[0][1:1+self.trend_len]
                    cat_gtrend = MinMaxScaler().fit_transform(cat_gtrend.reshape(-1, 1)).flatten()[:self.trend_len]
                    brand_gtrend = MinMaxScaler().fit_transform(brand_gtrend.reshape(-1, 1)).flatten()
                    
                    cat_gtrend = np.concatenate((cat_gtrend[delta:], cat_gtrend[:delta]))
                    brand_gtrend = np.concatenate((brand_gtrend[delta:], brand_gtrend[:delta]))
                    
                    
                    multitrends = np.vstack([cat_gtrend, brand_gtrend])
                    # Read images
                    # img = Image.open(os.path.join(self.img_root, img_path)).convert('RGB')
        
                    # Append them to the lists
                    gtrends.append(multitrends)
                    # image_features.append(img_transforms(img))
            else:
                tem_cat = data[data['keyword'] == keyword]['cat2'].values[0]
                cat_gtrend = self.gtrends.loc[self.gtrends['keyword'] == item_cat.replace('/', '')].values[0][1:1+self.trend_len]
                brand_gtrend = self.gtrends.loc[self.gtrends['keyword'] == keyword].values[0][1:1+self.trend_len]
                cat_gtrend = MinMaxScaler().fit_transform(cat_gtrend.reshape(-1, 1)).flatten()[:self.trend_len]
                brand_gtrend = MinMaxScaler().fit_transform(brand_gtrend.reshape(-1, 1)).flatten()
    
                multitrends = np.vstack([cat_gtrend, brand_gtrend])
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
        def extract_target_data(x):
            if self.do_data_aug:
                gt_tmp = self.gtrends[self.gtrends['keyword'] == x['keyword']].iloc[:, -52:].values[0]
                for delta in [0, 9, 22, 35]:
                    item_sale_li.append(np.concatenate((gt_tmp[delta:], gt_tmp[:delta])))
            else:
                item_sale_li.append(self.gtrends[self.gtrends['keyword'] == x['keyword']].iloc[:, -52:].values[0])
        data.apply(extract_target_data, axis=1)
        item_sales = torch.FloatTensor(item_sale_li)
        gtrends = torch.FloatTensor(gtrends)
        model = SentenceTransformer('beomi/KcELECTRA-base-v2022')
        
        res = 0
        words = []
        for k in range(len(data['keyword'].values)):
            if self.do_data_aug:
                text = reviews[reviews['keyword'] == k]['summ'].values
                for _ in range(4):
                    if len(text) == 0 or text[0].strip() == '':
                        words.append([k, data[data['keyword'] == k]['cat2'], data[data['keyword'] == k]['cat3']])
                    else:
                        words.append(text[0])
            else:
                text = reviews[reviews['keyword'] == k]['summ'].values
                if len(text) == 0 or text[0].strip() == '':
                    words.append([k, data[data['keyword'] == k]['cat2'], data[data['keyword'] == k]['cat3']])
                else:
                    words.append(text[0])
            # class_doc = reviews[reviews['keyword'] == k]
            # if len(class_doc.index) != 0:
            #     df2 = pd.DataFrame(class_doc.values, columns=['text'])
            
            #     tfidf_vector = TfidfVectorizer()
            #     tfidf_matrix = tfidf_vector.fit_transform(df2['text']).toarray()
            #     tfidf_feature = tfidf_vector.get_feature_names_out()
            #     result = pd.DataFrame(tfidf_matrix, columns=tfidf_feature)
            #     tmp = result.iloc[k]
            #     tmp.sort_values(ascending=False, inplace=True)
            #     words.append(' '.join(list(tmp[:5].index)+[k, data[data['keyword'] == k]['cat2']]))
            # else:
            #     words.append([k, data[data['keyword'] == k]['cat2'], data[data['keyword'] == k]['cat3']])
        word_embeddings = model.encode(words)
        # word_embeddings = np.zeros(len(data['keyword'].values)*768).reshape(len(data['keyword'].values), 768)
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

