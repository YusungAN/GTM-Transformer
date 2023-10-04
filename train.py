import os
import argparse
import wandb
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
from datetime import datetime
from models.GTM import GTM
from models.FCN import FCN
from utils.data_multitrends import ZeroShotDataset
import pickle

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(args):
    print(args)
    # Seeds for reproducibility (By default we use the number 21)
    pl.seed_everything(args.seed)

    # Load sales data
    train_df = pd.read_csv(Path(args.data_folder + 'itemscout_item_word.csv'))
    chosen_cat = ['식품']#['패션의류', '패션잡화', '화장품/미용', '가구/인테리어', '식품', '생활/건강']
    train_df = train_df[train_df.cat1.isin(chosen_cat)]
    gtrends = pd.read_csv(Path(args.data_folder + 'item_word_trend.csv'))
    reviews_df = pd.read_csv(Path(args.data_folder + 'reviews_summ_total.csv'))
    train_df = train_df[train_df.keyword.isin(reviews_df.groupby('keyword').count().index.tolist())]

    # test_df = pd.read_csv(Path(args.data_folder + 'test.csv'), parse_dates=['release_date'])

    # Load category and color encodings
    # cat_dict = torch.load(Path(args.data_folder + 'category_labels.pt'))
    # col_dict = torch.load(Path(args.data_folder + 'color_labels.pt'))
    # fab_dict = torch.load(Path(args.data_folder + 'fabric_labels.pt'))
    # print(cat_dict)
    # print(col_dict)
    # print(fab_dict)
    # Load Google trends
    train_df.info()
    chosen_word = reviews_df.groupby('keyword').count().index.tolist()
    train_df = train_df[train_df.cat1.isin(chosen_word)]
    train_df = train_df.sample(frac=1)
    cut = len(train_df.index)
    print('len', len(train_df.index), 'cut', cut)
    train_loader = ZeroShotDataset(train_df[:cut-3200], Path(args.data_folder + '/images'), gtrends, args.trend_len, reviews_df).get_loader(batch_size=args.batch_size, train=True)
    test_loader = ZeroShotDataset(train_df[cut-3200:], Path(args.data_folder + '/images'), gtrends, args.trend_len, reviews_df).get_loader(batch_size=args.batch_size, train=True)
    # with open("train_loader.pickle","rb") as fw:
    #     train_loader = pickle.load(fw)
    # with open("test_loader.pickle","rb") as fw:
    #     test_loader = pickle.load(fw)
    
    # Create model
    if args.model_type == 'FCN':
        model = FCN(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            use_trends=args.use_trends,
            use_text=args.use_text,
            use_img=args.use_img,
            trend_len=args.trend_len,
            num_trends=args.num_trends,
            use_encoder_mask=args.use_encoder_mask,
            gpu_num=args.gpu_num
        )
    else:
        model = GTM(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_heads=args.num_attn_heads,
            num_layers=args.num_hidden_layers,
            use_text=args.use_text,
            use_img=args.use_img,
            trend_len=args.trend_len,
            num_trends=args.num_trends,
            use_encoder_mask=args.use_encoder_mask,
            autoregressive=args.autoregressive,
            gpu_num=args.gpu_num
        )

    # Model Training
    # Define model saving procedure
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    model_savename = args.model_type + '_' + args.wandb_run

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.log_dir + '/'+args.model_type,
        filename=model_savename+'---{epoch}---'+dt_string,
        monitor='val_mae',
        mode='min',
        save_top_k=1
    )

    # wandb.init(entity=args.wandb_entity, project=args.wandb_proj, name=args.wandb_run)
    # wandb_logger = pl_loggers.WandbLogger()
    # wandb_logger.watch(model)

    # If you wish to use Tensorboard you can change the logger to:
    # tb_logger = pl_loggers.TensorBoardLogger(args.log_dir+'/', name=model_savename)
    trainer = pl.Trainer(accelerator="auto", max_epochs=args.epochs, check_val_every_n_epoch=5, callbacks=[checkpoint_callback])

    # Fit model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    # Print out path of best model
    print(checkpoint_callback.best_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot sales forecasting')

    # General arguments
    parser.add_argument('--data_folder', type=str, default='dataset/')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--gpu_num', type=int, default=0)

    # Model specific arguments
    parser.add_argument('--model_type', type=str, default='GTM', help='Choose between GTM or FCN')
    parser.add_argument('--use_trends', type=int, default=1)
    parser.add_argument('--use_img', type=int, default=0)
    parser.add_argument('--use_text', type=int, default=1)
    parser.add_argument('--trend_len', type=int, default=157)
    parser.add_argument('--num_trends', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=52)
    parser.add_argument('--use_encoder_mask', type=int, default=1)
    parser.add_argument('--autoregressive', type=int, default=0)
    parser.add_argument('--num_attn_heads', type=int, default=4)
    parser.add_argument('--num_hidden_layers', type=int, default=1)

    # wandb arguments
    parser.add_argument('--wandb_entity', type=str, default='username-here')
    parser.add_argument('--wandb_proj', type=str, default='GTM')
    parser.add_argument('--wandb_run', type=str, default='Run1')

    args = parser.parse_args()
    run(args)
