import argparse
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from models.GTM import GTM
from models.FCN import FCN
from utils.data_multitrends import ZeroShotDataset
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sentence_transformers import SentenceTransformer
from pathlib import Path


def cal_error_metrics(gt, forecasts):
    # Absolute errors
    mae = mean_absolute_error(gt, forecasts)
    wape = 100 * np.sum(np.sum(np.abs(gt - forecasts), axis=-1)) / np.sum(gt)

    return round(mae, 3), round(wape, 3)
    

def print_error_metrics(y_test, y_hat, rescaled_y_test, rescaled_y_hat):
    mae, wape = cal_error_metrics(y_test, y_hat)
    rescaled_mae, rescaled_wape = cal_error_metrics(rescaled_y_test, rescaled_y_hat)
    print(mae, wape, rescaled_mae, rescaled_wape)

def run(args):
    print(args)
    
    # Set up CUDA
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')

    # Seeds for reproducibility
    pl.seed_everything(args.seed)

    # # Load sales data    
    # test_df = pd.read_csv(Path(args.data_folder + 'test.csv'), parse_dates=['release_date'])
    # item_codes = test_df['external_code'].values

    #  # Load category and color encodings
    # cat_dict = torch.load(Path(args.data_folder + 'category_labels.pt'))
    # col_dict = torch.load(Path(args.data_folder + 'color_labels.pt'))
    # fab_dict = torch.load(Path(args.data_folder + 'fabric_labels.pt'))

    # Load Google trends
    # gtrends = pd.read_csv(Path(args.data_folder + 'gtrends.csv'), index_col=[0], parse_dates=True)
    
    # test_loader = ZeroShotDataset(test_df[:100], Path(args.data_folder + '/images'), gtrends, cat_dict, col_dict, \
    #         fab_dict, args.trend_len).get_loader(batch_size=1, train=False)


    model_savename = f'{args.wandb_run}_{args.output_dim}'
    
    # Create model
    model = None
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
    
    model.load_state_dict(torch.load(args.ckpt_path)['state_dict'], strict=False)

    # Forecast the testing set
    model.to(device)
    model.eval()
    gt, forecasts, attns = [], [],[]
    # for test_data in tqdm(test_loader, total=len(test_loader), ascii=True):
    #     with torch.no_grad():
    #         test_data = [tensor.to(device) for tensor in test_data]
    #         item_sales, category, color, textures, temporal_features, gtrends =  test_data
    #         y_pred, att = model(category, color,textures, temporal_features, gtrends)
    #         forecasts.append(y_pred.detach().cpu().numpy().flatten()[:args.output_dim])
    #         gt.append(item_sales.detach().cpu().numpy().flatten()[:args.output_dim])
    #         attns.append(att.detach().cpu().numpy())
    gtrends = [4.25469,4.41083,4.11319,3.20078,3.20078,3.33739,3.86435,3.73261,3.06416,3.5716,3.87899,3.48865,3.9717,3.66918,4.68894,4.43034,3.98633,3.62039,4.20102,3.64479,3.44962,2.7714,2.99585,3.77653,3.53744,3.33739,3.23005,3.1471,3.25933,3.07879,4.06928,37.72627,100,67.7775,41.59063,35.24762]
    sbert = SentenceTransformer('beomi/KcELECTRA-base-v2022')
    y_pred, att = model(sbert.encode(['다이어트 샐러드 건강 소스']}, gtrends)
    attns = np.stack(attns)
    forecasts = np.array(forecasts)
    gt = np.array(gt)
    print(y_pred.detach().cpu().numpy().flatten()[:args.output_dim])
    rescale_vals = np.load(args.data_folder + 'stfore_sales_norm_scalar.npy')
    rescaled_forecasts = forecasts * rescale_vals
    rescaled_gt = gt * rescale_vals
    print_error_metrics(gt, forecasts, rescaled_gt, rescaled_forecasts)

    
    torch.save({'results': forecasts* rescale_vals, 'gts': gt* rescale_vals, 'codes': item_codes.tolist()}, Path('GTM-Transformer/results/' + model_savename+'.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot sales forecasting')

    # General arguments
    parser.add_argument('--data_folder', type=str, default='dataset/')
    parser.add_argument('--ckpt_path', type=str, default='log/path-to-model.ckpt')
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--seed', type=int, default=21)

    # Model specific arguments
    parser.add_argument('--model_type', type=str, default='GTM', help='Choose between GTM or FCN')
    parser.add_argument('--use_trends', type=int, default=1)
    parser.add_argument('--use_img', type=int, default=0)
    parser.add_argument('--use_text', type=int, default=1)
    parser.add_argument('--trend_len', type=int, default=36)
    parser.add_argument('--num_trends', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=12)
    parser.add_argument('--use_encoder_mask', type=int, default=1)
    parser.add_argument('--autoregressive', type=int, default=0)
    parser.add_argument('--num_attn_heads', type=int, default=4)
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    
    # wandb arguments
    parser.add_argument('--wandb_run', type=str, default='Run1')

    args = parser.parse_args()
    run(args)
