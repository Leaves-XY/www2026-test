import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser()

    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config', 'basic.yaml')


    config = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))

    parser.add_argument('--model',type = str,default ='SASRec' ,choices = ['SASRec','FMLP','BSARec','BERT4Rec'],help='')


    parser.add_argument('--early_stop',type=int,default=15,help='')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (SAS.torch: 0.001)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (SAS.torch: 128)')
    parser.add_argument('--l2_reg', type=float, default=1e-6, help='L2 regularization coefficient')
    parser.add_argument('--l2_emb', type=float, default=1e-6, help='L2 regularization for embeddings (SAS.torch: 0.0)')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden layer dimensionality.')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate (SAS.torch: 0.2)')
    parser.add_argument('--epochs', type=int, default=1000000, help='')
    parser.add_argument('--dataset', type=str, default='ml-100k', help='Dataset name')
    parser.add_argument('--train_data', type=str, default='ml-100k.txt', help='train dataset')
    parser.add_argument('--num_layers', type=int, default=2, help='Transformer num_blocks: 2')
    parser.add_argument('--num_heads', type=int, default=1,)
    parser.add_argument('--inner_size', type=int, default=256, help='')
    parser.add_argument('--max_seq_len', type=int, default=200, help='maxlen: 200')
    parser.add_argument('--dim_s', type=int, default=16, help='')
    parser.add_argument('--dim_m', type=int, default=32, help='')
    parser.add_argument('--dim_l', type=int, default=64, help='')

    parser.add_argument('--decor_alpha', type=float, default=0.3, help='')
    parser.add_argument('--device_split', type=float, nargs=2, default=[0.5, 0.3], help='')

    parser.add_argument('--neg_num', type=int, default=99, help='')

    parser.add_argument('--skip_test_eval', action = 'store_true', default=False,
                        help='')

    parser.add_argument('--eval_freq', type=int, default=1,
                        help='')

    parser.add_argument ('--full_eval', action = 'store_true', default = True,
                         help = '')



    parser.add_argument('--c', type=int, default=9, help='')
    parser.add_argument('--alpha', type=float, default=0.3, help='')

    parser.add_argument('--mask_prob', type=float, default=0.15, help='')

    parser.add_argument ('--kd_ratio', type = float, default = 0.1,  help = '')
    parser.add_argument ('--kd_lr', type = float, default = 0.001,  help = '')
    parser.add_argument ('--distill_epochs', type = int, default = 10,  help = '')
    parser.add_argument ('--distill_freq', type = int, default = 3,  help = '')
    parser.add_argument ('--eval_k', type = int, default = 10,  help = '')

    parser.add_argument('--LDP_lambda', type=float, default=0, help='scale of laplace noise in LDP')

    parser.add_argument ('--top_k_ratio', type = float, default = 0.1,help = '')
    parser.add_argument ('--top_k_method', type = str, default = 'global',
                         choices = ['global', 'layer-wise'], help='')
    parser.add_argument ('--min_k', type = int, default = 1, help = '')

    args = parser.parse_args ()
    config.update (vars (args))

    return config

config = parse_args()