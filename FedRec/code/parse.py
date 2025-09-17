import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser()

    # 首先加载config文件
    import os
    # 獲取當前文件所在目錄
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config', 'basic.yaml')


    config = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))

    parser.add_argument('--model',type = str,default ='SASRec' ,choices = ['SASRec','FMLP','BSARec'],help='选择推荐模型')

    parser.add_argument ('--algorithm', type = str, default = 'UDL_DDR_RESKD',
                         choices = ['base','UDL','UDL_DDR','UDL_RESKD','UDL_DDR_RESKD','RESKD_I','DDR','RESKD_DDR','RESKD','UDL',
                                    'base_Top_k','UDL_Top_k','UDL_DDR_Top_k','UDL_DDR_RESKD_Top_k','DDR_Top_k','RESKD_Top_k','UDL_RESKD_Top_k','DDR_RESKD_Top_k','RESKD_I_Top_k',
                                    ], help = '算法选择')

    parser.add_argument('--early_stop',type=int,default=15,help='早停轮数')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (参考SAS.torch: 0.001)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (参考SAS.torch: 128)')
    parser.add_argument('--l2_reg', type=float, default=0, help='L2 regularization coefficient')
    parser.add_argument('--l2_emb', type=float, default=0.0, help='L2 regularization for embeddings (参考SAS.torch: 0.0)')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden layer dimensionality.')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate (参考SAS.torch: 0.2)')
    parser.add_argument('--epochs', type=int, default=1000000, help='训练轮数 ')
    parser.add_argument('--dataset', type=str, default='Steam', help='Dataset name')
    parser.add_argument('--train_data', type=str, default='Steam.txt', help='train dataset')
    parser.add_argument('--num_layers', type=int, default=2, help='Transformer层数 (参考SAS.torch num_blocks: 2)')
    parser.add_argument('--num_heads', type=int, default=1, help='注意力头数量 (参考SAS.torch: 1)')
    parser.add_argument('--inner_size', type=int, default=256, help='前馈网络内部大小')
    parser.add_argument('--max_seq_len', type=int, default=200, help='最大序列长度 (参考SAS.torch maxlen: 200)')
    parser.add_argument('--dim_s', type=int, default=16, help='小型设备嵌入维度')
    parser.add_argument('--dim_m', type=int, default=32, help='中型设备嵌入维度')
    parser.add_argument('--dim_l', type=int, default=64, help='大型设备嵌入维度')

    parser.add_argument('--decor_alpha', type=float, default=0.3, help='正则化参数')
    parser.add_argument('--device_split', type=float, nargs=2, default=[0.5, 0.3], help='小型和中型设备的分配比例阈值')

    parser.add_argument('--neg_num', type=int, default=99, help='负样本采样数量')

    # 测试集评估控制参数
    parser.add_argument('--skip_test_eval', action='store_true', default=True,
                        help='跳过测试集评估以节省训练时间，只进行验证集评估')

    # 评估频率控制参数
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='每隔多少个epoch进行一次评估，默认为20')

    parser.add_argument ('--full_eval', action = 'store_true', default = True,
                         help = '是否使用全量评估')



    parser.add_argument('--c', type=int, default=9, help='c for BSARec ,用来平衡模型对“从数据中学到的复杂模式（自注意力）”和“预设的通用规律（傅里叶归纳偏置）”的依赖程度')
    parser.add_argument('--alpha', type=float, default=0.3, help='alpha for BSARec ,用来在傅里叶变换后的频域中，区分什么是低频信号，什么是高频信号，低于C为低频信号，高于C为高频信息')

    # --- 知识蒸馏配置 ---
    parser.add_argument ('--kd_ratio', type = float, default = 0.1,  help = '蒸馏物品采样比例')
    parser.add_argument ('--kd_lr', type = float, default = 0.001,  help = '蒸馏学习率')
    parser.add_argument ('--distill_epochs', type = int, default = 10,  help = '蒸馏轮次')
    parser.add_argument ('--distill_freq', type = int, default = 3,  help = '蒸馏频率')
    parser.add_argument ('--eval_k', type = int, default = 10,  help = '评估')


    # Top-k配置
    parser.add_argument ('--top_k_ratio', type = float, default = 0.1,help = 'Top-k比例，默认保留10%的梯度')
    parser.add_argument ('--top_k_method', type = str, default = 'global',
                         choices = ['global', 'layer-wise'], help='Top-k类型')
    parser.add_argument ('--min_k', type = int, default = 1, help = '每层最少保留的梯度数量')

    args = parser.parse_args ()
    config.update (vars (args))  # 使用这个字典更新之前从 YAML 文件加载的配置字典 config

    return config

config = parse_args()