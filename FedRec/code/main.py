import sys
import os
# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the project root directory to the Python path
sys.path.insert(0, project_root)
# -*- coding: utf-8 -*-
from parse import config
from untils import Logger

if __name__ == '__main__':
    algorithm=config['algorithm']


    # 创建日志记录器
    logger = Logger(config)
    logger.info(f"算法: {algorithm}")
    logger.info("开始训练，配置参数如下：")
    for key, value in config.items():
        logger.info(f"{key}: {value}")

    # 打印数据集路径信息
    train_data_path = config['datapath'] + config['dataset'] + '/' + config['train_data']
    logger.info(f"训练数据: {train_data_path}")


    logger.info(f"最大序列长度: {config['max_seq_len']}")
    logger.info(f"批次大小: {config['batch_size']}")


    if algorithm == 'base_Top_k':
        from FedRec.code.Top_k_sparse.Fed_base_Top_k import Clients, Server
    elif algorithm == 'UDL_Top_k':
        from FedRec.code.Top_k_sparse.Fed_UDL_Top_k import Clients, Server
    elif algorithm == 'UDL_RESKD_Top_k':
        from FedRec.code.Top_k_sparse.Fed_UDL_RESKD_Top_k import Clients, Server
    elif algorithm == 'DDR_RESKD_Top_k':
        from FedRec.code.Top_k_sparse.Fed_DDR_RESKD_Top_k import Clients, Server
    elif algorithm == 'UDL_DDR_Top_k':
        from FedRec.code.Top_k_sparse.Fed_UDL_DDR_Top_k import Clients, Server
    elif algorithm == 'UDL_DDR_RESKD_Top_k':
        from FedRec.code.Top_k_sparse.Fed_UDL_DDR_RESKD_Top_k import Clients, Server
    elif algorithm == 'DDR_Top_k':
        from FedRec.code.Top_k_sparse.Fed_DDR_Top_k import Clients, Server
    elif algorithm == 'RESKD_Top_k':
        from FedRec.code.Top_k_sparse.Fed_RESKD_Top_k import Clients, Server
    elif algorithm == 'RESKD_I_Top_k':
        from FedRec.code.Top_k_sparse.Fed_RESKD_I_Top_k import Clients, Server
    elif algorithm=='base':
        from FedRec.code.base.Fed_base import Clients, Server
    elif algorithm=='UDL':
        from FedRec.code.base.Fed_UDL import Clients, Server
    elif algorithm=='DDR':
        from FedRec.code.base.Fed_DDR import Clients, Server
    elif algorithm=='RESKD':
        from FedRec.code.base.Fed_RESKD import Clients, Server
    elif algorithm=='RESKD_DDR':
        from FedRec.code.base.Fed_RESKD_DDR import Clients, Server
    elif algorithm=='RESKD_I':
        from FedRec.code.base.Fed_RESKD_I import Clients, Server
    elif algorithm == 'UDL_DDR':
        from FedRec.code.base.Fed_UDL_DDR import Clients, Server
    elif algorithm == 'UDL_RESKD':
        from FedRec.code.base.Fed_UDL_RESKD import Clients, Server
    elif algorithm == 'UDL_DDR_RESKD':
        from FedRec.code.base.Fed_UDL_DDR_RESKD import Clients, Server
    else:
        raise ValueError(f"不支持的算法: {algorithm}")
    # 构建客户端
    clients = Clients(config, logger)
    logger.info(f"用户数量: {clients.usernum}")
    logger.info(f"物品数量: {clients.itemnum}")

    # 构建服务器
    server = Server(config, clients, logger)

    # 开始训练
    server.train()