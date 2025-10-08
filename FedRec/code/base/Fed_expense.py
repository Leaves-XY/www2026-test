import numpy as np
import torch
import time
import os
import psutil
from FedRec.code.dataset import ClientsDataset, evaluate, evaluate_valid
from FedRec.code.metric import NDCG_binary_at_k_batch, AUC_at_k_batch, HR_at_k_batch
from thop import profile
from FedRec.code.untils import getModel, add_noise

def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)

class Clients:
    def __init__(self, config, logger):
        self.neg_num = config['neg_num']
        self.logger = logger
        self.config = config
        self.dim_s = config['dim_s']
        self.dim_m = config['dim_m']
        self.dim_l = config['dim_l']
        self.data_path = config['datapath'] + config['dataset'] + '/' + config['train_data']
        self.maxlen = config['max_seq_len']
        self.batch_size = config['batch_size']
        self.LDP_lambda = config['LDP_lambda']
        self.clients_data = ClientsDataset(self.data_path, maxlen=self.maxlen)
        self.dataset = self.clients_data.get_dataset()
        self.user_train, self.user_valid, self.user_test, self.usernum, self.itemnum = self.dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_types = {}
        self._init_device_types()
        config_s = self.config.copy()
        config_s['hidden_size'] = self.dim_s
        self.model_s = getModel(config_s, self.clients_data.get_maxid()).to(self.device)
        config_m = self.config.copy()
        config_m['hidden_size'] = self.dim_m
        self.model_m = getModel(config_m, self.clients_data.get_maxid()).to(self.device)
        config_l = self.config.copy()
        config_l['hidden_size'] = self.dim_l
        self.model_l = getModel(config_l, self.clients_data.get_maxid()).to(self.device)
        self.model = self.model_l
        self.logger.info(f"Heterogeneous device config: dim_s={self.dim_s}, dim_m={self.dim_m}, dim_l={self.dim_l}")
        self.logger.info(f"Device type distribution: "
                         f"Small:{sum(1 for t in self.device_types.values() if t == 's')}, "
                         f"Medium:{sum(1 for t in self.device_types.values() if t == 'm')}, "
                         f"Large:{sum(1 for t in self.device_types.values() if t == 'l')}")
        self._log_model_flops()

    def _profile_model_flops(self, model, max_len):
        dummy_input_seq = torch.randint(1, self.itemnum, (1, max_len), device=self.device)
        dummy_input_len = torch.tensor([max_len], device=self.device)
        inputs = (dummy_input_seq, dummy_input_len)
        macs, params = profile(model, inputs=inputs, verbose=False)
        mflops = macs * 2 / 1e6
        return mflops

    def _log_model_flops(self):
        self.logger.info("Start calculating model FLOPs (single forward pass)...")
        try:
            self.flops_s = self._profile_model_flops(self.model_s, self.maxlen)
            self.flops_m = self._profile_model_flops(self.model_m, self.maxlen)
            self.flops_l = self._profile_model_flops(self.model_l, self.maxlen)
            self.logger.info(f"Computation cost (MFLOPs): S={self.flops_s:.3f}, M={self.flops_m:.3f}, L={self.flops_l:.3f}")
        except Exception as e:
            self.logger.error(f"Error calculating FLOPs: {e}")

    def _init_device_types(self):
        user_set = self.clients_data.get_user_set()
        total_users = len(user_set)
        device_split = self.config['device_split']
        num_s = int(total_users * device_split[0])
        num_m = int(total_users * (device_split[0] + device_split[1]))
        if self.config['assign_by_interactions']:
            method = "Assigned by interaction order"
            user_interactions = {uid: len(self.user_train[uid]) for uid in user_set}
            sorted_users = sorted(user_set, key=lambda uid: user_interactions[uid])
        else:
            method = "Random assignment"
            sorted_users = list(user_set)
            np.random.shuffle(sorted_users)
        for i, uid in enumerate(sorted_users):
            if i < num_s:
                self.device_types[uid] = 's'
            elif i < num_m:
                self.device_types[uid] = 'm'
            else:
                self.device_types[uid] = 'l'
        count_s = sum(1 for t in self.device_types.values() if t == 's')
        count_m = sum(1 for t in self.device_types.values() if t == 'm')
        count_l = sum(1 for t in self.device_types.values() if t == 'l')
        self.logger.info(
            f"Device type assignment completed ({method}): "
            f"Small={count_s}({count_s / total_users:.1%}), "
            f"Medium={count_m}({count_m / total_users:.1%}), "
            f"Large={count_l}({count_l / total_users:.1%})"
        )

    def get_dim_for_client(self, uid):
        dev_type = self.device_types.get(uid, 'l')
        return {'s': self.dim_s, 'm': self.dim_m, 'l': self.dim_l}[dev_type]

    def load_server_model_params(self, client_model, server_state_dict):
        client_state_dict = client_model.state_dict()
        for name, server_param in server_state_dict.items():
            if name in client_state_dict:
                client_param = client_state_dict[name]
                if server_param.shape != client_param.shape:
                    slicing_indices = [slice(0, dim) for dim in client_param.shape]
                    client_param.copy_(server_param[tuple(slicing_indices)])
                else:
                    client_param.copy_(server_param)

    def train(self, uids, model_param_state_dict, epoch=0):
        clients_grads = {}
        clients_losses = {}
        clients_costs = {}
        for uid in uids:
            uid = uid.item()
            dev_type = self.device_types.get(uid, 'l')
            start_time = time.time()
            if dev_type == 's':
                client_model = self.model_s
                forward_flops = self.flops_s
            elif dev_type == 'm':
                client_model = self.model_m
                forward_flops = self.flops_m
            else:
                client_model = self.model_l
                forward_flops = self.flops_l
            client_model.train()
            self.load_server_model_params(client_model, model_param_state_dict)
            optimizer = torch.optim.Adam(client_model.parameters(), lr=self.config['lr'], betas=(0.9, 0.98), weight_decay=self.config['l2_reg'])
            input_seq = self.clients_data.train_seq[uid]
            target_seq = self.clients_data.valid_seq[uid]
            input_len = self.clients_data.seq_len[uid]
            cand = np.setdiff1d(self.clients_data.item_set, self.clients_data.seq[uid])
            prob = self.clients_data.item_prob[cand]
            prob = prob / prob.sum()
            neg_seq = np.random.choice(cand, (input_len, 100), p=prob)
            neg_seq = np.pad(neg_seq, ((input_seq.shape[0] - input_len, 0), (0, 0)))
            input_seq = torch.from_numpy(input_seq).unsqueeze(0).to(self.device)
            target_seq = torch.from_numpy(target_seq).unsqueeze(0).to(self.device)
            neg_seq = torch.from_numpy(neg_seq).unsqueeze(0).to(self.device)
            input_len = torch.tensor(input_len).unsqueeze(0).to(self.device)
            max_seq_length = client_model.max_seq_length
            if input_seq.size(1) > max_seq_length:
                input_seq = input_seq[:, -max_seq_length:]
                target_seq = target_seq[:, -max_seq_length:]
                neg_seq = neg_seq[:, -max_seq_length:, :]
                input_len = torch.clamp(input_len, max=max_seq_length)
            optimizer.zero_grad()
            seq_out = client_model(input_seq, input_len)
            padding_mask = (torch.not_equal(input_seq, 0)).float().unsqueeze(-1).to(self.device)
            loss = client_model.loss_function(seq_out, padding_mask, target_seq, neg_seq, input_len)
            clients_losses[uid] = loss.item()
            loss.backward()
            peak_memory_mb = get_process_memory_mb()
            optimizer.step()
            end_time = time.time()
            training_time = end_time - start_time
            total_mflops = forward_flops * 3.0
            clients_costs[uid] = {'mflops': total_mflops, 'time': training_time, 'peak_mem_mb': peak_memory_mb}
            gradients = {name: param.grad.clone() for name, param in client_model.named_parameters() if param.grad is not None}
            if self.LDP_lambda > 0:
                gradients = add_noise(gradients, self.LDP_lambda)
            clients_grads[uid] = gradients
        return clients_grads, clients_losses, clients_costs

class Server:
    def __init__(self, config, clients, logger):
        self.clients = clients
        self.config = config
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.early_stop = config['early_stop']
        self.maxlen = config['max_seq_len']
        self.skip_test_eval = config['skip_test_eval']
        self.eval_freq = config['eval_freq']
        self.early_stop_enabled = config['early_stop_enabled']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger
        self.dataset = self.clients.dataset
        self.dim_s = config['dim_s']
        self.dim_m = config['dim_m']
        self.dim_l = config['dim_l']
        self.eval_k = config['eval_k']
        config['hidden_size'] = self.dim_l
        self.model = getModel(config, self.clients.clients_data.get_maxid())
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'], betas=(0.9, 0.98), weight_decay=config['l2_reg'])
        self._init_comm_cost_calculation()
        logger.info("Server initialization completed")

    def _get_param_size(self, model_or_params):
        if isinstance(model_or_params, torch.nn.Module):
            params = model_or_params.parameters()
        elif isinstance(model_or_params, dict):
            params = model_or_params.values()
        else:
            params = model_or_params
        size = 0
        for p in params:
            if isinstance(p, torch.Tensor):
                size += p.nelement() * p.element_size()
        return size

    def _get_layer_size(self, model, layer_name):
        size = 0
        for name, param in model.named_parameters():
            if layer_name in name:
                size += param.nelement() * param.element_size()
        return size

    def _init_comm_cost_calculation(self):
        self.comm_costs = {'downlink': 0, 'uplink': 0}
        self.logger.info("Initializing communication cost calculator...")
        self.size_p_s = self._get_param_size(self.clients.model_s)
        self.size_p_m = self._get_param_size(self.clients.model_m)
        self.size_p_l = self._get_param_size(self.clients.model_l)
        self.size_e_s = self._get_layer_size(self.clients.model_s, 'item_embedding.weight')
        self.size_e_m = self._get_layer_size(self.clients.model_m, 'item_embedding.weight')
        self.size_e_l = self._get_layer_size(self.clients.model_l, 'item_embedding.weight')
        self.logger.info(f"Model total size (MB): S={self.size_p_s / 1e6:.3f}, M={self.size_p_m / 1e6:.3f}, L={self.size_p_l / 1e6:.3f}")
        self.logger.info(f"Embedding layer size (MB): S={self.size_e_s / 1e6:.3f}, M={self.size_e_m / 1e6:.3f}, L={self.size_e_l / 1e6:.3f}")

    def aggregate_gradients(self, clients_grads):
        clients_num = len(clients_grads)
        if clients_num == 0:
            self.logger.info("No client gradients received")
            return
        aggregated_gradients = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}
        for uid, grads_dict in clients_grads.items():
            for name, grad in grads_dict.items():
                if grad is None: continue
                target_shape = aggregated_gradients[name].shape
                if grad.shape != target_shape:
                    padded_grad = torch.zeros(target_shape, device=self.device)
                    slicing_indices = tuple(slice(0, dim) for dim in grad.shape)
                    padded_grad[slicing_indices] = grad
                    aggregated_gradients[name] += padded_grad
                else:
                    aggregated_gradients[name] += grad
        for name in aggregated_gradients:
            aggregated_gradients[name] /= clients_num
        for name, param in self.model.named_parameters():
            if name in aggregated_gradients:
                param.grad = aggregated_gradients[name]
            else:
                param.grad = None

    def train(self):
        user_set = self.clients.clients_data.get_user_set()
        uid_seq = []
        for i in range(0, len(user_set), self.batch_size):
            uid_seq.append(torch.tensor(user_set[i:i + self.batch_size]))
        best_val_ndcg, best_val_hr, best_test_ndcg, best_test_hr = 0.0, 0.0, 0.0, 0.0
        no_improve_count = 0
        for name, param in self.model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass
        if hasattr(self.model, 'position_embedding'): self.model.position_embedding.weight.data[0, :] = 0
        if hasattr(self.model, 'item_embedding'): self.model.item_embedding.weight.data[0, :] = 0
        T = 0.0
        t0 = time.time()
        early_stop_triggered = False
        for epoch in range(self.epochs):
            self.model.train()
            epoch_losses = []
            epoch_total_mflops, epoch_total_time = 0.0, 0.0
            epoch_max_peak_mem = 0.0
            for uids in uid_seq:
                downlink_cost_batch = 0
                for uid in uids:
                    dev_type = self.clients.device_types.get(uid.item(), 'l')
                    if dev_type == 's':
                        downlink_cost_batch += self.size_p_s
                    elif dev_type == 'm':
                        downlink_cost_batch += self.size_p_m
                    else:
                        downlink_cost_batch += self.size_p_l
                self.comm_costs['downlink'] += downlink_cost_batch
                self.optimizer.zero_grad()
                clients_grads, clients_losses, clients_costs = self.clients.train(uids, self.model.state_dict(), epoch)
                uplink_cost_batch = 0
                for uid in uids:
                    dev_type = self.clients.device_types.get(uid.item(), 'l')
                    if dev_type == 's':
                        uplink_cost_batch += self.size_p_s
                    elif dev_type == 'm':
                        uplink_cost_batch += self.size_p_m
                    else:
                        uplink_cost_batch += self.size_p_l
                self.comm_costs['uplink'] += uplink_cost_batch
                epoch_losses.extend(list(clients_losses.values()))
                for cost in clients_costs.values():
                    epoch_total_mflops += cost['mflops']
                    epoch_total_time += cost['time']
                    if cost['peak_mem_mb'] > epoch_max_peak_mem:
                        epoch_max_peak_mem = cost['peak_mem_mb']
                self.aggregate_gradients(clients_grads)
                self.optimizer.step()
            should_evaluate = (epoch + 1) % self.eval_freq == 0 or epoch == 0 or epoch == self.epochs - 1
            if should_evaluate:
                self.model.eval()
                t1 = time.time() - t0
                T += t1
                print('Evaluating', end='')
                t_valid = evaluate_valid(self.model, self.dataset, self.maxlen, self.clients.neg_num, self.eval_k, self.config['full_eval'], self.device)
                down_mb = self.comm_costs['downlink'] / (1024 ** 2)
                up_mb = self.comm_costs['uplink'] / (1024 ** 2)
                total_mb = down_mb + up_mb
                self.logger.info(f"---------- EPOCH {epoch + 1} COST SUMMARY ----------")
                self.logger.info(f"  COMMUNICATION: Downlink={down_mb:.3f}MB, Uplink={up_mb:.3f}MB, Total={total_mb:.3f}MB")
                self.logger.info(f"  COMPUTATION: Total Estimated mflops={epoch_total_mflops:.3f}, Total Client-Side Time={epoch_total_time:.3f}s")
                self.logger.info(f"  MEMORY: Max Peak Memory Usage per Client={epoch_max_peak_mem:.3f}MB")
                self.logger.info(f"-------------------------------------------")
                self.logger.info(f"EVALUATION - Epoch {epoch + 1}: NDCG@{self.eval_k}={t_valid[0]:.4f}, HR@{self.eval_k}={t_valid[1]:.4f}")
                if t_valid[0] >= 0.99 or t_valid[1] >= 0.99 or np.isnan(t_valid[0]) or np.isnan(t_valid[1]):
                    self.logger.info(f"Abnormal evaluation result detected: NDCG@{self.eval_k}={t_valid[0]:.4f}, HR@{self.eval_k}={t_valid[1]:.4f}")
                if self.early_stop_enabled:
                    if t_valid[0] > best_val_ndcg:
                        no_improve_count, best_val_ndcg = 0, t_valid[0]
                    else:
                        no_improve_count += 1
                    if no_improve_count >= self.early_stop:
                        self.logger.info(f"Early stopping triggered! NDCG did not improve for {self.early_stop} rounds.")
                        early_stop_triggered = True
                if not self.skip_test_eval:
                    t_test = evaluate(self.model, self.dataset, self.maxlen, self.clients.neg_num, self.eval_k, self.config['full_eval'], self.device)
                    if t_test[0] >= 1.0 or t_test[1] > 1.0 or np.isnan(t_test[0]) or np.isnan(t_test[1]):
                        self.logger.info(f"Abnormal test result detected: NDCG@{self.eval_k}={t_test[0]:.4f}, HR@{self.eval_k}={t_test[1]:.4f}")
                    self.logger.info('epoch:%d, time: %f(s), valid (NDCG@%d: %.4f, HR@%d: %.4f), test (NDCG@%d: %.4f, HR@%d: %.4f) all_time: %f(s)' % (epoch + 1, t1, self.eval_k, t_valid[0], self.eval_k, t_valid[1], self.eval_k, t_test[0], self.eval_k, t_test[1], T))
                else:
                    t_test = (0.0, 0.0)
                    self.logger.info('epoch:%d, time: %f(s), valid (NDCG@%d: %.4f, HR@%d: %.4f), test: SKIPPED, all_time: %f(s)' % (epoch + 1, t1, self.eval_k, t_valid[0], self.eval_k, t_valid[1], T))
                if not self.skip_test_eval:
                    if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr or t_test[0] > best_test_ndcg or t_test[1] > best_test_hr:
                        best_val_ndcg, best_val_hr = max(t_valid[0], best_val_ndcg), max(t_valid[1], best_val_hr)
                        best_test_ndcg, best_test_hr = max(t_test[0], best_test_ndcg), max(t_test[1], best_test_hr)
                        self.logger.info(f"New best performance: valid NDCG@{self.eval_k}={best_val_ndcg:.4f}, test NDCG@{self.eval_k}={best_test_ndcg:.4f}")
                else:
                    if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr:
                        best_val_ndcg, best_val_hr = max(t_valid[0], best_val_ndcg), max(t_valid[1], best_val_hr)
                        self.logger.info(f"New best performance: valid NDCG@{self.eval_k}={best_val_ndcg:.4f}, valid HR@{self.eval_k}={best_val_hr:.4f}")
                t0 = time.time()
                self.model.train()
                if early_stop_triggered: break
            if early_stop_triggered: break
        if not self.skip_test_eval:
            self.logger.info('[Federated Training] Best result: valid NDCG@{}={:.4f}, HR@{}={:.4f}, test NDCG@{}={:.4f}, HR@{}={:.4f}'.format(self.eval_k, best_val_ndcg, self.eval_k, best_val_hr, self.eval_k, best_test_ndcg, self.eval_k, best_test_hr))
        else:
            self.logger.info('[Federated Training] Best result: valid NDCG@{}={:.4f}, HR@{}={:.4f} (Test set evaluation skipped)'.format(self.eval_k, best_val_ndcg, self.eval_k, best_val_hr))
