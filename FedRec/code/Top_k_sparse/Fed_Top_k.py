import numpy as np
import torch
import time
from ..dataset import ClientsDataset, evaluate, evaluate_valid
from ..metric import NDCG_binary_at_k_batch, AUC_at_k_batch, HR_at_k_batch
from ..untils import getModel, FedDecorrLoss, add_noise


class Clients:
    """
    Federated Learning Client Class - Supports Top-k Gradient Selection
    Main Improvements:
    1. Supports different device types (Small/Medium/Large) and their corresponding embedding dimensions
    2. Supports local gradient Top-k selection, only transmitting the k most important gradient components
    3. Supports parameter equivalence maintenance
    """

    def __init__(self, config, logger):
        self.neg_num = config['neg_num']
        self.logger = logger
        self.config = config

        # Heterogeneous device dimension configuration
        self.dim_s = config['dim_s']  # Small device embedding dimension
        self.dim_m = config['dim_m']  # Medium device embedding dimension
        self.dim_l = config['dim_l']  # Large device embedding dimension

        # Top-k configuration
        self.top_k_ratio = config['top_k_ratio']  # Top-k ratio, default keeps 10% of gradients
        self.top_k_method = config['top_k_method']  # 'global' or 'layer-wise'
        self.min_k = config['min_k']  # Minimum number of gradients to keep for each layer

        # Top-k statistics
        self.top_k_stats = {
            'total_gradients': 0,
            'selected_gradients': 0,
            'compression_ratio': 0.0
        }

        # Data path
        self.data_path = config['datapath'] + config['dataset'] + '/' + config['train_data']
        self.maxlen = config['max_seq_len']
        self.batch_size = config['batch_size']

        # Load client dataset
        self.clients_data = ClientsDataset(self.data_path, maxlen=self.maxlen)
        self.dataset = self.clients_data.get_dataset()
        self.user_train, self.user_valid, self.user_test, self.usernum, self.itemnum = self.dataset

        # Device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Record client device types
        self.device_types = {}  # {uid: 's', 'm', 'l'}

        self._init_device_types()

        # Create models of three sizes
        config_s = self.config.copy()
        config_s['hidden_size'] = self.dim_s
        self.model_s = getModel(config_s, self.clients_data.get_maxid()).to(self.device)

        config_m = self.config.copy()
        config_m['hidden_size'] = self.dim_m
        self.model_m = getModel(config_m, self.clients_data.get_maxid()).to(self.device)

        config_l = self.config.copy()
        config_l['hidden_size'] = self.dim_l
        self.model_l = getModel(config_l, self.clients_data.get_maxid()).to(self.device)

        # The Server class needs a unified model reference, we let it reference the largest model
        self.model = self.model_l

        self.logger.info(f"Heterogeneous device config: dim_s={self.dim_s}, dim_m={self.dim_m}, dim_l={self.dim_l}")
        self.logger.info(f"Top-k config: ratio={self.top_k_ratio}, method={self.top_k_method}, min_k={self.min_k}")
        self.logger.info(f"Device type distribution: "
                         f"Small:{sum(1 for t in self.device_types.values() if t == 's')}, "
                         f"Medium:{sum(1 for t in self.device_types.values() if t == 'm')}, "
                         f"Large:{sum(1 for t in self.device_types.values() if t == 'l')}")

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
        return {
            's': self.dim_s,
            'm': self.dim_m,
            'l': self.dim_l
        }[dev_type]

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

    def apply_top_k_selection(self, gradients_dict):
        """
        Apply Top-k selection to the gradient dictionary
        Returns the compressed gradient dictionary and index information
        """
        compressed_gradients = {}
        gradient_indices = {}

        if self.top_k_method == 'global':
            # --- Corrected global Top-k logic ---
            # 1. Collect and flatten all gradients, while recording layer information
            all_gradients_flat = []
            layer_info = []  # Store layer name and original shape
            for name, grad in gradients_dict.items():
                if grad is not None:
                    all_gradients_flat.append(grad.flatten())
                    layer_info.append({'name': name, 'shape': grad.shape})

            if not all_gradients_flat:
                return compressed_gradients, gradient_indices

            # 2. Combine into a global gradient vector
            combined_gradients = torch.cat(all_gradients_flat)

            # 3. Calculate Top-k and get global indices
            k = max(self.min_k, int(len(combined_gradients) * self.top_k_ratio))
            
            # Handle the edge case of k=0
            if k == 0:
                for info in layer_info:
                    compressed_gradients[info['name']] = torch.zeros(info['shape'], device=self.device)
                    gradient_indices[info['name']] = torch.tensor([], dtype=torch.long, device=self.device)
                return compressed_gradients, gradient_indices

            _, top_indices_global = torch.topk(torch.abs(combined_gradients), k)

            # 4. Create a sparse global gradient vector, keeping only Top-k values
            sparse_combined_gradients = torch.zeros_like(combined_gradients)
            sparse_combined_gradients[top_indices_global] = combined_gradients[top_indices_global]

            # 5. Slice the sparse vector and rebuild it to the original shape of each layer
            start_idx = 0
            for i, info in enumerate(layer_info):
                grad_size = all_gradients_flat[i].numel()
                
                # Extract the current layer's part from the sparse global vector
                layer_grad_sparse_flat = sparse_combined_gradients[start_idx : start_idx + grad_size]
                
                # Reshape to the original shape
                compressed_gradients[info['name']] = layer_grad_sparse_flat.reshape(info['shape'])
                
                # Find the local indices of non-zero elements and flatten
                gradient_indices[info['name']] = torch.nonzero(layer_grad_sparse_flat, as_tuple=False).flatten()

                start_idx += grad_size

        else:  # layer-wise (selecting per layer)
            for name, grad in gradients_dict.items():
                if grad is not None:
                    grad_flat = grad.flatten()
                    num_elements = grad_flat.numel()

                    if num_elements == 0:
                        k = 0
                    else:
                        k = max(self.min_k, int(num_elements * self.top_k_ratio))
                        k = min(k, num_elements) # Ensure k does not exceed total number of elements

                    if k == 0:
                        compressed_grad = torch.zeros_like(grad_flat)
                        top_indices = torch.tensor([], dtype=torch.long, device=self.device)
                    else:
                        _, top_indices = torch.topk(torch.abs(grad_flat), k)
                        compressed_grad = torch.zeros_like(grad_flat)
                        compressed_grad[top_indices] = grad_flat[top_indices]

                    compressed_gradients[name] = compressed_grad.reshape(grad.shape)
                    gradient_indices[name] = top_indices

        return compressed_gradients, gradient_indices

    def train(self, uids, model_param_state_dict, epoch=0):
        clients_grads = {}
        clients_losses = {}
        clients_indices = {}  # Store gradient index information for each client

        for uid in uids:
            uid = uid.item()
            dev_type = self.device_types.get(uid, 'l')

            if dev_type == 's':
                client_model = self.model_s
            elif dev_type == 'm':
                client_model = self.model_m
            else:  # 'l'
                client_model = self.model_l

            client_model.train()
            alpha = self.config['decor_alpha']

            self.load_server_model_params(client_model, model_param_state_dict)

            optimizer = torch.optim.Adam(client_model.parameters(), lr=self.config['lr'],
                                         betas=(0.9, 0.98), weight_decay=self.config['l2_reg'])

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

            seq_out = client_model(input_seq, input_len)
            padding_mask = (torch.not_equal(input_seq, 0)).float().unsqueeze(-1).to(self.device)

            feddecorr = FedDecorrLoss ()

            # Calculate loss
            # Select loss function based on device type
            if dev_type == 's':
                # Small device: only calculate small model loss (using full user features)
                loss = self.model_s.loss_function(seq_out, padding_mask, target_seq, neg_seq, input_len)
            elif dev_type == 'm':
                # Medium device:
                # 1. Calculate partial loss of small model (using small feature subset Ns)
                loss_s = self.model_s.loss_function (seq_out, padding_mask, target_seq, neg_seq, input_len)
                # 2. Calculate overall loss of medium model (using full user features)
                loss_m = self.model_m.loss_function (seq_out, padding_mask, target_seq, neg_seq, input_len)
                # Add regularization loss (only for high-dimensional parts)
                loss_reg = feddecorr(client_model.item_embedding.weight[:, self.dim_s:])
                loss = loss_s + loss_m + alpha * loss_reg
            else:  # 'l'
                # Large device：
                # 1. Calculate partial loss of small model (using small feature subset Ns)
                loss_s = self.model_s.loss_function (seq_out, padding_mask, target_seq, neg_seq, input_len)
                # 2. Calculate overall loss of medium model (using full user features)
                loss_m = self.model_m.loss_function (seq_out, padding_mask, target_seq, neg_seq, input_len)
                # 3. Calculate overall loss of large model (using full user features)
                loss_l = self.model_l.loss_function (seq_out, padding_mask, target_seq, neg_seq, input_len)
                # Add regularization loss (only for high-dimensional parts)
                loss_reg = feddecorr (client_model.item_embedding.weight [:, self.dim_s:])
                loss =loss_l + loss_s + loss_m + alpha * loss_reg

            clients_losses[uid] = loss.item()

            optimizer.zero_grad()
            loss.backward()

            # Collect all gradients
            gradients = {name: param.grad.clone() for name, param in client_model.named_parameters() if
                         param.grad is not None}

            if self.config['LDP_lambda'] > 0:
                gradients = add_noise(gradients, self.config['LDP_lambda'])

            # Apply Top-k selection
            compressed_gradients, gradient_indices = self.apply_top_k_selection(gradients)

            # Update model parameters to compressed gradients
            for name, param in client_model.named_parameters():
                if name in compressed_gradients:
                    param.grad = compressed_gradients[name]

            optimizer.step()

            # Store compressed gradients and index information
            clients_grads[uid] = compressed_gradients
            clients_indices[uid] = gradient_indices

        # Update Top-k statistics
        total_gradients = sum(len(grad.flatten()) for grads in clients_grads.values() for grad in grads.values())
        selected_gradients = sum(
            len(indices) for indices_list in clients_indices.values() for indices in indices_list.values())

        self.top_k_stats['total_gradients'] = total_gradients
        self.top_k_stats['selected_gradients'] = selected_gradients
        if total_gradients > 0:
            self.top_k_stats['compression_ratio'] = selected_gradients / total_gradients

        return clients_grads, clients_losses, clients_indices


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

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'],
                                          betas=(0.9, 0.98), weight_decay=config['l2_reg'])

        logger.info("Server initialization completed")

        # --- Communication cost calculation module initialization ---
        self._init_comm_cost_calculation()

        # Add cumulative communication cost variables
        self.cumulative_uplink_cost = 0.0
        self.cumulative_downlink_cost = 0.0

    def _init_comm_cost_calculation(self):
        """Pre-calculate and store model sizes for cost statistics"""
        self.logger.info("Initializing communication cost calculator...")
        # Pre-calculate total parameter size of three model sizes (MB)
        self.size_p_s = self._calculate_dense_params_size(self.clients.model_s.state_dict())
        self.size_p_m = self._calculate_dense_params_size(self.clients.model_m.state_dict())
        self.size_p_l = self._calculate_dense_params_size(self.clients.model_l.state_dict())

        self.logger.info(f"Precomputed model size (MB): "
                         f"S={self.size_p_s:.4f}, "
                         f"M={self.size_p_m:.4f}, "
                         f"L={self.size_p_l:.4f}")

    def aggregate_gradients(self, clients_grads, clients_indices):
        clients_num = len(clients_grads)
        if clients_num == 0:
            self.logger.info("No client gradients received")
            return

        aggregated_gradients = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if
                                param.requires_grad}

        for uid, grads_dict in clients_grads.items():
            indices_dict = clients_indices.get(uid, {})

            for name, grad in grads_dict.items():
                if grad is None:
                    continue

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

    def _calculate_dense_params_size(self, params_dict):
        """Calculate the total size of dense parameter dictionary (MB)"""
        total_size = 0
        for param in params_dict.values():
            if isinstance(param, torch.Tensor):
                total_size += param.nelement() * param.element_size()
        return total_size / (1024 * 1024)  # Return in MB

    def _calculate_sparse_params_size(self, indices_dict):
        """Calculate the total size of sparse gradients (values + indices) (MB)"""
        total_elements = 0
        for indices in indices_dict.values():
            total_elements += len(indices)
        # Assuming transmission of each value (float32=4 bytes) and its corresponding index (int64=8 bytes)
        # In actual transmission, indices may be compressed, but this serves as an upper bound estimate
        total_size = total_elements * 4
        return total_size / (1024 * 1024)  # Return in MB

    def train(self):
        user_set = self.clients.clients_data.get_user_set()
        uid_seq = []
        for i in range(0, len(user_set), self.batch_size):
            batch_uids = user_set[i:i + self.batch_size]
            uid_seq.append(torch.tensor(batch_uids))

        best_val_ndcg, best_val_hr = 0.0, 0.0
        best_test_ndcg, best_test_hr = 0.0, 0.0

        # Initialize model parameters
        for name, param in self.model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass  # Ignore layers that fail to initialize

        # Set padding position of embedding layers to 0
        if hasattr(self.model, 'position_embedding'):
            self.model.position_embedding.weight.data[0, :] = 0
        if hasattr(self.model, 'item_embedding'):
            self.model.item_embedding.weight.data[0, :] = 0

        T = 0.0
        t0 = time.time()

        no_improve_count = 0
        early_stop_triggered = False  # Add early stop flag
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.model.train()

            # --- Communication cost calculation initialization ---
            total_uplink_cost = 0
            total_downlink_cost = 0
            # -------------------------

            batch_count = 0
            epoch_losses = []

            # Get the global model parameters to be sent
            model_state_to_send = self.model.state_dict()

            for uids in uid_seq:
                self.optimizer.zero_grad()

                # --- Calculate downlink cost ---
                # Calculate downlink cost based on the actual model size of each client
                downlink_cost_batch = 0
                for uid in uids:
                    dev_type = self.clients.device_types.get(uid.item(), 'l')
                    if dev_type == 's':
                        downlink_cost_batch += self.size_p_s
                    elif dev_type == 'm':
                        downlink_cost_batch += self.size_p_m
                    else:  # 'l'
                        downlink_cost_batch += self.size_p_l
                total_downlink_cost += downlink_cost_batch
                # -------------------------

                clients_grads, clients_losses, clients_indices = self.clients.train(uids, model_state_to_send, epoch)

                # --- Calculate uplink cost ---
                # Clients upload sparse gradients (values + indices)
                for client_uid, indices_dict in clients_indices.items():
                    total_uplink_cost += self._calculate_sparse_params_size(indices_dict)
                # -------------------------

                batch_losses = list(clients_losses.values())
                epoch_losses.extend(batch_losses)

                self.aggregate_gradients(clients_grads, clients_indices)

                self.optimizer.step()

                batch_count += 1

            # --- Update cumulative communication cost ---
            self.cumulative_uplink_cost += total_uplink_cost
            self.cumulative_downlink_cost += total_downlink_cost

            # --- Log communication cost (current epoch and cumulative total) ---

            self.logger.info(f"Cumulative communication cost: "
                             f"Uplink = {self.cumulative_uplink_cost:.4f} MB, "
                             f"Downlink = {self.cumulative_downlink_cost:.4f} MB, "
                             f"Total = {self.cumulative_uplink_cost + self.cumulative_downlink_cost:.4f} MB")
            # --------------------

            eval_freq = self.eval_freq
            should_evaluate = (epoch + 1) % eval_freq == 0 or epoch == 0 or epoch == self.epochs - 1
            if should_evaluate:
                self.model.eval()
                # t1 is the training time of this round, T is the total time
                t1 = time.time() - t0
                T += t1
                print('Evaluating', end='')

                t_valid = evaluate_valid(self.model, self.dataset, self.maxlen, self.clients.neg_num, self.eval_k,
                                         self.config['full_eval'], self.device)
                self.logger.info(
                    f"Evaluation result - Epoch {epoch + 1}: NDCG@{self.eval_k}={t_valid[0]:.4f}, HR@{self.eval_k}={t_valid[1]:.4f}")

                # Output Top-k statistics
                stats = self.clients.top_k_stats
                if stats['total_gradients'] > 0:
                    compression_ratio = stats['compression_ratio']
                    self.logger.info(f"Top-k compression stats: total_gradients={stats['total_gradients']}, "
                                     f"selected_gradients={stats['selected_gradients']}, "
                                     f"compression_ratio={compression_ratio:.2%}")

                # Check for abnormal evaluation results
                if t_valid[0] >= 0.99 or t_valid[1] >= 0.99 or np.isnan(t_valid[0]) or np.isnan(t_valid[1]):
                    self.logger.info(
                        f"Abnormal evaluation result detected: NDCG@{self.eval_k}={t_valid[0]:.4f}, HR@{self.eval_k}={t_valid[1]:.4f}")

                # Early stopping check - Check if NDCG has improved
                if self.early_stop_enabled:
                    if t_valid[0] > best_val_ndcg:
                        # NDCG improved, reset counter
                        no_improve_count = 0
                        best_val_ndcg = t_valid[0]
                    else:
                        # NDCG did not improve, increase counter
                        no_improve_count += 1

                    # If no improvement for several consecutive rounds, trigger early stopping
                    if no_improve_count >= self.early_stop:
                        self.logger.info(f"Early stopping triggered! NDCG did not improve for {self.early_stop} rounds.")
                        early_stop_triggered = True

                # Test set evaluation (optional) - also use heterogeneous evaluation
                if not self.skip_test_eval:
                    # Add test set heterogeneous evaluation here, temporarily using traditional method
                    t_test = evaluate(self.model, self.dataset, self.maxlen, self.clients.neg_num, self.eval_k,
                                      self.config['full_eval'], self.device)

                    # Check test set evaluation results for abnormalities
                    if t_test[0] >= 0.99 or t_test[1] >= 0.99 or np.isnan(t_test[0]) or np.isnan(t_test[1]):
                        self.logger.info(
                            f"Abnormal test result detected: NDCG@{self.eval_k}={t_test[0]:.4f}, HR@{self.eval_k}={t_test[1]:.4f}")

                    # Log results
                    self.logger.info(
                        'epoch:%d, time: %f(s), valid (NDCG@%d: %.4f, HR@%d: %.4f), test (NDCG@%d: %.4f, HR@%d: %.4f) all_time: %f(s)'
                        % (epoch + 1, t1, self.eval_k, t_valid[0], self.eval_k, t_valid[1], self.eval_k, t_test[0],
                           self.eval_k, t_test[1], T))
                else:
                    # Skip test set evaluation, set default values
                    t_test = (0.0, 0.0)
                    # Log results
                    self.logger.info(
                        'epoch:%d, time: %f(s), valid (NDCG@%d: %.4f, HR@%d: %.4f), test: SKIPPED, all_time: %f(s)'
                        % (epoch + 1, t1, self.eval_k, t_valid[0], self.eval_k, t_valid[1], T))

                # Update best results
                if not self.skip_test_eval:
                    # Include test set evaluation
                    if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr or t_test[0] > best_test_ndcg or \
                            t_test[1] > best_test_hr:
                        best_val_ndcg = max(t_valid[0], best_val_ndcg)
                        best_val_hr = max(t_valid[1], best_val_hr)
                        best_test_ndcg = max(t_test[0], best_test_ndcg)
                        best_test_hr = max(t_test[1], best_test_hr)
                        self.logger.info(
                            f"New best performance: valid NDCG@{self.eval_k}={best_val_ndcg:.4f}, test NDCG@{self.eval_k}={best_test_ndcg:.4f}")

                else:
                    # Skip test set evaluation
                    if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr:
                        best_val_ndcg = max(t_valid[0], best_val_ndcg)
                        best_val_hr = max(t_valid[1], best_val_hr)
                        self.logger.info(
                            f"New best performance: valid NDCG@{self.eval_k}={best_val_ndcg:.4f}, valid HR@{self.eval_k}={best_val_hr:.4f}")

                t0 = time.time()
                self.model.train()

                # Check early stop flag after evaluation
                if early_stop_triggered:
                    break

        # Record best results
        if not self.skip_test_eval:
            self.logger.info(
                '[Federated Training] Best result: valid NDCG@{}={:.4f}, HR@{}={:.4f}, test NDCG@{}={:.4f}, HR@{}={:.4f}'.format(
                    self.eval_k, best_val_ndcg, self.eval_k, best_val_hr, self.eval_k, best_test_ndcg, self.eval_k,
                    best_test_hr))
        else:
            self.logger.info('[Federated Training] Best result: valid NDCG@{}={:.4f}, HR@{}={:.4f} (Test set evaluation skipped)'.format(
                self.eval_k, best_val_ndcg, self.eval_k, best_val_hr))