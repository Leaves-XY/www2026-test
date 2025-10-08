import numpy as np
import torch
import time
from FedRec.code.dataset import ClientsDataset, evaluate, evaluate_valid
from FedRec.code.metric import NDCG_binary_at_k_batch, AUC_at_k_batch, HR_at_k_batch
from FedRec.code.untils import getModel, FedDecorrLoss, add_noise



class Clients:
    """
    Federated Learning Client Class - Supporting Heterogeneous Devices
    Major Improvements:
    1. Support for different device types (Small/Medium/Large) and their corresponding embedding dimensions
    2. Support for local gradient extraction submatrices
    3. Support for parameter equivalence maintenance
    """

    def __init__(self, config, logger):
        self.neg_num = config['neg_num']
        self.logger = logger
        self.config = config

        # Heterogeneous device dimension configuration
        self.dim_s = config['dim_s']  # Small device embedding dimension
        self.dim_m = config['dim_m']  # Medium device embedding dimension
        self.dim_l = config['dim_l']  # Large device embedding dimension

        # Data path
        self.data_path = config['datapath'] + config['dataset'] + '/' + config['train_data']
        self.maxlen = config['max_seq_len']
        self.batch_size = config['batch_size']

        # Load client dataset
        self.clients_data = ClientsDataset(self.data_path,  maxlen=self.maxlen)
        self.dataset = self.clients_data.get_dataset()
        self.user_train, self.user_valid, self.user_test, self.usernum, self.itemnum = self.dataset

        # Device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize SASRec model - Use large device dimension as the base
        config['embed_size'] = self.dim_l  # Use the maximum dimension as the base
        self.model = getModel(config, self.clients_data.get_maxid())
        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'],
                                          betas=(0.9, 0.98), weight_decay=config['l2_reg'])


        # Record client device type
        self.device_types = config.get('device_types', {})  # {uid: 's', 'm', 'l'}

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
        self.logger.info(f"Device type distribution: "
                         f"Small:{sum(1 for t in self.device_types.values() if t == 's')}, "
                         f"Medium:{sum(1 for t in self.device_types.values() if t == 'm')}, "
                         f"Large:{sum(1 for t in self.device_types.values() if t == 'l')}")

    def _init_device_types ( self ):
        user_set = self.clients_data.get_user_set ()
        total_users = len (user_set)

        device_split = self.config ['device_split']
        num_s = int (total_users * device_split [0])
        num_m = int (total_users * (device_split [0] + device_split [1]))

        if self.config ['assign_by_interactions']:
            # Assign by interaction order
            method = "Assigned by interaction order"
            user_interactions = {uid: len (self.user_train [uid]) for uid in user_set}
            sorted_users = sorted (user_set, key = lambda uid: user_interactions [uid])
        else:
            # Random assignment
            method = "Random assignment"
            sorted_users = list (user_set)
            np.random.shuffle (sorted_users)

        # Assign device types
        for i, uid in enumerate (sorted_users):
            if i < num_s:
                self.device_types [uid] = 's'
            elif i < num_m:
                self.device_types [uid] = 'm'
            else:
                self.device_types [uid] = 'l'

        # Statistics
        count_s = sum (1 for t in self.device_types.values () if t == 's')
        count_m = sum (1 for t in self.device_types.values () if t == 'm')
        count_l = sum (1 for t in self.device_types.values () if t == 'l')

        self.logger.info (
            f"Device type assignment completed ({method}): "
            f"Small={count_s}({count_s / total_users:.1%}), "
            f"Medium={count_m}({count_m / total_users:.1%}), "
            f"Large={count_l}({count_l / total_users:.1%})"
        )

    def get_dim_for_client(self, uid):
        """Get the embedding dimension for the client"""
        dev_type = self.device_types.get(uid, 'l')
        return {
            's': self.dim_s,
            'm': self.dim_m,
            'l': self.dim_l
        }[dev_type]

    def random_neq(self, l, r, s):
        """
        Generate a random number not in set s, reference to SAS.torch's random_neq function

        Parameters:
        - l: Lower bound
        - r: Upper bound
        - s: Excluded set

        Returns:
        - t: A random number not in set s
        """
        t = np.random.randint(l, r)
        while t in s:
            t = np.random.randint(l, r)
        return t

    def load_server_model_params(self, client_model, server_state_dict):
        """
        Load the global model parameters from the server into the client model.
        Only slice in the "feature dimension / hidden dimension", do not trim the index dimension (vocabulary size, position size).
        """
        client_state_dict = client_model.state_dict()

        for name, server_param in server_state_dict.items():
            if name in client_state_dict:
                client_param = client_state_dict[name]
                # If the server's parameter dimension is larger than the client's, slice it
                if server_param.shape != client_param.shape:
                    # Crop the part that matches the small model from the large parameter matrix
                    slicing_indices = [slice(0, dim) for dim in client_param.shape]
                    client_param.copy_(server_param[tuple(slicing_indices)])
                else:
                    # If the dimensions are the same, copy directly
                    client_param.copy_(server_param)

    def get_local_grads_for_device(self, uid, gradients):
        """
        Extract gradient submatrices according to device type
        Considering the nested relationship:
        - Small device: only extract the first 1/3 dimension
        - Medium device: extract the first 2/3 dimension
        - Large device: extract all dimensions
        """
        dev_type = self.device_types.get(uid, 'l')
        device_grads = {}

        for name, grad in gradients.items():
            if grad is None:
                continue

            # For item embedding, extract the subset of dimensions corresponding to the device
            if 'item_embedding' in name:
                if dev_type == 's':
                    # Small device: only take the first 1/3 dimension
                    device_grads[name] = grad[:, :self.dim_s].clone()
                elif dev_type == 'm':
                    # Medium device: take the first 2/3 dimensions
                    device_grads[name] = grad[:, :self.dim_m].clone()
                else:  # 'l'
                    # Large device: take all dimensions
                    device_grads[name] = grad.clone()
            else:
                device_grads[name] = grad.clone()

        return device_grads


    def sync_models_from_server(self, server):
        """Sync distilled model parameters from the server"""
        # self.logger.info("Syncing distilled model parameters from the server...")

        # Small model sync
        for param, server_param in zip(self.model_s.parameters(), server.model_s.parameters()):
            param.data.copy_(server_param.data)

        # Medium model sync
        for param, server_param in zip(self.model_m.parameters(), server.model_m.parameters()):
            param.data.copy_(server_param.data)

        # Large model sync
        for param, server_param in zip(self.model.parameters(), server.model.parameters()):
            param.data.copy_(server_param.data)

    def train(self, uids, model_param_state_dict, epoch=0):
        """
        Federated learning training method - Dynamic negative sampling version
        Implement standard federated learning process:
        1. Each client trains independently, protecting data privacy
        2. Dynamically generate negative samples to improve learning效果
        3. Separate private gradients (excluding embedding) and non-private gradients (including embedding)
        4. Private gradients are updated locally, non-private gradients are uploaded to the server
        5. Return the non-private gradient dictionary of all clients for server aggregation

        Parameters:
        - uids: List of client IDs to be trained in the current batch
        - model_param_state_dict: Latest global model parameters received from the server
        - epoch: Current training round, used for dynamic sampling strategy adjustment

        Returns:
        - clients_grads: Non-private gradient dictionary of all clients {uid: {param_name: grad}}
        - clients_losses: Loss value dictionary of all clients {uid: loss}
        """
        # Store client gradients and losses
        clients_grads = {}
        clients_losses = {}

        # Each client trains independently
        for uid in uids:
            uid = uid.item()
            dev_type = self.device_types.get(uid, 'l')

            # 1. Select the correct model based on device type
            if dev_type == 's':
                client_model = self.model_s
            elif dev_type == 'm':
                client_model = self.model_m
            else:  # 'l'
                client_model = self.model_l

            client_model.train()
            alpha = self.config['decor_alpha']

            # 2. Load server parameters into the selected client model
            self.load_server_model_params(client_model, model_param_state_dict)

            # 3. Create optimizer for the current model
            optimizer = torch.optim.Adam(client_model.parameters(), lr=self.config['lr'],
                                         betas=(0.9, 0.98), weight_decay=self.config['l2_reg'])

            # 4. Prepare the user's data
            input_seq = self.clients_data.train_seq[uid]
            target_seq = self.clients_data.valid_seq[uid]
            input_len = self.clients_data.seq_len[uid]
            # Negative sampling
            # Remove interacted items
            cand = np.setdiff1d(self.clients_data.item_set, self.clients_data.seq[uid])
            # Calculate sampling probability
            prob = self.clients_data.item_prob[cand]
            prob = prob / prob.sum()
            # Random sampling
            neg_seq = np.random.choice(cand, (input_len, 100), p=prob)
            # Padding
            neg_seq = np.pad(neg_seq, ((input_seq.shape[0] - input_len, 0), (0, 0)))
            # Convert to tensor
            input_seq = torch.from_numpy(input_seq).unsqueeze(0).to(self.device)
            target_seq = torch.from_numpy(target_seq).unsqueeze(0).to(self.device)
            neg_seq = torch.from_numpy(neg_seq).unsqueeze(0).to(self.device)
            input_len = torch.tensor(input_len).unsqueeze(0).to(self.device)
            max_seq_length = client_model.max_seq_length
            # Handle sequence length limitation
            if input_seq.size(1) > max_seq_length:
                input_seq = input_seq[:, -max_seq_length:]
                target_seq = target_seq[:, -max_seq_length:]
                neg_seq = neg_seq[:, -max_seq_length:, :]  # Add negative sample sequence truncation
                input_len = torch.clamp(input_len, max=max_seq_length)

            # 5. Train with the correctly sized client_model
            seq_out = client_model(input_seq, input_len)
            padding_mask = (torch.not_equal(input_seq, 0)).float().unsqueeze(-1).to(self.device)
            feddecorr = FedDecorrLoss()

            # Loss calculation - Maintain UDL-DDR structure
            if dev_type == 's':
                loss = self.model_s.loss_function(seq_out, padding_mask, target_seq, neg_seq, input_len)
            elif dev_type == 'm':
                loss_s = self.model_s.loss_function(seq_out, padding_mask, target_seq, neg_seq, input_len)
                loss_m = self.model_m.loss_function(seq_out, padding_mask, target_seq, neg_seq, input_len)
                loss_reg = feddecorr(client_model.item_embedding.weight[:, self.dim_s:])
                loss = loss_s + loss_m + alpha * loss_reg
            else:  # 'l'
                loss_s = self.model_s.loss_function(seq_out, padding_mask, target_seq, neg_seq, input_len)
                loss_m = self.model_m.loss_function(seq_out, padding_mask, target_seq, neg_seq, input_len)
                loss_l = self.model_l.loss_function(seq_out, padding_mask, target_seq, neg_seq, input_len)
                loss_reg = feddecorr(client_model.item_embedding.weight[:, self.dim_s:])
                loss = loss_l + loss_s + loss_m + alpha * loss_reg

            # Save loss value
            clients_losses[uid] = loss.item()

            # Backpropagation to calculate gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            gradients = {name: param.grad.clone() for name, param in client_model.named_parameters() if
                         param.grad is not None}

            if self.config['LDP_lambda'] > 0:
                gradients = add_noise(gradients, self.config['LDP_lambda'])

            clients_grads[uid] = gradients

        return clients_grads, clients_losses


class Server:
    """
    Federated Learning Server Class
    Main Functions:
    1. Implement FedAvg gradient aggregation algorithm
    2. Handle gradient averaging from multiple clients
    3. Maintain global model and perform evaluation
    """

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

        # Get heterogeneous device dimension configuration
        self.dim_s = config['dim_s']
        self.dim_m = config['dim_m']
        self.dim_l = config['dim_l']
        
        # Get evaluation k-value configuration
        self.eval_k = config['eval_k']

        # Initialize model - Use large device dimension
        config['embed_size'] = self.dim_l
        # Add knowledge distillation related parameters
        self.kd_ratio = config['kd_ratio']  # Distillation item sampling ratio
        self.kd_lr = config['kd_lr']  # Distillation learning rate
        self.distill_epochs = config['distill_epochs']  # Distillation rounds

        # Correction 1: Create the correct configuration object
        # Create small model configuration
        config_s = config.copy()
        config_s['hidden_size'] = self.dim_s
        # Create medium model configuration
        config_m = config.copy()
        config_m['hidden_size'] = self.dim_m
        # Create large model configuration (use global model configuration)
        config_l = config.copy()
        config_l['hidden_size'] = self.dim_l

        # Correction 2: Correctly initialize models of three sizes
        self.model_s = getModel(config_s, self.clients.clients_data.get_maxid())
        self.model_m = getModel(config_m, self.clients.clients_data.get_maxid())
        self.model_l = getModel(config_l, self.clients.clients_data.get_maxid())

        # Correction 3: Transfer models to device and initialize parameters
        for model in [self.model_s, self.model_m, self.model_l]:
            model.to(self.device)
            # Initialize model parameters
            for name, param in model.named_parameters():
                try:
                    torch.nn.init.xavier_normal_(param.data)
                except:
                    pass  # Ignore layers with initialization failure

        # Correction 4: Ensure global model is consistent with distilled large model
        self.model = self.model_l  # Let global model reference distilled large model
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=config['lr'],
                                          betas=(0.9, 0.98),
                                          weight_decay=config['l2_reg'])

        # Create distillation optimizer, including parameters of all three models
        kd_params = list(self.model_s.parameters()) + list(self.model_m.parameters()) + list(self.model_l.parameters())
        self.kd_optimizer = torch.optim.Adam(kd_params, lr=self.kd_lr)

        logger.info("Knowledge distillation module initialized")


    def _knowledge_distillation(self):
        """Perform relation-based integrated self-knowledge distillation - Fix size mismatch issue"""
        self.logger.info("Start knowledge distillation step...")

        # 1. Prepare three models
        models = {
            's': self.model_s,
            'm': self.model_m,
            'l': self.model_l
        }

        # 2. Randomly select a subset of distillation items
        item_set = self.clients.clients_data.item_set
        kd_items = self._select_distill_items(item_set, self.kd_ratio)
        kd_size = len(kd_items)
        self.logger.info(f"Distillation item subset size: {kd_size} items")

        # 3. Convert item ID to embedding index
        max_index = self.clients.itemnum - 1
        kd_index = torch.from_numpy(np.clip(kd_items - 1, 0, max_index)).long().to(self.device)

        # 4. Calculate item embeddings for each model - Only for the subset of distillation items
        similarity_matrices = {}
        with torch.no_grad():
            for model_type, model in models.items():
                embeddings = model.item_embedding(kd_index)

                # Normalization
                norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
                norm_emb = embeddings / norms

                # Calculate cosine similarity between items
                sim_matrix = torch.mm(norm_emb, norm_emb.t())
                similarity_matrices[model_type] = sim_matrix

        # 5. Calculate ensemble similarity - Ensure all matrix sizes are consistent
        ens_matrix = (similarity_matrices['s'] +
                      similarity_matrices['m'] +
                      similarity_matrices['l']) / 3.0

        # 6. Distillation training
        for epoch_idx in range(self.distill_epochs):
            self.kd_optimizer.zero_grad()

            # Recalculate the similarity of the current model
            current_similarities = {}
            for model_type, model in models.items():
                embeddings = model.item_embedding(kd_index)
                norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
                norm_emb = embeddings / norms
                sim_matrix = torch.mm(norm_emb, norm_emb.t())
                current_similarities[model_type] = sim_matrix

            # Calculate distillation loss - Use MSE
            loss = 0
            for model_type in models:
                loss += torch.nn.functional.mse_loss(
                    current_similarities[model_type],
                    ens_matrix.detach()
                )

            # Backpropagation
            loss.backward()
            self.kd_optimizer.step()

            self.logger.info(f"Distillation epoch [{epoch_idx + 1}/{self.distill_epochs}] loss: {loss.item():.4f}")

        # 7. Update client models
        self._update_client_models()

        self.logger.info("Knowledge distillation step completed")

    def _select_distill_items(self, item_set, ratio):
        """Randomly select a subset of distillation items"""
        num_items = len(item_set)
        kd_size = max(1, int(num_items * ratio))
        return np.random.choice(list(item_set), kd_size, replace=False)

    def _update_client_models(self):
        """Sync distilled model parameters to clients"""
        # Small model update
        client_s = self.clients.model_s
        for param, server_param in zip(client_s.parameters(), self.model_s.parameters()):
            param.data.copy_(server_param.data)

        # Medium model update
        client_m = self.clients.model_m
        for param, server_param in zip(client_m.parameters(), self.model_m.parameters()):
            param.data.copy_(server_param.data)

        # Large model update (global model)
        client_l = self.clients.model
        for param, server_param in zip(client_l.parameters(), self.model_l.parameters()):
            param.data.copy_(server_param.data)

    def aggregate_gradients(self, clients_grads):
        """Perform padding-based gradient aggregation - Support nested device types"""
        clients_num = len(clients_grads)
        if clients_num == 0:
            self.logger.warning("No client gradients received")
            return

        aggregated_gradients = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if
                                param.requires_grad}

        # Aggregate gradients from each client
        for uid, grads_dict in clients_grads.items():
            for name, grad in grads_dict.items():
                if grad is None:
                    continue

                target_shape = aggregated_gradients[name].shape

                # If the received gradient size does not match, pad it
                if grad.shape != target_shape:
                    padded_grad = torch.zeros(target_shape, device=self.device)
                    slicing_indices = tuple(slice(0, dim) for dim in grad.shape)
                    padded_grad[slicing_indices] = grad
                    aggregated_gradients[name] += padded_grad
                else:
                    # If the size matches, accumulate directly
                    aggregated_gradients[name] += grad

        # Calculate average gradient
        for name in aggregated_gradients:
            aggregated_gradients[name] /= clients_num

        # Apply gradients to the model
        for name, param in self.model.named_parameters():
            if name in aggregated_gradients:
                param.grad = aggregated_gradients[name]
            else:
                param.grad = None

    def train ( self ):
        """
        The real federated learning training loop
        """
        # When called in the Server's train method, change to:
        self.clients.sync_models_from_server(self)  # Pass the server instance
        # Create client batch sequence
        user_set = self.clients.clients_data.get_user_set ()
        uid_seq = []
        for i in range (0, len (user_set), self.batch_size):
            batch_uids = user_set [i:i + self.batch_size]
            uid_seq.append (torch.tensor (batch_uids))

        # Early stopping related variables
        best_val_ndcg, best_val_hr = 0.0, 0.0
        best_test_ndcg, best_test_hr = 0.0, 0.0

        # Early stopping counter - Record the number of consecutive rounds without NDCG improvement
        no_improve_count = 0

        # Initialize model parameters
        for name, param in self.model.named_parameters ():
            try:
                torch.nn.init.xavier_normal_ (param.data)
            except:
                pass  # Ignore initialization failure

        # Set the padding position of the embedding layer to 0
        if hasattr (self.model, 'position_embedding'):
            self.model.position_embedding.weight.data [0, :] = 0
        if hasattr (self.model, 'item_embedding'):
            self.model.item_embedding.weight.data [0, :] = 0

        T = 0.0
        t0 = time.time ()

        # Start multi-round training
        early_stop_triggered = False  # Add early stop flag
        distill_freq=self.config['distill_freq']
        for epoch in range (self.epochs):
            # Record the start time of the epoch
            epoch_start_time = time.time ()

            # Switch model to training mode
            self.model.train ()

            # Training statistics variables
            batch_count = 0
            epoch_losses = []

            # Traverse all client batches
            for uids in uid_seq:
                # Zero the gradients
                self.optimizer.zero_grad ()

                # Get the gradient dictionary and loss value dictionary of all clients in the batch
                clients_grads, clients_losses = self.clients.train (uids, self.model.state_dict (), epoch)

                # Collect the loss values of this batch
                batch_losses = list (clients_losses.values ())
                epoch_losses.extend (batch_losses)

                # Use FedAvg gradient aggregation method
                self.aggregate_gradients (clients_grads)

                # Update global model parameters
                self.optimizer.step ()

                # Record batch statistics
                batch_count += 1

            if (epoch + 1) % distill_freq == 0:
                self._knowledge_distillation()

            eval_freq = self.eval_freq
            # Evaluate on the first epoch, every eval_freq epochs, or the last epoch
            should_evaluate = (epoch + 1) % eval_freq == 0 or epoch == 0 or epoch == self.epochs - 1
            if should_evaluate:
                self.model.eval ()
                # t1 is the training time of this round, T is the total time
                t1 = time.time () - t0
                T += t1
                print ('Evaluating', end = '')


                t_valid = evaluate_valid(self.model, self.dataset, self.maxlen, self.clients.neg_num, self.eval_k, self.config['full_eval'], self.device)
                self.logger.info (
                    f"Evaluation result - Epoch {epoch + 1}: NDCG@{self.eval_k}={t_valid [0]:.4f}, HR@{self.eval_k}={t_valid [1]:.4f}")

                # Check for abnormal evaluation results
                if t_valid [0] >= 0.99 or t_valid [1] >= 0.99 or np.isnan (t_valid [0]) or np.isnan (t_valid [1]):
                    self.logger.info (f"Abnormal evaluation result detected: NDCG@{self.eval_k}={t_valid [0]:.4f}, HR@{self.eval_k}={t_valid [1]:.4f}")

                # Early stopping check - Check if NDCG has improved
                if self.early_stop_enabled:
                    if t_valid [0] > best_val_ndcg:
                        # NDCG improved, reset counter
                        no_improve_count = 0
                        best_val_ndcg = t_valid [0]
                    else:
                        # NDCG did not improve, increase counter
                        no_improve_count += 1

                    # If there is no improvement for many consecutive rounds, trigger early stopping
                    if no_improve_count >= self.early_stop:
                        self.logger.info (f"Early stopping triggered! NDCG did not improve for {self.early_stop} rounds.")
                        early_stop_triggered = True

                # Test set evaluation (optional) - Also use heterogeneous evaluation
                if not self.skip_test_eval:
                    # Add test set heterogeneous evaluation here, temporarily use traditional method
                    t_test = evaluate(self.model, self.dataset, self.maxlen, self.clients.neg_num, self.eval_k, self.config['full_eval'], self.device)

                    # Check for abnormal test set evaluation results
                    if t_test [0] > 1.0 or t_test [1] > 1.0 or np.isnan (t_test [0]) or np.isnan (t_test [1]):
                        self.logger.warning (f"Abnormal test result detected: NDCG@{self.eval_k}={t_test [0]:.4f}, HR@{self.eval_k}={t_test [1]:.4f}")

                    # Log the results
                    self.logger.info (
                                            'epoch:%d, time: %f(s), valid (NDCG@%d: %.4f, HR@%d: %.4f), test (NDCG@%d: %.4f, HR@%d: %.4f) all_time: %f(s)'
                    % (epoch + 1, t1, self.eval_k, t_valid [0], self.eval_k, t_valid [1], self.eval_k, t_test [0], self.eval_k, t_test [1], T))
                else:
                    # Skip test set evaluation, set default value
                    t_test = (0.0, 0.0)
                    # Log the results
                    self.logger.info (
                                            'epoch:%d, time: %f(s), valid (NDCG@%d: %.4f, HR@%d: %.4f), test: SKIPPED, all_time: %f(s)'
                    % (epoch + 1, t1, self.eval_k, t_valid [0], self.eval_k, t_valid [1], T))

                # Update best results
                if not self.skip_test_eval:
                    # Include test set evaluation
                    if t_valid [0] > best_val_ndcg or t_valid [1] > best_val_hr or t_test [0] > best_test_ndcg or \
                            t_test [1] > best_test_hr:
                        best_val_ndcg = max (t_valid [0], best_val_ndcg)
                        best_val_hr = max (t_valid [1], best_val_hr)
                        best_test_ndcg = max (t_test [0], best_test_ndcg)
                        best_test_hr = max (t_test [1], best_test_hr)
                        self.logger.info (
                            f"New best performance: valid NDCG@{self.eval_k}={best_val_ndcg:.4f}, test NDCG@{self.eval_k}={best_test_ndcg:.4f}")

                else:
                    # Skip test set evaluation, update based on validation set only
                    if t_valid [0] > best_val_ndcg or t_valid [1] > best_val_hr:
                        best_val_ndcg = max (t_valid [0], best_val_ndcg)
                        best_val_hr = max (t_valid [1], best_val_hr)
                        self.logger.info (
                            f"New best performance: valid NDCG@{self.eval_k}={best_val_ndcg:.4f}, valid HR@{self.eval_k}={best_val_hr:.4f}")

                t0 = time.time ()
                self.model.train ()

                # Check early stop flag after evaluation
                if early_stop_triggered:
                    break

            # If early stopping is triggered, exit the training loop
            if early_stop_triggered:
                break

        # Log best results
        if not self.skip_test_eval:
            self.logger.info (
                            '[Federated Training] Best result: valid NDCG@{}={:.4f}, HR@{}={:.4f}, test NDCG@{}={:.4f}, HR@{}={:.4f}'.format (
                self.eval_k, best_val_ndcg, self.eval_k, best_val_hr, self.eval_k, best_test_ndcg, self.eval_k, best_test_hr))
        else:
            self.logger.info ('[Federated Training] Best result: valid NDCG@{}={:.4f}, HR@{}={:.4f} (Test set evaluation skipped)'.format (
                self.eval_k, best_val_ndcg, self.eval_k, best_val_hr))
