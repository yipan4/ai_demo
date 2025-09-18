import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import time
import json
import socket
import csv

from model import CNN
from utils import send_msg, receive_msg, send_model_dict, receive_model_dict


# ENV
MODE = os.getenv('MODE', 'fedavg')
CLIENT_ID_STR = os.getenv('CLIENT_ID', '0')
SERVER_IP = os.getenv('SERVER_IP', '127.0.0.1')
SERVER_PORT = int(os.getenv('SERVER_PORT', 4651))
NUM_TOTAL_CLIENTS_IN_FL = int(os.getenv('NUM_TOTAL_CLIENTS', 3))
SEED = int(os.getenv('SEED', 4651))

L1_REGULARIZATION_LAMBDA = float(os.getenv('L1_REG_LAMBDA', 1e-5))
ETA_MU_THRESHOLDING_PARAM = float(os.getenv('ETA_MU_THRESHOLDING', 0.2))
THRESHOLD_ADJUSTMENT_STRENGTH_PARAM = float(os.getenv('THRESHOLD_ADJUSTMENT_STRENGTH', 0.1))
LOCAL_TRAINING_EPOCHS = int(os.getenv('LOCAL_EPOCHS', 1))
CLIENT_LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.01))
CLIENT_NOISE_STD_DEV = float(os.getenv('NOISE_STD_DEV', 0.3))

CLIENT_BATCH_SIZE = 64
LOCAL_DATA_PATH = './data'


MAX_CONNECTION_RETRIES = 5
RETRY_DELAY_SECONDS = 15
INTER_ROUND_DELAY_SECONDS = 5


GROUP_A_IDENTIFIER_KEY = 'group_a'
GROUP_B_IDENTIFIER_KEY = 'group_b'
GROUP_A_CLASS_LABELS = [0,2,3,4,6]
GROUP_B_CLASS_LABELS = [1,5,7,8,9]


try:
    CLIENT_ID = int(CLIENT_ID_STR)
except ValueError:
    print(f"FATAL: CLIENT_ID '{CLIENT_ID_STR}' is not a valid integer. Please set CLIENT_ID environment variable correctly.", flush=True)
    exit(1)

CLIENT_LOG_FIELDNAMES = [
    "round", "mode", "local_epochs", "learning_rate",
    "avg_epoch_loss_fedavg", "accuracy_test", "avg_tpr_a_test", "avg_tpr_b_test",
    "loss_group_a_mwr", "loss_group_b_mwr"
]
CLIENT_LOG_FILENAME = f"client_{CLIENT_ID}_results_seed{SEED}_mode{MODE}.csv"


np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

class TempDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)
    

def add_gaussian_noise(image_tensor, mean=0.0, std_dev=0.1) -> torch.Tensor:
    noise = torch.randn_like(image_tensor) * std_dev + mean
    return torch.clamp(image_tensor + noise, 0.0, 1.0)

def get_fashion_mnist_data(current_client_id, total_num_clients, is_training_set=True, noise_standard_deviation=0.3) -> tuple:
    base_image_transforms = [transforms.ToTensor()]
    apply_noise_for_this_client = False
    target_class_labels_for_client = []

    if current_client_id == 0:
        target_class_labels_for_client = GROUP_A_CLASS_LABELS
    elif current_client_id == 1:
        target_class_labels_for_client = GROUP_B_CLASS_LABELS
    elif current_client_id == 2 and total_num_clients >= 3:
        target_class_labels_for_client = GROUP_B_CLASS_LABELS
        apply_noise_for_this_client = True
    else:
        target_class_labels_for_client = GROUP_A_CLASS_LABELS if current_client_id % 2 == 0 else GROUP_B_CLASS_LABELS

    image_transform_pipeline = list(base_image_transforms)
    if apply_noise_for_this_client:
        image_transform_pipeline.append(transforms.Lambda(lambda img: add_gaussian_noise(img, std_dev=noise_standard_deviation)))
    image_transform_pipeline.append(transforms.Normalize((0.2860,), (0.3530,)))
    final_image_transform = transforms.Compose(image_transform_pipeline)

    full_dataset_raw = datasets.FashionMNIST(LOCAL_DATA_PATH, train=is_training_set, download=True, transform=None)
    client_sample_indices = [idx for idx, label in enumerate(full_dataset_raw.targets) if label.item() in target_class_labels_for_client]
    client_subset_raw = Subset(full_dataset_raw, client_sample_indices)
    client_dataset_transformed = TempDataset(client_subset_raw, final_image_transform)
    data_loader = DataLoader(client_dataset_transformed, batch_size=CLIENT_BATCH_SIZE, shuffle=is_training_set, num_workers=0)
    
    return data_loader, target_class_labels_for_client


def evaluate_model(model, eval_data_loader, classes_evaluated_by_client, device="cpu", is_internal_threshold_calc=False, current_round_num=0) -> tuple:
    model.eval()
    model.to(device)
    class_true_positives = torch.zeros(10, dtype=torch.float32).to(device)
    class_actual_samples = torch.zeros(10, dtype=torch.float32).to(device)

    with torch.no_grad():
        for images, actual_labels in eval_data_loader:
            images, actual_labels = images.to(device), actual_labels.to(device)
            log_probabilities = model(images)
            predicted_labels = log_probabilities.argmax(dim=1, keepdim=True)
            for i in range(len(actual_labels)):
                actual_class_idx, predicted_class_idx = actual_labels[i].item(), predicted_labels[i].item()
                class_actual_samples[actual_class_idx] += 1
                if actual_class_idx == predicted_class_idx:
                    class_true_positives[actual_class_idx] += 1

    tpr_per_class = torch.zeros(10, dtype=torch.float32).to(device)
    for i in range(10):
        if class_actual_samples[i] > 0:
            tpr_per_class[i] = class_true_positives[i] / class_actual_samples[i]

    accuracy = 0.0
    total_correct_predictions = class_true_positives.sum().item()
    total_samples_evaluated = class_actual_samples.sum().item()
    if total_samples_evaluated > 0:
        accuracy = 100. * total_correct_predictions / total_samples_evaluated

    if not is_internal_threshold_calc:
        print(f'\n--- C{CLIENT_ID} R{current_round_num} Evaluation (Mode {MODE}) ---', flush=True)
        if total_samples_evaluated > 0:
            print(f'Accuracy: {accuracy:.2f}% ({int(total_correct_predictions)}/{int(total_samples_evaluated)})', flush=True)

    group_a_tprs_list = [tpr_per_class[c_label].item() for c_label in GROUP_A_CLASS_LABELS
                        if c_label in classes_evaluated_by_client and class_actual_samples[c_label] > 0]
    group_b_tprs_list = [tpr_per_class[c_label].item() for c_label in GROUP_B_CLASS_LABELS
                        if c_label in classes_evaluated_by_client and class_actual_samples[c_label] > 0]
    average_tpr_group_a = np.mean(group_a_tprs_list) if group_a_tprs_list else float('nan')
    average_tpr_group_b = np.mean(group_b_tprs_list) if group_b_tprs_list else float('nan')

    if not is_internal_threshold_calc:
        if not np.isnan(average_tpr_group_a): 
            print(f'Avg TPR GrpA (relevant): {average_tpr_group_a:.4f}', flush=True)
        if not np.isnan(average_tpr_group_b): 
            print(f'Avg TPR GrpB (relevant): {average_tpr_group_b:.4f}', flush=True)

    return tpr_per_class.cpu(), average_tpr_group_a, average_tpr_group_b, class_actual_samples.cpu(), accuracy


def train_local(model, data_loader, optimizer, num_local_epochs, device, current_round_num=0) -> float:
    model.train()
    model.to(device)
    final_epoch_avg_loss = 0.0
    for epoch_num in range(num_local_epochs):
        sum_epoch_loss = 0.0
        num_batches_in_epoch = 0
        for batch_idx, (batch_images, batch_labels) in enumerate(data_loader):
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            output_log_probs = model(batch_images)
            loss = F.nll_loss(output_log_probs, batch_labels)
            loss.backward()
            optimizer.step()
            sum_epoch_loss += loss.item()
            num_batches_in_epoch +=1
        final_epoch_avg_loss = sum_epoch_loss / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
        print(f'C{CLIENT_ID} R{current_round_num} FedAvg E{epoch_num+1}/{num_local_epochs} AvgLoss: {final_epoch_avg_loss:.4f}', flush=True)
        
    return final_epoch_avg_loss


def train_mwr(model, train_data_loader, test_data_loader, optimizer, num_local_epochs, device,
              current_server_lambdas, client_group_config,
              client_target_classes,
              l1_strength, eta_thresh_param, thresh_adj_factor, mode, current_round_num=0) -> dict:
    model.to(device)
    initial_training_weights = {}
    sum_of_lambdas = sum(val for val in current_server_lambdas.values() if val is not None and val > 0)
    if sum_of_lambdas > 1e-9:
        for group_key, lambda_val in current_server_lambdas.items(): 
            initial_training_weights[group_key] = lambda_val / sum_of_lambdas
    else:
        initial_training_weights = {key: 1.0 / len(current_server_lambdas) if current_server_lambdas else 0.5 for key in current_server_lambdas}
    final_training_weights = initial_training_weights.copy()

    if mode == 'mwr_thresh' and eta_thresh_param > 0 and test_data_loader:
        model.eval()
        tpr_per_class_values, _, _, actual_positives_per_class, _ = evaluate_model( 
            model, test_data_loader, client_target_classes,
            device=device, is_internal_threshold_calc=True, current_round_num=current_round_num
        )
        model.train()
        relevant_tprs = [tpr_per_class_values[c_label].item() for c_label in client_target_classes if actual_positives_per_class[c_label] > 0]
        if len(relevant_tprs) >= 2:
            tpr_best_client, tpr_worst_client = max(relevant_tprs), min(relevant_tprs)
            target_tpr_threshold = tpr_best_client - eta_thresh_param * (tpr_best_client - tpr_worst_client)
            best_class_label_idx, worst_class_label_idx = -1, -1
            for class_idx in client_target_classes:
                if actual_positives_per_class[class_idx] > 0:
                    if abs(tpr_per_class_values[class_idx].item() - tpr_best_client) < 1e-6: 
                        best_class_label_idx = class_idx
                    if abs(tpr_per_class_values[class_idx].item() - tpr_worst_client) < 1e-6: 
                        worst_class_label_idx = class_idx
            best_performing_meta_group_key, worst_performing_meta_group_key = None, None
            if best_class_label_idx != -1:
                for gk_iter, cl_iter in client_group_config.items():
                    if cl_iter and best_class_label_idx in cl_iter: 
                        best_performing_meta_group_key = gk_iter
                        break
            if worst_class_label_idx != -1:
                for gk_iter, cl_iter in client_group_config.items():
                    if cl_iter and worst_class_label_idx in cl_iter: 
                        worst_performing_meta_group_key = gk_iter
                        break
            current_fairness_gap = tpr_best_client - tpr_worst_client
            SIGNIFICANT_GAP_THRESHOLD = 0.05
            if best_performing_meta_group_key and worst_performing_meta_group_key and best_performing_meta_group_key != worst_performing_meta_group_key and current_fairness_gap > SIGNIFICANT_GAP_THRESHOLD:
                if tpr_best_client > target_tpr_threshold + 1e-3 and best_performing_meta_group_key in final_training_weights:
                    final_training_weights[best_performing_meta_group_key] = max(0.01, final_training_weights[best_performing_meta_group_key] * (1.0 - thresh_adj_factor))
                if worst_performing_meta_group_key in final_training_weights:
                    final_training_weights[worst_performing_meta_group_key] *= (1.0 + thresh_adj_factor)
                sum_adj_weights = sum(w for w in final_training_weights.values() if w is not None and w > 0)
                if sum_adj_weights > 1e-9:
                    for gk_norm in final_training_weights:
                        if final_training_weights[gk_norm] is not None: 
                            final_training_weights[gk_norm] /= sum_adj_weights

    epoch_unweighted_losses_per_group = {key: [] for key, cl in client_group_config.items() if cl is not None}
    for epoch_num in range(num_local_epochs):
        sum_batch_weighted_loss_for_epoch, num_samples_processed_in_epoch = 0.0, 0
        for group_id_key_to_clear in epoch_unweighted_losses_per_group: epoch_unweighted_losses_per_group[group_id_key_to_clear] = []
        for batch_idx, (batch_images, batch_labels) in enumerate(train_data_loader):
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            output_log_probs = model(batch_images)
            unweighted_sample_losses = F.nll_loss(output_log_probs, batch_labels, reduction='none')
            current_batch_total_weighted_loss = torch.tensor(0.0).to(device)
            for i in range(len(batch_labels)):
                actual_class_label = batch_labels[i].item()
                unweighted_loss = unweighted_sample_losses[i]
                assigned_meta_group_key = None
                for gk_iter, cl_iter in client_group_config.items():
                    if cl_iter and actual_class_label in cl_iter: 
                        assigned_meta_group_key = gk_iter
                        break
                weighted_loss = unweighted_loss
                if assigned_meta_group_key and assigned_meta_group_key in final_training_weights and final_training_weights[assigned_meta_group_key] is not None:
                    group_weight = final_training_weights[assigned_meta_group_key]
                    weighted_loss = group_weight * unweighted_loss
                    epoch_unweighted_losses_per_group[assigned_meta_group_key].append(unweighted_loss.item())
                elif assigned_meta_group_key and assigned_meta_group_key in epoch_unweighted_losses_per_group:
                    epoch_unweighted_losses_per_group[assigned_meta_group_key].append(unweighted_loss.item())
                current_batch_total_weighted_loss += weighted_loss
            avg_batch_weighted_loss = current_batch_total_weighted_loss / len(batch_labels) if len(batch_labels) > 0 else torch.tensor(0.0).to(device)
            l1_penalty = torch.tensor(0.0).to(device)
            if l1_strength > 0:
                for param in model.parameters(): 
                    l1_penalty += torch.norm(param, 1)
            total_loss_for_update = avg_batch_weighted_loss + l1_strength * l1_penalty
            total_loss_for_update.backward()
            optimizer.step()
            sum_batch_weighted_loss_for_epoch += total_loss_for_update.item() * len(batch_labels)
            num_samples_processed_in_epoch += len(batch_labels)
        avg_epoch_weighted_loss = sum_batch_weighted_loss_for_epoch / num_samples_processed_in_epoch if num_samples_processed_in_epoch > 0 else 0
        print(f'C{CLIENT_ID} R{current_round_num} {mode.upper()} E{epoch_num+1}/{num_local_epochs} AvgWLoss:{avg_epoch_weighted_loss:.4f}', flush=True)
    reportable_avg_unweighted_losses = {
        gk_report: np.mean(losses_report) if losses_report else None for gk_report, losses_report in epoch_unweighted_losses_per_group.items()
    }
    
    return reportable_avg_unweighted_losses


def setup_client_logging_file() -> None:
    if not os.path.exists(CLIENT_LOG_FILENAME):
        with open(CLIENT_LOG_FILENAME, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CLIENT_LOG_FIELDNAMES)
            writer.writeheader()
    print(f"C{CLIENT_ID}: Logging results to {CLIENT_LOG_FILENAME}", flush=True)


def log_client_round_data_to_file(data_dict) -> None:
    try:
        with open(CLIENT_LOG_FILENAME, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CLIENT_LOG_FIELDNAMES)
            writer.writerow(data_dict)
    except Exception as e:
        print(f"C{CLIENT_ID}: Error writing to log file {CLIENT_LOG_FILENAME}: {e}", flush=True)


def run_one_fl_round(current_round_num_for_logging: int) -> str | bool: 
    print(f'C{CLIENT_ID} starting round {current_round_num_for_logging}. Server: {SERVER_IP}:{SERVER_PORT}', flush=True)
    device = torch.device('cpu')
    client_local_model = CNN().to(device)
    client_socket = None
    avg_loss_this_round = None 
    mwr_group_losses_this_round = {}

    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(30.0)
        client_socket.connect((SERVER_IP, SERVER_PORT))
        print(f'C{CLIENT_ID} (R{current_round_num_for_logging}): Connected.', flush=True)
        client_socket.settimeout(300.0)

        send_msg(client_socket, "REQINFO", {"client_id": CLIENT_ID, "round": current_round_num_for_logging})
        msg_type_config, round_config_from_server = receive_msg(client_socket)
        if msg_type_config != "RCONFIG" or not round_config_from_server:
            print(f"C{CLIENT_ID} (R{current_round_num_for_logging}): Expected RCONFIG, got {msg_type_config}. Aborting.", flush=True)
            return False
        
        global_model_state_dict = receive_model_dict(client_socket)
        client_local_model.load_state_dict(global_model_state_dict)

        current_mode_for_round = round_config_from_server.get('mode', MODE)
        local_epochs_for_round = round_config_from_server.get('local_epochs', LOCAL_TRAINING_EPOCHS)
        lr_for_round = round_config_from_server.get('learning_rate', CLIENT_LEARNING_RATE)
        l1_reg_lambda_for_round = round_config_from_server.get('l1_reg_lambda', L1_REGULARIZATION_LAMBDA)
        eta_mu_thresh_for_round = round_config_from_server.get('eta_mu_thresholding', ETA_MU_THRESHOLDING_PARAM)
        thresh_adj_strength_for_round = round_config_from_server.get('threshold_adjustment_strength', THRESHOLD_ADJUSTMENT_STRENGTH_PARAM)
        noise_std_dev_for_round = round_config_from_server.get('noise_std_dev', CLIENT_NOISE_STD_DEV)
        
        print(f"C{CLIENT_ID} (R{current_round_num_for_logging}): Config - Mode: {current_mode_for_round}, Epochs: {local_epochs_for_round}, LR: {lr_for_round}", flush=True)

        lambdas_from_server = {}
        if current_mode_for_round in ['mwr', 'mwr_thresh']:
            msg_type_lambda, lambda_payload_data = receive_msg(client_socket)
            if msg_type_lambda != 'LAMBDAS':
                lambdas_from_server = {GROUP_A_IDENTIFIER_KEY: 0.5, GROUP_B_IDENTIFIER_KEY: 0.5}
            else:
                lambdas_from_server = lambda_payload_data
                if not lambdas_from_server or (GROUP_A_IDENTIFIER_KEY not in lambdas_from_server or GROUP_B_IDENTIFIER_KEY not in lambdas_from_server):
                    lambdas_from_server = {GROUP_A_IDENTIFIER_KEY: 0.5, GROUP_B_IDENTIFIER_KEY: 0.5}
        
        client_specific_group_config = {} 
        client_all_effective_target_classes_list = []
        if CLIENT_ID == 0:
            client_specific_group_config[GROUP_A_IDENTIFIER_KEY] = list(GROUP_A_CLASS_LABELS)
            client_all_effective_target_classes_list.extend(GROUP_A_CLASS_LABELS)
        elif CLIENT_ID == 1:
            client_specific_group_config[GROUP_B_IDENTIFIER_KEY] = list(GROUP_B_CLASS_LABELS)
            client_all_effective_target_classes_list.extend(GROUP_B_CLASS_LABELS)
        elif CLIENT_ID == 2 and NUM_TOTAL_CLIENTS_IN_FL >= 3:
            client_specific_group_config[GROUP_B_IDENTIFIER_KEY] = list(GROUP_B_CLASS_LABELS)
            client_all_effective_target_classes_list.extend(GROUP_B_CLASS_LABELS)
        else:
            if CLIENT_ID % 2 == 0:
                client_specific_group_config[GROUP_A_IDENTIFIER_KEY] = list(GROUP_A_CLASS_LABELS)
                client_all_effective_target_classes_list.extend(GROUP_A_CLASS_LABELS)
            else:
                client_specific_group_config[GROUP_B_IDENTIFIER_KEY] = list(GROUP_B_CLASS_LABELS)
                client_all_effective_target_classes_list.extend(GROUP_B_CLASS_LABELS)
        client_all_effective_target_classes = sorted(list(set(client_all_effective_target_classes_list)))

        train_data_loader, _ = get_fashion_mnist_data(CLIENT_ID, NUM_TOTAL_CLIENTS_IN_FL, True, noise_std_dev_for_round)
        test_data_loader, _ = get_fashion_mnist_data(CLIENT_ID, NUM_TOTAL_CLIENTS_IN_FL, False, noise_std_dev_for_round)
        sgd_optimizer = optim.SGD(client_local_model.parameters(), lr=lr_for_round)
        
        if current_mode_for_round == 'fedavg':
            avg_loss_this_round = train_local(client_local_model, train_data_loader, sgd_optimizer, local_epochs_for_round, device, current_round_num_for_logging)
        elif current_mode_for_round in ['mwr', 'mwr_thresh']:
            mwr_group_losses_this_round = train_mwr(
                client_local_model, train_data_loader, test_data_loader, sgd_optimizer, local_epochs_for_round, device,
                lambdas_from_server, client_specific_group_config, client_all_effective_target_classes,
                l1_reg_lambda_for_round, eta_mu_thresh_for_round, thresh_adj_strength_for_round, current_mode_for_round, current_round_num_for_logging
            )
        
        tpr_metrics_final_eval, avg_tpr_group_a_final_eval, avg_tpr_group_b_final_eval, actual_positives_final_eval, accuracy_val = evaluate_model(
            client_local_model, test_data_loader, client_all_effective_target_classes, device, False, current_round_num_for_logging
        )
        

        log_payload = {key: None for key in CLIENT_LOG_FIELDNAMES} 
        log_payload.update({
            "round": current_round_num_for_logging,
            "mode": current_mode_for_round,
            "local_epochs": local_epochs_for_round,
            "learning_rate": lr_for_round,
            "accuracy_test": f"{accuracy_val:.4f}" if accuracy_val is not None else None,
            "avg_tpr_a_test": f"{avg_tpr_group_a_final_eval:.4f}" if not np.isnan(avg_tpr_group_a_final_eval) else None,
            "avg_tpr_b_test": f"{avg_tpr_group_b_final_eval:.4f}" if not np.isnan(avg_tpr_group_b_final_eval) else None,
        })
        if current_mode_for_round == 'fedavg' and avg_loss_this_round is not None:
            log_payload["avg_epoch_loss_fedavg"] = f"{avg_loss_this_round:.4f}"
        if current_mode_for_round in ['mwr', 'mwr_thresh'] and mwr_group_losses_this_round:
            log_payload["loss_group_a_mwr"] = f"{mwr_group_losses_this_round.get(GROUP_A_IDENTIFIER_KEY):.4f}" if mwr_group_losses_this_round.get(GROUP_A_IDENTIFIER_KEY) is not None else None
            log_payload["loss_group_b_mwr"] = f"{mwr_group_losses_this_round.get(GROUP_B_IDENTIFIER_KEY):.4f}" if mwr_group_losses_this_round.get(GROUP_B_IDENTIFIER_KEY) is not None else None
        log_client_round_data_to_file(log_payload)


        client_report_for_server = {
            'client_id': CLIENT_ID, 'round': current_round_num_for_logging, 'mode': current_mode_for_round,
            'group_losses': mwr_group_losses_this_round if current_mode_for_round in ['mwr', 'mwr_thresh'] else ({"overall_loss": avg_loss_this_round} if avg_loss_this_round is not None else {}),
            'avg_tpr_a_final': avg_tpr_group_a_final_eval if not np.isnan(avg_tpr_group_a_final_eval) else None,
            'avg_tpr_b_final': avg_tpr_group_b_final_eval if not np.isnan(avg_tpr_group_b_final_eval) else None,
        }
        for group_id_key_report, class_list_report in client_specific_group_config.items():
            if class_list_report:
                tprs_for_group_report = [tpr_metrics_final_eval[c_label].item() for c_label in class_list_report if actual_positives_final_eval[c_label] > 0]
                if tprs_for_group_report:
                    btpr_meta_group, wtpr_meta_group = max(tprs_for_group_report), min(tprs_for_group_report)
                    tprsd_meta_group = np.std(tprs_for_group_report).item() if len(tprs_for_group_report) > 1 else 0.0
                    client_report_for_server[f'BTPR_{group_id_key_report}'] = btpr_meta_group
                    client_report_for_server[f'WTPR_{group_id_key_report}'] = wtpr_meta_group
                    client_report_for_server[f'TPRSD_{group_id_key_report}'] = tprsd_meta_group
                    client_report_for_server[f'TPRD_{group_id_key_report}'] = btpr_meta_group - wtpr_meta_group
        
        send_msg(client_socket, "REPORT", client_report_for_server)
        send_model_dict(client_socket, client_local_model.cpu().state_dict())
        msg_type_ack, ack_data = receive_msg(client_socket)
        
        if msg_type_ack == 'ACKUPD':
            if isinstance(ack_data, dict) and ack_data.get("status") == "FL_COMPLETE":
                print(f'C{CLIENT_ID} (R{current_round_num_for_logging}): FL_COMPLETE signal received. Will terminate after this round.', flush=True)
                return "FL_COMPLETE"
            print(f'C{CLIENT_ID} (R{current_round_num_for_logging}): Update acknowledged. Round complete.', flush=True)
            return True
        else:
            print(f'C{CLIENT_ID} (R{current_round_num_for_logging}): [Warning] Expected ACKUPD, got {msg_type_ack}. Assuming success.', flush=True)
            return True

    except ConnectionRefusedError: 
        print(f'C{CLIENT_ID} (R{current_round_num_for_logging}) FATAL: Conn refused.', flush=True)
        return False
    except socket.timeout: 
        print(f'C{CLIENT_ID} (R{current_round_num_for_logging}) FATAL: Socket timeout.', flush=True)
        return False
    except ConnectionError as e: 
        print(f'C{CLIENT_ID} (R{current_round_num_for_logging}) FATAL: ConnectionError: {e}', flush=True)
        return False
    except socket.error as e: 
        print(f'C{CLIENT_ID} (R{current_round_num_for_logging}) FATAL: SocketError: {e}', flush=True)
        return False
    except Exception as e:
        print(f'C{CLIENT_ID} (R{current_round_num_for_logging}) FATAL: Unexpected error: {e}', flush=True)
        import traceback
        traceback.print_exc()
        return False
    finally:
        if client_socket: 
            client_socket.close()


def start_client_daemon():
    current_round_attempt = 0
    connection_retries_count = 0

    while True:
        current_round_attempt += 1
        print(f"\nC{CLIENT_ID}: Attempting FL Round (Client attempt: {current_round_attempt})...", flush=True)
        round_outcome = run_one_fl_round(current_round_attempt)

        if round_outcome == "FL_COMPLETE":
            print(f"C{CLIENT_ID}: FL process complete as signaled by server. Client shutting down.", flush=True)
            break
        elif round_outcome is True: 
            connection_retries_count = 0
            print(f"C{CLIENT_ID}: Round {current_round_attempt} completed successfully. Waiting {INTER_ROUND_DELAY_SECONDS}s...", flush=True)
            time.sleep(INTER_ROUND_DELAY_SECONDS)
        else: 
            print(f"C{CLIENT_ID}: Round {current_round_attempt} failed or connection issue.", flush=True)
            connection_retries_count += 1
            if connection_retries_count > MAX_CONNECTION_RETRIES:
                print(f"C{CLIENT_ID}: Max connection retries ({MAX_CONNECTION_RETRIES}) exceeded. Client shutting down.", flush=True)
                break
            print(f"C{CLIENT_ID}: Retry in {RETRY_DELAY_SECONDS}s (Retry {connection_retries_count}/{MAX_CONNECTION_RETRIES}).", flush=True)
            time.sleep(RETRY_DELAY_SECONDS)
    print(f"Client {CLIENT_ID} has finished its execution.", flush=True)

if __name__ == '__main__':
    if not SERVER_IP or not SERVER_PORT:
        print(f'Client {CLIENT_ID_STR}: FATAL - SERVER_IP or SERVER_PORT not set.', flush=True)
        exit(1)
    
    setup_client_logging_file()
    start_client_daemon()