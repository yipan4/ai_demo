import torch
import socket
import threading
import numpy as np
import random
from collections import OrderedDict
import time
import argparse
import csv
import os

from model import CNN
from utils import send_msg, receive_msg, send_model_dict, receive_model_dict

DEFAULT_SEED = 4651
GROUP_A_KEY = 'group_a'
GROUP_B_KEY = 'group_b'

global_model_state_dict_server = None
current_lambda_groups_state_server = None
client_updates_for_current_round = []
client_updates_access_lock = threading.Lock()
ARGS_PARSED = None

SERVER_LAMBDA_LOG_FILENAME_TEMPLATE = "server_lambdas_log_seed{seed}_mode{mode}.csv"

def setup_server_lambda_logging_file(args):
    if args.mode not in ['mwr', 'mwr_thresh']:
        return
    filename = SERVER_LAMBDA_LOG_FILENAME_TEMPLATE.format(seed=args.seed, mode=args.mode)
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["round", f"lambda_{GROUP_A_KEY}", f"lambda_{GROUP_B_KEY}"])
    print(f"Server: Logging lambdas to {filename}", flush=True)


def log_server_lambdas_data_to_file(round_num, lambdas_dict, args):
    if args.mode not in ['mwr', 'mwr_thresh'] or not lambdas_dict:
        return
    filename = SERVER_LAMBDA_LOG_FILENAME_TEMPLATE.format(seed=args.seed, mode=args.mode)
    try:
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            lambda_a = lambdas_dict.get(GROUP_A_KEY)
            lambda_b = lambdas_dict.get(GROUP_B_KEY)
            writer.writerow([
                round_num,
                f"{lambda_a:.6f}" if isinstance(lambda_a, float) else 'N/A',
                f"{lambda_b:.6f}" if isinstance(lambda_b, float) else 'N/A'
            ])
    except Exception as e:
        print(f"Server: Error writing to lambda log file {filename}: {e}", flush=True)


def set_seed(seed) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f'Server: Seed set to {seed}', flush=True)


def initialize_lambda_groups(num_meta_groups=2) -> dict:
    initial_value = 0.5
    if num_meta_groups > 0:
        initial_value = 1.0 / num_meta_groups
    initial_lambdas_map = {GROUP_A_KEY: initial_value, GROUP_B_KEY: initial_value}
    
    return initial_lambdas_map


def update_lambda_groups_on_server(current_lambda_map, list_of_client_reports, eta_mu_for_lambda_update) -> dict:
    all_group_losses_map = {GROUP_A_KEY: [], GROUP_B_KEY: []}
    for client_report_item in list_of_client_reports:
        if 'group_losses' in client_report_item and isinstance(client_report_item['group_losses'], dict):
            for group_id_key, reported_loss_value in client_report_item['group_losses'].items():
                if reported_loss_value is not None and isinstance(reported_loss_value, (int, float)) and group_id_key in all_group_losses_map:
                    all_group_losses_map[group_id_key].append(reported_loss_value)

    if not any(loss_list for loss_list in all_group_losses_map.values()):
        return current_lambda_map

    new_lambdas_map = current_lambda_map.copy()
    for group_id_key, list_of_losses in all_group_losses_map.items():
        if list_of_losses:
            average_group_loss = np.mean(list_of_losses)
            lambda_update_factor_val = np.exp(-average_group_loss * eta_mu_for_lambda_update)
            new_lambdas_map[group_id_key] = current_lambda_map.get(group_id_key, 0.5) * lambda_update_factor_val
            
    return new_lambdas_map


def aggregate_models_from_updates(list_of_client_model_state_dicts) -> OrderedDict | None:
    if not list_of_client_model_state_dicts:
        return None
    aggregated_model_weights = OrderedDict()
    reference_model_state_dict = list_of_client_model_state_dicts[0]
    for layer_key in reference_model_state_dict.keys():
        tensors_for_layer_key = [client_model_sd[layer_key].float().cpu() 
                                for client_model_sd in list_of_client_model_state_dicts 
                                if layer_key in client_model_sd and client_model_sd[layer_key] is not None]
        if tensors_for_layer_key:
            aggregated_model_weights[layer_key] = torch.stack(tensors_for_layer_key, 0).mean(0)
        elif layer_key in reference_model_state_dict:
            aggregated_model_weights[layer_key] = reference_model_state_dict[layer_key].float().cpu()
    
    return aggregated_model_weights if aggregated_model_weights else None

def handle_client_connection(client_comm_socket, client_inet_address, server_runtime_args, current_fl_round_num) -> None:
    global global_model_state_dict_server, current_lambda_groups_state_server, client_updates_for_current_round, client_updates_access_lock
    client_identifier_str = "UnknownClient"
    
    try:
        msg_type_req, client_req_data = receive_msg(client_comm_socket)
        if msg_type_req != "REQINFO":
            print(f'SVR_HANDLER ({client_inet_address}): Unexpected init message: "{msg_type_req}". Expected "REQINFO". Closing.', flush=True)
            return

        client_identifier_str = str(client_req_data.get('client_id', 'UnknownClient'))
        client_reported_round = client_req_data.get('round', 'N/A')
        print(f'\nSVR_HANDLER C{client_identifier_str} (ClientR: {client_reported_round}, SvrR: {current_fl_round_num}): Connected.', flush=True)

        # Prepare round configuration to send to client
        round_config_for_client = {'mode': server_runtime_args.mode, 'seed': server_runtime_args.seed}
        round_config_for_client.update({
            k: getattr(server_runtime_args, k) for k in [
                'l1_reg_lambda', 'eta_mu_thresholding', 'threshold_adjustment_strength',
                'local_epochs', 'learning_rate', 'noise_std_dev', 'num_total_clients'
            ] if hasattr(server_runtime_args, k)
        })

        send_msg(client_comm_socket, 'RCONFIG', round_config_for_client)
        send_model_dict(client_comm_socket, global_model_state_dict_server)

        # Send lambdas if in MWR mode
        if server_runtime_args.mode in ['mwr', 'mwr_thresh'] and current_lambda_groups_state_server:
            send_msg(client_comm_socket, 'LAMBDAS', current_lambda_groups_state_server)

        # Receive client's report and model
        msg_type_report, client_submitted_report_data = receive_msg(client_comm_socket)
        if msg_type_report != "REPORT":
            print(f'SVR_HANDLER C{client_identifier_str}: Unexpected report msg_type: "{msg_type_report}". Discarding.', flush=True)
            return
        client_submitted_model_state_dict = receive_model_dict(client_comm_socket)

        # Store client update (thread-safe)
        with client_updates_access_lock:
            client_updates_for_current_round.append({
                'client_id': client_identifier_str,
                'model_state_dict': client_submitted_model_state_dict,
                'report_data': client_submitted_report_data
            })
        print(f'SVR_HANDLER C{client_identifier_str}: Received updates for R{current_fl_round_num}.', flush=True)

        # Acknowledge update and signal FL completion if it's the last round
        ack_payload = None
        if current_fl_round_num == server_runtime_args.num_rounds:
            ack_payload = {"status": "FL_COMPLETE"}
            print(f"SVR_HANDLER C{client_identifier_str}: This is the final round ({current_fl_round_num}/{server_runtime_args.num_rounds}). Sending FL_COMPLETE signal.", flush=True)
        send_msg(client_comm_socket, 'ACKUPD', ack_payload)

    except ConnectionError as e:
        print(f'SVR_HANDLER C{client_identifier_str} ({client_inet_address}): ConnectionError: {e}', flush=True)
    except ValueError as e: # Can be raised by receive_msg on malformed data
        print(f'SVR_HANDLER C{client_identifier_str} ({client_inet_address}): ValueError: {e}', flush=True)
    except Exception as e:
        print(f'SVR_HANDLER C{client_identifier_str} ({client_inet_address}): Unexpected error: {e}', flush=True)
        import traceback
        traceback.print_exc()
    finally:
        client_comm_socket.close()


def run_server(cmd_args) -> None:
    global global_model_state_dict_server, current_lambda_groups_state_server, client_updates_for_current_round
    set_seed(cmd_args.seed)
    print(f'Server starting. Mode: {cmd_args.mode}, Seed: {cmd_args.seed}', flush=True)
    if cmd_args.verbose:
        print(f'Hyperparams: {vars(cmd_args)}', flush=True)

    setup_server_lambda_logging_file(cmd_args)

    # Initialize global model
    temp_cnn_model = CNN()
    global_model_state_dict_server = temp_cnn_model.state_dict()
    del temp_cnn_model

    # Initialize lambdas if in MWR mode and log initial state
    if cmd_args.mode in ['mwr', 'mwr_thresh']:
        current_lambda_groups_state_server = initialize_lambda_groups(cmd_args.num_meta_groups)
        log_server_lambdas_data_to_file(0, current_lambda_groups_state_server, cmd_args) # Log initial lambdas as round 0

    main_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    main_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        main_server_socket.bind((cmd_args.server_ip, cmd_args.server_port))
        main_server_socket.listen(cmd_args.num_clients + 5) # Allow some buffer for connections
        print(f'Server: Listening on {cmd_args.server_ip}:{cmd_args.server_port}', flush=True)
    except Exception as e:
        print(f'Server FATAL: Could not bind: {e}', flush=True)
        return

    for current_round_iterator in range(cmd_args.num_rounds):
        current_fl_round_num = current_round_iterator + 1
        round_start_timestamp = time.time()
        print(f'\n-- Round {current_fl_round_num}/{cmd_args.num_rounds} (Mode: {cmd_args.mode}) --', flush=True)
        if current_lambda_groups_state_server and cmd_args.mode in ['mwr', 'mwr_thresh']:
            print(f'Server Lambdas (start of R{current_fl_round_num}): {current_lambda_groups_state_server}', flush=True)

        # Clear updates from previous round
        with client_updates_access_lock:
            client_updates_for_current_round.clear()

        active_client_handler_threads = []
        num_clients_connected_for_round = 0
        # Set timeout for accepting new connections
        main_server_socket.settimeout(cmd_args.client_connection_timeout)

        print(f'Server: Waiting for {cmd_args.num_clients} clients (connection timeout: {cmd_args.client_connection_timeout}s)...', flush=True)
        while num_clients_connected_for_round < cmd_args.num_clients:
            try:
                client_comm_socket_new, client_inet_address_new = main_server_socket.accept()
                num_clients_connected_for_round += 1
                client_thread = threading.Thread(target=handle_client_connection,
                                                 args=(client_comm_socket_new, client_inet_address_new,
                                                       cmd_args, current_fl_round_num),
                                                 daemon=True)
                active_client_handler_threads.append(client_thread)
                client_thread.start()
            except socket.timeout:
                print(f'Server: Connection Timeout. Connected {num_clients_connected_for_round}/{cmd_args.num_clients} clients for this round.', flush=True)
                break
            except Exception as e:
                print(f'Server ERROR: Accepting new connection: {e}', flush=True)
                break
        # Remove connection timeout for the rest of the round processing
        main_server_socket.settimeout(None)

        # Wait for client submissions
        submission_start_timestamp = time.time()
        last_printed_submission_count = -1
        last_print_time = time.time()
        PRINT_LIVENESS_INTERVAL_SECONDS = 1.0

        while (time.time() - submission_start_timestamp) < cmd_args.round_timeout:
            current_loop_time = time.time()
            with client_updates_access_lock:
                current_submission_count = len(client_updates_for_current_round)

            # If all connected clients have submitted, break
            if num_clients_connected_for_round > 0 and current_submission_count >= num_clients_connected_for_round:
                if current_submission_count != last_printed_submission_count or last_printed_submission_count == -1: 
                    print(f'\rServer: Submissions R{current_fl_round_num}: {current_submission_count}/{num_clients_connected_for_round}. Elapsed: {int(current_loop_time - submission_start_timestamp)}s   ', end="", flush=True)
                break

            # Print if submission count changed or liveness interval passed
            if current_submission_count != last_printed_submission_count or \
               (current_loop_time - last_print_time) >= PRINT_LIVENESS_INTERVAL_SECONDS:
                print(f'\rServer: Submissions R{current_fl_round_num}: {current_submission_count}/{num_clients_connected_for_round}. Elapsed: {int(current_loop_time - submission_start_timestamp)}s   ', end="", flush=True)
                last_printed_submission_count = current_submission_count
                last_print_time = current_loop_time

            time.sleep(0.2)
        print()

        # Wait for handler threads to complete
        for handler_thread in active_client_handler_threads:
            handler_thread.join(timeout=10)

        list_of_client_reports_this_round = []
        list_of_model_state_dicts_this_round = []
        with client_updates_access_lock:
            if not client_updates_for_current_round and num_clients_connected_for_round > 0 :
                 print(f'Server Warn: No client updates successfully processed, though {num_clients_connected_for_round} clients connected.', flush=True)
            for client_update_item in client_updates_for_current_round:
                list_of_client_reports_this_round.append(client_update_item['report_data'])
                list_of_model_state_dicts_this_round.append(client_update_item['model_state_dict'])

        print(f'Server: Processed updates from {len(list_of_model_state_dicts_this_round)} clients for R{current_fl_round_num}.', flush=True)

        # Update lambdas if in MWR mode
        if cmd_args.mode in ['mwr', 'mwr_thresh'] and current_lambda_groups_state_server and list_of_client_reports_this_round:
            current_lambda_groups_state_server = update_lambda_groups_on_server(
                current_lambda_groups_state_server, list_of_client_reports_this_round, cmd_args.eta_mu_lambda_update
            )
            log_server_lambdas_data_to_file(current_fl_round_num, current_lambda_groups_state_server, cmd_args)

        # Aggregate models
        if cmd_args.mode != 'no_fl_placeholder' and list_of_model_state_dicts_this_round:
            new_aggregated_global_weights = aggregate_models_from_updates(list_of_model_state_dicts_this_round)
            if new_aggregated_global_weights:
                global_model_state_dict_server = new_aggregated_global_weights
                print(f'Server: New global model aggregated.', flush=True)
            else:
                print(f'Server Warn: Aggregation failed. Global model not updated.', flush=True)
        elif not list_of_model_state_dicts_this_round and cmd_args.mode != 'no_fl_placeholder':
             print(f'Server: No client models to aggregate this round.', flush=True)

        print(f'-- Round {current_fl_round_num} completed in {(time.time() - round_start_timestamp):.2f}s --', flush=True)

    # Final outputs after all rounds
    if current_lambda_groups_state_server and cmd_args.mode in ['mwr', 'mwr_thresh']:
        print(f'Server: Final lambda groups: {current_lambda_groups_state_server}', flush=True)

    print("\n-- Final Client Metrics Summary (from last successful round's reports) --", flush=True)
    final_round_reports = []
    with client_updates_access_lock:
        for update_item in client_updates_for_current_round:
            final_round_reports.append(update_item['report_data'])

    if not final_round_reports:
        print(f'Server: No client reports to summarize from the final round.', flush=True)
    else:
        for report in sorted(final_round_reports, key=lambda x: x.get('client_id', -1)):
            client_id = report.get('client_id', 'UnknownClient')
            print(f'Client {client_id}:', flush=True)
            for key, value in report.items():
                if key != 'client_id':
                    if isinstance(value, float):
                        value = f'{value:.4f}'
                    print(f'  {key}: {value}', flush=True)

    main_server_socket.close()
    print(f'Server: Socket closed. All {cmd_args.num_rounds} rounds completed. Server shutting down.', flush=True)


ARGS_PARSED = None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning Server (Socket Version)')
    parser.add_argument('--mode', type=str, default='fedavg', choices=['fedavg', 'mwr', 'mwr_thresh', 'no_fl_placeholder'], help='FL mode')
    parser.add_argument('--num_rounds', type=int, default=3, help='Number of rounds')
    parser.add_argument('--num_clients', type=int, default=2, help='Number of clients expected per round')
    parser.add_argument('--local_epochs', type=int, default=1, help='Client local epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Client LR')
    parser.add_argument('--noise_std_dev', type=float, default=0.3, help='Noise std for noisy client')

    parser.add_argument('--eta_mu_lambda_update', type=float, default=0.05, help='Eta_mu for lambda updates')
    parser.add_argument('--l1_reg_lambda', type=float, default=0.00001, help='L1 Reg strength')
    parser.add_argument('--eta_mu_thresholding', type=float, default=0.2, help='Eta_mu for BTPR/WTPR thresholding')
    parser.add_argument('--threshold_adjustment_strength', type=float, default=0.1, help='Strength of threshold adjustment')

    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose server-side logging')

    parser.add_argument('--server_ip', type=str, default='0.0.0.0', help='IP address for server to bind')
    parser.add_argument('--server_port', type=int, default=4651, help='Port for server to listen on')
    parser.add_argument('--client_connection_timeout', type=int, default=60, help='Timeout for server accepting client conns (s)')
    parser.add_argument('--round_timeout', type=int, default=300, help='Max timeout for clients to submit updates in a round (s)')
    parser.add_argument('--num_meta_groups', type=int, default=2, help='Number of meta groups for lambda init')

    ARGS_PARSED = parser.parse_args()

    if ARGS_PARSED.mode == 'no_fl_placeholder':
        print('Server: Mode "no_fl_placeholder". For centralized training, run a separate script.', flush=True)
    else:
        run_server(ARGS_PARSED)