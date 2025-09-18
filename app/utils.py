import socket
import struct
import json
import io
import torch


def socket_error_handler_decorator(func_to_decorate):
    # For debugging and logging purpose
    def wrapper_function(sock: socket.socket, *args, **kwargs):
        try:
            return func_to_decorate(sock, *args, **kwargs)
        except socket.timeout:
            print(f"UTILS: Socket timeout in {func_to_decorate.__name__} for {sock.getpeername() if hasattr(sock, 'getpeername') else sock}", flush=True)
            raise 
        except ConnectionError as e: 
            print(f"UTILS: ConnectionError in {func_to_decorate.__name__} for {sock.getpeername() if hasattr(sock, 'getpeername') else sock}: {e}", flush=True)
            raise
        except socket.error as e: 
            print(f"UTILS: Socket.error in {func_to_decorate.__name__} for {sock.getpeername() if hasattr(sock, 'getpeername') else sock}: {e}", flush=True)
            raise
        except Exception as e: 
            print(f"UTILS: UNEXPECTED error in {func_to_decorate.__name__} for {sock.getpeername() if hasattr(sock, 'getpeername') else sock}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
    return wrapper_function
        

@socket_error_handler_decorator
def send_msg(sock: socket.socket, msg_type_str: str, data=None) -> None:
    """
    Sends a structured message over a socket.
    Message format: Type (10 bytes) | DataType (5 bytes) | PayloadLength (4 bytes) | Payload
    msg_type_str: String, truncated/padded to 10 bytes.
    data: dict/list (for JSON), bytes, or other simple JSON-serializable types.
    """
    payload_bytes = b''
    data_type_field_str = 'none'

    if data is not None:
        if isinstance(data, (dict, list)):
            payload_bytes = json.dumps(data).encode('utf-8')
            data_type_field_str = 'json'
        elif isinstance(data, bytes):
            payload_bytes = data
            data_type_field_str = 'bytes'
        else: 
            try:
                payload_bytes = json.dumps(data).encode('utf-8') # Handles str directly, others via default JSON
                data_type_field_str = 'json'
            except TypeError: 
                 raise TypeError(f"Data for send_msg must be dict, list, bytes, or JSON-serializable. Got {type(data)}")
    
    msg_type_field_processed = msg_type_str[:10].ljust(10) 
    data_type_field_processed = data_type_field_str[:5].ljust(5)

    msg_type_header_bytes = msg_type_field_processed.encode('utf-8')
    data_type_header_bytes = data_type_field_processed.encode('utf-8')
    payload_len_header_bytes = struct.pack('>I', len(payload_bytes))

    full_message_to_send = msg_type_header_bytes + data_type_header_bytes + payload_len_header_bytes + payload_bytes

    sock.sendall(full_message_to_send) 
    

@socket_error_handler_decorator
def receive_msg(sock: socket.socket) -> tuple[str, any]:
    """
    Receives a structured message from a socket.
    Returns: (msg_type_str, data_payload)
    data_payload can be dict/list (if JSON), bytes, or None.
    """
    header_buffer = b''
    # msg_type(10) + data_type(5) + payload_len(4)
    header_total_len = 19 
    bytes_received_for_header = 0

    while bytes_received_for_header < header_total_len:
        remaining_header_bytes_to_recv = header_total_len - bytes_received_for_header
        chunk = sock.recv(remaining_header_bytes_to_recv) 
        if not chunk:
            raise ConnectionError("Socket connection broken while receiving header (empty chunk)")
        header_buffer += chunk
        bytes_received_for_header += len(chunk)
    
    msg_type_from_header = header_buffer[:10].decode('utf-8').strip() 
    data_type_from_header = header_buffer[10:15].decode('utf-8').strip()
    payload_actual_len = struct.unpack('>I', header_buffer[15:19])[0]

    payload_buffer = b''
    if payload_actual_len > 0:
        bytes_received_for_payload = 0
        while bytes_received_for_payload < payload_actual_len:
            remaining_payload_bytes_to_recv = payload_actual_len - bytes_received_for_payload
            chunk_size_for_payload_recv = min(remaining_payload_bytes_to_recv, 4096) 
            chunk = sock.recv(chunk_size_for_payload_recv)
            if not chunk: 
                raise ConnectionError("Socket connection broken while receiving payload (empty chunk)")
            payload_buffer += chunk
            bytes_received_for_payload += len(chunk)
            
    if data_type_from_header == 'json':
        try:
            if not payload_buffer: 
                return msg_type_from_header, None 
            return msg_type_from_header, json.loads(payload_buffer.decode('utf-8'))
        except json.JSONDecodeError as e:
            print(f"UTILS_JSON_DECODE_ERROR for msg_type '{msg_type_from_header}', payload (first 100 bytes): {payload_buffer[:100]}... Error: {e}", flush=True)
            raise 
    elif data_type_from_header == 'bytes':
        return msg_type_from_header, payload_buffer
    elif data_type_from_header == 'none':
        return msg_type_from_header, None
    else:
        print(f"UTILS_WARNING: Unknown data_type '{data_type_from_header}' received for msg_type '{msg_type_from_header}'. Returning raw payload.", flush=True)
        return msg_type_from_header, payload_buffer


@socket_error_handler_decorator
def send_model_dict(sock: socket.socket, state_dict: dict) -> None:
    """Serializes a PyTorch state_dict to bytes and sends it using send_msg with type 'MODELDICT'."""
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0) 
    model_bytes = buffer.read()
    send_msg(sock, "MODELDICT", model_bytes) 


@socket_error_handler_decorator
def receive_model_dict(sock: socket.socket, target_device: str = 'cpu') -> dict:
    """Receives a message expected to be 'MODELDICT' and loads the state_dict."""
    msg_type_received, model_bytes_payload = receive_msg(sock) 
    
    if msg_type_received != 'MODELDICT': 
        raise ValueError(f"Expected msg_type 'MODELDICT', got '{msg_type_received}'")
    if not isinstance(model_bytes_payload, bytes):
        raise ValueError(f"Expected model_bytes_payload to be bytes for MODELDICT, got {type(model_bytes_payload)}")
    
    buffer = io.BytesIO(model_bytes_payload)
    buffer.seek(0)
    state_dict_loaded = torch.load(buffer, map_location=torch.device(target_device))
    return state_dict_loaded