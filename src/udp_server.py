# Function for UDP 

def send_udp_data(sock, server_address_port, data, log=False):
    """
    Sends data over UDP to a specified server and port.
    It is used to send the data over to the Unreal Engine Client. 
    
    - sock (socket): The socket object used for communication.
    - server_address_port: A tuple containing the server address and port. 
    - data: The data to send.
    - log: If the function should log information in the console
    """
    encoded_data = str.encode(str(data))
    sock.sendto(encoded_data, server_address_port)

    if log:
        print(f"[INFO] Sent {encoded_data} to {server_address_port}")