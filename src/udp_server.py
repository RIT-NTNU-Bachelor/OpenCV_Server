import socket

def send_udp_data(sock:socket, server_address_port: tuple, data, log=False):
    """Sends data over UDP to a specified server and port.

    Parameters:
        sock (socket.socket): The socket object used for UDP communication.
        server_address_port (tuple): A tuple containing the server address and port, formatted as (address, port).
        data (Any): The data to be sent over UDP. This can be any data type that is serializable and supported by the socket.
        log (bool): If set to True, the function logs the details of the sent data to the console. Defaults to False.

    Note:
        Ensure that the socket ip and port is set to match the Unreal Engine client ip and port. 
        See the code for the [Unreal Engine Client code here.](https://github.com/RIT-NTNU-Bachelor/Unreal-facetracking-client)

        Default ip and port of the client is (127.0.0.1, 5052)
    """
    encoded_data = str.encode(str(data))
    sock.sendto(encoded_data, server_address_port)

    if log:
        print(f"[INFO] Sent {encoded_data} to {server_address_port}")