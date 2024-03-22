import socket

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the port
clientAddressPort = ('127.0.0.1', 5052)  # Choose a port for the client
sock.bind(clientAddressPort)

# Receive data from the server
while True:
    data, serverAddress = sock.recvfrom(1024)  # Buffer size is 1024 bytes
    print("Received message:", data.decode())
