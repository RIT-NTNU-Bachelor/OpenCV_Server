# Import packages for viewing the image, sending the data and tracking the face
from datetime import datetime
import time
import socket
import csv

# Importing the UDP Function for transmitting data
from udp_server import send_udp_data

# Setup for the information for the UDP server. 
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address_port = ('127.0.0.1', 5052)

def main():
    send_index = 1

    # Open the CSV file in append mode and create a CSV writer
    with open('mock_server.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Never ending loop
        while True:
            # Get current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            
            # Construct the mock_face data with the timestamp
            mock_face = [100, 100, 50, send_index]  # Convert numerical values to strings
            # Write the mock_face data to the CSV file

            csv_writer.writerow([send_index, timestamp])  # Prepend timestamp to the mock_face data
            # Write the mock_face data to the CSV file

            send_index = send_index + 1
            send_udp_data(sock, server_address_port, mock_face, log=True)  # Exclude timestamp from data sent via UDP
            time.sleep(0.2)


# Run the main function when the file is run
if __name__ == "__main__":
    main()