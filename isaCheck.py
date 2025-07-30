from isabelle_client import start_isabelle_server
from isabelle_client import get_isabelle_client

import logging
import json
import pickle
import os

if __name__ == "__main__":

    # first, we start Isabelle server
    server_info, _ = start_isabelle_server(
        name="test", port=9999, log_file="server.log"
    )
    # then we create Python client to Isabelle server
    isabelle = get_isabelle_client(server_info)
    # we will log all the messages from the server to a file
    isabelle.logger = logging.getLogger()
    isabelle.logger.setLevel(logging.INFO)
    isabelle.logger.addHandler(logging.FileHandler("session.log"))
    
    list_file_path = "thy.full"

    try:
        # Read the main file list, filtering out blank lines
        with open(list_file_path, 'r') as file:
            file_paths = [line.strip() for line in file if line.strip()]

    except FileNotFoundError:
        print(f"Error: The list file '{list_file_path}' was not found.")
        exit
    except Exception as e:
        print(f"An error occurred while reading the list file: {e}")
        exit

    print("--- Starting to Process Files from List ---")

    # Iterate through each file path in the list
    for path in file_paths:
        try:
            if os.path.exists(path):
                print(f"\n[INFO] checking '{path}'...")
                isabelle.use_theories(theories=[f"{path[:-4]}"], master_dir=".", watchdog_timeout=0)
            else:
                print(f"\n[WARNING] File not found: '{path}'. Skipping this entry.")

        except Exception as e:
            print(f"\n[ERROR] An error occurred while reading '{path}': {e}")

    print("\n--- Finished Processing All Files ---")

    isabelle.shutdown()
