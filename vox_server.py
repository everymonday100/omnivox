import socket
import struct
import time
import random

SERVER_HOST = "127.0.0.1" 
SERVER_PORT = 9999
PACKET_FORMAT = "=BBfffffffII"

def run_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    
    connected_players = set()
    global_ammo_counter = 0

    print("=== VOXEL AUTOCANNON SERVER STARTED ===")
    print(f"Listening on {SERVER_HOST}:{SERVER_PORT}...")

    while True:
        try:
            data, client_address = server_socket.recvfrom(1024)
            
            # 🔥 ФИКС: Регистрируем игрока СРАЗУ, как только пришел ЛЮБОЙ пакет
            if client_address not in connected_players:
                connected_players.add(client_address)
                print(f"[CONNECT] Player registered: {client_address}. Total: {len(connected_players)}")

            if len(data) >= 38:
                packet_type, player_id, ox, oy, oz, vx, vy, vz, client_seed, client_ts, client_ammo = struct.unpack(PACKET_FORMAT, data[:38])
                
                # Тип 1: Выстрел
                if packet_type == 1:
                    global_ammo_counter += 1
                    server_seed = random.uniform(0.1, 1000.0)
                    server_timestamp = int(time.time() * 1000) & 0xFFFFFFFF
                    
                    approved_packet = struct.pack(
                        PACKET_FORMAT, 1, player_id, ox, oy, oz, vx, vy, vz, server_seed, server_timestamp, global_ammo_counter
                    )
                    
                    for player_address in connected_players:
                        server_socket.sendto(approved_packet, player_address)
                
                # Тип 2: Сброс стены на R
                elif packet_type == 2:
                    print(f"[RESET] Map reset requested by ID: {player_id}")
                    global_ammo_counter = 0
                    reset_packet = struct.pack(PACKET_FORMAT, 2, player_id, 0,0,0, 0,0,0, 0.0, 0, 0)
                    for player_address in connected_players:
                        server_socket.sendto(reset_packet, player_address)

        except Exception as e:
            print(f"[ERROR] Server tick error: {e}")

if __name__ == "__main__":
    run_server()
