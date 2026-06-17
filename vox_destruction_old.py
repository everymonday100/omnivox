import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
import pygame
from pygame.locals import *
import moderngl
import cupy as cp
from numba import cuda
import math
import numpy as np
import socket
import struct
import threading
import queue
import time
import sys
import traceback
import win32gui
import win32con

WIDTH, HEIGHT = 800, 600
VOXEL_RES = 128  
MAX_PARTICLES = 15000 
MAX_BULLETS = 50       
BLOCK_RES = VOXEL_RES // 2

SERVER_IP = "127.0.0.1"
SERVER_PORT = 9999
PACKET_FORMAT = "=BBfffffffII"

# Физические свойства частиц
SPARK_SPEED = 2.603
SPARK_DECAY = 0.056
SPARK_CHANCE = 3.794
BLOCK_GRAV = 0.131
BLOCK_DECAY = 0.016

# --- 💎 НАСТРОЙКА БАЛЛИСТИКИ ОРУЖИЯ (ЗАПЕЧЕННЫЙ ПРЕСЕТ) 💎 ---
BULLET_CALIBER = 2.0   # Калибр: физический радиус дыры (в вокселях)
EXPLOSIVE_POWER = 3.33  # Фугасность: урон взрывной волны по макро-блокам

@cuda.jit
def omni_destructor_and_gravity_kernel(building_grid, block_hp, b_w, b_h, b_t, bullets_pos, bullets_vel, bullets_active, 
                                       caliber, explosive, reset_scene, p_pos, p_vel, p_active, p_life, p_size, p_count_array, seed):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
    if x >= VOXEL_RES or y >= VOXEL_RES or z >= VOXEL_RES: return
    bx, by, bz = x // 2, y // 2, z // 2
    
    x_min, x_max = (VOXEL_RES / 2.0) - b_t / 2.0, (VOXEL_RES / 2.0) + b_t / 2.0
    y_min, y_max = 1.0, 1.0 + b_h  
    z_min, z_max = (VOXEL_RES / 2.0) - b_w / 2.0, (VOXEL_RES / 2.0) + b_w / 2.0

    if reset_scene:
        is_wall = (x_min <= x <= x_max) and (y_min <= y <= y_max) and (z_min <= z <= z_max)
        is_floor_ceil = (y == 1 or y == VOXEL_RES-2) and (x % 16 == 0 or z % 16 == 0)
        
        if is_wall or is_floor_ceil:
            building_grid[x, y, z] = 1.0 if is_wall else 0.15 
            block_hp[bx, by, bz] = 4.0 if is_wall else 999.0  
        else:
            building_grid[x, y, z] = 0.0
            block_hp[bx, by, bz] = 0.0
        return

    if block_hp[bx, by, bz] <= 0.0:
        building_grid[x, y, z] = 0.0
        return

    hash_val = (x * 73129 + y * 95121 + z * 15413 + int(seed * 1000))
    raw_rand = hash_val % 100

    for b_idx in range(MAX_BULLETS):
        if bullets_active[b_idx] < 0.5: continue
        b_x, b_y, b_z = bullets_pos[b_idx, 0], bullets_pos[b_idx, 1], bullets_pos[b_idx, 2]
        v_x, v_y, v_z = bullets_vel[b_idx, 0], bullets_vel[b_idx, 1], bullets_vel[b_idx, 2]

        # Оптимальные 35 шагов субстеппинга для честного пробития на дистанции
        for step in range(35):
            t = float(step) * 0.08
            dmg_x, dmg_y, dmg_z = b_x + v_x * t, b_y + v_y * t, b_z + v_z * t
            dx, dy, dz = x - dmg_x, y - dmg_y, z - dmg_z
            dmg_dist_sq = dx*dx + dy*dy + dz*dz

            # Эпицентр калибра пули (прямой урон)
            if dmg_dist_sq < (caliber * caliber):
                if block_hp[bx, by, bz] < 500.0: 
                    # Наносим урон, масштабированный фугасностью снаряда
                    block_hp[bx, by, bz] = max(0.0, block_hp[bx, by, bz] - 0.150 * explosive) 
            
            # Зона фугасного поражения (взрывная сопутствующая волна)
            elif dmg_dist_sq < (caliber * caliber * 5.0):
                if block_hp[bx, by, bz] < 500.0:
                    block_hp[bx, by, bz] = max(0.0, block_hp[bx, by, bz] - 0.045 * explosive)

            if block_hp[bx, by, bz] <= 0.0:
                building_grid[x, y, z] = 0.0
                if x_max - 1.0 <= dmg_x <= x_max + 1.0 and v_x < 0.0:
                    bullets_vel[b_idx, 0] = -v_x * 0.6 
                    bullets_vel[b_idx, 1] *= 0.9
                    bullets_vel[b_idx, 2] *= 0.9
                if float(raw_rand) < SPARK_CHANCE:
                    idx = cuda.atomic.add(p_count_array, 0, 1)
                    if idx < MAX_PARTICLES:
                        p_pos[idx, 0], p_pos[idx, 1], p_pos[idx, 2] = float(x), float(y), float(z)
                        spark_hash = hash_val + b_idx * 997
                        cone_angle = float(spark_hash % 360) * 0.0174533
                        cone_radius = float((spark_hash // 3) % 25) * 0.14 
                        p_vel[idx, 0] = -SPARK_SPEED + (dx / (math.sqrt(dmg_dist_sq) + 0.1)) * 0.8  
                        p_vel[idx, 1] = (math.sin(cone_angle) * cone_radius) + 0.35 
                        p_vel[idx, 2] = (math.cos(cone_angle) * cone_radius)
                        p_active[idx], p_life[idx] = 1.0, 1.0
                        p_size[idx, 0] = 1.0 + float(hash_val % 3)          
                        p_size[idx, 1] = 1.0 + float((hash_val // 3) % 2)   
                        p_size[idx, 2] = 1.0                                
                break

    if block_hp[bx, by, bz] > 0.0 and block_hp[bx, by, bz] < 500.0:
        building_grid[x, y, z] = block_hp[bx, by, bz] / 4.0

    if y > 1.0 and block_hp[bx, by, bz] > 2.8 and block_hp[bx, by, bz] < 500.0:
        has_support = False
        if by > 0 and block_hp[bx, by - 1, bz] > 1.0: has_support = True
        if not has_support and bx > 0 and block_hp[bx - 1, by, bz] > 1.0: has_support = True
        if not has_support and bx < (BLOCK_RES - 1) and block_hp[bx + 1, by, bz] > 1.0: has_support = True
        if not has_support and bz > 0 and block_hp[bx, by, bz - 1] > 1.0: has_support = True
        if not has_support and bz < (BLOCK_RES - 1) and block_hp[bx, by, bz + 1] > 1.0: has_support = True
        if not has_support:
            if raw_rand < 35: 
                block_hp[bx, by, bz] = 0.0
                building_grid[x, y, z] = 0.0 
                idx = cuda.atomic.add(p_count_array, 0, 1)
                if idx < MAX_PARTICLES:
                    p_pos[idx, 0], p_pos[idx, 1], p_pos[idx, 2] = float(x), float(y), float(z)
                    p_vel[idx, 0] = float((hash_val % 11) - 5) * 0.03
                    p_vel[idx, 1] = -0.32 
                    p_vel[idx, 2] = float((hash_val % 13) - 6) * 0.03
                    p_active[idx], p_life[idx] = 1.0, 0.95
                    p_size[idx, 0] = 2.0 + float(hash_val % 2)
                    p_size[idx, 1] = 1.0 + float((hash_val // 2) % 2)
                    p_size[idx, 2] = 1.0
@cuda.jit
def update_particles_physics_kernel(p_pos, p_vel, p_active, p_life, p_size, count, building_grid, b_t): 
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= count or p_active[idx] < 0.5: return
    ox, oy, oz = int(p_pos[idx, 0]), int(p_pos[idx, 1]), int(p_pos[idx, 2])
    sz_x, sz_y, sz_z = int(p_size[idx, 0]), int(p_size[idx, 1]), int(p_size[idx, 2])

    if sz_x == 1:
        if 0 <= ox < VOXEL_RES and 0 <= oy < VOXEL_RES and 0 <= oz < VOXEL_RES:
            if building_grid[ox, oy, oz] <= 0.45: building_grid[ox, oy, oz] = 0.0
    else:
        for sx in range(sz_x):
            for sy in range(sz_y):
                for sz_d in range(sz_z):
                    v_x, v_y, v_z = ox + sx, oy + sy, oz + sz_d
                    if 0 <= v_x < VOXEL_RES and 0 <= v_y < VOXEL_RES and 0 <= v_z < VOXEL_RES:
                        if building_grid[v_x, v_y, v_z] <= 0.45: building_grid[v_x, v_y, v_z] = 0.0

    front_plane_x = (VOXEL_RES / 2.0) + (b_t / 2.0)
    is_smoke_zone = p_pos[idx, 0] > front_plane_x
    p_life[idx] -= SPARK_DECAY if is_smoke_zone else BLOCK_DECAY  
    if p_life[idx] <= 0.0:
        p_active[idx] = 0.0
        return

    p_vel[idx, 0] *= 0.96  
    p_vel[idx, 1] *= 0.98 if is_smoke_zone else 0.96
    if not is_smoke_zone: p_vel[idx, 1] -= BLOCK_GRAV 
    p_vel[idx, 2] *= 0.96  

    nx, ny, nz = p_pos[idx, 0] + p_vel[idx, 0], p_pos[idx, 1] + p_vel[idx, 1], p_pos[idx, 2] + p_vel[idx, 2]
    if ny < 1.0:
        ny = 1.0
        p_vel[idx, 1] = -p_vel[idx, 1] * 0.35  
        p_vel[idx, 0] *= 0.6                   
        p_vel[idx, 2] *= 0.6

    if nx < 0 or nx >= VOXEL_RES or ny >= VOXEL_RES or nz < 0 or nz >= VOXEL_RES:
        p_active[idx] = 0.0
        return

    rx, ry, rz = int(nx), int(ny), int(nz)
    if 0 <= rx < VOXEL_RES and 0 <= ry < VOXEL_RES and 0 <= rz < VOXEL_RES:
        if building_grid[rx, ry, rz] > 0.8:
            p_vel[idx, 0] = -p_vel[idx, 0] * 0.4 
            p_vel[idx, 1] *= 0.8 
            p_vel[idx, 2] *= 0.8
            nx, ny = p_pos[idx, 0] + p_vel[idx, 0], p_pos[idx, 1] + p_vel[idx, 1]

    p_pos[idx, 0], p_pos[idx, 1], p_pos[idx, 2] = nx, ny, nz
    rx, ry, rz = int(nx), int(ny), int(nz)

    if sz_x == 1:
        if 0 <= rx < VOXEL_RES and 0 <= ry < VOXEL_RES and 0 <= rz < VOXEL_RES:
            if building_grid[rx, ry, rz] <= 0.8:
                building_grid[rx, ry, rz] = min(0.04, building_grid[rx, ry, rz] + 0.03) if is_smoke_zone else min(0.45, building_grid[rx, ry, rz] + 0.20 * p_life[idx])
    else:
        for sx in range(sz_x):
            for sy in range(sz_y):
                for sz_d in range(sz_z):
                    v_x, v_y, v_z = rx + sx, ry + sy, rz + sz_d
                    if 0 <= v_x < VOXEL_RES and 0 <= v_y < VOXEL_RES and 0 <= v_z < VOXEL_RES:
                        if building_grid[v_x, v_y, v_z] > 0.8: continue
                        building_grid[v_x, v_y, v_z] = min(0.45, building_grid[v_x, v_y, v_z] + 0.35 * p_life[idx])

VERTEX_SHADER = """
#version 330
in vec2 in_vert;
out vec2 uvs;
void main() {
    uvs = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
in vec2 uvs;
out vec4 fragColor;
uniform sampler3D volumeTex; 
uniform float camAngleX; uniform float camAngleY;     
uniform vec3 camPos;     
const int MAX_BULLETS = 50;
uniform vec3 bulletsPos[MAX_BULLETS]; uniform float bulletsActive[MAX_BULLETS];

bool intersectBox(vec3 ro, vec3 rd, out float t0, out float t1) {
    vec3 invR = 1.0 / (rd + 1e-6);
    vec3 tbot = invR * (vec3(0.0) - ro); vec3 ttop = invR * (vec3(1.0) - ro);
    vec3 tmin = min(tbot, ttop); vec3 tmax = max(tbot, ttop);
    t0 = max(tmin.x, max(tmin.y, tmin.z)); t1 = min(tmax.x, min(tmax.y, tmax.z));
    return t0 < t1 && t1 > 0.0;
}
void main() {
    vec3 ro = camPos;
    vec3 direction = vec3(
        cos(camAngleY) * sin(camAngleX),
        sin(camAngleY),
        cos(camAngleY) * cos(camAngleX)
    );
    vec3 ww = normalize(direction); vec3 uu = normalize(cross(ww, vec3(0.0, 1.0, 0.0))); vec3 vv = normalize(cross(uu, ww));
    vec2 p = uvs * 2.0 - 1.0; p.x *= 1.333; 
    vec3 rd = normalize(p.x * uu + p.y * vv + 1.5 * ww);
    float t0, t1; vec3 color = vec3(0.04, 0.05, 0.08); 
    if (intersectBox(ro, rd, t0, t1)) {
        t0 = max(t0, 0.0); vec3 pos = ro + rd * t0;
        float stepSize = 0.0025; vec3 stepDir = rd * stepSize; 
        float T = 1.0; vec3 volumeColor = vec3(0.0);
        for (int i = 0; i < 350; i++) { 
            if (t0 > t1 || T < 0.01) break;
            float density = texture(volumeTex, vec3(pos.z, pos.y, pos.x)).r;

            for (int b = 0; b < MAX_BULLETS; b++) {
                if (bulletsActive[b] > 0.5) {
                    if (distance(pos * 128.0, bulletsPos[b]) < 0.43) { 
                        volumeColor += T * vec3(4.0, 3.5, 1.5) * 0.9; T *= 0.02;
                    }
                }
            }
            if (density > 0.01) {
                vec3 voxelCol = vec3(0.4, 0.44, 0.48); float alpha = density * 0.35;
                if (density <= 0.04) { voxelCol = vec3(3.0, 3.0, 3.0); alpha = density * 3.5; } 
                else if (density <= 0.25) { voxelCol = vec3(0.0, 0.8, 1.0); alpha = 0.015; } 
                else if (density < 0.28) { voxelCol = vec3(1.0, 0.45, 0.05); } 
                else if (density < 0.44) { voxelCol = vec3(0.12, 0.13, 0.15); } 
                else { alpha = 0.018; }
                volumeColor += T * voxelCol * alpha; T *= (1.0 - alpha); 
            }
            pos += stepDir; t0 += stepSize;
        }
        color = volumeColor + T * color;
    }

    vec2 scrCoord = uvs * vec2(800.0, 600.0);
    vec2 center = vec2(400.0, 300.0);
    float distToCenterHor = abs(scrCoord.x - center.x);
    float distToCenterVer = abs(scrCoord.y - center.y);
    if ((distToCenterHor < 6.0 && distToCenterVer < 1.0) || (distToCenterVer < 6.0 && distToCenterHor < 1.0)) {
        if (distToCenterHor > 1.0 || distToCenterVer > 1.0) {
            color = vec3(0.0, 1.0, 0.2); 
        }
    }

    fragColor = vec4(color, 1.0);
}
"""
running = True
is_holding_trigger = False
server_reset_flag = False
reset_scene = True       
fire_cooldown = 0
total_ammo_fired = 0 
network_seed = 0.5 

PLAYER_ID = np.random.randint(1, 255)
net_queue = queue.Queue()
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.setblocking(False)

def incoming_packets_listener(sock, q):
    while running:
        try:
            data, _ = sock.recvfrom(1024)
            if len(data) >= 38: q.put(data[:38])
        except BlockingIOError: time.sleep(0.001)
        except Exception as e:
            log_error("Network Thread Listener Failure", e)
            break

pygame.init()
pygame.display.set_caption(f"Voxel Player {PLAYER_ID}")
pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
ctx = moderngl.create_context()

hwnd = win32gui.FindWindow(None, f"Voxel Player {PLAYER_ID}")
if hwnd:
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, 
                          win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

quad_buffer = ctx.buffer(np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4'))
prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
vao = ctx.vertex_array(prog, [(quad_buffer, '2f', 'in_vert')])

grid_cupy = cp.zeros((VOXEL_RES, VOXEL_RES, VOXEL_RES), dtype=cp.float32)
block_hp = cp.zeros((BLOCK_RES, BLOCK_RES, BLOCK_RES), dtype=cp.float32)

part_pos = cp.zeros((MAX_PARTICLES, 3), dtype=cp.float32)
part_vel = cp.zeros((MAX_PARTICLES, 3), dtype=cp.float32)
part_active = cp.zeros(MAX_PARTICLES, dtype=cp.float32)
part_life = cp.zeros(MAX_PARTICLES, dtype=cp.float32) 
part_size = cp.zeros((MAX_PARTICLES, 3), dtype=cp.float32) 
part_count = cp.zeros(1, dtype=cp.int32) 

bullets_pos = cp.zeros((MAX_BULLETS, 3), dtype=cp.float32)
bullets_vel = cp.zeros((MAX_BULLETS, 3), dtype=cp.float32)
bullets_active = cp.zeros(MAX_BULLETS, dtype=cp.float32)

texture_3d = ctx.texture3d((VOXEL_RES, VOXEL_RES, VOXEL_RES), 1, dtype='f4')
texture_3d.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

threads = (8, 8, 8)
blocks = (VOXEL_RES // 8, VOXEL_RES // 8, VOXEL_RES // 8)

wall_width, wall_height, wall_thickness = 88.0, 60.0, 8.0 

# Высота глаз 0.22 (уполовиненный рост)
cam_x, cam_y, cam_z = 0.5, 0.22, 0.78 
camera_angle_x, camera_angle_y = 0.0, 0.0
move_speed = 0.015

try:
    client_socket.sendto(struct.pack(PACKET_FORMAT, 0, PLAYER_ID, 0,0,0, 0,0,0, 0.0, 0, 0), (SERVER_IP, SERVER_PORT))
except Exception as e:
    log_error("Initial Server Connection Registration Failure", e)

net_thread = threading.Thread(target=incoming_packets_listener, args=(client_socket, net_queue), daemon=True)
net_thread.start()

pygame.mouse.set_visible(True)
pygame.event.set_grab(False)
clock = pygame.time.Clock()

while running:
    while not net_queue.empty():
        packet = net_queue.get()
        p_type, p_uid, ox, oy, oz, vx, vy, vz, r_seed, r_ts, r_ammo = struct.unpack(PACKET_FORMAT, packet)
        if p_type == 1: 
            for b_idx in range(MAX_BULLETS):
                if bullets_active[b_idx] < 0.5:
                    bullets_active[b_idx] = 1.0
                    bullets_pos[b_idx, 0], bullets_pos[b_idx, 1], bullets_pos[b_idx, 2] = ox, oy, oz
                    bullets_vel[b_idx, 0], bullets_vel[b_idx, 1], bullets_vel[b_idx, 2] = vx, vy, vz
                    network_seed = r_seed 
                    break
        elif p_type == 2: server_reset_flag = True

    mouse_click = pygame.mouse.get_pressed()
    if mouse_click: 
        rel_x, rel_y = pygame.mouse.get_rel()
        camera_angle_x -= rel_x * 0.004 
        camera_angle_y = max(-1.4, min(1.4, camera_angle_y - rel_y * 0.004))
    else:
        pygame.mouse.get_rel() 

    keys = pygame.key.get_pressed()
    forward_x = math.sin(camera_angle_x)
    forward_z = math.cos(camera_angle_x)
    
    if keys[K_w]: cam_x += forward_x * move_speed; cam_z += forward_z * move_speed
    if keys[K_s]: cam_x -= forward_x * move_speed; cam_z -= forward_z * move_speed
    if keys[K_a]: cam_x += forward_z * move_speed; cam_z -= forward_x * move_speed
    if keys[K_d]: cam_x -= forward_z * move_speed; cam_z += forward_x * move_speed

    cam_x = max(0.05, min(0.95, cam_x))
    cam_z = max(0.05, min(0.95, cam_z))

    for event in pygame.event.get():
        if event.type == QUIT: running = False
        if event.type == KEYDOWN:
            if event.key == K_r:
                try:
                    req = struct.pack(PACKET_FORMAT, 2, PLAYER_ID, 0,0,0, 0,0,0, 0.0, 0, 0)
                    client_socket.sendto(req, (SERVER_IP, SERVER_PORT))
                except Exception as e: log_error("Reset Packet Send Error", e)
            elif event.key == K_ESCAPE: running = False
        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 1: is_holding_trigger = True
        elif event.type == MOUSEBUTTONUP:
            if event.button == 1: is_holding_trigger = False

    if is_holding_trigger:
        if fire_cooldown <= 0:
            ox_req = cam_x * 128.0
            oy_req = cam_y * 128.0
            oz_req = cam_z * 128.0
            
            vx_req = math.cos(camera_angle_y) * math.sin(camera_angle_x) * 110.0
            vy_req = math.sin(camera_angle_y) * 110.0
            vz_req = math.cos(camera_angle_y) * math.cos(camera_angle_x) * 110.0
            
            try:
                shot_request = struct.pack(PACKET_FORMAT, 1, PLAYER_ID, ox_req, oy_req, oz_req, vx_req, vy_req, vz_req, 0.0, 0, 0)
                client_socket.sendto(shot_request, (SERVER_IP, SERVER_PORT))
                total_ammo_fired += 1
                fire_cooldown = 3 
            except Exception as e:
                log_error("Ballistic Vector Packet Assembly/Send Failure", e)
    
    if fire_cooldown > 0: fire_cooldown -= 1

    bullets_pos += bullets_vel * 0.035 * bullets_active[:, np.newaxis]
    bullets_active[(bullets_pos[:, 0] < 0.0) | (bullets_pos[:, 0] > 128.0) | (bullets_pos[:, 1] < 0.0) | (bullets_pos[:, 1] > 128.0) | (bullets_pos[:, 2] < 0.0) | (bullets_pos[:, 2] > 128.0)] = 0.0

    actual_reset = reset_scene or server_reset_flag
    if server_reset_flag:
        bullets_active.fill(0); part_active.fill(0); part_life.fill(0); part_size.fill(0); part_count.fill(0)
        total_ammo_fired = 0; server_reset_flag = False

    # В ядро передаются независимые BULLET_CALIBER и EXPLOSIVE_POWER вместо общего радиуса взрыва
    omni_destructor_and_gravity_kernel[blocks, threads](
        grid_cupy, block_hp, wall_width, wall_height, wall_thickness, bullets_pos, bullets_vel, bullets_active, 
        BULLET_CALIBER, EXPLOSIVE_POWER, actual_reset, part_pos, part_vel, part_active, part_life, part_size, part_count, network_seed
    )
    reset_scene = False

    current_p_count = int(part_count.item()) 
    if current_p_count > 0:
        p_blocks = math.ceil(min(current_p_count, MAX_PARTICLES) / 256)
        update_particles_physics_kernel[p_blocks, 256](
            part_pos, part_vel, part_active, part_life, part_size, min(current_p_count, MAX_PARTICLES), grid_cupy, wall_thickness
        )
        cp.cuda.stream.get_current_stream().synchronize()

    texture_3d.write(grid_cupy.tobytes())
    texture_3d.build_mipmaps()

    ctx.clear(0.0, 0.0, 0.0)
    texture_3d.use(location=0)
    
    prog['volumeTex'].value = 0
    prog['camAngleX'].value = camera_angle_x
    prog['camAngleY'].value = camera_angle_y
    prog['camPos'].value = (cam_x, cam_y, cam_z)
    prog['bulletsPos'].write(bullets_pos.tobytes())
    prog['bulletsActive'].write(bullets_active.tobytes())

    vao.render(moderngl.TRIANGLE_STRIP)
    pygame.display.flip()
    
    current_fps = int(clock.get_fps())
    pygame.display.set_caption(f"Player {PLAYER_ID} | FPS: {current_fps} | AMMO: {total_ammo_fired}")
    clock.tick(60)

pygame.quit()
