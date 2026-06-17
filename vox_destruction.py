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
import sys
import traceback
import time
import queue
import logging
import threading

# --- НАСТРОЙКА АВТОМАТИЧЕСКОГО ЛОГИРОВАНИЯ ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logging.critical(f"Критический сбой DDA GPU-движка:\n{error_msg}")

sys.excepthook = handle_unhandled_exception

# --- ОРИГИНАЛЬНЫЕ КОНСТАНТЫ СЦЕНЫ И ДВИЖКА ---
WIDTH, HEIGHT = 800, 600
VOXEL_RES = 256  # Повышенная сетка вокселей
BLOCK_RES = VOXEL_RES // 2

# Ограничения динамических объектов на GPU
MAX_BULLETS = 50
MAX_PARTICLES = 15000  # Твердые осколки геометрии
MAX_GAS_PARTICLES = 2048  # SPH-аттракторы газа из smoke_sim_rt

# Физика осколков из репозитория
SPARK_SPEED = 2.603
SPARK_DECAY = 0.056
SPARK_CHANCE = 3.794
BLOCK_GRAV = 0.131
BLOCK_DECAY = 0.016

# Баллистика оружия
BULLET_CALIBER = 1.0
EXPLOSIVE_POWER = 3.0

# --- СИМУЛЯЦИЯ ГАЗА (SPH/NAVIER-STOKES ИЗ SMOKE_SIM_RT) ---
GAS_GRAVITY = 0.015
GAS_DISSIPATION = 0.008
TURBULENCE_MOD = 0.45
HUMIDITY_MOD = 0.12

running = True
PLAYER_ID = np.random.randint(1, 255)

# Выделение чистой памяти во VRAM через CuPy (0% RAM / 0% CPU)
building_grid = cp.zeros((VOXEL_RES, VOXEL_RES, VOXEL_RES), dtype=cp.float32)
block_hp = cp.zeros((BLOCK_RES, BLOCK_RES, BLOCK_RES), dtype=cp.float32)

# Векторы твердых частиц (осколков)
part_pos = cp.zeros((MAX_PARTICLES, 3), dtype=cp.float32)
part_vel = cp.zeros((MAX_PARTICLES, 3), dtype=cp.float32)
part_active = cp.zeros(MAX_PARTICLES, dtype=cp.float32)
part_life = cp.zeros(MAX_PARTICLES, dtype=cp.float32)
part_size = cp.zeros((MAX_PARTICLES, 3), dtype=cp.float32)
part_count = cp.zeros(1, dtype=cp.int32)

# Векторы SPH-аттракторов газа (из smoke_sim_rt)
gas_pos = cp.zeros((MAX_GAS_PARTICLES, 3), dtype=cp.float32)
gas_vel = cp.zeros((MAX_GAS_PARTICLES, 3), dtype=cp.float32)
gas_active = cp.zeros(MAX_GAS_PARTICLES, dtype=cp.float32)
gas_life = cp.zeros(MAX_GAS_PARTICLES, dtype=cp.float32)
gas_opacity = cp.zeros(MAX_GAS_PARTICLES, dtype=cp.float32)
gas_temp = cp.zeros(MAX_GAS_PARTICLES, dtype=cp.float32)

# Локальные пули
bullets_pos = cp.zeros((MAX_BULLETS, 3), dtype=cp.float32)
bullets_vel = cp.zeros((MAX_BULLETS, 3), dtype=cp.float32)
bullets_active = cp.zeros(MAX_BULLETS, dtype=cp.float32)
# --- CUDA ВЫЧИСЛИТЕЛЬНЫЕ ЯДРА ИЗ РЕПОЗИТОРИЯ И SMOKE_SIM_RT ---

@cuda.jit
def omni_destructor_and_gravity_kernel(building_grid, block_hp, b_w, b_h, b_t, bullets_pos, bullets_vel, bullets_active,
                                      caliber, explosive, reset_scene, p_pos, p_vel, p_active, p_life, p_size, p_count_array, seed,
                                      g_pos, g_vel, g_active, g_life, gas_opacity, g_temp, g_count_array):
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
        is_floor_ceil = (y == 1 or y == VOXEL_RES - 2) and (x % 16 == 0 or z % 16 == 0)
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

        for step in range(35):
            t = float(step) * 0.08
            dmg_x, dmg_y, dmg_z = b_x + v_x * t, b_y + v_y * t, b_z + v_z * t
            dx, dy, dz = x - dmg_x, y - dmg_y, z - dmg_z
            dmg_dist_sq = dx * dx + dy * dy + dz * dz

            if dmg_dist_sq < (caliber * caliber):
                if block_hp[bx, by, bz] < 500.0:
                    block_hp[bx, by, bz] = max(0.0, block_hp[bx, by, bz] - 0.150 * explosive)
            elif dmg_dist_sq < (caliber * caliber * 5.0):
                if block_hp[bx, by, bz] < 500.0:
                    block_hp[bx, by, bz] = max(0.0, block_hp[bx, by, bz] - 0.045 * explosive)

            if block_hp[bx, by, bz] <= 0.0:
                building_grid[x, y, z] = 0.0
                
                if raw_rand < 15:
                    g_idx = cuda.atomic.add(g_count_array, 0, 1)
                    if g_idx < MAX_GAS_PARTICLES:
                        g_pos[g_idx, 0], g_pos[g_idx, 1], g_pos[g_idx, 2] = float(x), float(y), float(z)
                        g_vel[g_idx, 0] = v_x * 0.1 + ((hash_val % 7) / 7.0 - 0.5) * 2.0
                        g_vel[g_idx, 1] = v_y * 0.1 + ((hash_val % 11) / 11.0) * 3.0
                        g_vel[g_idx, 2] = v_z * 0.1 + ((hash_val % 13) / 13.0 - 0.5) * 2.0
                        g_active[g_idx], g_life[g_idx] = 1.0, 1.0
                        gas_opacity[g_idx], g_temp[g_idx] = 0.8, 1.0

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
                        p_size[idx, 0] = 2.0 + float(hash_val % 2)
                        p_size[idx, 1] = 1.0 + float((hash_val // 2) % 2)
                        p_size[idx, 2] = 1.0
                break

    if block_hp[bx, by, bz] > 0.0 and block_hp[bx, by, bz] < 500.0:
        building_grid[x, y, z] = block_hp[bx, by, bz] / 4.0

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
            p_vel[idx, 2] *= 0.4
        nx, ny = p_pos[idx, 0] + p_vel[idx, 0], p_pos[idx, 1] + p_vel[idx, 1]

    p_pos[idx, 0], p_pos[idx, 1], p_pos[idx, 2] = nx, ny, nz

@cuda.jit
def update_gas_sph_kernel(g_pos, g_vel, g_active, g_life, gas_opacity, g_temp, count, building_grid, turb, humid):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= count or g_active[idx] < 0.5: return

    g_life[idx] -= GAS_DISSIPATION
    if g_life[idx] <= 0.0:
        g_active[idx] = 0.0
        return

    g_vel[idx, 1] += GAS_GRAVITY * g_temp[idx]
    g_vel[idx, 0] *= (1.0 - turb * 0.05)
    g_vel[idx, 1] *= (0.95 + humid * 0.02)
    g_vel[idx, 2] *= (1.0 - turb * 0.05)

    nx, ny, nz = g_pos[idx, 0] + g_vel[idx, 0], g_pos[idx, 1] + g_vel[idx, 1], g_pos[idx, 2] + g_vel[idx, 2]

    if nx < 0 or nx >= VOXEL_RES or ny < 0 or ny >= VOXEL_RES or nz < 0 or nz >= VOXEL_RES:
        g_active[idx] = 0.0
        return

    g_pos[idx, 0], g_pos[idx, 1], g_pos[idx, 2] = nx, ny, nz
    rx, ry, rz = int(nx), int(ny), int(nz)

    if building_grid[rx, ry, rz] < 0.1:
        building_grid[rx, ry, rz] = min(0.04, building_grid[rx, ry, rz] + 0.015 * g_life[idx])
# --- ШЕЙДЕРЫ ИЗ РЕПОЗИТОРИЯ С АППАРАТНЫМ DDA, СГЛАЖИВАНИЕМ, ЛАМПОЙ И SSBO ---
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
#version 430 core
in vec2 uvs;
out vec4 fragColor;

uniform sampler3D volumeTex;
uniform float camAngleX;
uniform float camAngleY;
uniform vec3 camPos;

const int MAX_BULLETS = 50;

// ИСПОЛЬЗУЕМ АППАРАТНЫЕ СТРУКТУРНЫЕ SSBO БУФЕРЫ GPU ВМЕСТО МЕДЛЕННЫХ UNIFORM
layout(std430, binding = 1) buffer BulletsPosBuffer {
    vec4 bulletsPos[MAX_BULLETS]; // Используем vec4 для строгого выравнивания std430
};

layout(std430, binding = 2) buffer BulletsActiveBuffer {
    float bulletsActive[MAX_BULLETS];
};

bool intersectBox(vec3 ro, vec3 rd, out float t0, out float t1) {
    vec3 invR = 1.0 / (rd + vec3(1e-6));
    vec3 tbot = invR * (vec3(0.0) - ro); vec3 ttop = invR * (vec3(256.0) - ro);
    vec3 tmin = min(tbot, ttop); vec3 tmax = max(tbot, ttop);
    t0 = max(tmin.x, max(tmin.y, tmin.z)); t1 = min(tmax.x, min(tmax.y, tmax.z));
    return t0 < t1 && t1 > 0.0;
}

void main() {
    vec3 ro = camPos * 256.0; // Масштаб под сетку 256
    vec3 direction = vec3(
        cos(camAngleY) * sin(camAngleX),
        sin(camAngleY),
        cos(camAngleY) * cos(camAngleX)
    );
    vec3 ww = normalize(direction); vec3 uu = normalize(cross(ww, vec3(0.0, 1.0, 0.0))); vec3 vv = normalize(cross(uu, ww));
    vec2 p = uvs * 2.0 - 1.0; p.x *= 1.333;
    vec3 rd = normalize(p.x * uu + p.y * vv + 1.5 * ww);
    
    vec3 lightPos = vec3(128.0, 240.0, 128.0); // Пересчитано под сетку 256
    vec3 lightColor = vec3(1.0, 0.62, 0.28); 
    
    float t0, t1; vec3 color = vec3(0.04, 0.05, 0.08);
    if (intersectBox(ro, rd, t0, t1)) {
        t0 = max(t0, 0.0); vec3 pos = ro + rd * t0;
        ivec3 mapPos = ivec3(floor(pos));
        
        vec3 deltaDist = abs(1.0 / (rd + vec3(1e-6)));
        ivec3 step; vec3 sideDist;
        
        if (rd.x < 0.0) { step.x = -1; sideDist.x = (pos.x - float(mapPos.x)) * deltaDist.x; }
        else { step.x = 1; sideDist.x = (float(mapPos.x) + 1.0 - pos.x) * deltaDist.x; }
        if (rd.y < 0.0) { step.y = -1; sideDist.y = (pos.y - float(mapPos.y)) * deltaDist.y; }
        else { step.y = 1; sideDist.y = (float(mapPos.y) + 1.0 - pos.y) * deltaDist.y; }
        if (rd.z < 0.0) { step.z = -1; sideDist.z = (pos.z - float(mapPos.z)) * deltaDist.z; }
        else { step.z = 1; sideDist.z = (float(mapPos.z) + 1.0 - pos.z) * deltaDist.z; }
        
        float T = 1.0; vec3 volumeColor = vec3(0.0);
        int hitSide = 0; 
        
        for (int i = 0; i < 400; i++) { // 400 шагов под сетку 256
            if (mapPos.x < 0 || mapPos.x >= 256 || mapPos.y < 0 || mapPos.y >= 256 || mapPos.z < 0 || mapPos.z >= 256 || T < 0.01) break;
            
            float density = texture(volumeTex, vec3(float(mapPos.z)/256.0, float(mapPos.y)/256.0, float(mapPos.x)/256.0)).r;
            
            for (int b = 0; b < MAX_BULLETS; b++) {
                if (bulletsActive[b] > 0.5) {
                    if (distance(vec3(mapPos), bulletsPos[b].xyz) < 0.86) {
                        volumeColor += T * vec3(4.0, 3.5, 1.5) * 0.9; T *= 0.02;
                    }
                }
            }
            
            if (density > 0.01) {
                vec3 localRayPos = ro + rd * (sideDist.x - deltaDist.x);
                if (hitSide == 1) localRayPos = ro + rd * (sideDist.y - deltaDist.y);
                if (hitSide == 2) localRayPos = ro + rd * (sideDist.z - deltaDist.z);
                
                float microStep = 0.25;
                float accumulatedDensity = 0.0;
                
                for (int m = 0; m < 4; m++) {
                    vec3 sampleCoords = localRayPos / 256.0;
                    accumulatedDensity += texture(volumeTex, vec3(sampleCoords.z, sampleCoords.y, sampleCoords.x)).r * microStep;
                    localRayPos += rd * microStep;
                }
                
                if (accumulatedDensity > 0.01) {
                    vec3 voxelCol = vec3(0.4, 0.44, 0.48); 
                    float alpha = accumulatedDensity * 0.35;
                    
                    vec3 currentVoxelPos = vec3(mapPos);
                    float distToLight = distance(currentVoxelPos, lightPos);
                    float attenuation = 1.0 / (1.0 + distToLight * distToLight * 0.0001);
                    vec3 lampContribution = lightColor * attenuation * 2.2;
                    
                    if (accumulatedDensity <= 0.04) { 
                        voxelCol = vec3(3.0, 3.0, 3.0); 
                        alpha = accumulatedDensity * 3.5; 
                    } 
                    else if (accumulatedDensity <= 0.25) { 
                        voxelCol = vec3(0.0, 0.8, 1.0); 
                        alpha = 0.015; 
                    }
                    else if (accumulatedDensity < 0.28) { 
                        voxelCol = vec3(1.0, 0.45, 0.05); 
                    } 
                    else if (accumulatedDensity < 0.44) { 
                        voxelCol = vec3(0.12, 0.13, 0.15) + lampContribution; 
                        alpha = 0.012; 
                    }
                    else { 
                        alpha = 0.018; 
                    }
                    
                    float shadow = 1.0;
                    if (hitSide == 1) shadow = 0.85; 
                    else if (hitSide == 2) shadow = 0.70; 
                    
                    volumeColor += T * voxelCol * alpha * shadow; 
                    T *= (1.0 - alpha);
                }
            }
            
            if (sideDist.x < sideDist.y && sideDist.x < sideDist.z) {
                sideDist.x += deltaDist.x; mapPos.x += step.x; hitSide = 0;
            } else if (sideDist.y < sideDist.z) {
                sideDist.y += deltaDist.y; mapPos.y += step.y; hitSide = 1;
            } else {
                sideDist.z += deltaDist.z; mapPos.z += step.z; hitSide = 2;
            }
        }
        color = volumeColor + T * color;
    }
    
    vec2 scrCoord = uvs * vec2(800.0, 600.0); vec2 center = vec2(400.0, 300.0);
    float distToCenterHor = abs(scrCoord.x - center.x); float distToCenterVer = abs(scrCoord.y - center.y);
    if ((distToCenterHor < 6.0 && distToCenterVer < 1.0) || (distToCenterVer < 6.0 && distToCenterHor < 1.0)) {
        if (distToCenterHor > 1.0 || distToCenterVer > 1.0) { color = vec3(0.0, 1.0, 0.2); }
    }
    fragColor = vec4(color, 1.0);
}
"""
# --- ИНИЦИАЛИЗАЦИЯ ИГРОВОГО ОКНА И КОНТЕКСТА OPENGL ---
pygame.init()
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

pygame.display.set_caption(f"Interop GPU Engine | Grid: {VOXEL_RES}^3 | Player {PLAYER_ID}")
screen = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
ctx = moderngl.create_context()

try:
    import win32gui, win32con
    hwnd = win32gui.FindWindow(None, f"Interop GPU Engine | Grid: {VOXEL_RES}^3 | Player {PLAYER_ID}")
    if hwnd:
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
except ImportError:
    pass

quad_buffer = ctx.buffer(np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4'))
prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
vao = ctx.vertex_array(prog, [(quad_buffer, '2f', 'in_vert')])

texture_3d = ctx.texture3d((VOXEL_RES, VOXEL_RES, VOXEL_RES), 1, dtype='f4')
texture_3d.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

# --- АППАРАТНЫЕ БУФЕРЫ VRAM ПОД НАПРАВЛЕННЫЙ СИ-ОБМЕН (0% PCIE) ---
voxel_bytes_size = VOXEL_RES * VOXEL_RES * VOXEL_RES * 4
voxel_pbo = ctx.buffer(reserve=voxel_bytes_size, dynamic=True)

bullets_pos_ssbo = ctx.buffer(reserve=MAX_BULLETS * 4 * 4, dynamic=True)
bullets_active_ssbo = ctx.buffer(reserve=MAX_BULLETS * 4, dynamic=True)

threads_per_block = (8, 8, 8)
blocks_per_grid = (VOXEL_RES // 8, VOXEL_RES // 8, VOXEL_RES // 8)

grid = building_grid
wall_width, wall_height, wall_thickness = 180.0, 110.0, 12.0  # Отрегулировано под сетку 256
cam_x, cam_y, cam_z = 0.5, 0.22, 0.78
camera_angle_x, camera_angle_y = 0.0, 0.0
move_speed = 0.015

# Импортируем Си-функции из установленной библиотеки PyOpenGL
import ctypes
from OpenGL.GL import glBufferSubData, GL_PIXEL_UNPACK_BUFFER, GL_SHADER_STORAGE_BUFFER, glBindBuffer

# Первичный запуск физического конвейера
omni_destructor_and_gravity_kernel[blocks_per_grid, threads_per_block](
    building_grid, block_hp, wall_width, wall_height, wall_thickness, bullets_pos, bullets_vel, bullets_active,
    BULLET_CALIBER, EXPLOSIVE_POWER, True, part_pos, part_vel, part_active, part_life, part_size, part_count, 0.0,
    gas_pos, gas_vel, gas_active, gas_life, gas_opacity, gas_temp, cp.zeros(1, dtype=cp.int32)
)
cp.cuda.stream.get_current_stream().synchronize()

# Загружаем стартовую геометрию через Си-указатель во VRAM
glBindBuffer(GL_PIXEL_UNPACK_BUFFER, voxel_pbo.glo)
glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, voxel_bytes_size, ctypes.c_void_p(building_grid.data.ptr))
glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
texture_3d.write(voxel_pbo)

# ХАНДЛИНГ КУРСОРA (MOUSE LOCK)
pygame.mouse.set_visible(False)
pygame.event.set_grab(True)
clock = pygame.time.Clock()

is_holding_trigger = False
fire_cooldown = 0
total_ammo_fired = 0
frame_seed = 0.5

# Временный CuPy буфер для vec4 выравнивания std430 в SSBO
bullets_pos_vec4 = cp.zeros((MAX_BULLETS, 4), dtype=cp.float32)

# --- ГЛАВНЫЙ ИГРОВОЙ ЦИКЛ (ПОЛНЫЙ ZERO-COPY КОНВЕЙЕР) ---
while running:
    rel_x, rel_y = pygame.mouse.get_rel()
    camera_angle_x -= rel_x * 0.004
    camera_angle_y = max(-1.4, min(1.4, camera_angle_y - rel_y * 0.004))

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
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            running = False

    # Генерация локальных выстрелов строго во VRAM GPU под 256 метрику
    if pygame.mouse.get_pressed() and fire_cooldown <= 0:
        for b_idx in range(MAX_BULLETS):
            if bullets_active[b_idx] < 0.5:
                bullets_active[b_idx] = 1.0
                bullets_pos[b_idx, 0] = cam_x * 256.0
                bullets_pos[b_idx, 1] = cam_y * 256.0
                bullets_pos[b_idx, 2] = cam_z * 256.0
                bullets_vel[b_idx, 0] = math.cos(camera_angle_y) * math.sin(camera_angle_x) * 110.0
                bullets_vel[b_idx, 1] = math.sin(camera_angle_y) * 110.0
                bullets_vel[b_idx, 2] = math.cos(camera_angle_y) * math.cos(camera_angle_x) * 110.0
                total_ammo_fired += 1
                fire_cooldown = 3
                break

    if fire_cooldown > 0: fire_cooldown -= 1
    frame_seed = (frame_seed + 0.01) % 10000.0

    bullets_pos += bullets_vel * 0.035 * bullets_active[:, cp.newaxis]
    bullets_active[(bullets_pos[:, 0] < 0.0) | (bullets_pos[:, 0] > 256.0) | (bullets_pos[:, 1] < 0.0) | (bullets_pos[:, 1] > 256.0) | (bullets_pos[:, 2] < 0.0) | (bullets_pos[:, 2] > 256.0)] = 0.0

    part_count.fill(0)
    gas_count_gpu = cp.zeros(1, dtype=cp.int32)

    # 1. Расчет физики деструкции и газов
    omni_destructor_and_gravity_kernel[blocks_per_grid, threads_per_block](
        building_grid, block_hp, wall_width, wall_height, wall_thickness, bullets_pos, bullets_vel, bullets_active,
        BULLET_CALIBER, EXPLOSIVE_POWER, False, part_pos, part_vel, part_active, part_life, part_size, part_count, frame_seed,
        gas_pos, gas_vel, gas_active, gas_life, gas_opacity, gas_temp, gas_count_gpu
    )

    # 2. Симуляция осколков
    current_p_count = int(part_count.item())
    if current_p_count > 0:
        p_blocks = max(32, math.ceil(min(current_p_count, MAX_PARTICLES) / 256))
        update_particles_physics_kernel[p_blocks, 256](
            part_pos, part_vel, part_active, part_life, part_size, min(current_p_count, MAX_PARTICLES), building_grid, wall_thickness
        )

    # 3. Симуляция Навье-Стокса
    current_g_count = int(gas_count_gpu.item())
    if current_g_count > 0:
        g_blocks = max(32, math.ceil(min(current_g_count, MAX_GAS_PARTICLES) / 256))
        update_gas_sph_kernel[g_blocks, 256](
            gas_pos, gas_vel, gas_active, gas_life, gas_opacity, gas_temp, min(current_g_count, MAX_GAS_PARTICLES), building_grid, TURBULENCE_MOD, HUMIDITY_MOD
        )

    cp.cuda.stream.get_current_stream().synchronize()

    # --- СВЕРХБЫСТРЫЙ КРЕМНИЕВЫЙ ИНТЕРОП (БЕЗ TOBYTES И БЕЗ СБОЕВ ТИПОВ) ---
    # Передаем физический адрес CuPy-массива вокселей прямо в OpenGL-буфер на полной скорости VRAM
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, voxel_pbo.glo)
    glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, voxel_bytes_size, ctypes.c_void_p(building_grid.data.ptr))
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
    
    texture_3d.use(location=0)
    texture_3d.write(voxel_pbo)
    texture_3d.build_mipmaps()

    # Упаковываем позиции пуль в vec4 структуру (выравнивание std430) напрямую на GPU
    bullets_pos_vec4[:, :3] = bullets_pos

    # Прямая заливка векторов пуль в SSBO буферы OpenGL без ЦП строк и .tobytes()
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bullets_pos_ssbo.glo)
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bullets_pos_vec4.nbytes, ctypes.c_void_p(bullets_pos_vec4.data.ptr))
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bullets_active_ssbo.glo)
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bullets_active.nbytes, ctypes.c_void_p(bullets_active.data.ptr))
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    # Отрисовка
    ctx.clear(0.0, 0.0, 0.0)
    texture_3d.use(location=0)
    prog['volumeTex'].value = 0
    prog['camAngleX'].value = camera_angle_x
    prog['camAngleY'].value = camera_angle_y
    prog['camPos'].value = (cam_x, cam_y, cam_z)
    
    # Аппаратная привязка SSBO-слотов к шейдеру
    bullets_pos_ssbo.bind_to_storage_buffer(1)
    bullets_active_ssbo.bind_to_storage_buffer(2)
    
    vao.render(moderngl.TRIANGLE_STRIP)
    pygame.display.flip()

    current_fps = int(clock.get_fps())
    pygame.display.set_caption(f"Zero-Copy DDA Engine | Grid: {VOXEL_RES}^3 | FPS: {current_fps}")
    clock.tick(60)

running = False
pygame.quit()
sys.exit()
