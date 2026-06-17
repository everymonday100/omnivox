import math
import numpy as np
from numba import cuda
import random
import pygame
import moderngl
import tkinter as tk
from tkinter import ttk
import threading

# =============================================================================
# 1. СЕТКА 128^3 И ПАРАМЕТРЫ АНАЛИТИЧЕСКОГО СТЕНДА
# =============================================================================
NUM_ATTRACTORS = 100
VOXEL_GRID_RES = 128        # Подняли разрешение до 128^3 (Честный HD-каркас)
WINDOW_RES = (900, 400)     # Три ортогональные проекции

pygame.init()
pygame.display.set_mode(WINDOW_RES, pygame.OPENGL | pygame.DOUBLEBUF)
ctx = moderngl.create_context()

# =============================================================================
# 2. GLSL КОНВЕЙЕР (MIP RAYMARCHING + ТЕПЛЫЙ ДИНАМИЧЕСКИЙ СВЕТ)
# =============================================================================
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

uniform sampler3D volumeTexture;  
uniform float gammaValue;        
uniform vec3 lightPos;           

void main() {
    vec3 rayOrigin = vec3(0.0);
    vec3 rayDir = vec3(0.0);
    
    // Шаг адаптирован под плотность 128 для сохранения 60+ FPS
    float stepSize = 0.0012; 
    
    if (uvs.x < 0.333) {
        vec2 local_uv = vec2(uvs.x / 0.333, uvs.y);
        rayOrigin = vec3(local_uv.x, 1.0, local_uv.y); 
        rayDir = vec3(0.0, -1.0, 0.0);
    } else if (uvs.x >= 0.333 && uvs.x < 0.666) {
        vec2 local_uv = vec2((uvs.x - 0.333) / 0.333, uvs.y);
        rayOrigin = vec3(local_uv.x, local_uv.y, 0.0); 
        rayDir = vec3(0.0, 0.0, 1.0);
    } else {
        vec2 local_uv = vec2((uvs.x - 0.666) / 0.334, uvs.y);
        rayOrigin = vec3(0.0, local_uv.y, local_uv.x); 
        rayDir = vec3(1.0, 0.0, 0.0);
    }
    
    float maxLightEnergy = 0.0;    
    float maxSmokeDensity = 0.0;   
    vec3 p = rayOrigin;
    
    // Плотный маршинг луча по сглаженной сетке
    for(int i = 0; i < 450; i++) {
        p += rayDir * stepSize;
        if(p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z < 0.0 || p.z > 1.0) continue;
        
        float density = texture(volumeTexture, p).r;
        if(density > 0.01) {
            if (density > maxSmokeDensity) {
                maxSmokeDensity = density;
            }
            
            vec3 lDir = normalize(lightPos - p);
            float lStep = 0.012;
            float opticalThickness = 0.0;
            vec3 lp = p;
            
            // Теневой проход к динамической лампе
            for(int j = 0; j < 12; j++) {
                lp += lDir * lStep;
                if(lp.x < 0.0 || lp.x > 1.0 || lp.y < 0.0 || lp.y > 1.0 || lp.z < 0.0 || lp.z > 1.0) break;
                opticalThickness += texture(volumeTexture, lp).r * lStep;
            }
            
            float lightDepth = exp(-opticalThickness * 12.0);
            float currentLightEnergy = density * lightDepth;
            
            if (currentLightEnergy > maxLightEnergy) {
                maxLightEnergy = currentLightEnergy;
            }
        }
    }
    
    maxLightEnergy = pow(maxLightEnergy, gammaValue);
    maxSmokeDensity = pow(maxSmokeDensity, gammaValue);
    
    // Монохромный пепельный каркас газа
    vec3 smokeBodyColor = vec3(maxSmokeDensity * 0.32); 
    
    // Теплый спектр фотонов прожектора (Янтарный: R=1.0, G=0.74, B=0.44)
    vec3 warmLightColor = vec3(1.0, 0.74, 0.44) * maxLightEnergy * 0.85;
    
    vec3 finalColor = smokeBodyColor + warmLightColor;
    
    if (abs(uvs.x - 0.333) < 0.002 || abs(uvs.x - 0.666) < 0.002) {
        fragColor = vec4(0.2, 0.2, 0.25, 1.0);
    } else {
        fragColor = vec4(finalColor, 1.0);
    }
}
"""

prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

quad_vertices = np.array([
    -1.0, -1.0,   1.0, -1.0,  -1.0,  1.0,
    -1.0,  1.0,   1.0, -1.0,   1.0,  1.0,
], dtype='f4')

vbo = ctx.buffer(quad_vertices)
vao = ctx.vertex_array(prog, [(vbo, '2f', 'in_vert')])

volume_tex = ctx.texture3d((VOXEL_GRID_RES, VOXEL_GRID_RES, VOXEL_GRID_RES), 1, data=None, dtype='f4')
volume_tex.filter = (moderngl.LINEAR, moderngl.LINEAR) 
volume_tex.wrap = (False, False, False)
# =============================================================================
# 3. CUDA-ЯДРО НАВЬЕ-СТОКСА ПОД СЕТКУ 128
# =============================================================================
@cuda.jit(device=True, inline=True)
def gpu_rand3d(x, y, z):
    dot = float(x) * 12.9898 + float(y) * 78.233 + float(z) * 45.164
    sin_val = math.sin(dot) * 43758.5453
    return sin_val - math.floor(sin_val)

@cuda.jit(device=True, inline=True)
def lerp(a, b, t):
    return a + t * (b - a)

@cuda.jit(device=True, inline=True)
def get_smoothed_noise3d(x, y, z, frequency_shift):
    scale = float(1 << frequency_shift)
    fx0 = math.floor(float(x) / scale)
    fy0 = math.floor(float(y) / scale)
    fz0 = math.floor(float(z) / scale)
    x0, y0, z0 = int(fx0), int(fy0), int(fz0)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    tx = (float(x) / scale) - fx0
    ty = (float(y) / scale) - fy0
    tz = (float(z) / scale) - fz0
    tx = tx * tx * (3.0 - 2.0 * tx)
    ty = ty * ty * (3.0 - 2.0 * ty)
    tz = tz * tz * (3.0 - 2.0 * tz)
    return lerp(lerp(lerp(gpu_rand3d(x0, y0, z0), gpu_rand3d(x1, y0, z0), tx), lerp(gpu_rand3d(x0, y1, z0), gpu_rand3d(x1, y1, z0), tx), ty), lerp(lerp(gpu_rand3d(x0, y0, z1), gpu_rand3d(x1, y0, z1), tx), lerp(gpu_rand3d(x0, y1, z1), gpu_rand3d(x1, y1, z1), tx), ty), tz)

@cuda.jit(fastmath=True)
def h_kinematic_navier_stokes_kernel(attractors, voxel_grid, seed_offset, gui_turb_mod, gui_humid_mod):
    x, y, z = cuda.grid(3)
    if x >= VOXEL_GRID_RES or y >= VOXEL_GRID_RES or z >= VOXEL_GRID_RES: return
    accumulated_density = 0.0
    nx, ny, nz = float(x) / VOXEL_GRID_RES, float(y) / VOXEL_GRID_RES, float(z) / VOXEL_GRID_RES
    
    # Константы шума масштабированы под сетку 128
    noise_large = get_smoothed_noise3d(x, y, z + seed_offset, 3)
    noise_medium = get_smoothed_noise3d(x, y, z + seed_offset, 1)
    noise_fine = gpu_rand3d(x, y, z + seed_offset)
    mass_stochastic_cut = (noise_large * 0.55 + noise_medium * 0.3 + noise_fine * 0.15) - 0.5
    
    for i in range(NUM_ATTRACTORS):
        a_lifetime = attractors[i, 8]
        if a_lifetime <= 0.0: continue
        
        ax, ay, az = attractors[i, 0] / VOXEL_GRID_RES, attractors[i, 1] / VOXEL_GRID_RES, attractors[i, 2] / VOXEL_GRID_RES
        a_opacity     = attractors[i, 7]
        a_temperature = attractors[i, 10]
        spin          = attractors[i, 11]
        
        dx, dy, dz = nx - ax, ny - ay, nz - az
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        r_max = (0.05 + (1.0 - a_lifetime) * 0.14) * (1.0 + a_temperature * 0.2)
        
        if dist < r_max:
            norm_dist = dist / r_max
            local_time_shift = noise_fine * 25.0
            vorticity = math.sin(dy * 110.0 + mass_stochastic_cut * 12.0 + local_time_shift) * math.cos(dx * 110.0) * 0.25
            
            viscosity_brake = 1.0 / (1.0 + gui_humid_mod * 2.5)
            angle = (spin * (1.0 - norm_dist) * 6.28 + vorticity) * viscosity_brake
            
            s_x = dx * math.cos(angle) - dz * math.sin(angle)
            base_density = (1.0 - math.sqrt(s_x*s_x + dy*dy + (dx * math.sin(angle) + dz * math.cos(angle))**2)/r_max) * a_opacity
            
            if base_density > 0.0:
                local_density = max(0.0, base_density * (1.0 + mass_stochastic_cut * gui_turb_mod))
                if local_density > 0.0:
                    accumulated_density += local_density

    final_density = min(accumulated_density, 1.0)
    
    # --- КУБИЧЕСКИЙ БУФЕР ЗАТУХАНИЯ КРАЕВ (БЕСШОВНОСТЬ) ---
    dist_to_edge_x = min(nx, 1.0 - nx)
    dist_to_edge_y = min(ny, 1.0 - ny)
    dist_to_edge_z = min(nz, 1.0 - nz)
    min_dist_to_edge = min(dist_to_edge_x, min(dist_to_edge_y, dist_to_edge_z))
    
    edge_buffer = 0.10
    boundary_mask = min_dist_to_edge / edge_buffer
    boundary_mask = max(0.0, min(boundary_mask, 1.0))
    boundary_mask = boundary_mask * boundary_mask * (3.0 - 2.0 * boundary_mask)
    
    voxel_grid[x, y, z] = final_density * boundary_mask

# Выделение буферов памяти VRAM под сетку 128
attractors_cpu = np.zeros((NUM_ATTRACTORS, 12), dtype=np.float32)
voxel_grid_cpu = np.zeros((VOXEL_GRID_RES, VOXEL_GRID_RES, VOXEL_GRID_RES), dtype=np.float32)
voxel_grid_gpu = cuda.to_device(voxel_grid_cpu)
attractors_gpu = cuda.to_device(attractors_cpu)

threads_per_block = (8, 8, 4)  
blocks_per_grid = (math.ceil(VOXEL_GRID_RES/8), math.ceil(VOXEL_GRID_RES/8), math.ceil(VOXEL_GRID_RES/4))

# Координаты траектории Безье
P0, P1, P2 = np.array([15.,25.,64.]), np.array([64.,20.,15.]), np.array([112.,90.,110.])
# =============================================================================
# 4. GUI ПАНЕЛЬ TKINTER С ПОДВИЖНЫМ ПРОЖЕКТОРОМ
# =============================================================================
gui_opacity, gui_turb, gui_humidity, gui_temp, gui_gamma, gui_buoyancy = 0.95, 2.4, 0.4, 1.0, 0.90, 0.25
gui_light_x, gui_light_y, gui_light_z = 0.5, 1.2, 0.5

def launch_tkinter_gui():
    global gui_opacity, gui_turb, gui_humidity, gui_temp, gui_gamma, gui_buoyancy, gui_light_x, gui_light_y, gui_light_z
    root = tk.Tk(); root.title("CFD Стенд 128^3"); root.geometry("340x620+950+50"); root.attributes("-topmost", True)  
    def update_vals(*args):
        global gui_opacity, gui_turb, gui_humidity, gui_temp, gui_gamma, gui_buoyancy, gui_light_x, gui_light_y, gui_light_z
        gui_opacity, gui_turb, gui_humidity, gui_temp, gui_gamma, gui_buoyancy = float(s1.get()), float(s2.get()), float(s3.get()), float(s4.get()), float(s5.get()), float(s6.get())
        gui_light_x, gui_light_y, gui_light_z = float(l_x.get()), float(l_y.get()), float(l_z.get())
    tk.Label(root, text="Плотность (Opacity):").pack()
    s1 = ttk.Scale(root, from_=0.1, to=2.0, value=0.95, command=update_vals); s1.pack(fill='x', padx=15)
    tk.Label(root, text="Турбулентность Резака:").pack()
    s2 = ttk.Scale(root, from_=0.0, to=5.0, value=2.4, command=update_vals); s2.pack(fill='x', padx=15)
    tk.Label(root, text="Липкость (Взаимный Импульс):").pack()
    s3 = ttk.Scale(root, from_=0.0, to=2.0, value=0.4, command=update_vals); s3.pack(fill='x', padx=15)
    tk.Label(root, text="Температура (Радиус):").pack()
    s4 = ttk.Scale(root, from_=0.1, to=3.0, value=1.0, command=update_vals); s4.pack(fill='x', padx=15)
    tk.Label(root, text="Мягкость (Гамма-Контраст):").pack()
    s5 = ttk.Scale(root, from_=0.4, to=3.5, value=0.90, command=update_vals); s5.pack(fill='x', padx=15)
    tk.Label(root, text="Конвекция (Всплытие):").pack()
    s6 = ttk.Scale(root, from_=0.0, to=0.8, value=0.25, command=update_vals); s6.pack(fill='x', padx=15)
    tk.Label(root, text="[ПРОЖЕКТОР] Источник X:", fg="blue").pack()
    l_x = ttk.Scale(root, from_=-0.5, to=1.5, value=0.5, command=update_vals); l_x.pack(fill='x', padx=15)
    tk.Label(root, text="[ПРОЖЕКТОР] Источник Y:", fg="blue").pack()
    l_y = ttk.Scale(root, from_=0.0, to=2.0, value=1.2, command=update_vals); l_y.pack(fill='x', padx=15)
    tk.Label(root, text="[ПРОЖЕКТОР] Источник Z:", fg="blue").pack()
    l_z = ttk.Scale(root, from_=-0.5, to=1.5, value=0.5, command=update_vals); l_z.pack(fill='x', padx=15)
    root.mainloop()

gui_thread = threading.Thread(target=launch_tkinter_gui, daemon=True); gui_thread.start()

# =============================================================================
# 5. ГЛАВНЫЙ ИГРОВОЙ ЦИКЛ С ИМПУЛЬСНЫМ ЗАСАСЫВАНИЕМ СРЕДЫ
# =============================================================================
clock = pygame.time.Clock()
frame, running, session_seed = 0, True, random.randint(0, 5000)
print("\n[ЗАПУСК СТЕНДА] Сетка 128^3 + Волновой импульсный след активны. Мониторинг 3 проекций.")

while running:
    t = (frame % 100) / 100.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False

    # -------------------------------------------------------------------------
    # CPU: КИНЕМАТИЧЕСКИЙ ВОЛНОВОЙ СОЛВЕР: "КЛУБОК ВЫСТР ЕЛИВАЕТ К КЛУБКУ"
    # -------------------------------------------------------------------------
    for i in range(1, NUM_ATTRACTORS):
        if attractors_cpu[i, 8] > 0.0:
            if i > 1 and attractors_cpu[i-1, 8] > 0.0:
                # Находим направленный вектор выстреливания импульса
                suction_dx = attractors_cpu[i, 0] - attractors_cpu[i-1, 0]
                suction_dy = attractors_cpu[i, 1] - attractors_cpu[i-1, 1]
                suction_dz = attractors_cpu[i, 2] - attractors_cpu[i-1, 2]
                
                # Коэффициент сцепления завязан на ползунок липкости (gui_humidity)
                # Передаем не просто позицию, а векторный толчок импульса скорости выхлопа [C11]
                impulse_wave = 0.12 * (1.0 + gui_humidity * 2.0)
                
                # Старый клуб выстреливает вперед, склеиваясь с траекторией нового [C11]
                attractors_cpu[i-1, 0] += suction_dx * impulse_wave
                attractors_cpu[i-1, 1] += suction_dy * impulse_wave
                attractors_cpu[i-1, 2] += suction_dz * impulse_wave
                
                # Закручиваем "генетическую" траекторию в спиральный жгут
                attractors_cpu[i-1, 0] += math.sin(float(frame)*0.1) * gui_humidity * 0.15
                attractors_cpu[i-1, 2] += math.cos(float(frame)*0.1) * gui_humidity * 0.15

            # Всплытие и диффузия
            attractors_cpu[i, 1] += gui_buoyancy
            attractors_cpu[i, 0] += random.uniform(-0.15, 0.15)
            attractors_cpu[i, 2] += random.uniform(-0.15, 0.15)
            attractors_cpu[i, 8] -= 0.012  
            attractors_cpu[i, 7] *= 0.975   
            if attractors_cpu[i, 8] <= 0.0: attractors_cpu[i, 8] = 0.0

    # Спавн нового ведущего импульса по кривой Безье
    tank_pos = (1-t)**2 * P0 + 2*(1-t)*t * P1 + t**2 * P2
    if frame % 2 == 0:
        for i in range(NUM_ATTRACTORS):
            if attractors_cpu[i, 8] <= 0.0:
                attractors_cpu[i, 0:3] = tank_pos
                attractors_cpu[i, 6], attractors_cpu[i, 7], attractors_cpu[i, 8] = 1.0, gui_opacity, 1.0
                attractors_cpu[i, 9], attractors_cpu[i, 10] = gui_humidity, gui_temp
                attractors_cpu[i, 11] = random.choice([-1.0, 1.0]) * random.uniform(5.0, 8.5) 
                break

    # CUDA-расчет
    attractors_gpu.copy_to_device(attractors_cpu)
    seed_offset_step = session_seed + frame
    h_kinematic_navier_stokes_kernel[blocks_per_grid, threads_per_block](attractors_gpu, voxel_grid_gpu, seed_offset_step, gui_turb, gui_humidity)
    cuda.synchronize()
    
    # Пересылка 8 Мегабайт во VRAM по PCIe
    voxel_grid_cpu = voxel_grid_gpu.copy_to_host()
    volume_tex.write(voxel_grid_cpu.tobytes()) 
    
    # Отрисовка трех проекций
    ctx.clear(0.0, 0.0, 0.0)
    volume_tex.use(0)
    prog['gammaValue'].value = gui_gamma  
    prog['lightPos'].value = (gui_light_x, gui_light_y, gui_light_z)
    
    vao.render()
    pygame.display.flip()
    frame += 1
    clock.tick(60)

pygame.quit()
