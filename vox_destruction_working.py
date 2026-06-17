import warnings, pygame, moderngl, cupy as cp, numba.cuda as cuda, math, numpy as np, sys, traceback, logging
from numba.core.errors import NumbaPerformanceWarning
from pygame.locals import *

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

# --- 💎 ВОССТАНОВЛЕНЫ ОРИГИНАЛЬНЫЕ КОНСТАНТЫ ИЗ VOX_DESTRUCTION_OLD.PY 💎 ---
WIDTH, HEIGHT = 1920, 1080
VOXEL_RES = 128          # Возвращаем честное оригинальное разрешение сетки
BLOCK_RES = VOXEL_RES // 2
MAX_PARTICLES = 15000
MAX_BULLETS = 50

SPARK_SPEED = 2.603
SPARK_DECAY = 0.056
SPARK_CHANCE = 3.794
BLOCK_GRAV = 0.131
BLOCK_DECAY = 0.016
BULLET_CALIBER = 2.0      
EXPLOSIVE_POWER = 3.33    

# Аллокация строго упорядоченных массивов во VRAM
grid_cupy = cp.zeros((VOXEL_RES, VOXEL_RES, VOXEL_RES), dtype=cp.float32, order='C')
block_hp = cp.zeros((BLOCK_RES, BLOCK_RES, BLOCK_RES), dtype=cp.float32, order='C')

part_pos = cp.zeros((MAX_PARTICLES, 3), dtype=cp.float32)
part_vel = cp.zeros((MAX_PARTICLES, 3), dtype=cp.float32)
part_active = cp.zeros(MAX_PARTICLES, dtype=cp.float32)
part_life = cp.zeros(MAX_PARTICLES, dtype=cp.float32)
part_size = cp.zeros((MAX_PARTICLES, 3), dtype=cp.float32)
part_count = cp.zeros(1, dtype=cp.int32)

bullets_pos = cp.zeros((MAX_BULLETS, 3), dtype=cp.float32)
bullets_vel = cp.zeros((MAX_BULLETS, 3), dtype=cp.float32)
bullets_active = cp.zeros(MAX_BULLETS, dtype=cp.float32)
# --- ЧАСТЬ 2: CUDA ВЫЧИСЛИТЕЛЬНЫЕ ЯДРА СИМУЛЯЦИИ И ДЕСТРУКЦИИ ОРИГИНАЛЬНОГО ДВИЖКА ---

@cuda.jit
def omni_destructor_and_gravity_kernel(building_grid, block_hp, bullets_pos, bullets_vel, bullets_active,
                                      caliber, explosive, reset_scene, p_pos, p_vel, p_active, p_life, p_size, p_count_array, seed):
    """
    Честный деструктор: рассчитывает пробитие, деформирует макро-блоки HP,
    и спавнит частицы в зависимости от калибра пули.
    """
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
    
    if x >= 128 or y >= 128 or z >= 128: return # Защита под VOXEL_RES = 128
    bx, by, bz = x // 2, y // 2, z // 2
    
    # Сценарий генерации оригинальной тестовой стены при старте или сбросе
    if reset_scene:
        is_wall = (54.0 <= x <= 74.0) and (1.0 <= y <= 61.0) and (20.0 <= z <= 108.0)
        is_floor_ceil = (y == 1 or y == 126) and (x % 16 == 0 or z % 16 == 0)
        if is_wall or is_floor_ceil:
            building_grid[x, y, z] = 0.45 if is_wall else 0.15
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
    
    for b_idx in range(50): # MAX_BULLETS
        if bullets_active[b_idx] < 0.5: continue
        
        b_x, b_y, b_z = bullets_pos[b_idx, 0], bullets_pos[b_idx, 1], bullets_pos[b_idx, 2]
        v_x, v_y, v_z = bullets_vel[b_idx, 0], bullets_vel[b_idx, 1], bullets_vel[b_idx, 2]
        
        # Оригинальные 35 шагов субстеппинга для предотвращения сквозного пролета пуль
        for step in range(35):
            t = float(step) * 0.08
            dmg_x, dmg_y, dmg_z = b_x + v_x * t, b_y + v_y * t, b_z + v_z * t
            dx, dy, dz = x - dmg_x, y - dmg_y, z - dmg_z
            dmg_dist_sq = dx*dx + dy*dy + dz*dz
            
            # Эпицентр калибра пули (прямой урон)
            if dmg_dist_sq < (caliber * caliber):
                if block_hp[bx, by, bz] < 500.0:
                    block_hp[bx, by, bz] = max(0.0, block_hp[bx, by, bz] - 0.150 * explosive)
            # Сопутствующая взрывная волна (фугасный урон)
            elif dmg_dist_sq < (caliber * caliber * 5.0):
                if block_hp[bx, by, bz] < 500.0:
                    block_hp[bx, by, bz] = max(0.0, block_hp[bx, by, bz] - 0.045 * explosive)
                    
            if block_hp[bx, by, bz] <= 0.0:
                building_grid[x, y, z] = 0.0
                
                # Спавн искр и дыма по оригинальному шансу SPARK_CHANCE
                if float(raw_rand) < 3.794:
                    idx = cuda.atomic.add(p_count_array, 0, 1)
                    if idx < 15000:
                        p_pos[idx, 0], p_pos[idx, 1], p_pos[idx, 2] = float(x), float(y), float(z)
                        spark_hash = hash_val + b_idx * 997
                        cone_angle = float(spark_hash % 360) * 0.0174533
                        cone_radius = float((spark_hash // 3) % 25) * 0.14
                        
                        # Скорость разлета на основе SPARK_SPEED
                        p_vel[idx, 0] = -2.603 + (dx / (math.sqrt(dmg_dist_sq) + 0.1)) * 0.8
                        p_vel[idx, 1] = (math.sin(cone_angle) * cone_radius) + 0.35
                        p_vel[idx, 2] = (math.cos(cone_angle) * cone_radius)
                        
                        p_active[idx], p_life[idx] = 1.0, 1.0
                        p_size[idx, 0] = 1.0 + float(hash_val % 3)
                        p_size[idx, 1] = 1.0 + float((hash_val // 3) % 2)
                        p_size[idx, 2] = 1.0
                break
                
        if block_hp[bx, by, bz] > 0.0:
            building_grid[x, y, z] = block_hp[bx, by, bz] / 4.0

@cuda.jit
def update_particles_physics_kernel(p_pos, p_vel, p_active, p_life, p_size, count, building_grid):
    """
    Оригинальная физика: затирает старый след, рассчитывает сопротивление,
    гравитацию BLOCK_GRAV и запекает в сетку искры (0.45) или дым (<=0.04).
    """
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= count or p_active[idx] < 0.5: return
    
    ox, oy, oz = int(p_pos[idx, 0]), int(p_pos[idx, 1]), int(p_pos[idx, 2])
    sz_x, sz_y, sz_z = int(p_size[idx, 0]), int(p_size[idx, 1]), int(p_size[idx, 2])
    
    # Очищаем старый аналоговый след частицы перед шагом физики
    for sx in range(sz_x):
        for sy in range(sz_y):
            for sz_d in range(sz_z):
                vx, vy, vz = ox + sx, oy + sy, oz + sz_d
                if 0 <= vx < 128 and 0 <= vy < 128 and 0 <= vz < 128:
                    if building_grid[vx, vy, vz] <= 0.45: building_grid[vx, vy, vz] = 0.0
                    
    # Триггер зоны дыма: за пределами фронтальной плоскости стены частицы уходят в дым
    is_smoke_zone = p_pos[idx, 0] > 74.0
    p_life[idx] -= 0.056 if is_smoke_zone else 0.016 # SPARK_DECAY / BLOCK_DECAY
    
    if p_life[idx] <= 0.0:
        p_active[idx] = 0.0
        return
        
    p_vel[idx, 0] *= 0.96
    p_vel[idx, 1] *= 0.98 if is_smoke_zone else 0.96
    if not is_smoke_zone: p_vel[idx, 1] -= 0.131 # BLOCK_GRAV
    p_vel[idx, 2] *= 0.96
    
    nx = p_pos[idx, 0] + p_vel[idx, 0]
    ny = p_pos[idx, 1] + p_vel[idx, 1]
    nz = p_pos[idx, 2] + p_vel[idx, 2]
    
    if ny < 1.0:
        ny = 1.0
        p_vel[idx, 1] = -p_vel[idx, 1] * 0.35
        p_vel[idx, 0] *= 0.6
        p_vel[idx, 2] *= 0.6
        
    if nx < 0 or nx >= 128 or ny >= 128 or nz < 0 or nz >= 128:
        p_active[idx] = 0.0
        return
        
    rx, ry, rz = int(nx), int(ny), int(nz)
    if building_grid[rx, ry, rz] > 0.8:
        p_vel[idx, 0] = -p_vel[idx, 0] * 0.4
        p_vel[idx, 1] *= 0.8
        p_vel[idx, 2] *= 0.4
        nx, ny = p_pos[idx, 0] + p_vel[idx, 0], p_pos[idx, 1] + p_vel[idx, 1]
        
    p_pos[idx, 0], p_pos[idx, 1], p_pos[idx, 2] = nx, ny, nz
    rx, ry, rz = int(nx), int(ny), int(nz)
    
    # Запекание следа частицы с сохранением строгого баланса плотностей
    for sx in range(sz_x):
        for sy in range(sz_y):
            for sz_d in range(sz_z):
                v_x, v_y, v_z = rx + sx, ry + sy, rz + sz_d
                if 0 <= v_x < 128 and 0 <= v_y < 128 and 0 <= v_z < 128:
                    if building_grid[v_x, v_y, v_z] > 0.8: continue
                    if is_smoke_zone:
                        building_grid[v_x, v_y, v_z] = min(0.04, building_grid[v_x, v_y, v_z] + 0.03)
                    else:
                        building_grid[v_x, v_y, v_z] = min(0.45, building_grid[v_x, v_y, v_z] + 0.20 * p_life[idx])
# --- ЧАСТЬ 3: ШЕЙДЕРЫ АППАРАТНОГО РЕНДЕРИНГА 3D ТЕКСТУРНОГО РЭЙМАРШИНГА (ModernGL / GLSL) ---

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
uniform float camAngleX; 
uniform float camAngleY;
uniform vec3 camPos;

const int MAX_BULLETS = 50;
uniform vec3 bulletsPos[MAX_BULLETS]; 
uniform float bulletsActive[MAX_BULLETS];

bool intersectBox(vec3 ro, vec3 rd, out float t0, out float t1) {
    vec3 invR = 1.0 / (rd + 1e-6);
    vec3 tbot = invR * (vec3(0.0) - ro); 
    vec3 ttop = invR * (vec3(1.0) - ro);
    vec3 tmin = min(tbot, ttop); 
    vec3 tmax = max(tbot, ttop);
    
    // ИСПРАВЛЕНО: Функции min/max разложены на пары по 2 аргумента, убрана лишняя закрывающая скобка
    t0 = max(tmin.x, max(tmin.y, tmin.z)); 
    t1 = min(tmax.x, min(tmax.y, tmax.z));
    
    return t0 < t1 && t1 > 0.0;
}

void main() {
    vec3 ro = camPos;
    vec3 direction = vec3(
        cos(camAngleY) * sin(camAngleX),
        sin(camAngleY),
        cos(camAngleY) * cos(camAngleX)
    );
    
    vec3 ww = normalize(direction); 
    vec3 uu = normalize(cross(ww, vec3(0.0, 1.0, 0.0))); 
    vec3 vv = normalize(cross(uu, ww));
    
    vec2 p = uvs * 2.0 - 1.0; 
    p.x *= 1.333; 
    
    vec3 rd = normalize(p.x * uu + p.y * vv + 1.5 * ww);
    
    float t0, t1; 
    vec3 color = vec3(0.04, 0.05, 0.08); 
    
    if (intersectBox(ro, rd, t0, t1)) {
        t0 = max(t0, 0.0); 
        vec3 pos = ro + rd * t0;
        
        float stepSize = 0.0025; 
        vec3 stepDir = rd * stepSize;
        
        float T = 1.0; 
        vec3 volumeColor = vec3(0.0);
        
        for (int i = 0; i < 350; i++) {
            if (t0 > t1 || T < 0.01) break;
            
            float density = texture(volumeTex, vec3(pos.z, pos.y, pos.x)).r;
            
            for (int b = 0; b < MAX_BULLETS; b++) {
                if (bulletsActive[b] > 0.5) {
                    if (distance(pos * 128.0, bulletsPos[b]) < 0.43) {
                        volumeColor += T * vec3(4.0, 3.5, 1.5) * 0.9; 
                        T *= 0.02;
                    }
                }
            }
            
            if (density > 0.01) {
                vec3 voxelCol = vec3(0.4, 0.44, 0.48); 
                float alpha = density * 0.35;
                
                if (density <= 0.04) { 
                    voxelCol = vec3(3.0, 3.0, 3.0); 
                    alpha = density * 3.5; 
                }
                else if (density <= 0.25) { 
                    voxelCol = vec3(0.0, 0.8, 1.0); 
                    alpha = 0.015; 
                }
                else if (density < 0.28) { 
                    voxelCol = vec3(1.0, 0.45, 0.05); 
                }
                else if (density < 0.44) { 
                    voxelCol = vec3(0.12, 0.13, 0.15); 
                }
                else { 
                    alpha = 0.018; 
                }
                
                volumeColor += T * voxelCol * alpha; 
                T *= (1.0 - alpha);
            }
            pos += stepDir; 
            t0 += stepSize;
        }
        color = volumeColor + T * color;
    }
    
    vec2 scrCoord = uvs * vec2(1920.0, 1080.0);
    vec2 center = vec2(1920.0 / 2.0, 1080.0 / 2.0);
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
# --- ЧАСТЬ 4: ГЛАВНЫЙ FPS-ЦИКЛ АВТОМАТИЧЕСКОГО ПУЛЕМЕТА И СИНХРОНИЗАЦИЯ С ТЕКСТУРОЙ ---

class OmnivoxEngine:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(f"Voxel Player FPS-142 Mod")
        pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
        self.ctx = moderngl.create_context()
        
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        
        quad_buffer = self.ctx.buffer(np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4'))
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.vao = self.ctx.vertex_array(self.prog, [(quad_buffer, '2f', 'in_vert')])
        
        self.texture_3d = self.ctx.texture3d((VOXEL_RES, VOXEL_RES, VOXEL_RES), 1, dtype='f4')
        self.texture_3d.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        
        # Буфер в оперативной памяти (RAM) для исключения bytes-вызовов
        self.cpu_grid_buffer = np.zeros((VOXEL_RES, VOXEL_RES, VOXEL_RES), dtype=np.float32, order='C')
        
        self.bullets_pos_cpu = np.zeros((MAX_BULLETS, 3), dtype=np.float32)
        self.bullets_act_cpu = np.zeros(MAX_BULLETS, dtype=np.float32)
        
        self.cam_x, self.cam_y, self.cam_z = 0.5, 0.22, 0.78
        self.camera_angle_x, self.camera_angle_y = 0.0, 0.0
        self.move_speed = 0.015
        
        self.shoot_cooldown = 0.0
        self.fire_rate = 0.04  
        self.total_ammo_fired = 0
        self.time_elapsed = 0.0
        
        self.clock = pygame.time.Clock()
        
        global grid_cupy, block_hp
        threads = (8, 8, 8)
        blocks = (VOXEL_RES // 8, VOXEL_RES // 8, VOXEL_RES // 8)
        omni_destructor_and_gravity_kernel[blocks, threads](
            grid_cupy, block_hp, bullets_pos, bullets_vel, bullets_active,
            BULLET_CALIBER, EXPLOSIVE_POWER, True, part_pos, part_vel, part_active, part_life, part_size, part_count, 0.5
        )

    def process_input(self, dt):
        global bullets_pos, bullets_vel, bullets_active, part_pos, part_vel, part_active, part_life, part_size, part_count
        
        rel_x, rel_y = pygame.mouse.get_rel()
        self.camera_angle_x -= rel_x * 0.003
        self.camera_angle_y = max(-1.4, min(1.4, self.camera_angle_y - rel_y * 0.003))
        
        keys = pygame.key.get_pressed()
        forward_x = math.sin(self.camera_angle_x)
        forward_z = math.cos(self.camera_angle_x)
        
        if keys[K_w]: self.cam_x += forward_x * self.move_speed; self.cam_z += forward_z * self.move_speed
        if keys[K_s]: self.cam_x -= forward_x * self.move_speed; self.cam_z -= forward_z * self.move_speed
        if keys[K_a]: self.cam_x += forward_z * self.move_speed; self.cam_z -= forward_x * self.move_speed
        if keys[K_d]: self.cam_x -= forward_z * self.move_speed; self.cam_z += forward_x * self.move_speed
        
        self.cam_x = max(0.05, min(0.95, self.cam_x))
        self.cam_z = max(0.05, min(0.95, self.cam_z))

        mouse_click = pygame.mouse.get_pressed()
        if self.shoot_cooldown > 0.0:
            self.shoot_cooldown -= dt
            
        if mouse_click[0] and self.shoot_cooldown <= 0.0: # Чтение ЛКМ без конфликтов кортежа mouse_click
            slots = cp.flatnonzero(bullets_active == 0.0)
            if slots.size > 0:
                # ИСПРАВЛЕНО: Извлекаем строго первый свободный элемент [0] во избежание ошибки маштабирования массивов
                idx = int(slots[0])
                bullets_pos[idx, 0] = self.cam_x * 128.0
                bullets_pos[idx, 1] = self.cam_y * 128.0
                bullets_pos[idx, 2] = self.cam_z * 128.0
                
                bullets_vel[idx, 0] = math.cos(self.camera_angle_y) * math.sin(self.camera_angle_x) * 110.0
                bullets_vel[idx, 1] = math.sin(self.camera_angle_y) * 110.0
                bullets_vel[idx, 2] = math.cos(self.camera_angle_y) * math.cos(self.camera_angle_x) * 110.0
                
                bullets_active[idx] = 1.0
                self.total_ammo_fired += 1
                self.shoot_cooldown = self.fire_rate

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()

    def run(self):
        threads = (8, 8, 8)
        blocks = (VOXEL_RES // 8, VOXEL_RES // 8, VOXEL_RES // 8)
        
        global grid_cupy, block_hp, part_pos, part_vel, part_active, part_life, part_size, part_count, bullets_pos, bullets_vel, bullets_active

        while True:
            dt = self.clock.tick(142) * 0.001
            self.time_elapsed += dt
            
            self.process_input(dt)
            
            bullets_pos += bullets_vel * dt * bullets_active[:, np.newaxis]
            bullets_active[(bullets_pos[:, 0] < 0.0) | (bullets_pos[:, 0] > 128.0) | 
                           (bullets_pos[:, 1] < 0.0) | (bullets_pos[:, 1] > 128.0) | 
                           (bullets_pos[:, 2] < 0.0) | (bullets_pos[:, 2] > 128.0)] = 0.0

            omni_destructor_and_gravity_kernel[blocks, threads](
                grid_cupy, block_hp, bullets_pos, bullets_vel, bullets_active,
                BULLET_CALIBER, EXPLOSIVE_POWER, False, part_pos, part_vel, part_active, part_life, part_size, part_count, self.time_elapsed
            )

            current_p_count = int(part_count.item())
            if current_p_count > 0:
                p_blocks = math.ceil(min(current_p_count, MAX_PARTICLES) / 256)
                update_particles_physics_kernel[p_blocks, 256](
                    part_pos, part_vel, part_active, part_life, part_size, min(current_p_count, MAX_PARTICLES), grid_cupy
                )

            # Безопасная выгрузка вокселей из VRAM в RAM-буфер по правилу C-contiguous
            grid_cupy.get(out=self.cpu_grid_buffer)
            self.texture_3d.write(self.cpu_grid_buffer)
            self.texture_3d.build_mipmaps()

            bullets_pos.get(out=self.bullets_pos_cpu)
            bullets_active.get(out=self.bullets_act_cpu)

            self.ctx.clear(0.0, 0.0, 0.0)
            self.texture_3d.use(location=0)
            self.prog['volumeTex'].value = 0
            self.prog['camAngleX'].value = self.camera_angle_x
            self.prog['camAngleY'].value = self.camera_angle_y
            self.prog['camPos'].value = (self.cam_x, self.cam_y, self.cam_z)
            self.prog['bulletsPos'].write(self.bullets_pos_cpu)
            self.prog['bulletsActive'].write(self.bullets_act_cpu)
            
            self.vao.render(moderngl.TRIANGLE_STRIP)
            pygame.display.flip()
            
            current_fps = int(self.clock.get_fps())
            pygame.display.set_caption(f"Omnivox FPS Engine | FPS: {current_fps} | AMMO FIRED: {self.total_ammo_fired}")

if __name__ == "__main__":
    engine = OmnivoxEngine()
    engine.run()
