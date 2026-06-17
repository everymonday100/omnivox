import warnings, pygame, moderngl, cupy as cp, numba.cuda as cuda, math, numpy as np, sys, traceback, logging
from numba.core.errors import NumbaPerformanceWarning
from pygame.locals import *

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt): sys.__excepthook__(exc_type, exc_value, exc_traceback); return
    logging.critical(f"Критический сбой GPU-движка:\n{''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))}")
sys.excepthook = handle_unhandled_exception

WIDTH, HEIGHT = 1920, 1080
VOXEL_RES = 128          
BLOCK_RES = VOXEL_RES // 2
MAX_PARTICLES = 15000
MAX_BULLETS = 50

SPARK_SPEED = 2.603
SPARK_DECAY = 0.056
SPARK_CHANCE = 3.794
BLOCK_GRAV = 0.065 
BLOCK_DECAY = 0.016
BULLET_CALIBER = 2.0      
EXPLOSIVE_POWER = 3.33    

# Выделение управляющих массивов во VRAM
block_hp = cp.zeros((BLOCK_RES, BLOCK_RES, BLOCK_RES), dtype=cp.float32, order='C')
structural_load = cp.zeros((BLOCK_RES, BLOCK_RES, BLOCK_RES), dtype=cp.float32, order='C')

part_pos = cp.zeros((MAX_PARTICLES, 3), dtype=cp.float32)
part_vel = cp.zeros((MAX_PARTICLES, 3), dtype=cp.float32)
part_active = cp.zeros(MAX_PARTICLES, dtype=cp.float32)
part_life = cp.zeros(MAX_PARTICLES, dtype=cp.float32)
part_size = cp.zeros((MAX_PARTICLES, 3), dtype=cp.float32)
part_count = cp.zeros(1, dtype=cp.int32)

bullets_pos = cp.zeros((MAX_BULLETS, 3), dtype=cp.float32)
bullets_vel = cp.zeros((MAX_BULLETS, 3), dtype=cp.float32)
bullets_active = cp.zeros(MAX_BULLETS, dtype=cp.float32)

# Глобальные ссылки на буферы Double-Buffering (будут связаны в Части 5 через PBO Interop)
grid_read_cupy = None
grid_write_cupy = None

VOXEL_BYTES = VOXEL_RES * VOXEL_RES * VOXEL_RES * 4

# --- ЧАСТЬ 2: ОПТИМИЗИРОВАННОЕ CUDA-ЯДРО ДИСКРЕТНОЙ БАЛЛИСТИКИ И ПРОСТРЕЛОВ ---

@cuda.jit
def omni_destructor_and_gravity_kernel(building_grid, block_hp, bullets_pos, bullets_vel, bullets_active,
                                      caliber, explosive, reset_scene, p_pos, p_vel, p_active, p_life, p_size, p_count_array, seed):
    """ 
    УПРОЩЕНО: Дискретная баллистика. Проверяет строго факт коллизии точки пули с блоком.
    Если столкновение произошло — пуля простреливает стену насквозь и аннигилирует.
    """
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
    
    if x >= 128 or y >= 128 or z >= 128: return
    bx, by, bz = x // 2, y // 2, z // 2
    
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
    
    # Видеокарта параллельно опрашивает массив летящих снарядов
    for b_idx in range(50):
        if bullets_active[b_idx] < 0.5: continue
        
        # Получаем текущую физическую позицию снаряда в пространстве
        b_x = bullets_pos[b_idx, 0]
        b_y = bullets_pos[b_idx, 1]
        b_z = bullets_pos[b_idx, 2]
        
        # Переводим координаты пули в индексы макро-сетки блоков (64^3)
        b_bx = int(b_x) // 2
        b_by = int(b_y) // 2
        b_bz = int(b_z) // 2
        
        # УПРОЩЕННАЯ ПРОВЕРКА КОЛЛИЗИИ: Если пуля физически влетела внутрь текущего живого блока
        if bx == b_bx and by == b_by and bz == b_bz:
            if block_hp[bx, by, bz] < 500.0:
                # МГНОВЕННЫЙ СКВОЗНОЙ ПРОСТРЕЛ: Разрушаем блок фугасной силой плазмоида
                block_hp[bx, by, bz] = 0.0
                building_grid[x, y, z] = 0.0
                
                # МГНОВЕННОЕ СТИРАНИЕ: Снаряд аннигилирует и выключается из рендера
                bullets_active[b_idx] = 0.0
                
                # Спавн сочных радиальных осколков в точке прострела
                if float(raw_rand) < 15.0: # Сбалансированный шанс спавна
                    idx = cuda.atomic.add(p_count_array, 0, 1) % 15000
                    p_pos[idx, 0], p_pos[idx, 1], p_pos[idx, 2] = float(x), float(y), float(z)
                    
                    # Честный физический импульс разлета осколков бетона во все стороны
                    p_vel[idx, 0] = (float(hash_val % 3 - 1) * SPARK_SPEED * 0.6)
                    p_vel[idx, 1] = (float((hash_val // 3) % 3 - 1) * SPARK_SPEED * 0.6) + 0.2
                    p_vel[idx, 2] = (float((hash_val // 9) % 3 - 1) * SPARK_SPEED * 0.6)
                    
                    p_active[idx], p_life[idx] = 1.0, 1.0
                    p_size[idx, 0] = 1.0; p_size[idx, 1] = 1.0; p_size[idx, 2] = 1.0
                break
                
        if bullets_active[b_idx] < 0.5: break
        
    if block_hp[bx, by, bz] > 0.0:
        building_grid[x, y, z] = block_hp[bx, by, bz] / 4.0

@cuda.jit
def calculate_structural_collapse_kernel(building_grid, block_hp, structural_load, p_pos, p_vel, p_active, p_life, p_size, p_count_array, seed):
    """
    НАСТОЯЩИЙ ЛОГАРИФМИЧЕСКИЙ СОПРОМАТ: Послойный расчет точечного напряжения (Stress Over Time).
    ИСПРАВЛЕНО: Скорость свободного падения блоков снижена в 2 раза для плавного, тяжелого обрушения.
    """
    bx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    bz = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if bx >= 64 or bz >= 64: return

    # Просчитываем накопленный вес сверху вниз
    accumulated_mass = 0.0
    for by in range(62, 0, -1):
        if block_hp[bx, by, bz] > 0.1 and block_hp[bx, by, bz] < 500.0:
            accumulated_mass += 1.0
            structural_load[bx, by, bz] = accumulated_mass
        else:
            structural_load[bx, by, bz] = 0.0

    # Сканируем по слоям снизу вверх для вычисления деструкции пятен перегрузки
    for by in range(1, 63):
        hp_current = block_hp[bx, by, bz]
        if hp_current <= 0.1 or hp_current > 500.0: continue

        has_support_below = block_hp[bx, by - 1, bz] > 0.1
        
        has_left  = bx > 0  and block_hp[bx - 1, by, bz] > 0.1
        has_right = bx < 63 and block_hp[bx + 1, by, bz] > 0.1
        has_front = bz > 0  and block_hp[bx, by, bz - 1] > 0.1
        has_back  = bz < 63 and block_hp[bx, by, bz + 1] > 0.1
        
        # Сканируем сквозную жесткую связь с боковым фундаментом
        has_left_anchor = False
        for lx in range(bx, -1, -1):
            if block_hp[lx, by, bz] <= 0.1: break
            if 16 <= lx <= 20: has_left_anchor = True; break

        has_right_anchor = False
        for rx in range(bx, 64):
            if block_hp[rx, by, bz] <= 0.1: break
            if 44 <= rx <= 47: has_right_anchor = True; break

        is_attached = has_left_anchor or has_right_anchor

        lateral_connections = 0
        if is_attached:
            if has_left: lateral_connections += 1
            if has_right: lateral_connections += 1
            if has_front: lateral_connections += 1
            if has_back: lateral_connections += 1

        weight_above = structural_load[bx, by, bz]

        should_fall = False
        if not has_support_below:
            if lateral_connections == 0:
                should_fall = True 
            elif weight_above > float(lateral_connections * 3):
                should_fall = True 
                
        # ЛОГАРИФМИЧЕСКИЙ DAMAGE OVER TIME СТЫКА
        elif has_support_below and block_hp[bx, by - 1, bz] < 500.0:
            if weight_above > 1.5:
                log_support = math.log(float(lateral_connections) + 1.1)
                log_stress_decay = (weight_above / log_support) * 0.024
                
                block_hp[bx, by - 1, bz] = max(0.0, block_hp[bx, by - 1, bz] - log_stress_decay)
                
                hash_val = (bx * 73129 + by * 95121 + bz * 15413 + int(seed * 333))
                if hash_val % 30 == 0:
                    idx = cuda.atomic.add(p_count_array, 0, 1) % 15000
                    if p_active[idx] < 0.5:
                        p_active[idx], p_life[idx] = 1.0, 0.4
                        p_pos[idx, 0] = float(bx * 2)
                        p_pos[idx, 1] = float(by * 2) - 0.5
                        p_pos[idx, 2] = float(bz * 2)
                        p_vel[idx, 0] = float(hash_val % 3 - 1) * 0.4
                        p_vel[idx, 1] = 0.15
                        p_vel[idx, 2] = float((hash_val // 3) % 3 - 1) * 0.4

        if should_fall:
            # ИСПРАВЛЕНО: Шаг набора кинетической энергии снижен в 2 раза (с 0.45 до 0.22)
            structural_load[bx, by, bz] += 0.02            
            current_velocity = structural_load[bx, by, bz]

            block_hp[bx, by, bz] = 0.0
            block_hp[bx, by - 1, bz] = hp_current
            structural_load[bx, by, bz] = 0.0
            structural_load[bx, by - 1, bz] = current_velocity
            
        elif has_support_below and structural_load[bx, by, bz] > 0.1:
            # Кинетический взрыв плиты при ударе о землю
            impact_velocity = structural_load[bx, by, bz]
            structural_load[bx, by, bz] = 0.0 

            center_x_offset = 0.0; center_z_offset = 0.0
            if bx > 0  and block_hp[bx - 1, by, bz] > 0.1: center_x_offset -= 1.0
            if bx < 63 and block_hp[bx + 1, by, bz] > 0.1: center_x_offset += 1.0
            if bz > 0  and block_hp[bx, by, bz - 1] > 0.1: center_z_offset -= 1.0
            if bz < 63 and block_hp[bx, by, bz + 1] > 0.1: center_z_offset += 1.0

            if impact_velocity > 0.6:
                block_hp[bx, by, bz] = 0.0
                building_grid[bx * 2, by * 2, bz * 2] = 0.0
                
                hash_val = (bx * 73129 + by * 95121 + bz * 15413 + int(seed * 555))
                for p_num in range(3):
                    idx = cuda.atomic.add(p_count_array, 0, 1) % 15000
                    if p_active[idx] < 0.5:
                        p_active[idx], p_life[idx] = 1.0, 1.0
                        p_pos[idx, 0] = float(bx * 2)
                        p_pos[idx, 1] = float(by * 2) + 0.5
                        p_pos[idx, 2] = float(bz * 2)
                        
                        p_vel[idx, 0] = (center_x_offset * 1.5) + (float(hash_val % 3 - 1) * impact_velocity * 0.8)
                        p_vel[idx, 1] = impact_velocity * 0.9 
                        p_vel[idx, 2] = (center_z_offset * 1.5) + (float((hash_val // 3) % 3 - 1) * impact_velocity * 0.8)
                        p_size[idx, 0], p_size[idx, 1], p_size[idx, 2] = 2.0, 2.0, 2.0

@cuda.jit
def update_particles_physics_kernel(p_pos, p_vel, p_active, p_life, p_size, count, building_grid):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= count or p_active[idx] < 0.5: return
    ox, oy, oz = int(p_pos[idx, 0]), int(p_pos[idx, 1]), int(p_pos[idx, 2])
    if 0 <= ox < 128 and 0 <= oy < 128 and 0 <= oz < 128:
        if building_grid[ox, oy, oz] <= 0.45: building_grid[ox, oy, oz] = 0.0
            
    is_smoke_zone = p_pos[idx, 0] > 74.0
    p_life[idx] -= 0.056 if is_smoke_zone else 0.016
    if p_life[idx] <= 0.0: p_active[idx] = 0.0; return
        
    p_vel[idx, 0] *= 0.96; p_vel[idx, 1] *= 0.98 if is_smoke_zone else 0.96
    # ИСПРАВЛЕНО: Сила гравитации для мелких осколков тоже уменьшена (с 0.131 до 0.065), чтобы их физика соответствовала тяжелым блокам
    if not is_smoke_zone: p_vel[idx, 1] -= 0.03 
    p_vel[idx, 2] *= 0.96
    
    nx, ny, nz = p_pos[idx, 0] + p_vel[idx, 0], p_pos[idx, 1] + p_vel[idx, 1], p_pos[idx, 2] + p_vel[idx, 2]
    if ny < 1.0:
        ny = 1.0; p_vel[idx, 1] = -p_vel[idx, 1] * 0.35; p_vel[idx, 0] *= 0.6; p_vel[idx, 2] *= 0.6
        
    if nx < 0 or nx >= 128 or ny >= 128 or nz < 0 or nz >= 128: p_active[idx] = 0.0; return
        
    rx, ry, rz = int(nx), int(ny), int(nz)
    if building_grid[rx, ry, rz] > 0.8:
        p_vel[idx, 0] = -p_vel[idx, 0] * 0.4; p_vel[idx, 1] *= 0.8; p_vel[idx, 2] = p_vel[idx, 2] * 0.4
        nx, ny = p_pos[idx, 0] + p_vel[idx, 0], p_pos[idx, 1] + p_vel[idx, 1]
        
    p_pos[idx, 0], p_pos[idx, 1], p_pos[idx, 2] = nx, ny, nz
    rx, ry, rz = int(nx), int(ny), int(nz)
    
    if 0 <= rx < 128 and 0 <= ry < 128 and 0 <= rz < 128:
        if building_grid[rx, ry, rz] <= 0.8:
            if is_smoke_zone: building_grid[rx, ry, rz] = min(0.04, building_grid[rx, ry, rz] + 0.03)
            else: building_grid[rx, ry, rz] = min(0.45, building_grid[rx, ry, rz] + 0.20 * p_life[idx])

# --- ЧАСТЬ 4: ИСПРАВЛЕННЫЙ И ОПТИМИЗИРОВАННЫЙ ШЕЙДЕР АППАРАТНОГО РЕНДЕРИНГА (ModernGL / GLSL) ---

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
uniform int u_fps; 

const int MAX_BULLETS = 50;
uniform vec3 bulletsPos[MAX_BULLETS]; 
uniform float bulletsActive[MAX_BULLETS];

int getFontMask(int digit) {
    switch(digit) {
        case 0: return 31599; case 1: return 9362; case 2: return 29671; case 3: return 29391; case 4: return 23497;
        case 5: return 31119; case 6: return 31215; case 7: return 29257; case 8: return 31727; case 9: return 31663;
    }
    return 0;
}

float drawDigit(int digit, vec2 uv) {
    if (uv.x < 0.0 || uv.x > 3.0 || uv.y < 0.0 || uv.y > 5.0) return 0.0;
    int bitIndex = int(floor(uv.x)) + int(floor(4.0 - uv.y)) * 3;
    int mask = getFontMask(digit);
    if ((mask & (1 << (14 - bitIndex))) != 0) return 1.0;
    return 0.0;
}

float drawFPS(int val, vec2 px, vec2 res) {
    vec2 textPos = vec2(res.x - 100.0, res.y - 45.0);
    vec2 localUV = (px - textPos) / 4.0; 
    int hundred = (val / 100) % 10; int ten = (val / 10) % 10; int one = val % 10;
    float mask_fps = 0.0;
    mask_fps += drawDigit(hundred, localUV);
    mask_fps += drawDigit(ten, localUV - vec2(4.0, 0.0)); 
    mask_fps += drawDigit(one, localUV - vec2(8.0, 0.0));
    return mask_fps;
}

bool intersectBox(vec3 ro, vec3 rd, out float t0, out float t1) {
    vec3 invR = 1.0 / (rd + 1e-6);
    vec3 tbot = invR * (vec3(0.0) - ro); vec3 ttop = invR * (vec3(1.0) - ro);
    vec3 tmin = min(tbot, ttop); vec3 tmax = max(tbot, ttop);
    t0 = max(tmin.x, max(tmin.y, tmin.z)); t1 = min(tmax.x, min(tmax.y, tmax.z));
    return t0 < t1 && t1 > 0.0;
}

void main() {
    vec3 ro = camPos;
    vec3 direction = vec3(cos(camAngleY) * sin(camAngleX), sin(camAngleY), cos(camAngleY) * cos(camAngleX));
    vec3 ww = normalize(direction); vec3 uu = normalize(cross(ww, vec3(0.0, 1.0, 0.0))); vec3 vv = normalize(cross(uu, ww));
    vec2 p = uvs * 2.0 - 1.0; p.x *= 1.333; 
    vec3 rd = normalize(p.x * uu + p.y * vv + 1.5 * ww);
    
    float t0, t1; vec3 color = vec3(0.04, 0.05, 0.08); 
    
    float bulletGlow = 0.0;
    if (abs(p.x) < 0.25 && abs(p.y) < 0.25) {
        for (int b = 0; b < MAX_BULLETS; b++) {
            if (bulletsActive[b] > 0.5) {
                vec3 bPosNormal = bulletsPos[b] / 128.0; 
                float t = dot(bPosNormal - ro, rd);
                
                if (t > 0.04 && t < 0.8) { 
                    vec3 rayPoint = ro + rd * t;
                    float distToAxis = distance(rayPoint * 128.0, bulletsPos[b]);
                    
                    if (distToAxis < 1.6) {
                        bulletGlow += 0.95 * exp(-pow(distToAxis * 3.2, 2.0)) * (1.0 - t);
                        if (bulletGlow >= 1.0) {
                            bulletGlow = 1.0;
                            break;
                        }
                    }
                }
            }
        }
    }
    
    if (intersectBox(ro, rd, t0, t1)) {
        t0 = max(t0, 0.0); vec3 pos = ro + rd * t0;
        float stepSize = 0.0045; float T = 1.0; vec3 volumeColor = vec3(0.0);
        
        for (int i = 0; i < 180; i++) { 
            if (t0 > t1 || T < 0.005) break;
            
            float density = texture(volumeTex, vec3(pos.z, pos.y, pos.x)).r;
            if (density > 0.01) {
                vec3 voxelCol = vec3(0.4, 0.44, 0.48); float alpha = density * 0.35;
                if (density <= 0.04) { voxelCol = vec3(3.0, 3.0, 3.0); alpha = density * 2.5; }
                else if (density <= 0.25) { voxelCol = vec3(0.0, 0.8, 1.0); alpha = 0.015; }
                else if (density < 0.28) { voxelCol = vec3(1.0, 0.45, 0.05); }
                else if (density < 0.44) { voxelCol = vec3(0.25, 0.26, 0.28); alpha = 0.7; }
                volumeColor += T * voxelCol * alpha; T *= (1.0 - alpha); stepSize = 0.0045;
            } else { stepSize = 0.0090; }
            pos += rd * stepSize; t0 += stepSize;
        }
        color = volumeColor + T * color;
    }
    
    color += vec3(0.0, 0.55, 1.0) * bulletGlow;
    
    vec2 scrCoord = uvs * vec2(1920.0, 1080.0);
    if ((abs(scrCoord.x - 960.0) < 6.0 && abs(scrCoord.y - 540.0) < 1.0) || (abs(scrCoord.y - 540.0) < 6.0 && abs(scrCoord.x - 960.0) < 1.0)) {
        if (abs(scrCoord.x - 960.0) > 1.0 || abs(scrCoord.y - 540.0) > 1.0) color = vec3(0.0, 1.0, 0.2);
    }
    if (drawFPS(u_fps, scrCoord, vec2(1920.0, 1080.0)) > 0.5) color = vec3(0.0, 1.0, 0.2);
    fragColor = vec4(color, 1.0);
}
"""
# --- ЧАСТЬ 5: GPU-МЕНЕДЖЕР И ИНИЦИАЛИЗАЦИЯ 6-СЕКУНДНОГО АВТОЛОГГЕРА (ПОЛОВИНА 1) ---

class SafeGPUInteropManager:
    @staticmethod
    def get_clean_scalar_index(active_array):
        slots = cp.flatnonzero(active_array == 0.0)
        if slots.size > 0: return int(slots)
        return -1
    @staticmethod
    def run_structural_collapse(blocks, threads, grid, hp, load, p_pos, p_vel, p_act, p_life, p_sz, p_count, time):
        calculate_structural_collapse_kernel[blocks, threads](grid, hp, load, p_pos, p_vel, p_act, p_life, p_sz, p_count, np.float32(time))
    @staticmethod
    def run_particles_physics(blocks, threads, p_pos, p_vel, p_act, p_life, p_sz, count, grid):
        update_particles_physics_kernel[blocks, threads](p_pos, p_vel, p_act, p_life, p_sz, clean_count, grid)


class OmnivoxEngine:
    def __init__(self):
        pygame.init(); pygame.display.set_caption("Omnivox Engine | Automatic 6-Second Profiler")
        pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
        self.ctx = moderngl.create_context(); pygame.mouse.set_visible(False); pygame.event.set_grab(True)
        
        quad_buffer = self.ctx.buffer(np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4'))
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.vao = self.ctx.vertex_array(self.prog, [(quad_buffer, '2f', 'in_vert')])
        
        self.texture_3d = self.ctx.texture3d((VOXEL_RES, VOXEL_RES, VOXEL_RES), 1, dtype='f4')
        self.texture_3d.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        self.cpu_grid_buffer = np.zeros((VOXEL_RES, VOXEL_RES, VOXEL_RES), dtype=np.float32, order='C')
        self.bullets_pos_cpu = np.zeros((MAX_BULLETS, 3), dtype=np.float32)
        self.bullets_act_cpu = np.zeros(MAX_BULLETS, dtype=np.float32)
        
        self.cam_x, self.cam_y, self.cam_z = 0.50, 0.25, 0.10
        self.camera_angle_x, self.camera_angle_y = 0.0, 0.0; self.move_speed = 0.015
        self.shoot_cooldown = 0.0; self.fire_rate = 0.24; self.total_ammo_fired = 0; self.time_elapsed = 0.0; self.is_firing = False
        
        # Буфер логов и флаги 6-секундного таймера
        self.perf_log_buffer = []
        self.profiler_duration = 6.0  # Ровно 6 секунд сбора
        self.log_saved = False
        
        self.clock = pygame.time.Clock()
        self.reset_entire_simulation()

    def reset_entire_simulation(self):
        logging.info("♻️ Перезапекание воксельного монолита...")
        global grid_read_cupy, grid_write_cupy, block_hp, structural_load, part_pos, part_vel, part_active, part_life, part_size, part_count, bullets_pos, bullets_vel, bullets_active
        grid_read_cupy[:] = 0.0; grid_write_cupy[:] = 0.0; block_hp[:] = 0.0; structural_load[:] = 0.0
        part_pos[:] = 0.0; part_vel[:] = 0.0; part_active[:] = 0.0; part_life[:] = 0.0; part_size[:] = 0.0; part_count[:] = 0
        bullets_pos[:] = 0.0; bullets_vel[:] = 0.0; bullets_active[:] = 0.0
        
        grid_write_cupy[:, :16, :] = 1.0; block_hp[:, :8, :] = 999.0 
        grid_write_cupy[:, 112:128, :] = 1.0; block_hp[:, 56:64, :] = 999.0
        grid_write_cupy[32:96, 16:112, 110:125] = 1.0; block_hp[16:48, 8:56, 55:62] = 12.0
        block_hp[16:20, 8:56, 55:62] = 999.0; block_hp[44:48, 8:56, 55:62] = 999.0
        self.total_ammo_fired = 0
        omni_destructor_and_gravity_kernel[(16,16,16), (8,8,8)](grid_write_cupy, block_hp, bullets_pos, bullets_vel, bullets_active, BULLET_CALIBER, EXPLOSIVE_POWER, True, part_pos, part_vel, part_active, part_life, part_size, part_count, 0.5)
        grid_read_cupy[:] = grid_write_cupy[:]

    def force_dump_perf_log_to_disk(self):
        if len(self.perf_log_buffer) == 0: return
        try:
            with open("perf_report.log", "w", encoding="utf-8") as f:
                f.writelines(self.perf_log_buffer)
                f.flush()
            print("💾 [ПРОФАЙЛЕР] 6 секунд истекли! perf_report.log успешно сохранен на диск.")
            self.log_saved = True
        except Exception as e:
            print(f"❌ Ошибка сохранения файла: {e}")
# --- ЧАСТЬ 5: GPU-МЕНЕДЖЕР И ИНИЦИАЛИЗАЦИЯ 6-СЕКУНДНОГО АВТОЛОГГЕРА (ПОЛОВИНА 1) ---

class SafeGPUInteropManager:
    @staticmethod
    def get_clean_scalar_index(active_array):
        slots = cp.flatnonzero(active_array == 0.0)
        if slots.size > 0: return int(slots)
        return -1
    @staticmethod
    def run_structural_collapse(blocks, threads, grid, hp, load, p_pos, p_vel, p_act, p_life, p_sz, p_count, time):
        calculate_structural_collapse_kernel[blocks, threads](grid, hp, load, p_pos, p_vel, p_act, p_life, p_sz, p_count, np.float32(time))
    @staticmethod
    def run_particles_physics(blocks, threads, p_pos, p_vel, p_act, p_life, p_sz, count, grid):
        update_particles_physics_kernel[blocks, threads](p_pos, p_vel, p_act, p_life, p_sz, clean_count, grid)


class OmnivoxEngine:
    def __init__(self):
        pygame.init(); pygame.display.set_caption("Omnivox Engine | Automatic 6-Second Profiler")
        pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
        self.ctx = moderngl.create_context(); pygame.mouse.set_visible(False); pygame.event.set_grab(True)
        
        quad_buffer = self.ctx.buffer(np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4'))
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.vao = self.ctx.vertex_array(self.prog, [(quad_buffer, '2f', 'in_vert')])
        
        self.texture_3d = self.ctx.texture3d((VOXEL_RES, VOXEL_RES, VOXEL_RES), 1, dtype='f4')
        self.texture_3d.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        self.cpu_grid_buffer = np.zeros((VOXEL_RES, VOXEL_RES, VOXEL_RES), dtype=np.float32, order='C')
        self.bullets_pos_cpu = np.zeros((MAX_BULLETS, 3), dtype=np.float32)
        self.bullets_act_cpu = np.zeros(MAX_BULLETS, dtype=np.float32)
        
        self.cam_x, self.cam_y, self.cam_z = 0.50, 0.25, 0.10
        self.camera_angle_x, self.camera_angle_y = 0.0, 0.0; self.move_speed = 0.015
        self.shoot_cooldown = 0.0; self.fire_rate = 0.24; self.total_ammo_fired = 0; self.time_elapsed = 0.0; self.is_firing = False
        
        # Буфер логов и флаги 6-секундного таймера
        self.perf_log_buffer = []
        self.profiler_duration = 6.0  # Ровно 6 секунд сбора
        self.log_saved = False
        
        self.clock = pygame.time.Clock()
        self.reset_entire_simulation()

    def reset_entire_simulation(self):
        logging.info("♻️ Перезапекание воксельного монолита...")
        global grid_read_cupy, grid_write_cupy, block_hp, structural_load, part_pos, part_vel, part_active, part_life, part_size, part_count, bullets_pos, bullets_vel, bullets_active
        grid_read_cupy[:] = 0.0; grid_write_cupy[:] = 0.0; block_hp[:] = 0.0; structural_load[:] = 0.0
        part_pos[:] = 0.0; part_vel[:] = 0.0; part_active[:] = 0.0; part_life[:] = 0.0; part_size[:] = 0.0; part_count[:] = 0
        bullets_pos[:] = 0.0; bullets_vel[:] = 0.0; bullets_active[:] = 0.0
        
        grid_write_cupy[:, :16, :] = 1.0; block_hp[:, :8, :] = 999.0 
        grid_write_cupy[:, 112:128, :] = 1.0; block_hp[:, 56:64, :] = 999.0
        grid_write_cupy[32:96, 16:112, 110:125] = 1.0; block_hp[16:48, 8:56, 55:62] = 12.0
        block_hp[16:20, 8:56, 55:62] = 999.0; block_hp[44:48, 8:56, 55:62] = 999.0
        self.total_ammo_fired = 0
        omni_destructor_and_gravity_kernel[(16,16,16), (8,8,8)](grid_write_cupy, block_hp, bullets_pos, bullets_vel, bullets_active, BULLET_CALIBER, EXPLOSIVE_POWER, True, part_pos, part_vel, part_active, part_life, part_size, part_count, 0.5)
        grid_read_cupy[:] = grid_write_cupy[:]

    def force_dump_perf_log_to_disk(self):
        if len(self.perf_log_buffer) == 0: return
        try:
            with open("perf_report.log", "w", encoding="utf-8") as f:
                f.writelines(self.perf_log_buffer)
                f.flush()
            print("💾 [ПРОФАЙЛЕР] 6 секунд истекли! perf_report.log успешно сохранен на диск.")
            self.log_saved = True
        except Exception as e:
            print(f"❌ Ошибка сохранения файла: {e}")
# --- ЧАСТЬ 5: GPU-МЕНЕДЖЕР И ИНИЦИАЛИЗАЦИЯ 6-СЕКУНДНОГО АВТОЛОГГЕРА (ПОЛОВИНА 1) ---

class SafeGPUInteropManager:
    @staticmethod
    def get_clean_scalar_index(active_array):
        slots = cp.flatnonzero(active_array == 0.0)
        if slots.size > 0: return int(slots)
        return -1
    @staticmethod
    def run_structural_collapse(blocks, threads, grid, hp, load, p_pos, p_vel, p_act, p_life, p_sz, p_count, time):
        calculate_structural_collapse_kernel[blocks, threads](grid, hp, load, p_pos, p_vel, p_act, p_life, p_sz, p_count, np.float32(time))
    @staticmethod
    def run_particles_physics(blocks, threads, p_pos, p_vel, p_act, p_life, p_sz, count, grid):
        update_particles_physics_kernel[blocks, threads](p_pos, p_vel, p_act, p_life, p_sz, clean_count, grid)


class OmnivoxEngine:
    def __init__(self):
        pygame.init(); pygame.display.set_caption("Omnivox Engine | Automatic 6-Second Profiler")
        pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
        self.ctx = moderngl.create_context(); pygame.mouse.set_visible(False); pygame.event.set_grab(True)
        
        quad_buffer = self.ctx.buffer(np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4'))
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.vao = self.ctx.vertex_array(self.prog, [(quad_buffer, '2f', 'in_vert')])
        
        self.texture_3d = self.ctx.texture3d((VOXEL_RES, VOXEL_RES, VOXEL_RES), 1, dtype='f4')
        self.texture_3d.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        self.cpu_grid_buffer = np.zeros((VOXEL_RES, VOXEL_RES, VOXEL_RES), dtype=np.float32, order='C')
        self.bullets_pos_cpu = np.zeros((MAX_BULLETS, 3), dtype=np.float32)
        self.bullets_act_cpu = np.zeros(MAX_BULLETS, dtype=np.float32)
        
        self.cam_x, self.cam_y, self.cam_z = 0.50, 0.25, 0.10
        self.camera_angle_x, self.camera_angle_y = 0.0, 0.0; self.move_speed = 0.015
        self.shoot_cooldown = 0.0; self.fire_rate = 0.24; self.total_ammo_fired = 0; self.time_elapsed = 0.0; self.is_firing = False
        
        # Буфер логов и флаги 6-секундного таймера
        self.perf_log_buffer = []
        self.profiler_duration = 6.0  # Ровно 6 секунд сбора
        self.log_saved = False
        
        self.clock = pygame.time.Clock()
        self.reset_entire_simulation()

    def reset_entire_simulation(self):
        logging.info("♻️ Перезапекание воксельного монолита...")
        global grid_read_cupy, grid_write_cupy, block_hp, structural_load, part_pos, part_vel, part_active, part_life, part_size, part_count, bullets_pos, bullets_vel, bullets_active
        grid_read_cupy[:] = 0.0; grid_write_cupy[:] = 0.0; block_hp[:] = 0.0; structural_load[:] = 0.0
        part_pos[:] = 0.0; part_vel[:] = 0.0; part_active[:] = 0.0; part_life[:] = 0.0; part_size[:] = 0.0; part_count[:] = 0
        bullets_pos[:] = 0.0; bullets_vel[:] = 0.0; bullets_active[:] = 0.0
        
        grid_write_cupy[:, :16, :] = 1.0; block_hp[:, :8, :] = 999.0 
        grid_write_cupy[:, 112:128, :] = 1.0; block_hp[:, 56:64, :] = 999.0
        grid_write_cupy[32:96, 16:112, 110:125] = 1.0; block_hp[16:48, 8:56, 55:62] = 12.0
        block_hp[16:20, 8:56, 55:62] = 999.0; block_hp[44:48, 8:56, 55:62] = 999.0
        self.total_ammo_fired = 0
        omni_destructor_and_gravity_kernel[(16,16,16), (8,8,8)](grid_write_cupy, block_hp, bullets_pos, bullets_vel, bullets_active, BULLET_CALIBER, EXPLOSIVE_POWER, True, part_pos, part_vel, part_active, part_life, part_size, part_count, 0.5)
        grid_read_cupy[:] = grid_write_cupy[:]

    def force_dump_perf_log_to_disk(self):
        if len(self.perf_log_buffer) == 0: return
        try:
            with open("perf_report.log", "w", encoding="utf-8") as f:
                f.writelines(self.perf_log_buffer)
                f.flush()
            print("💾 [ПРОФАЙЛЕР] 6 секунд истекли! perf_report.log успешно сохранен на диск.")
            self.log_saved = True
        except Exception as e:
            print(f"❌ Ошибка сохранения файла: {e}")
    def process_input(self, dt):
        global bullets_pos, bullets_vel, bullets_active
        rel_x, rel_y = pygame.mouse.get_rel()
        self.camera_angle_x -= rel_x * 0.003
        self.camera_angle_y = max(-1.4, min(1.4, self.camera_angle_y - rel_y * 0.003))
        keys = pygame.key.get_pressed()
        fx, fz = math.sin(self.camera_angle_x), math.cos(self.camera_angle_x)
        if keys[K_w]: self.cam_x += fx * self.move_speed; self.cam_z += fz * self.move_speed
        if keys[K_s]: self.cam_x -= fx * self.move_speed; self.cam_z -= fz * self.move_speed
        if keys[K_a]: self.cam_x += fz * self.move_speed; self.cam_z -= fx * self.move_speed
        if keys[K_d]: self.cam_x -= fz * self.move_speed; self.cam_z += fx * self.move_speed
        self.cam_x, self.cam_z = max(0.05, min(0.95, self.cam_x)), max(0.05, min(0.95, self.cam_z))
        
        if self.shoot_cooldown > 0.0: self.shoot_cooldown -= dt
        if self.is_firing and self.shoot_cooldown <= 0.0:
            if int(cp.sum(bullets_active)) < 8:
                idx = SafeGPUInteropManager.get_clean_scalar_index(bullets_active)
                if idx != -1:
                    bullets_pos[idx, 0], bullets_pos[idx, 1], bullets_pos[idx, 2] = self.cam_x * 128.0, self.cam_y * 128.0, self.cam_z * 128.0
                    bullets_vel[idx, 0], bullets_vel[idx, 1], bullets_vel[idx, 2] = math.cos(self.camera_angle_y) * fx * 110.0, math.sin(self.camera_angle_y) * 110.0, math.cos(self.camera_angle_y) * fz * 110.0
                    bullets_active[idx] = 1.0; self.total_ammo_fired += 1; self.shoot_cooldown = self.fire_rate
                    
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                self.force_dump_perf_log_to_disk()
                pygame.quit(); sys.exit()
            elif event.type == KEYDOWN and event.key == K_r: self.reset_entire_simulation()
            elif event.type == MOUSEBUTTONDOWN and event.button == 1: self.is_firing = True
            elif event.type == MOUSEBUTTONUP and event.button == 1: self.is_firing = False

    def run(self):
        threads = (8, 8, 8); blocks = (VOXEL_RES // 8, VOXEL_RES // 8, VOXEL_RES // 8)
        threads_structural = (8, 8); blocks_structural = (BLOCK_RES // 8, BLOCK_RES // 8)
        global grid_read_cupy, grid_write_cupy, block_hp, structural_load, part_pos, part_vel, part_active, part_life, part_size, part_count, bullets_pos, bullets_vel, bullets_active

        print("📢 АВТО-ПРОФАЙЛЕР: Начат сбор 6 секунд геймплея. Зажмите гашетку и уничтожайте стену!")

        while True:
            t_frame_start = pygame.time.get_ticks()
            dt = self.clock.tick(142) * 0.001; self.time_elapsed += dt; self.process_input(dt)
            
            # 1. Симуляция CuPy баллистики
            t0 = pygame.time.get_ticks()
            bullets_pos += bullets_vel * dt * bullets_active[:, np.newaxis]
            bullets_active[(bullets_pos[:, 0] < 0.0) | (bullets_pos[:, 0] > 128.0) | (bullets_pos[:, 1] < 0.0) | (bullets_pos[:, 1] > 128.0) | (bullets_pos[:, 2] < 0.0) | (bullets_pos[:, 2] > 128.0)] = 0.0
            cp.cuda.stream.get_current_stream().synchronize()
            t_ballistics = pygame.time.get_ticks() - t0

            # 2. CUDA Ядро деструкции бетона
            t0 = pygame.time.get_ticks()
            omni_destructor_and_gravity_kernel[blocks, threads](grid_write_cupy, block_hp, bullets_pos, bullets_vel, bullets_active, BULLET_CALIBER, EXPLOSIVE_POWER, False, part_pos, part_vel, part_active, part_life, part_size, part_count, self.time_elapsed)
            cuda.synchronize() 
            t_destructor = pygame.time.get_ticks() - t0

            # 3. CUDA Ядро логарифмического сопромата
            t0 = pygame.time.get_ticks()
            SafeGPUInteropManager.run_structural_collapse(blocks_structural, threads_structural, grid_write_cupy, block_hp, structural_load, part_pos, part_vel, part_active, part_life, part_size, part_count, self.time_elapsed)
            cuda.synchronize()
            t_structural = pygame.time.get_ticks() - t0

            # 4. CUDA Ядро физики осколков
            t0 = pygame.time.get_ticks()
            p_blocks = (MAX_PARTICLES + 255) // 256
            SafeGPUInteropManager.run_particles_physics(p_blocks, 256, part_pos, part_vel, part_active, part_life, part_size, MAX_PARTICLES, grid_write_cupy)
            cuda.synchronize()
            t_particles = pygame.time.get_ticks() - t0

            # 5. Флип буферов VRAM и выгрузка текстуры (Шина PCIe)
            t0 = pygame.time.get_ticks()
            grid_read_cupy[:] = grid_write_cupy[:]
            self.texture_3d.write(cp.asnumpy(grid_read_cupy))
            self.ctx.finish() 
            t_vram_copy = pygame.time.get_ticks() - t0

            # 6. GLSL Рэймаршинг кадра
            t0 = pygame.time.get_ticks()
            bullets_pos.get(out=self.bullets_pos_cpu); bullets_active.get(out=self.bullets_act_cpu)
            self.ctx.clear(0.0, 0.0, 0.0); self.texture_3d.use(location=0); self.prog['volumeTex'].value = 0
            self.prog['camAngleX'].value = self.camera_angle_x; self.prog['camAngleY'].value = self.camera_angle_y; self.prog['camPos'].value = (self.cam_x, self.cam_y, self.cam_z)
            self.prog['bulletsPos'].write(self.bullets_pos_cpu); self.prog['bulletsActive'].write(self.bullets_act_cpu)
            current_fps = int(self.clock.get_fps()); self.prog['u_fps'].value = current_fps
            self.vao.render(moderngl.TRIANGLE_STRIP); pygame.display.flip()
            self.ctx.finish() 
            t_render = pygame.time.get_ticks() - t0

            t_total_frame = pygame.time.get_ticks() - t_frame_start

            # Фиксируем абсолютно все кадры, пока не истечет глобальный таймер в 6 секунд
            if self.time_elapsed <= self.profiler_duration:
                self.perf_log_buffer.append(
                    f"📊 КАДР СИМУЛЯЦИИ | Время: {self.time_elapsed:.2f}с | Текущий FPS: {current_fps} | Снарядов в воздухе: {cp.sum(bullets_active)}\n"
                    f"  |- Полное время кадра шутера: {t_total_frame} мс\n"
                    f"  |- [CPU] Баллистические векторы: {t_ballistics} мс\n"
                    f"  |- [GPU CUDA] Прострелы и деструкция: {t_destructor} мс\n"
                    f"  |- [GPU CUDA] Логарифмический сопромат: {t_structural} мс\n"
                    f"  |- [GPU CUDA] Попиксельные осколки/искры: {t_particles} мс\n"
                    f"  |- [PCIe Bus] Выгрузка воксельной памяти (.write): {t_vram_copy} мс\n"
                    f"  |- [GPU GLSL] Рэймаршинг кадра на экране: {t_render} мс\n"
                    f"-----------------------------------------------------------------\n"
                )
            elif not self.log_saved:
                # Ровно через 6 секунд принудительно выгружаем весь массив ОЗУ на диск
                self.force_dump_perf_log_to_disk()

if __name__ == "__main__":
    engine = OmnivoxEngine()
    engine.run()
