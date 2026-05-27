import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

def crear_campo_vectorial(config):
    xF, yF, zF = config['fuente']
    xS, yS, zS = config['sumidero']
    fuerza_ext = config['fisica']['fuerza_extractor']
    vel_max = config['fisica']['v_max']
    
    obstaculos = config.get('obstaculos', [])

    def motor_física(x, y, z):
        singularidad_evitar = 0.3 
        
        dx_vent = xS - x
        dy_vent = yS - y
        dz_vent = zS - z
        dist_vent2 = dx_vent**2 + dy_vent**2 + dz_vent**2 + singularidad_evitar
        
        magnitud_pull = fuerza_ext / dist_vent2
        
        p_vent = magnitud_pull * (dx_vent / np.sqrt(dist_vent2))
        q_vent = magnitud_pull * (dy_vent / np.sqrt(dist_vent2))
        r_vent = magnitud_pull * (dz_vent / np.sqrt(dist_vent2))
        
        p_obs = np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
        q_obs = np.zeros_like(y) if isinstance(y, np.ndarray) else 0.0
        r_obs = np.zeros_like(z) if isinstance(z, np.ndarray) else 0.0

        for (xMin, xMax, yMin, yMax, zMin, zMax) in obstaculos:
            dist_x_min = x - xMin
            dist_x_max = xMax - x
            dist_y_min = y - yMin
            dist_y_max = yMax - y
            dist_z_min = z - zMin
            dist_z_max = zMax - z
            
            ejes_distancia = np.stack([dist_x_min, dist_x_max, dist_y_min, dist_y_max, dist_z_min, dist_z_max], axis=-1)
            eje_critico_idx = np.argmin(np.abs(ejes_distancia), axis=-1)
            dist_critica = np.min(np.abs(ejes_distancia), axis=-1)
            
            fuerzaRepulsion_motor = 200.0 * np.exp(-5.0 * dist_critica)
            
            for i, axis_idx in enumerate([0, 1, 2, 3, 4, 5]):
                indices = (eje_critico_idx == axis_idx)
                if axis_idx == 0: p_obs[indices] += fuerzaRepulsion_motor[indices]  
                elif axis_idx == 1: p_obs[indices] -= fuerzaRepulsion_motor[indices] 
                elif axis_idx == 2: q_obs[indices] += fuerzaRepulsion_motor[indices] 
                elif axis_idx == 3: q_obs[indices] -= fuerzaRepulsion_motor[indices] 
                elif axis_idx == 4: r_obs[indices] += fuerzaRepulsion_motor[indices] 
                elif axis_idx == 5: r_obs[indices] -= fuerzaRepulsion_motor[indices] 

        p_Final = p_vent + p_obs
        q_Final = q_vent + q_obs
        r_Final = r_vent + r_obs

        v_limite = 3.5 

        p_Final = np.clip(p_Final, -v_limite, v_limite)
        q_Final = np.clip(q_Final, -v_limite, v_limite)
        r_Final = np.clip(r_Final, -v_limite, v_limite)

        return p_Final, q_Final, r_Final
        
    return motor_física

def evaluar_eficiencia_volumetrica(config):
    campo_func = crear_campo_vectorial(config)
    
    resolucion_control = 0.8 
    xs = np.arange(0.5, config['limites']['x'], resolucion_control)
    ys = np.arange(0.5, config['limites']['y'], resolucion_control)
    zs = np.arange(0.5, config['limites']['z'], resolucion_control)
    X, Y, Z = np.meshgrid(xs, ys, zs)
    X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()
    
    dx_eval = config['sumidero'][0] - X_flat
    dy_eval = config['sumidero'][1] - Y_flat
    dz_eval = config['sumidero'][2] - Z_flat
    dist_eval = np.sqrt(dx_eval**2 + dy_eval**2 + dz_eval**2 + 10**-5)
    
    ux, uy, uz = dx_eval / dist_eval, dy_eval / dist_eval, dz_eval / dist_eval
    
    vx, vy, vz = campo_func(X_flat, Y_flat, Z_flat)
    
    eficiencia_flujo_util = (vx * ux) + (vy * uy) + (vz * uz)
    
    return np.mean(eficiencia_flujo_util)

def optimizar_sumidero_realista(config_base, resolucion=0.5):
    print(f"\n--- INICIANDO OPTIMIZACIÓN VOLUMÉTRICA INDUSTRIAL: {config_base['title'].upper()} ---")
    print(f"Mesa de trabajo (Fuente): {config_base['fuente']}")
    inicio_tiempo = time.time()
    
    x_max, y_max, z_max = config_base['limites']['x'], config_base['limites']['y'], config_base['limites']['z']
    
    xs_test = np.arange(0.5, x_max, resolucion)
    ys_test = np.arange(0.5, y_max, resolucion)
    zs_test = np.arange(0.5, z_max, resolucion)
    
    restricciones = config_base['fisica']['restricciones_sumidero']
    posiciones_a_evaluar = set()
    margen_frontera = 0.25 
    
    if restricciones.get('techo', True):
        for x in xs_test:
            for y in ys_test: 
                posiciones_a_evaluar.add((x, y, z_max - margen_frontera))
            
    if restricciones.get('paredes_x', True):
        for y in ys_test:
            for z in zs_test:
                posiciones_a_evaluar.add((margen_frontera, y, z))
                posiciones_a_evaluar.add((x_max - margen_frontera, y, z))
            
    if restricciones.get('paredes_y', True):
        for x in xs_test:
            for z in zs_test:
                posiciones_a_evaluar.add((x, margen_frontera, z))
                posiciones_a_evaluar.add((x, y_max - margen_frontera, z))

    posiciones_a_evaluar = list(posiciones_a_evaluar)
    total_iteraciones = len(posiciones_a_evaluar)
    
    mejor_alineacion_motor = -float('inf') 
    mejor_posicion = None
    iteracion_actual = 0
    obstaculos = config_base.get('obstaculos', [])
    
    for (x, y, z) in posiciones_a_evaluar:
        iteracion_actual += 1
        
        esta_dentro = False
        for (xMin, xMax, yMin, yMax, zMin, zMax) in obstaculos:
            if (xMin <= x <= xMax) and (yMin <= y <= yMax) and (zMin <= z <= zMax):
                esta_dentro = True
                break
        if esta_dentro: continue 
        
        config_base['sumidero'] = (x, y, z)
        alineacion_flujo = evaluar_eficiencia_volumetrica(config_base)
        
        if alineacion_flujo > mejor_alineacion_motor:
            mejor_alineacion_motor = alineacion_flujo
            mejor_posicion = (float(x), float(y), float(z)) 
            
        if iteracion_actual % max(1, (total_iteraciones // 5)) == 0:
            progreso = (iteracion_actual / total_iteraciones) * 100
            print(f"Progreso: {progreso:.0f}% de la estructura industrial evaluada...")

    fin_tiempo = time.time()
    
    print("\n================ RESULTADOS DE LA OPTIMIZACIÓN ================")
    print(f"Tiempo de cómputo: {fin_tiempo - inicio_tiempo:.2f} segundos")
    print(f"Posiciones reales evaluadas: {total_iteraciones}")
    print(f"--> MEJOR UBICACIÓN ESTRUCTURAL (Sumidero): {mejor_posicion}")
    print(f"--> ARRASTRE VOLUMÉTRICO LOGRADO EN CUARTO : {mejor_alineacion_motor:.4f} m/s útiles")
    print("===============================================================\n")
    
    return mejor_posicion

def ejecutar_simulacion(config):
    xLim, yLim, zLim = config['limites']['x'], config['limites']['y'], config['limites']['z']
    dt = config['dt']
    num_flechas = config['num_flechas']

    xF, yF, zF = config['fuente']
    xS, yS, zS = config['sumidero']
    
    obstaculos = list(config.get('obstaculos', [])) 
    pared = config.get('pared', None)
    
    if pared and len(obstaculos) == 0:
        xP, yP = pared['centro_x'], pared['centro_y']
        w, l = pared['ancho_x'] / 2.0, pared['largo_y'] / 2.0
        obstaculos.append([xP - w, xP + w, yP - l, yP + l, 0, zLim])

    part_lig = config['particulas']['ligeras']
    part_air = config['particulas']['aire']
    part_pes = config['particulas']['pesadas']
    num_fuente = part_lig + part_air + part_pes
    
    part_rand = config['particulas'].get('aleatorias', 0)
    num_total = num_fuente + part_rand

    def en_obstaculo(px, py, pz):
        mask = np.zeros_like(px, dtype=bool)
        for (xMin, xMax, yMin, yMax, zMin, zMax) in obstaculos:
            dentro = (px > xMin) & (px < xMax) & (py > yMin) & (py < yMax) & (pz > zMin) & (pz < zMax)
            mask = mask | dentro
        return mask

    xP_f = np.random.normal(xF, 0.04, num_fuente)
    yP_f = np.random.normal(yF, 0.04, num_fuente)
    zP_f = np.random.normal(zF, 0.04, num_fuente)
    
    malos_f = en_obstaculo(xP_f, yP_f, zP_f)
    while np.any(malos_f):
        xP_f[malos_f] = np.random.normal(xF, 0.04, np.sum(malos_f))
        yP_f[malos_f] = np.random.normal(yF, 0.04, np.sum(malos_f))
        zP_f[malos_f] = np.random.normal(zF, 0.04, np.sum(malos_f))
        malos_f = en_obstaculo(xP_f, yP_f, zP_f)

    xP_r = np.random.uniform(0.2, xLim - 0.2, part_rand)
    yP_r = np.random.uniform(0.2, yLim - 0.2, part_rand)
    zP_r = np.random.uniform(0.2, zLim - 0.2, part_rand)
    
    malos_r = en_obstaculo(xP_r, yP_r, zP_r)
    while np.any(malos_r):
        xP_r[malos_r] = np.random.uniform(0.2, xLim - 0.2, np.sum(malos_r))
        yP_r[malos_r] = np.random.uniform(0.2, yLim - 0.2, np.sum(malos_r))
        zP_r[malos_r] = np.random.uniform(0.2, zLim - 0.2, np.sum(malos_r))
        malos_r = en_obstaculo(xP_r, yP_r, zP_r)

    xParticulas = np.concatenate([xP_f, xP_r])
    yParticulas = np.concatenate([yP_f, yP_r])
    zParticulas = np.concatenate([zP_f, zP_r])

    flotabilidad = np.zeros(num_total)
    colores = np.empty(num_total, dtype=object)
    
    flotabilidad[0:part_lig] = config['fisica']['flot_ligera']
    colores[0:part_lig] = 'dimgray'
    flotabilidad[part_lig:part_lig + part_air] = 0.0
    colores[part_lig:part_lig + part_air] = 'royalblue'
    flotabilidad[part_lig + part_air:num_fuente] = config['fisica']['flot_pesada']
    colores[part_lig + part_air:num_fuente] = 'sienna'
    flotabilidad[num_fuente:] = 0.0
    colores[num_fuente:] = 'purple'

    frames_fuente = np.random.randint(0, 180, num_fuente)
    frames_rand = np.zeros(part_rand)
    frames_aparicion = np.concatenate([frames_fuente, frames_rand])

    activas = np.ones(num_total, dtype=bool)
    tiempo_final_fijo = 0.0
    fuerza_ext = config['fisica']['fuerza_extractor']

    def campoVectorialFuncion(x, y, z):
        singularidad_evitar = 0.3
        
        dx_vent, dy_vent, dz_vent = xS - x, yS - y, zS - z
        dist_vent2 = dx_vent**2 + dy_vent**2 + dz_vent**2 + singularidad_evitar
        magnitud_pull = fuerza_ext / dist_vent2
        
        p = magnitud_pull * (dx_vent / np.sqrt(dist_vent2))
        q = magnitud_pull * (dy_vent / np.sqrt(dist_vent2))
        r = magnitud_pull * (dz_vent / np.sqrt(dist_vent2))
        
        for (xMin, xMax, yMin, yMax, zMin, zMax) in obstaculos:
            cx = np.clip(x, xMin, xMax)
            cy = np.clip(y, yMin, yMax)
            cz = np.clip(z, zMin, zMax)

            vx, vy, vz = x - cx, y - cy, z - cz
            dist = np.sqrt(vx**2 + vy**2 + vz**2) + 1e-5
            nx, ny, nz = vx / dist, vy / dist, vz / dist

            dot_product = p * nx + q * ny + r * nz
            
            hacia_adentro = (dist < 0.1) & (dot_product < 0)

            p = np.where(hacia_adentro, p - dot_product * nx, p)
            q = np.where(hacia_adentro, q - dot_product * ny, q)
            r = np.where(hacia_adentro, r - dot_product * nz, r)

            push = 1.0 * np.exp(-10.0 * dist)
            p += push * nx
            q += push * ny
            r += push * nz

        v_limite = 3
        return np.clip(p, -v_limite, v_limite), np.clip(q, -v_limite, v_limite), np.clip(r, -v_limite, v_limite)

    fig = plt.figure(figsize=(12, 6))

    axis3d = fig.add_subplot(1, 2, 1, projection='3d')
    axis3d.set_box_aspect((xLim, yLim, zLim))
    axis3d.set_title('Visualización 3D')
    axis3d.set_xlim(0, xLim); axis3d.set_ylim(0, yLim); axis3d.set_zlim(0, zLim)
    axis3d.scatter([xF], [yF], [zF], color='green', s=200, alpha=0.6, label="Fuente")
    axis3d.scatter([xS], [yS], [zS], color='red', s=200, alpha=0.6, label="Extractor")

    axis2d = fig.add_subplot(1, 2, 2)
    axis2d.set_aspect('equal')
    axis2d.set_title('Planta 2D (Campo de Succión)')
    axis2d.set_xlim(0, xLim); axis2d.set_ylim(0, yLim)
    axis2d.scatter([xF], [yF], color='green', s=200, alpha=0.6)
    axis2d.scatter([xS], [yS], color='red', s=200, alpha=0.6)

    for (xMin, xMax, yMin, yMax, zMin, zMax) in obstaculos:
        axis3d.plot([xMin, xMax, xMax, xMin, xMin], [yMin, yMin, yMax, yMax, yMin], [zMin]*5, color='black', alpha=0.8)
        axis3d.plot([xMin, xMax, xMax, xMin, xMin], [yMin, yMin, yMax, yMax, yMin], [zMax]*5, color='black', alpha=0.8)
        
        for vx, vy in [(xMin, yMin), (xMax, yMin), (xMax, yMax), (xMin, yMax)]:
            axis3d.plot([vx, vx], [vy, vy], [zMin, zMax], color='black', alpha=0.8)
        axis2d.fill([xMin, xMax, xMax, xMin, xMin], [yMin, yMin, yMax, yMax, yMin], color='black', alpha=0.5)

    xGrid, yGrid = np.meshgrid(np.linspace(0.5, xLim - 0.5, num_flechas), np.linspace(0.5, yLim - 0.5, num_flechas))
    zGrid = np.full_like(xGrid, (zF + zS) / 2)
    pGrid, qGrid, _ = campoVectorialFuncion(xGrid, yGrid, zGrid)
    magnitud = np.sqrt(pGrid**2 + qGrid**2) + 1e-5
    axis2d.quiver(xGrid, yGrid, pGrid / magnitud, qGrid / magnitud, color='gray', alpha=0.3, pivot='middle')

    vivas_iniciales = activas & (0 >= frames_aparicion)
    animatedScatter3d = axis3d.scatter(xParticulas[vivas_iniciales], yParticulas[vivas_iniciales], zParticulas[vivas_iniciales], c=colores[vivas_iniciales], alpha=0.6, s=15)
    animatedScatter2d = axis2d.scatter(xParticulas[vivas_iniciales], yParticulas[vivas_iniciales], c=colores[vivas_iniciales], alpha=0.6, s=15)

    def update(frame):
        nonlocal xParticulas, yParticulas, zParticulas, activas, tiempo_final_fijo

        vivas_y_visibles = activas & (frame >= frames_aparicion)
        num_vivas = np.sum(vivas_y_visibles)

        if num_vivas > 0:
            p, q, r = campoVectorialFuncion(xParticulas[vivas_y_visibles], yParticulas[vivas_y_visibles], zParticulas[vivas_y_visibles])
            r += flotabilidad[vivas_y_visibles]

            ruido_x = np.random.normal(0, 0.015, num_vivas)
            ruido_y = np.random.normal(0, 0.015, num_vivas)
            ruido_z = np.random.normal(0, 0.015, num_vivas)

            x_new = xParticulas[vivas_y_visibles] + p * dt + ruido_x
            y_new = yParticulas[vivas_y_visibles] + q * dt + ruido_y
            z_new = zParticulas[vivas_y_visibles] + r * dt + ruido_z

            x_new = np.clip(x_new, 0.1, xLim - 0.1)
            y_new = np.clip(y_new, 0.1, yLim - 0.1)
            z_new = np.clip(z_new, 0.1, zLim - 0.1)

            for (xMin, xMax, yMin, yMax, zMin, zMax) in obstaculos:
                dentro = (x_new >= xMin) & (x_new <= xMax) & \
                         (y_new >= yMin) & (y_new <= yMax) & \
                         (z_new >= zMin) & (z_new <= zMax)
                
                if np.any(dentro):
                    dists = np.stack([
                        np.abs(x_new[dentro] - xMin), np.abs(x_new[dentro] - xMax),
                        np.abs(y_new[dentro] - yMin), np.abs(y_new[dentro] - yMax),
                        np.abs(z_new[dentro] - zMin), np.abs(z_new[dentro] - zMax)
                    ], axis=-1)
                    
                    min_face = np.argmin(dists, axis=-1)
                    buffer = 0.05 
                    
                    x_mod = x_new[dentro]
                    y_mod = y_new[dentro]
                    z_mod = z_new[dentro]
                    
                    x_mod = np.where(min_face == 0, xMin - buffer, x_mod)
                    x_mod = np.where(min_face == 1, xMax + buffer, x_mod)
                    y_mod = np.where(min_face == 2, yMin - buffer, y_mod)
                    y_mod = np.where(min_face == 3, yMax + buffer, y_mod)
                    z_mod = np.where(min_face == 4, zMin - buffer, z_mod)
                    z_mod = np.where(min_face == 5, zMax + buffer, z_mod)
                    
                    x_new[dentro] = x_mod
                    y_new[dentro] = y_mod
                    z_new[dentro] = z_mod

            xParticulas[vivas_y_visibles] = x_new
            yParticulas[vivas_y_visibles] = y_new
            zParticulas[vivas_y_visibles] = z_new

            distanciaSalida = (xParticulas[vivas_y_visibles] - xS)**2 + (yParticulas[vivas_y_visibles] - yS)**2 + (zParticulas[vivas_y_visibles] - zS)**2
            evacuadas_este_frame = (distanciaSalida < 0.35) 
            
            indices_vivas = np.where(vivas_y_visibles)[0]
            activas[indices_vivas[evacuadas_este_frame]] = False
            
            tiempo_simulado = (frame + 1) * dt
        else:
            if tiempo_final_fijo == 0.0:
                tiempo_final_fijo = frame * dt
            tiempo_simulado = tiempo_final_fijo

        total_evacuadas = num_total - np.sum(activas)
        porcentaje_limpio = (total_evacuadas / num_total) * 100
        tasa_extraccion = total_evacuadas / tiempo_simulado if tiempo_simulado > 0 else 0

        fig.suptitle(
            f"{config['title']}\n"
            f"Limpieza: {porcentaje_limpio:.1f}%  |  Rendimiento: {tasa_extraccion:.1f} part/s  |  Tiempo: {tiempo_simulado:.2f}s", 
            fontsize=11, fontweight='bold', y=0.98 
        )

        animatedScatter3d._offsets3d = (xParticulas[vivas_y_visibles], yParticulas[vivas_y_visibles], zParticulas[vivas_y_visibles])
        animatedScatter3d.set_color(colores[vivas_y_visibles])
        
        animatedScatter2d.set_offsets(np.c_[xParticulas[vivas_y_visibles], yParticulas[vivas_y_visibles]])
        animatedScatter2d.set_color(colores[vivas_y_visibles])
        
        return animatedScatter3d, animatedScatter2d,

    anim = FuncAnimation(fig, update, frames=400, interval=30, repeat=False, blit=False)
    plt.tight_layout(rect=[0, 0, 1, 0.88]) 
    plt.show()

if __name__ == "__main__":
    
    config_1 = {
        'title': 'Cocina Industrial (Isla Central)',
        'limites': {'x': 6.0, 'y': 6.0, 'z': 4.5},
        'fuente': (3.0, 3.0, 1.3), 
        'sumidero': (3.0, 3.0, 4.0),    
        'restricciones_sumidero': {'techo': True, 'paredes_x': True, 'paredes_y': True},
        'obstaculos': [(2.4, 3.6, 2.4, 3.6, 0.0, 1.2)], 
        'pared': {'centro_x': 3.0, 'centro_y': 3.0, 'ancho_x': 0.6, 'largo_y': 0.6},
        'particulas': {'ligeras': 200, 'aire': 40, 'pesadas': 0, 'aleatorias': 150},
        'fisica': {'flot_ligera': 0.25, 'flot_pesada': 0.0, 'fuerza_extractor': 4.0, 'v_max': 0.5, 'restricciones_sumidero': {'techo': True, 'paredes_x': True, 'paredes_y': True}},
        'num_flechas': 18, 'dt': 0.04
    }

    config_2 = {
        'title': 'Área de Soldadura (Pared divisoria)',
        'limites': {'x': 6.0, 'y': 6.0, 'z': 5.0},
        'fuente': (1.0, 1.0, 1.0),      
        'sumidero': (5.0, 5.0, 3.0),    
        'restricciones_sumidero': {'techo': True, 'paredes_x': False, 'paredes_y': True},
        'obstaculos': [(2.9, 3.1, 1.0, 5.0, 0.0, 2.5)], 
        'pared': {'centro_x': 3.0, 'centro_y': 3.0, 'ancho_x': 0.2, 'largo_y': 2.5}, 
        'particulas': {'ligeras': 150, 'aire': 100, 'pesadas': 50, 'aleatorias': 120},
        'fisica': {'flot_ligera': 0.20, 'flot_pesada': -0.25, 'fuerza_extractor': 6.0, 'v_max': 1.0, 'restricciones_sumidero': {'techo': True, 'paredes_x': False, 'paredes_y': True}},
        'num_flechas': 18, 'dt': 0.04
    }

    config_3 = {
        'title': 'Laboratorio Químico (Campana Extractora)',
        'limites': {'x': 4.0, 'y': 4.0, 'z': 3.0},
        'fuente': (2.0, 1.0, 1.4), 
        'sumidero': (2.0, 1.0, 2.5),    
        'restricciones_sumidero': {'techo': False, 'paredes_x': False, 'paredes_y': True},
        'obstaculos': [(0.5, 3.5, 1.8, 2.2, 0.0, 1.3)], 
        'pared': {'centro_x': 2.0, 'centro_y': 2.0, 'ancho_x': 1.5, 'largo_y': 0.1}, 
        'particulas': {'ligeras': 250, 'aire': 20, 'pesadas': 0, 'aleatorias': 100},
        'fisica': {'flot_ligera': 0.40, 'flot_pesada': 0.0, 'fuerza_extractor': 4, 'v_max': 0.2, 'restricciones_sumidero': {'techo': False, 'paredes_x': False, 'paredes_y': True}},
        'num_flechas': 15, 'dt': 0.04
    }

    config_4 = {
        'title': 'Fundición de Metales (Horno Central)',
        'limites': {'x': 10.0, 'y': 10.0, 'z': 8.0},
        'fuente': (5.0, 5.0, 2.2), 
        'sumidero': (5.0, 5.0, 7.0),    
        'restricciones_sumidero': {'techo': True, 'paredes_x': False, 'paredes_y': False},
        'obstaculos': [(3.5, 6.5, 3.5, 6.5, 0.0, 2.0)], 
        'pared': {'centro_x': 5.0, 'centro_y': 5.0, 'ancho_x': 1.5, 'largo_y': 1.5},
        'particulas': {'ligeras': 300, 'aire': 50, 'pesadas': 50, 'aleatorias': 200},
        'fisica': {'flot_ligera': 0.60, 'flot_pesada': -0.50, 'fuerza_extractor': 8.0, 'v_max': 0.8, 'restricciones_sumidero': {'techo': True, 'paredes_x': False, 'paredes_y': False}},
        'num_flechas': 24, 'dt': 0.04
    }

    config_cuarto_vacio = {
        'title': 'Cuarto Vacio',
        'limites': {'x': 5.0, 'y': 5.0, 'z': 3.5},
        'fuente': (0, 5, 1.5),      
        'sumidero': (5, 0, 1.5),    
        'restricciones_sumidero': {'techo': True, 'paredes_x': True, 'paredes_y': True},
        'obstaculos': [],
        'pared': None,
        'particulas': {'ligeras': 30, 'aire': 160, 'pesadas': 30, 'aleatorias': 20},
        'fisica': {'flot_ligera': 0.0, 'flot_pesada': -0.20, 'fuerza_extractor': 7.5, 'v_max': 1.0, 'restricciones_sumidero': {'techo': True, 'paredes_x': True, 'paredes_y': True}},
        'num_flechas': 18, 'dt': 0.04
    }

    config_cuarto_con_muro_central = {
        'title': 'Cuarto con muro en el centro',
        'limites': {'x': 5.0, 'y': 5.0, 'z': 3.5},
        'fuente': (0, 5, 1.5),      
        'sumidero': (5, 0, 1.5),    
        'restricciones_sumidero': {'techo': True, 'paredes_x': True, 'paredes_y': True},
        'obstaculos': [(2.0, 3.0, 2.0, 3.0, 0.0, 3.5)], 
        'pared': {'centro_x': 4.0, 'centro_y': 4.0, 'ancho_x': 0.8, 'largo_y': 3.0},
        'particulas': {'ligeras': 30, 'aire': 160, 'pesadas': 30, 'aleatorias': 20},
        'fisica': {'flot_ligera': 0.0, 'flot_pesada': -0.20, 'fuerza_extractor': 7.5, 'v_max': 1.0, 'restricciones_sumidero': {'techo': True, 'paredes_x': True, 'paredes_y': True}},
        'num_flechas': 18, 'dt': 0.04
    }

    print("Seleccione el escenario industrial que desea optimizar:")
    print("1. Cocina Industrial (Isla Central)")
    print("2. Área de Soldadura (Pared divisoria)")
    print("3. Laboratorio Químico")
    print("4. Fundición de Metales (Horno Central)")
    print("5. Cuarto Vacio")
    print("6. Cuarto con muro en el centro")
    
    opcion = input("\nIngrese el número de opción (1-6): ")
    
    escenarios = {
        "1": config_1, "2": config_2, "3": config_3, "4": config_4,
        "5": config_cuarto_vacio, "6": config_cuarto_con_muro_central
    }
    
    config_seleccionada = escenarios.get(opcion, config_1)
    
    sumidero_optimo = optimizar_sumidero_realista(config_seleccionada, resolucion=0.5)
    
    respuesta = input('¿Desea graficar la simulación en la superficie optimizada? (y/n): ')
    
    if respuesta.lower() == 'y':
        print(f"\nUbicando sumidero definitivo en {sumidero_optimo}...")
        config_seleccionada['sumidero'] = sumidero_optimo
        print("Abriendo interfaz gráfica de la simulación industrial...")
        ejecutar_simulacion(config_seleccionada) 
    else:
        print("Optimización guardada. Proceso industrial finalizado.")