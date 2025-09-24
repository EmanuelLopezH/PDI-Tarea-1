import cv2 as cv
import math
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

def cargar_video(ruta_video):
    cap = cv.VideoCapture(ruta_video)
    
    if not cap.isOpened():
        print("Error: No se puede abrir el video")
        return None
    
    # Set the video to start at frame 39
    if cap.set(cv.CAP_PROP_POS_FRAMES, 39):
        # Get video properties
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        print(f"Video cargado correctamente")
        print(f"FPS: {fps}, Frames totales: {frame_count}")
        print(f"Resolución: {width}x{height}")
        print(f"Iniciando desde frame 39")
            
    return cap, fps, frame_count, width, height
def segmentacion(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # Valores predeterminados para segmentación (pueden ajustarse)
    lower = np.array([90, 60, 60])
    upper = np.array([130,255,255])
    
    mask = cv.inRange(hsv, lower, upper)
    
    # Operaciones morfológicas para limpiar la máscara
    img = cv.dilate(mask, None, iterations=5)
    img = cv.erode(img, None, iterations=5)
    
    return img
def centroide(contours):
    """
    Función modificada para usar el origen establecido
    """
    if not contours:
        return None, None, None, None
    
    cx = cy = None
    c = max(contours, key=cv.contourArea)
    M = cv.moments(c)

    if M["m00"] > 1e-3:  # Evitar división por cero
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Convertir píxeles a metros usando el origen establecido
        x_m = (cx - origen_x) / px_per_m  
        y_m = (origen_y - cy) / px_per_m  # Y crece hacia arriba desde el origen

        positions.append((t, cx, cy, x_m, y_m))
        
        # Dibujar centroide
        cv.circle(img, (int(cx), int(cy)), 6, (0, 0, 255), -1)
        
        # Dibujar origen para referencia
        cv.circle(frame, (int(origen_x), int(origen_y)), 8, (0, 255, 0), -1)
        cv.putText(frame, "O(0,0)", (int(origen_x)+10, int(origen_y)-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Mostrar coordenadas relativas al origen
        cv.putText(frame, f"({x_m:.2f}, {y_m:.2f})m", (int(cx)+10, int(cy)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        print(f"Frame {frame_num}: Pelota en ({cx:.1f}, {cy:.1f}) px = ({x_m:.3f}, {y_m:.3f}) m desde origen")
        return cx, cy, x_m, y_m
    
    return None, None, None, None
def calibrar_origen(first_frame):
    """
    Permite al usuario seleccionar un punto de origen en el primer frame.
    """
    global origen_x, origen_y
    puntos_origen = []
    img_display = first_frame.copy()
    
    def mouse_callback_origen(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            puntos_origen.clear()  # Solo un punto
            puntos_origen.append((x, y))
            img_temp = first_frame.copy()
            cv.circle(img_temp, (x, y), 8, (0, 255, 0), -1)
            cv.putText(img_temp, "ORIGEN (0,0)", (x+10, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.imshow('Calibrar Origen', img_temp)
    
    cv.namedWindow('Calibrar Origen')
    cv.setMouseCallback('Calibrar Origen', mouse_callback_origen)
    
    print("=" * 50)
    print("CALIBRACIÓN DEL ORIGEN DE COORDENADAS")
    print("=" * 50)
    print("Haz clic en el punto que será el origen (0,0)")
    print("Sugerencias:")
    print("- Esquina de la mesa")
    print("- Punto de lanzamiento")
    print("- Cualquier referencia fija")
    print("Presiona ESC cuando termines")
    
    while True:
        cv.imshow('Calibrar Origen', img_display)
        key = cv.waitKey(1) & 0xFF
        if key == 27 or len(puntos_origen) > 0:  # ESC o click
            break
    
    cv.destroyWindow('Calibrar Origen')
    
    if puntos_origen:
        origen_x, origen_y = puntos_origen[0]
        print(f"✓ Origen establecido en píxel: ({origen_x}, {origen_y})")
        return origen_x, origen_y
    else:
        # Origen por defecto (esquina inferior izquierda)
        origen_x, origen_y = 0, first_frame.shape[0]
        print(f"⚠ Usando origen por defecto: ({origen_x}, {origen_y})")
        return origen_x, origen_y
def dibujar_trayectoria(frame, positions, color=(0, 255, 255), grosor=2):
    """
    Dibuja la trayectoria completa del centroide
    """
    if len(positions) < 2:
        return
    
    # Dibujar líneas conectando todas las posiciones
    for i in range(1, len(positions)):
        if positions[i-1][1] is not None and positions[i][1] is not None:
            pt1 = (int(positions[i-1][1]), int(positions[i-1][2]))  # (cx_anterior, cy_anterior)
            pt2 = (int(positions[i][1]), int(positions[i][2]))      # (cx_actual, cy_actual)
            cv.line(frame, pt1, pt2, color, grosor)
    
    # Dibujar puntos en cada posición
    for pos in positions:
        if pos[1] is not None:
            cv.circle(frame, (int(pos[1]), int(pos[2])), 3, color, -1)
def dibujar_vector_velocidad(frame, cx, cy, vx, vy, escala=50, color=(255, 0, 255), grosor=3):
    """
    Dibuja el vector de velocidad como una flecha
    """
    if cx is None or cy is None or vx is None or vy is None:
        return
    
    # Punto inicial (centroide actual)
    start_point = (int(cx), int(cy))
    
    # Punto final (proporcional a la velocidad)
    end_x = int(cx + vx * escala)
    end_y = int(cy - vy * escala)
    end_point = (end_x, end_y)
    
    # Dibujar línea principal del vector
    cv.arrowedLine(frame, start_point, end_point, color, grosor, tipLength=0.3)
    
    # Mostrar magnitud del vector
    magnitud = math.sqrt(vx**2 + vy**2)
    cv.putText(frame, f"|v|={magnitud:.2f}m/s", 
            (end_x + 10, end_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
def mostrar_informacion_fisica(frame, positions, vels, accs, t):
    """
    Muestra información física en el frame como en tu imagen
    """
    # Fondo semi-transparente para el texto
    overlay = frame.copy()
    cv.rectangle(overlay, (frame.shape[1] - 200, 10), (frame.shape[1] - 10, 200), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    y_pos = 30
    color_texto = (255, 255, 255)
    
    # Tiempo actual
    cv.putText(frame, f"Tiempo: {t:.2f}s", (frame.shape[1] - 190, y_pos), 
            cv.FONT_HERSHEY_SIMPLEX, 0.5, color_texto, 1)
    y_pos += 25
    
    # Velocidades actuales
    if vels:
        vx, vy = vels[-1][1], vels[-1][2]
        vmag = math.sqrt(vx**2 + vy**2)
        cv.putText(frame, f"Vel X: {vx:.2f}m/s", (frame.shape[1] - 190, y_pos), 
                cv.FONT_HERSHEY_SIMPLEX, 0.4, color_texto, 1)
        y_pos += 20
        cv.putText(frame, f"Vel Y: {vy:.2f}m/s", (frame.shape[1] - 190, y_pos), 
                cv.FONT_HERSHEY_SIMPLEX, 0.4, color_texto, 1)
        y_pos += 20
        cv.putText(frame, f"Vel Total: {vmag:.2f}m/s", (frame.shape[1] - 190, y_pos), 
                cv.FONT_HERSHEY_SIMPLEX, 0.4, color_texto, 1)
        y_pos += 30
    
    # Parámetros iniciales (calculados)
    if len(positions) > 5:  # Después de algunos frames
        pos_inicial = positions[0]
        pos_actual = positions[-1]
        
        cv.putText(frame, "PARAMETROS INICIALES", (frame.shape[1] - 190, y_pos), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        y_pos += 20
        
        if vels:
            # Estimar velocidad inicial (promedio de primeros frames)
            v0_estimada = math.sqrt(vels[0][1]**2 + vels[0][2]**2) if vels else 0
            angulo_estimado = math.degrees(math.atan2(vels[0][2], vels[0][1])) if vels else 0
            
            cv.putText(frame, f"Vel inicial: {v0_estimada:.1f}m/s", (frame.shape[1] - 190, y_pos), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, color_texto, 1)
            y_pos += 15
            cv.putText(frame, f"Angulo: {angulo_estimado:.1f} grados", (frame.shape[1] - 190, y_pos), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, color_texto, 1)
            y_pos += 15
            cv.putText(frame, f"H inicial: {pos_inicial[4]:.2f}m", (frame.shape[1] - 190, y_pos), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, color_texto, 1)
def calcular_datos_teoricos():
    """
    Calcula datos teóricos del tiro parabólico de forma simple
    """
    print("="*50)
    print("📊 CÁLCULO TEÓRICO SIMPLE - TIRO PARABÓLICO")
    print("="*50)
    
    # ========== DATOS EXPERIMENTALES ==========
    x0, y0 = 0.0, 0.53      # Posición inicial (m)
    xf, yf = 0.36, 0.0     # Posición final (m)
    tiempo_total = 0.35     # Tiempo experimental (s)
    g = 9.81                # Gravedad (m/s²)
    
    print(f"📍 Posición inicial: ({x0}, {y0}) m")
    print(f"📍 Posición final: ({xf}, {yf}) m")
    print(f"⏱️ Tiempo total: {tiempo_total} s")
    
    # ========== CALCULAR VELOCIDADES INICIALES ==========
    # Desde las ecuaciones básicas:
    # x = x0 + v0x * t  →  v0x = (xf - x0) / t
    # y = y0 + v0y * t - 0.5 * g * t²  →  v0y = (yf - y0 + 0.5 * g * t²) / t
    
    v0x = (xf - x0) / tiempo_total
    v0y = 0  # Asumimos lanzamiento horizontal para simplificar
    v0_total = math.sqrt(v0x**2 + v0y**2)
    angulo = math.degrees(math.atan2(v0y, v0x))
    
    print(f"\n🚀 VELOCIDADES INICIALES:")
    print(f"   • v0x = {v0x:.3f} m/s")
    print(f"   • v0y = {v0y:.3f} m/s")
    print(f"   • v0 total = {v0_total:.3f} m/s")
    print(f"   • Ángulo = {angulo:.1f}°")
    
    # ========== GENERAR PUNTOS TEÓRICOS ==========
    num_puntos = 100
    t = np.linspace(0, tiempo_total, num_puntos)
    
    # Posiciones
    x = x0 + v0x * t
    y = y0 + v0y * t - 0.5 * g * t**2
    
    # Velocidades
    vx = v0x * np.ones_like(t)
    vy = v0y - g * t
    v_total = np.sqrt(vx**2 + vy**2)
    
    # Aceleraciones
    ax = np.zeros_like(t)
    ay = -g * np.ones_like(t)
    
    # ========== CREAR DATAFRAME ==========
    datos = pd.DataFrame({
        'tiempo_s': t,
        'x_m': x,
        'y_m': y,
        'vx_ms': vx,
        'vy_ms': vy,
        'v_total_ms': v_total,
        'ax_ms2': ax,
        'ay_ms2': ay
    })
    
    # Filtrar solo puntos válidos (y >= 0)
    datos = datos[datos['y_m'] >= 0]
    
    # ========== GUARDAR DATOS ==========
    datos.to_csv('datos_teoricos.csv', index=False)
    print(f"\n✅ {len(datos)} puntos guardados en: datos_teoricos.csv")
    
    # ========== VERIFICACIÓN ==========
    print(f"\n🔍 VERIFICACIÓN:")
    print(f"   • Posición final calculada: ({datos['x_m'].iloc[-1]:.3f}, {datos['y_m'].iloc[-1]:.3f}) m")
    print(f"   • Error X: {abs(datos['x_m'].iloc[-1] - xf):.6f} m")
    print(f"   • Error Y: {abs(datos['y_m'].iloc[-1] - yf):.6f} m")
    
    return datos, {
        'v0x': v0x,
        'v0y': v0y,
        'v0_total': v0_total,
        'angulo': angulo
    }
def generar_graficas_comparativas():
    """
    Genera gráficas comparando datos teóricos vs experimentales
    """
    print("\n" + "="*60)
    print("📊 GENERANDO GRÁFICAS COMPARATIVAS")
    print("="*60)
    
    # Cargar datos experimentales
    try:
        df_exp = pd.read_csv('output/datos_experimentales.csv')
        print(f"✅ Datos experimentales cargados: {len(df_exp)} puntos")
    except FileNotFoundError:
        print("❌ No se encontraron datos experimentales")
        return
    
    # Cargar datos teóricos
    try:
        df_teo = pd.read_csv('datos_teoricos.csv')
        print(f"✅ Datos teóricos cargados: {len(df_teo)} puntos")
    except FileNotFoundError:
        print("❌ No se encontraron datos teóricos")
        return
    
    # Configurar estilo de gráficas
    plt.style.use('default')
    fig = plt.figure(figsize=(15, 12))
    
    # 1. TRAYECTORIA EN EL PLANO XY
    plt.subplot(2, 3, 1)
    plt.plot(df_exp['x_m'], df_exp['y_m'], 'ro-', label='Experimental', 
             markersize=4, linewidth=2, alpha=0.8)
    plt.plot(df_teo['x_m'], df_teo['y_m'], 'b--', label='Teórico', 
             linewidth=2, alpha=0.7)
    plt.xlabel('Posición X (m)')
    plt.ylabel('Posición Y (m)')
    plt.title('Trayectoria Completa')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 2. POSICIÓN X vs TIEMPO
    plt.subplot(2, 3, 2)
    plt.plot(df_exp['tiempo_s'], df_exp['x_m'], 'ro-', label='Experimental', 
             markersize=4, linewidth=2)
    plt.plot(df_teo['tiempo_s'], df_teo['x_m'], 'b--', label='Teórico', 
             linewidth=2)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Posición X (m)')
    plt.title('Movimiento Horizontal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. POSICIÓN Y vs TIEMPO
    plt.subplot(2, 3, 3)
    plt.plot(df_exp['tiempo_s'], df_exp['y_m'], 'ro-', label='Experimental', 
             markersize=4, linewidth=2)
    plt.plot(df_teo['tiempo_s'], df_teo['y_m'], 'b--', label='Teórico', 
             linewidth=2)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Posición Y (m)')
    plt.title('Movimiento Vertical')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # # 4. VELOCIDAD EN X
    # plt.subplot(2, 3, 4)
    # if vels:  # Si hay datos de velocidad experimental
    #     t_vel = [v[0] for v in vels]
    #     vx_exp = [v[1] for v in vels]
    #     plt.plot(t_vel, vx_exp, 'ro-', label='Experimental', 
    #              markersize=4, linewidth=2)
    # plt.plot(df_teo['tiempo_s'], df_teo['vx_ms'], 'b--', label='Teórico', 
    #          linewidth=2)
    # plt.xlabel('Tiempo (s)')
    # plt.ylabel('Velocidad X (m/s)')
    # plt.title('Velocidad Horizontal')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    
    # # 5. VELOCIDAD EN Y
    # plt.subplot(2, 3, 5)
    # if vels:  # Si hay datos de velocidad experimental
    #     vy_exp = [v[2] for v in vels]
    #     plt.plot(t_vel, vy_exp, 'ro-', label='Experimental', 
    #              markersize=4, linewidth=2)
    # plt.plot(df_teo['tiempo_s'], df_teo['vy_ms'], 'b--', label='Teórico', 
    #          linewidth=2)
    # plt.xlabel('Tiempo (s)')
    # plt.ylabel('Velocidad Y (m/s)')
    # plt.title('Velocidad Vertical')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    
    # 6. VELOCIDAD TOTAL
    plt.subplot(2, 3, 5)
    if vels:  # Si hay datos de velocidad experimental
        t_vel = [v[0] for v in vels]
        vx_exp = [v[1] for v in vels]
        vy_exp = [v[2] for v in vels]
        v_total_exp = [math.sqrt(vx**2 + vy**2) for vx, vy in zip(vx_exp, vy_exp)]
        plt.plot(t_vel, v_total_exp, 'ro-', label='Experimental', 
                 markersize=4, linewidth=2)
    plt.plot(df_teo['tiempo_s'], df_teo['v_total_ms'], 'b--', label='Teórico', 
             linewidth=2)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad Total (m/s)')
    plt.title('Magnitud de Velocidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparacion_teorico_experimental.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ANÁLISIS ESTADÍSTICO
    print("\n📈 ANÁLISIS ESTADÍSTICO DE ERRORES:")
    print("-" * 40)
    
    # Interpollar datos teóricos a tiempos experimentales para comparación
    
    # Verificar que los tiempos experimentales estén dentro del rango teórico
    t_exp = df_exp['tiempo_s'].values
    t_teo = df_teo['tiempo_s'].values
    
    if t_exp.max() <= t_teo.max() and t_exp.min() >= t_teo.min():
        # Interpolación de posiciones
        f_x = interp1d(t_teo, df_teo['x_m'], kind='cubic')
        f_y = interp1d(t_teo, df_teo['y_m'], kind='cubic')
        
        x_teo_interp = f_x(t_exp)
        y_teo_interp = f_y(t_exp)
        
        # Calcular errores
        error_x = df_exp['x_m'].values - x_teo_interp
        error_y = df_exp['y_m'].values - y_teo_interp
        error_total = np.sqrt(error_x**2 + error_y**2)
        
        print(f"🎯 Error promedio en X: {np.mean(np.abs(error_x)):.4f} ± {np.std(error_x):.4f} m")
        print(f"🎯 Error promedio en Y: {np.mean(np.abs(error_y)):.4f} ± {np.std(error_y):.4f} m")
        print(f"🎯 Error total promedio: {np.mean(error_total):.4f} ± {np.std(error_total):.4f} m")
        print(f"🎯 Error máximo: {np.max(error_total):.4f} m")
        
        # Coeficiente de correlación
        r_x = np.corrcoef(df_exp['x_m'], x_teo_interp)[0,1]
        r_y = np.corrcoef(df_exp['y_m'], y_teo_interp)[0,1]
        
        print(f"📊 Correlación X: R² = {r_x**2:.4f}")
        print(f"📊 Correlación Y: R² = {r_y**2:.4f}")
    
    print(f"\n💾 Gráfica guardada como: comparacion_teorico_experimental.png")
    print("="*60)

# Llamar la función para generar el archivo CSV
datos_teoricos, parametros = calcular_datos_teoricos()

# Parámetros
SCALE="output/scale.txt"
VIDEO_PATH = 'V3.mp4'
OUT_FOLDER = 'output'
t=0.0; prev=None; v_prev=None
positions=[]   # (t, x_px, y_px, x_m, y_m)
vels=[]        # (t_mid, vx, vy)
accs=[]       # (t_mid2, ax, ay)
px_per_m = float(open(SCALE).read().strip())
cap, fps, frame_count, width, height = cargar_video(VIDEO_PATH)
dt = 1.0 / fps # Tiempo entre frames

if cap is not None:    
    # Leer primer frame para calibrar origen
    ok, first_frame = cap.read()
    if ok:
        first_frame = cv.resize(first_frame, (320, 480))
        origen_x, origen_y = calibrar_origen(first_frame)
        frame_num = 0
        
        while frame_num < 21:
            ok, frame = cap.read()
            if not ok:
                print("Fin del video o error al leer frame")
                break
                
            t = frame_num * dt  # Tiempo actual
            frame = cv.resize(frame, (320, 480))
            img = segmentacion(frame)

            # Encontrar contornos
            contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            cx, cy, x_m, y_m = centroide(contours)

            # CÁLCULO DE VELOCIDADES Y ACELERACIONES
            current_vx = current_vy = None
            if prev is not None:
                dt_calc = t - prev[0]
                if dt_calc > 0:
                    current_vx = (x_m - prev[3]) / dt_calc
                    current_vy = (y_m - prev[4]) / dt_calc
                    vels.append((t - 0.5 * dt_calc, current_vx, current_vy))
                    
                    if v_prev is not None:
                        ax = (current_vx - v_prev[1]) / dt_calc
                        ay = (current_vy - v_prev[2]) / dt_calc
                        accs.append((t - dt_calc, ax, ay))
                    v_prev = (t - 0.5 * dt_calc, current_vx, current_vy)
            prev = (t, cx, cy, x_m, y_m)
            
            # ✅ DIBUJAR TRAYECTORIA COMPLETA
            dibujar_trayectoria(frame, positions, color=(0, 255, 255), grosor=2)
            
            # ✅ DIBUJAR VECTOR DE VELOCIDAD
            if current_vx is not None and current_vy is not None and cx is not None:
                dibujar_vector_velocidad(frame, cx, cy, current_vx, current_vy, 
                                    escala=30, color=(255, 0, 255), grosor=3)
            
            # ✅ MOSTRAR INFORMACIÓN FÍSICA
            mostrar_informacion_fisica(img, positions, vels, accs, t)
                
            # SUPERPOSICIÓN DE INFORMACIÓN EN LA MÁSCARA
            def txt(y, s, img=img): 
                cv.putText(img, s, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            t += dt

            # Mostrar las ventanas
            cv.imshow('Video con Trayectoria', frame)
            cv.imshow('Mask', img)

            frame_num += 1
            key = cv.waitKey(100) & 0xFF
            if key == 27:  # ESC para salir
                break
            elif key == 32:  # ESPACIO para pausar
                cv.waitKey(0)
                
        cv.destroyAllWindows()
        
        # ✅ GUARDAR DATOS FINALES
        print(f"\n🎉 PROCESAMIENTO COMPLETADO:")
        print(f"📊 {len(positions)} posiciones registradas")
        print(f"🏃 {len(vels)} velocidades calculadas")
        print(f"⚡ {len(accs)} aceleraciones calculadas")
        
        # Guardar en archivo CSV
        if positions:
            df_positions = pd.DataFrame(positions, columns=['tiempo_s', 'cx_px', 'cy_px', 'x_m', 'y_m'])
                # Agregar velocidades si existen
        if vels:
            # Crear DataFrame de velocidades
            df_velocidades = pd.DataFrame(vels, 
                columns=['t_mid', 'vx_ms', 'vy_ms'])
            
            # Interpolar velocidades a los tiempos de posición
            from scipy.interpolate import interp1d
            
            # Solo si hay suficientes puntos para interpolación
            if len(vels) > 1:
                try:
                    # Interpolación para vx
                    f_vx = interp1d(df_velocidades['t_mid'], df_velocidades['vx_ms'], 
                                kind='linear', fill_value='extrapolate')
                    df_positions['vx_ms'] = f_vx(df_positions['tiempo_s'])
                    
                    # Interpolación para vy
                    f_vy = interp1d(df_velocidades['t_mid'], df_velocidades['vy_ms'], 
                                kind='linear', fill_value='extrapolate')
                    df_positions['vy_ms'] = f_vy(df_positions['tiempo_s'])
                    
                    # Velocidad total
                    df_positions['v_total_ms'] = np.sqrt(df_positions['vx_ms']**2 + df_positions['vy_ms']**2)
                    
                    print(f"✅ Velocidades interpoladas correctamente")
                except Exception as e:
                    print(f"⚠️ Error en interpolación: {e}")
                    # Rellenar con NaN si falla la interpolación
                    df_positions['vx_ms'] = np.nan
                    df_positions['vy_ms'] = np.nan
                    df_positions['v_total_ms'] = np.nan
            else:
                # Si no hay suficientes velocidades, rellenar con NaN
                df_positions['vx_ms'] = np.nan
                df_positions['vy_ms'] = np.nan
                df_positions['v_total_ms'] = np.nan
            df_positions.to_csv('output/datos_experimentales.csv', index=False)
            print(f"💾 Datos guardados en: output/datos_experimentales.csv")
            generar_graficas_comparativas()
            
