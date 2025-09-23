# --- requisitos ---
# pip install pandas matplotlib numpy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Lee tu CSV (debe tener estas columnas: tiempo_s,x_m,y_m,vx_ms,vy_ms,v_total_ms,ax_ms2,ay_ms2)
CSV_PATH = "output/datos_experimentales.csv"   # <- pon aquí tu archivo
CSV_PATH1 = "datos_teoricos.csv"   # <- pon aquí tu archivo
df = pd.read_csv(CSV_PATH)

# (opcional) recorta a la parte por encima del suelo
df = df[df["y_m"] >= 0].reset_index(drop=True)
print("DataFrame columns:", df.columns)

# 2) Datos base
x = df["x_m"].values
y = df["y_m"].values
vx = df["vx_ms"].values
vy = df["vy_ms"].values

# 3) Trazo de la trayectoria x–y
fig, ax = plt.subplots(figsize=(12, 12), dpi=120)
ax.plot(x, y, linewidth=3, label="Trayectoria (x vs y)", color='lightblue', zorder=2)

# Marcar inicio y final
ax.scatter(x[0], y[0], color='green', s=80, label='Inicio', zorder=5, marker='o')
ax.scatter(x[-1], y[-1], color='red', s=80, label='Final', zorder=5, marker='X')

# 4) Flechas de velocidad (submuestrear para no saturar la figura)
step = max(1, len(df)//8)     # ~8 grupos de flechas (menos que antes)
idx = np.arange(0, len(df), step)

# 🔧 AJUSTAR ESCALAS PARA REDUCIR TAMAÑO DE FLECHAS:
# Marcar todos los puntos de la trayectoria
ax.scatter(x[idx], y[idx], color='green', s=20, alpha=0.6, zorder=2, label='Puntos trayectoria')

# Vector velocidad total v (REDUCIDO)
ax.quiver(x[idx], y[idx], vx[idx], vy[idx],
          angles="xy", scale_units="xy", scale=15,  # ↑ Aumentado de 4 a 15
          width=0.002, color='black', label="v total", zorder=4)

# Componente horizontal vx (REDUCIDO)
ax.quiver(x[idx], y[idx], vx[idx], np.zeros_like(idx, dtype=float),
          angles="xy", scale_units="xy", scale=20,  # ↑ Aumentado de 4 a 20
          width=0.0015, alpha=0.7, color='red', label="vₓ", zorder=2)

# Componente vertical vy (REDUCIDO)
ax.quiver(x[idx], y[idx], np.zeros_like(idx, dtype=float), vy[idx],
          angles="xy", scale_units="xy", scale=20,  # ↑ Aumentado de 4 a 20
          width=0.0015, alpha=0.7, color='orange', label="vᵧ", zorder=2)

# 5) Líneas guía: suelo y eje de lanzamiento
ax.axhline(0, linestyle="--", linewidth=1, alpha=0.5, color='gray')
ax.axvline(0, linestyle=":",  linewidth=1, alpha=0.5, color='gray')

# 6) Mejorar visualización
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Posición X (m)", fontweight='bold')
ax.set_ylabel("Posición Y (m)", fontweight='bold')
ax.set_title("Movimiento Parabólico - Trayectoria y Vectores de Velocidad", 
          fontweight='bold', fontsize=14)
ax.grid(True, linestyle=":", alpha=0.3)
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

# Añadir información de parámetros
v0_total = np.sqrt(vx[0]**2 + vy[0]**2)
angulo = np.degrees(np.arctan2(vy[0], vx[0]))
alcance = x[-2] - x[0]
altura_max = y.max()
tiempo_vuelo = df['tiempo_s'].iloc[-1] - df['tiempo_s'].iloc[0]

info_text = f"""Parámetros:
v₀ = {v0_total:.2f} m/s
θ = {angulo:.1f}°
Alcance = {alcance:.3f} m
H máx = {altura_max:.3f} m
t vuelo = {tiempo_vuelo:.3f} s"""

ax.text(0.02, 0.2, info_text, transform=ax.transAxes, 
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('trayectoria_vectores_reducidos.png', dpi=300, bbox_inches='tight')
plt.show()

# 7) Mostrar estadísticas
print(f"\n📊 ESTADÍSTICAS DEL MOVIMIENTO:")
print(f"   • Velocidad inicial: {v0_total:.3f} m/s")
print(f"   • Ángulo de lanzamiento: {angulo:.1f}°")
print(f"   • Alcance horizontal: {alcance:.3f} m")
print(f"   • Altura máxima: {altura_max:.3f} m")
print(f"   • Tiempo de vuelo: {tiempo_vuelo:.3f} s")
print(f"   • Número de puntos: {len(df)}")