import gymnasium as gym
from stable_baselines3 import PPO
from traffic_env_sumo import TrafficSumoEnv
import os

# --- CREAR DIRECTORIOS ---
models_dir = "models/ppo_multi_agent"
log_dir = "tensorboard/multi_agent_logs"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# --- INICIAR ENTORNO ---
# gui=False para entrenar rápido (sin ventana gráfica)
print("🏗️  Creando entorno Multi-Agente Centralizado...")
env = TrafficSumoEnv(gui=False) 

# --- CONFIGURAR AGENTE PPO ---
# Usamos MlpPolicy porque la entrada es un vector plano de números
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=log_dir,
    learning_rate=0.0003, # Tasa de aprendizaje estándar
    n_steps=2048,         # Pasos antes de actualizar la red (mayor es mejor para entornos complejos)
    batch_size=64,
    gamma=0.99            # Factor de descuento (importancia del futuro)
)

print("🚀 Iniciando entrenamiento...")
print("El sistema aprenderá a coordinar TODOS los semáforos para minimizar colas y esperas.")

# --- CICLO DE ENTRENAMIENTO ---
TIMESTEPS = 10000
for i in range(1, 51): # 50 iteraciones * 10,000 pasos = 500,000 pasos totales
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_City_Multi")
    
    # Guardar checkpoint
    save_path = f"{models_dir}/ppo_multi_{TIMESTEPS*i}"
    model.save(save_path)
    print(f"💾 Modelo guardado en: {save_path}")

print("✅ Entrenamiento finalizado.")
env.close()