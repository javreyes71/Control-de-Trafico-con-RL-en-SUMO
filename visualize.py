# visualize.py
import time
import os
from stable_baselines3 import PPO
from traffic_env_sumo import TrafficSumoEnv

MODEL_PATH = "models/ppo_final"

def main():
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        print(f"❌ No encuentro {MODEL_PATH}.zip. Verifica la carpeta models/"); return

    print("🔵 Generando tráfico nuevo y abriendo visualizador...")
    env = TrafficSumoEnv(gui=True)
    env.setup()
    
    try:
        model = PPO.load(MODEL_PATH)
    except:
        print("❌ Error cargando modelo. Asegúrate de que coincida con el código actual.")
        return

    obs, _ = env.reset()
    done = False
    
    print("🟢 Simulación iniciada.")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        time.sleep(0.05) # Control de velocidad visual

    env.close()

if __name__ == "__main__":
    main()