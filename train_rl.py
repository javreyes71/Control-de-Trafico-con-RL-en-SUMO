# train_rl.py
import os
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from traffic_env_sumo import TrafficSumoEnv

warnings.filterwarnings("ignore")

MODEL_DIR = "models"
TB_DIR = "tensorboard"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)

class SimpleLogCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(SimpleLogCallback, self).__init__(verbose)
        self.ep_count = 0

    def _on_step(self) -> bool:
        if self.locals.get("dones", [False])[0]:
            self.ep_count += 1
            info = self.locals.get("infos", [{}])[0]
            
            # Extracción de datos
            rew = info.get("episode", {}).get("r", 0)
            length = info.get("episode", {}).get("l", 0)
            
            # Extraer último número de vehiculos 
            mins = (length * 2.0) / 60 
            
            print(f"--- Ep {self.ep_count} | Reward: {rew:.1f} | Duración: {mins:.1f} min ---")
            if mins < 55:
                print(f"    ⚠️ Episodio terminado por baja densidad (<10 autos).")
                
        return True

def make_env():
    env = TrafficSumoEnv(gui=False)
    env.setup()
    return Monitor(env)

if __name__ == "__main__":
    print("🚀 Iniciando Entrenamiento Optimizado (SmartLock + Min 10 Autos)")
    
    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy", env, verbose=1,
        learning_rate=3e-4,
        n_steps=2048, 
        batch_size=64, 
        gamma=0.99, 
        ent_coef=0.01,
        tensorboard_log=TB_DIR
    )

    TOTAL_TIMESTEPS = 300000
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, 
        callback=[
            CheckpointCallback(save_freq=10000, save_path=MODEL_DIR, name_prefix="ppo_smart"),
            SimpleLogCallback()
        ]
    )
    
    model.save(os.path.join(MODEL_DIR, "ppo_final"))
    env.close()
    print("✅ Entrenamiento finalizado.")