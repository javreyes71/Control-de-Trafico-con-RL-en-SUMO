import time
import traci
from traffic_env_sumo import TrafficSumoEnv
import random

def test_control():
    print("🧪 INICIANDO TEST DE CONTROL DE SEMÁFOROS...")
    print("Objetivo: Verificar si los semáforos realmente cambian de color.")
    
    # 1. Iniciamos el entorno con GUI para que TÚ lo veas
    env = TrafficSumoEnv(gui=True)
    obs, _ = env.reset()
    
    # Vamos a monitorear un semáforo específico (el primero de la lista)
    target_tls = env.tls_ids[0]
    print(f"👀 Monitoreando semáforo testigo: {target_tls}")
    
    # Intentaremos forzar cambios
    print("🟢 Iniciando bucle de prueba. Mira la ventana de SUMO...")
    
    for step in range(100): # 100 pasos de prueba
        # Elegimos acciones aleatorias para todos
        actions = env.action_space.sample()
        
        # Obtenemos la fase ANTES de aplicar la acción
        phase_before = traci.trafficlight.getPhase(target_tls)
        
        # Aplicamos paso (aquí el entorno intenta cambiar las luces)
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # Obtenemos la fase DESPUÉS
        phase_after = traci.trafficlight.getPhase(target_tls)
        
        # Verificamos si hubo cambio
        if phase_before != phase_after:
            print(f"✅ ¡ÉXITO! Semáforo {target_tls} cambió: {phase_before} -> {phase_after}")
        
        # Dormimos un poco para que te dé tiempo a ver la ventana
        time.sleep(0.1)
        
        if terminated:
            env.reset()

    print("🏁 Test finalizado.")
    env.close()

if __name__ == "__main__":
    test_control()
