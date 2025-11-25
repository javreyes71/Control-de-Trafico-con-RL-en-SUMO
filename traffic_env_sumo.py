import gymnasium as gym
from gymnasium import spaces
import traci
import sumolib
import numpy as np
import os
import sys

# --- CONFIGURACIÓN ---
DELTA_TIME = 5.0        
MIN_PHASE_TIME = 10.0   
YELLOW_TIME = 3.0       
MAX_WAIT_TIME = 20.0    

class TrafficSumoEnv(gym.Env):
    def __init__(self, gui=False):
        super(TrafficSumoEnv, self).__init__()
        self.gui = gui
        
        # 1. Configurar SUMO
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("❌ Error: Por favor, declara la variable de entorno 'SUMO_HOME'")
        
        self._sumo_binary = sumolib.checkBinary('sumo-gui') if self.gui else sumolib.checkBinary('sumo')
        self.config_file = "simulation.sumocfg"
        self.net_file = "network.net.xml" 

        # 2. ANÁLISIS DEL MAPA (Versión Tolerante 🛡️)
        print("🔍 Analizando semáforos en el mapa...")
        net = sumolib.net.readNet(self.net_file)
        
        self.tls_ids = []
        self.green_phases_map = {} 
        self.action_dims = []      

        all_tls = net.getTrafficLights()
        
        for tls in all_tls:
            tls_id = tls.getID()
            programs = tls.getPrograms()
            
            green_indices = []

            # CASO 1: Semáforo con lógica legible
            if programs:
                # Intentamos leer el programa '0' o el primero disponible
                if '0' in programs: prog = programs['0']
                else: prog = next(iter(programs.values()))
                
                phases = prog.getPhases()
                for i, p in enumerate(phases):
                    state = p.state
                    if ('G' in state or 'g' in state) and 'y' not in state:
                         green_indices.append(i)
            
            # CASO 2: Semáforo "Vacío" o sin fases verdes claras (El error que tenías)
            # Estrategia: Asumimos por defecto fases 0 y 2 (Norte-Sur / Este-Oeste estándar)
            if len(green_indices) == 0:
                print(f"   ⚠️ ID: {tls_id} sin lógica clara. Aplicando configuración por defecto [0, 2].")
                green_indices = [0, 2] # Asumimos 2 fases básicas

            # Agregamos el semáforo a la lista de controlables
            self.tls_ids.append(tls_id)
            self.green_phases_map[tls_id] = green_indices
            self.action_dims.append(len(green_indices)) 

        num_tls = len(self.tls_ids)
        print(f"✅ Total semáforos controlados: {num_tls}")
        
        if num_tls == 0:
            raise ValueError("❌ No se encontraron semáforos. Revisa que network.net.xml tenga Traffic Lights (<tlLogic>).")

        # 3. Definir Espacios
        self.action_space = spaces.MultiDiscrete(self.action_dims)

        obs_per_tls = 5 
        total_obs_size = num_tls * obs_per_tls
        
        self.observation_space = spaces.Box(
            low=0, 
            high=999, 
            shape=(total_obs_size,), 
            dtype=np.float32
        )

        self.step_count = 0
        self.last_switch_time = {tls: 0 for tls in self.tls_ids}
        self.connection = None

    def setup(self):
        try: traci.close()
        except: pass

        cmd = [self._sumo_binary, "-c", self.config_file, "--start"]
        traci.start(cmd)
        self.connection = traci
        
        for tls in self.tls_ids:
            self.last_switch_time[tls] = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.setup()
        self.step_count = 0
        return self._get_observation(), {}

    def step(self, actions):
        self.step_count += 1
        current_time = traci.simulation.getTime()
        
        tls_to_switch = []
        
        # --- MAPEO INTELIGENTE ---
        for i, tls_id in enumerate(self.tls_ids):
            action_index = actions[i]
            valid_phases = self.green_phases_map[tls_id]
            
            # Protección de rango
            if action_index < len(valid_phases):
                target_sumo_phase = valid_phases[action_index]
            else:
                target_sumo_phase = valid_phases[0]

            try:
                current_phase = traci.trafficlight.getPhase(tls_id)
                if current_phase != target_sumo_phase:
                    if (current_time - self.last_switch_time[tls_id]) >= MIN_PHASE_TIME:
                        tls_to_switch.append((tls_id, target_sumo_phase))
            except Exception as e:
                pass

        # Ejecutar cambios (Auto-Yellow)
        if len(tls_to_switch) > 0:
            for tls_id, _ in tls_to_switch:
                try:
                    curr = traci.trafficlight.getPhase(tls_id)
                    traci.trafficlight.setPhase(tls_id, curr + 1)
                except: pass
            
            steps_yellow = int(YELLOW_TIME)
            for _ in range(steps_yellow):
                traci.simulationStep()
            
            for tls_id, target in tls_to_switch:
                try:
                    traci.trafficlight.setPhase(tls_id, target)
                    self.last_switch_time[tls_id] = traci.simulation.getTime()
                except: pass

        steps_to_do = int(DELTA_TIME)
        for _ in range(steps_to_do):
            traci.simulationStep()

        # Datos
        observation = self._get_observation()
        reward = self._calculate_global_reward()
        
        total_vehicles = traci.vehicle.getIDCount()
        terminated = (total_vehicles == 0 and self.step_count > 50)
        truncated = self.step_count > 2000
        
        return observation, reward, terminated, truncated, {"vehicles": total_vehicles}

    def _get_observation(self):
        full_obs = []
        for tls_id in self.tls_ids:
            try:
                lanes = traci.trafficlight.getControlledLanes(tls_id)
                halt_counts = [traci.lane.getLastStepHaltingNumber(lane) for lane in lanes]
            except:
                halt_counts = []

            while len(halt_counts) < 4: halt_counts.append(0)
            local_obs = halt_counts[:4]
            
            try: phase = traci.trafficlight.getPhase(tls_id)
            except: phase = 0
            local_obs.append(phase)
            full_obs.extend(local_obs)
            
        return np.array(full_obs, dtype=np.float32)

    def _calculate_global_reward(self):
        total_halt = 0
        penalty_wait = 0
        
        for tls_id in self.tls_ids:
            try:
                lanes = traci.trafficlight.getControlledLanes(tls_id)
                for lane in lanes:
                    total_halt += traci.lane.getLastStepHaltingNumber(lane)
                    wait_time = traci.lane.getWaitingTime(lane)
                    if wait_time > MAX_WAIT_TIME:
                        penalty_wait += (wait_time - MAX_WAIT_TIME) * 1.5
            except: pass

        return - (total_halt * 0.5) - (penalty_wait * 1.0)

    def close(self):
        try: traci.close()
        except: pass