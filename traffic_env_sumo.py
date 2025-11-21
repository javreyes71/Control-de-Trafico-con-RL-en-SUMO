# traffic_env_sumo.py
import time
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
import xml.etree.ElementTree as ET
from pathlib import Path
from traffic_generator import generate_routefile

NETWORK_FILE = Path("network.net.xml")
ROUTES_FILE = "routes.rou.xml"
SUMO_CFG = "simulation.sumocfg"

# --- CONFIGURACIÓN ---
STEP_LENGTH = 0.4       
DELTA_TIME = 2.0        # Revisión rápida
MIN_PHASE_TIME = 8.0    # Bloqueo de 8s
MAX_STEPS = 3600        # 1 hora simulada

# --- UMBRALES DE TERMINACIÓN ---
WARMUP_STEPS = 60       # Primeros 2 min (60 steps) no cortamos nada
MIN_VEHICLES = 10       # Si hay menos de 10 autos, reiniciamos

def parse_topology(net_file: Path):
    tree = ET.parse(str(net_file))
    root = tree.getroot()
    junctions = {}
    edges_map = {}
    for j in root.findall("junction"):
        jid = j.get("id")
        inc = list({l.split("_")[0] for l in j.get("incLanes","").split() if not l.startswith(":")})
        junctions[jid] = {"incoming": inc, "outgoing": []}
    for e in root.findall("edge"):
        eid = e.get("id")
        if eid.startswith(":"): continue 
        fr = e.get("from"); to = e.get("to")
        edges_map[eid] = {"from":fr, "to":to}
        if fr in junctions: junctions[fr]["outgoing"].append(eid)
    return junctions, edges_map

class TrafficSumoEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, gui=False):
        super().__init__()
        self.gui = gui
        self.junctions, self.edges_map = parse_topology(NETWORK_FILE)
        
        self.tls_ids = []
        self.tls_to_j = {}
        self.tls_nph = {} 
        self.last_change_time = {} 
        self.last_actions = {}     

        self.action_space = None
        self.observation_space = None
        self.step_count = 0
        self.sumo_steps_per_rl_step = int(DELTA_TIME / STEP_LENGTH)

    def setup(self):
        binpath = "sumo-gui" if self.gui else "sumo"
        label = f"init_{random.randint(0,99999)}"
        try:
            traci.start([binpath, "-c", SUMO_CFG, "--start", "--quit-on-end"], label=label)
            conn = traci.getConnection(label)
        except traci.exceptions.FatalTraCIError:
            conn = traci

        raw_tls = conn.trafficlight.getIDList()
        mapped = {}
        for t in raw_tls:
            lanes = conn.trafficlight.getControlledLanes(t)
            tgt = None
            for lane in lanes:
                e = lane.split("_")[0]
                if e in self.edges_map:
                    tgt = self.edges_map[e]["to"]
                    if tgt in self.junctions: break
            if tgt is None: tgt = next(iter(self.junctions)) 
            mapped[t] = tgt

        self.tls_ids = list(mapped.keys())
        self.tls_to_j = mapped

        for t in self.tls_ids:
            try:
                logic = conn.trafficlight.getCompleteRedYellowGreenDefinition(t)[0]
                self.tls_nph[t] = len(logic.phases)
                self.last_change_time[t] = 0.0
                self.last_actions[t] = 0
            except:
                self.tls_nph[t] = 4 

        if len(self.tls_ids) > 0:
            self.action_space = spaces.MultiDiscrete([self.tls_nph[t] for t in self.tls_ids])
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(len(self.tls_ids)*2,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(1)
            self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        conn.close()

    def _get_obs(self):
        obs = []
        for t in self.tls_ids:
            j = self.tls_to_j[t]
            inc = self.junctions[j]["incoming"]
            q_total = 0; w_total = 0
            for e in inc:
                try:
                    q_total += traci.edge.getLastStepHaltingNumber(e)
                    w_total += traci.edge.getWaitingTime(e)
                except: pass
            obs.append(q_total)
            obs.append(w_total / 60.0) 
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self):
        total_queue = 0; total_waiting = 0
        for t in self.tls_ids:
            j = self.tls_to_j[t]
            inc = self.junctions[j]["incoming"]
            for e in inc:
                try:
                    total_queue += traci.edge.getLastStepHaltingNumber(e)
                    total_waiting += traci.edge.getWaitingTime(e)
                except: pass
        return float(-1.0 * total_queue - 0.05 * total_waiting)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Generamos tráfico nuevo SIEMPRE para garantizar variedad
        try:
            # end_time debe ser al menos igual a MAX_STEPS (segundos reales)
            generate_routefile(str(NETWORK_FILE), ROUTES_FILE, end_time=MAX_STEPS + 500)
        except Exception as e:
            print(f"⚠️ Error generando tráfico: {e}")

        try: traci.close()
        except: pass
        time.sleep(0.5)

        binpath = "sumo-gui" if self.gui else "sumo"
        traci.start([binpath, "-c", SUMO_CFG, "--start", "--quit-on-end"])

        self.step_count = 0
        for t in self.tls_ids:
            self.last_change_time[t] = 0.0
            self.last_actions[t] = 0

        # Warm-up pequeño de simulación
        for _ in range(5): traci.simulationStep()
        return self._get_obs(), {}

    def step(self, action):
        current_sim_time = self.step_count * DELTA_TIME

        # 1. Lógica Smart Locking
        for i, t in enumerate(self.tls_ids):
            try:
                proposed = int(action[i])
                current = traci.trafficlight.getPhase(t)
                time_since = current_sim_time - self.last_change_time[t]

                if proposed != current:
                    if time_since >= MIN_PHASE_TIME:
                        traci.trafficlight.setPhase(t, proposed)
                        self.last_change_time[t] = current_sim_time
            except: pass

        # 2. Avanzar Simulación
        for _ in range(self.sumo_steps_per_rl_step):
            traci.simulationStep()
            if self.gui: time.sleep(0.01) 
        
        self.step_count += 1
        
        # 3. Verificación de Densidad y Término
        terminated = (self.step_count * DELTA_TIME) >= MAX_STEPS
        
        # --- REGLA DE ORO: MÍNIMO 10 AUTOS ---
        vehicles_on_road = traci.vehicle.getIDCount()
        
        # Solo aplicamos la regla después del calentamiento (WARMUP_STEPS)
        # para dar tiempo a que los autos entren al mapa.
        if self.step_count > WARMUP_STEPS:
            if vehicles_on_road < MIN_VEHICLES:
                # Terminamos prematuramente porque no hay suficiente tráfico
                terminated = True
        
        if terminated: self.close()
        
        # Pasamos info extra para que el log sepa por qué terminó o cuántos autos había
        info = {"step": self.step_count, "vehs": vehicles_on_road}
        
        return self._get_obs(), self._compute_reward(), terminated, False, info

    def close(self):
        try: traci.close()
        except: pass