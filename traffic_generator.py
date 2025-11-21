# traffic_generator.py
import os
import sys
import subprocess

def get_sumo_tools_path():
    if 'SUMO_HOME' in os.environ:
        return os.path.join(os.environ['SUMO_HOME'], 'tools')
    else:
        paths = [
            "/usr/share/sumo/tools",
            "/usr/local/share/sumo/tools",
            "C:\\Program Files (x86)\\Eclipse\\Sumo\\tools"
        ]
        for p in paths:
            if os.path.exists(p):
                return p
        raise ImportError("❌ Error: No se encuentra SUMO_HOME.")

def generate_routefile(net_file="network.net.xml", output_file="routes.rou.xml", end_time=3600):
    tools = get_sumo_tools_path()
    random_trips = os.path.join(tools, "randomTrips.py")
    

    cmd = [
        "python", random_trips,
        "-n", net_file,
        "-r", output_file,
        "-e", str(end_time),    
        "-p", "1.5",            
        "--fringe-factor", "10", 
        "--min-distance", "300", 
        "--validate",
        "--random"              
    ]
    
    subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    generate_routefile()