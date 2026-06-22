import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from src.agents.simulation_runner import SimulationRunner

def test_sim():
    print("Initializing SimulationRunner...")
    try:
        runner = SimulationRunner()
        print("Runner initialized. Running 5 ticks...")
        for i in range(5):
            frame = runner.tick()
            print(f"Tick {frame['tick']} - Load: {frame['agents']['scheduler']['load_pct']}% - Peak Temp: {frame['racks'][0]['peak_temp']}C")
        print("Success: SimulationRunner works correctly.")
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sim()
