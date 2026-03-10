#!/usr/bin/env bash
set -e

# Optionnel : activer l'environnement virtuel si tu en utilises un
# source .venv/bin/activate

cleanup() {
  echo
  echo "Arrêt des processus V2X et de Mosquitto..."
  [[ -n "${PID_V1:-}" ]] && kill "$PID_V1" 2>/dev/null || true
  [[ -n "${PID_V2:-}" ]] && kill "$PID_V2" 2>/dev/null || true
  [[ -n "${PID_MOSQ:-}" ]] && kill "$PID_MOSQ" 2>/dev/null || true
}

trap cleanup INT TERM

echo "Démarrage du broker Mosquitto local..."
mosquitto -v >/dev/null 2>&1 &
PID_MOSQ=$!
echo "Mosquitto PID: $PID_MOSQ"

# Lancement V1
python src/stereo_v2x_node.py \
  --vid V1 \
  --video-left Data/Simulation_4/video_v1_cam1_HD.mp4 \
  --video-right Data/Simulation_4/video_v1_cam2_HD.mp4 \
  --csv Data/Simulation_4/distances_scenario_v4.csv &
PID_V1=$!

# Lancement V2
python src/stereo_v2x_node.py \
  --vid V2 \
  --video-left Data/Simulation_4/video_v2_cam1_HD.mp4 \
  --video-right Data/Simulation_4/video_v2_cam2_HD.mp4 \
  --csv Data/Simulation_4/distances_scenario_v4.csv &
PID_V2=$!

echo "V1 PID: $PID_V1"
echo "V2 PID: $PID_V2"
echo "Mosquitto PID: $PID_MOSQ"
echo "Appuie sur Ctrl+C pour tout arrêter."

# Attendre que V1 et V2 se terminent
wait "$PID_V1" "$PID_V2"

# Une fois les deux scripts terminés, arrêter Mosquitto
echo "Les deux nœuds V2X sont terminés, arrêt de Mosquitto..."
[[ -n "${PID_MOSQ:-}" ]] && kill "$PID_MOSQ" 2>/dev/null || true
