#!/usr/bin/env bash
set -e

# Optionnel : activer l'environnement virtuel si tu en utilises un
# source .venv/bin/activate

# Vérifier que mosquitto tourne (optionnel, à adapter si besoin)
# Si tu veux le démarrer ici en mode background :
# brew services start mosquitto >/dev/null 2>&1 || true

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
echo "Appuie sur Ctrl+C pour tout arrêter."

# Attendre que l'un des deux se termine
wait