#!/usr/bin/env bash
# start wsl_validate_emulators.sh detached, log to ~/runs/logs/tmp_cfg_validate_emulators.log
H=/mnt/d/Hydra/layer_attribution/research/config_faults
setsid nohup bash "$H/wsl_validate_emulators.sh" > "$HOME/runs/logs/tmp_cfg_validate_emulators.log" 2>&1 < /dev/null &
sleep 1
echo launched; pgrep -f validate_emulators_default | tr '\n' ' '; echo
