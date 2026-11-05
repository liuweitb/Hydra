#!/usr/bin/env bash
# remove the temporary runs and logs of the apply-vs-ignore test
rm -rf "$HOME"/runs/uh2/tmp_dk_*
rm -f "$HOME"/runs/logs/tmp_dk_*.log "$HOME"/runs/logs/tmp_dk_*.err
ls "$HOME"/runs/uh2 | grep tmp_dk || echo "cleaned"
