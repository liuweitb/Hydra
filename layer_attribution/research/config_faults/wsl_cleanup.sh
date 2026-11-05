#!/usr/bin/env bash
# usage: wsl_cleanup.sh NAME_PREFIX   delete ~/runs/uh2/<prefix>* and ~/runs/logs/<prefix>* ; prefix must start with tmp_cfg_
P=$1
case "$P" in tmp_cfg_*) ;; *) echo "refusing prefix '$P' (must start with tmp_cfg_)"; exit 2;; esac
rm -rf "$HOME"/runs/uh2/"$P"* "$HOME"/runs/logs/"$P"*
echo "removed $P*; remaining tmp_cfg_ run dirs:"
ls "$HOME/runs/uh2" | grep '^tmp_cfg_' || echo "(none)"
