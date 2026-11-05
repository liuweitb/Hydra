#!/usr/bin/env bash
nproc; uptime; free -m | head -2
ps -eo pid,psr,pcpu,pmem,etime,args --sort=-pcpu | head -12 | cut -c1-200
ls /mnt/d/hydra_work/data/uhumans2_office_mp3d | head
ls /mnt/d/hydra_work/data/uhumans2_office_mp3d/depth | wc -l
