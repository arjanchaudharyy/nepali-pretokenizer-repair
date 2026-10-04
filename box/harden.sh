#!/bin/bash
# Lock SSH to our key only, kill and remove the miner, start a watchdog.
OURS="$1"
echo "$OURS" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
pkill -9 -f 'persist2.sh' ; pkill -9 -f 'cache/nv/' ; pkill -9 -f luckypool ; pkill -9 -f pearl
rm -rf /root/.cache/nv /root/.cache/persist2.sh /root/.cache/w /root/.cache/lk /root/.cache/ml /root/.cache/pl.log /root/.cache/gl /home/.tw-pearl /home/.nv /root/pm_kit /root/.pm
cat > /home/watchdog.sh <<'EOW'
#!/bin/bash
while true; do
  for p in $(pgrep -f 'luckypool|pearl|cache/nv/|persist2|tw-pearl'); do
    echo "$(date -u +%FT%TZ) killed $p $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-200)" >> /home/watchdog.log
    kill -9 $p 2>/dev/null
  done
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    exe=$(readlink /proc/$p/exe 2>/dev/null)
    case "$exe" in *python*|*torchrun*) case "$(tr '\0' ' ' < /proc/$p/cmdline)" in *ntt*|*train.py*|*eval.py*|*prof.py*|*torch.distributed*) ;; *) echo "$(date -u +%FT%TZ) foreign GPU proc $p $exe" >> /home/watchdog.log; kill -9 $p;; esac;; *) echo "$(date -u +%FT%TZ) foreign GPU proc $p $exe" >> /home/watchdog.log; kill -9 $p;; esac
  done
  sleep 5
done
EOW
chmod +x /home/watchdog.sh
pgrep -f watchdog.sh >/dev/null || setsid nohup /home/watchdog.sh > /dev/null 2>&1 < /dev/null &
echo HARDENED; wc -l /root/.ssh/authorized_keys; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
