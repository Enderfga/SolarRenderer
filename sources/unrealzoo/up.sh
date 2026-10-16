#!/bin/bash
# usage: up.sh GPU MAP [WAIT_S]  -> (re)start the UE server on GPU with MAP (UnrealCV port 9000+GPU),
# wait until the map is loaded, then start bridge.py (relay port 19000+GPU).
# The UnrealCV server stops accepting connections once its first client disconnects, so all
# clients go through the bridge, which holds that one connection for the server's lifetime.
# Scripts (launch.sh, bridge.py) are taken from this script's directory; the package, venv and logs
# live in $UZ_HOME (default /opt/uzoo, see deploy.sh).
G=$1; M=$2; WAIT=${3:-300}
D=$(cd "$(dirname "$0")" && pwd)
R=${UZ_HOME:-/opt/uzoo}
cd $R; mkdir -p logs
for p in $(pgrep -f "bridge.py $((9000+G))"); do kill -9 $p; done
for p in $(pgrep -f "graphicsadapter=$G -ResX"); do kill -9 $p; done
sleep 1
# Start servers one at a time on this machine, at least UZ_LAUNCH_GAP s apart (several UE Vulkan servers starting at
# once has crashed nodes with GPU XID errors). The lock is held for the gap after the launch.
exec 9> $R/logs/.launch.lock
flock 9
(setsid nohup $D/launch.sh $G $M > logs/ue$G.log 2>&1 < /dev/null 9>&- &)
sleep ${UZ_LAUNCH_GAP:-25}
flock -u 9; exec 9>&-
# Ready = the nav point exporter finished ("Sampling complete"), or, on maps where it never reports, the map is
# loaded ("Load map complete") and the log has been quiet for 15 s. The log is flushed per line (-FORCELOGFLUSH).
quiet=0; last=-1
for i in $(seq $WAIT); do
  grep -aq "Sampling complete" logs/ue$G.log && break
  if ! pgrep -f "graphicsadapter=$G -ResX" > /dev/null; then echo "DIED $G $M"; tail -5 logs/ue$G.log; exit 2; fi
  sz=$(stat -c %s logs/ue$G.log)
  if [ "$sz" = "$last" ]; then quiet=$((quiet+1)); else quiet=0; last=$sz; fi
  [ $quiet -ge 15 ] && grep -aq "Load map complete" logs/ue$G.log && break
  sleep 1
done
sleep 4
(setsid nohup venv/bin/python -u $D/bridge.py $((9000+G)) > logs/bridge$G.log 2>&1 < /dev/null &)
for i in $(seq 30); do grep -q "bridge up" logs/bridge$G.log && echo UP $G $M && exit 0; sleep 1; done
echo FAIL $G $M; tail -3 logs/bridge$G.log; exit 1
