"""Hold the single UnrealCV connection a server accepts (it stops accepting after the first client
disconnects) and relay requests from any number of short-lived clients.
usage: bridge.py UE_PORT  -> listens on UE_PORT+10000 (multiprocessing.connection, authkey b'uz')"""
import sys, threading
from multiprocessing.connection import Listener
from unrealcv import Client
port = int(sys.argv[1])
c = Client(('127.0.0.1', port)); c.connect(timeout=30)
assert c.isconnected(), 'no connection'
lock = threading.Lock()
print('bridge up', port, flush=True)
def serve(conn):
    try:
        while True:
            cmds, timeout = conn.recv()
            out = []
            with lock:
                for cmd in cmds:
                    out.append(c.request(cmd, timeout=timeout))
            conn.send(out)
    except EOFError:
        pass
    finally:
        conn.close()
L = Listener(('127.0.0.1', port + 10000), authkey=b'uz')
while True:
    threading.Thread(target=serve, args=(L.accept(),), daemon=True).start()
