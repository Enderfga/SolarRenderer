"""Client side of bridge.py. As a CLI: uc.py UE_PORT 'cmd1' 'cmd2' ... prints replies."""
import sys
from multiprocessing.connection import Client as MPClient

class UC:
    def __init__(self, port):
        self.conn = MPClient(('127.0.0.1', port + 10000), authkey=b'uz')
    def batch(self, cmds, timeout=120):
        self.conn.send((list(cmds), timeout)); return self.conn.recv()
    def __call__(self, cmd, timeout=120):
        return self.batch([cmd], timeout)[0]

if __name__ == '__main__':
    u = UC(int(sys.argv[1]))
    for cmd, res in zip(sys.argv[2:], u.batch(sys.argv[2:])):
        s = res if isinstance(res, str) else f'<bytes {len(res)}>'
        print(f'{cmd} -> {s[:3000]}')
