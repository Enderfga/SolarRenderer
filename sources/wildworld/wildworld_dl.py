"""Copy WildWorld (https://huggingface.co/datasets/AlayaLab/WildWorld, gated) into storage.

    HF_TOKEN=<token of an account granted access> WW_SRC=<storage>/.../wildworld python3 wildworld_dl.py
Polls every 10 minutes until a data file answers, then copies every clip's rgb, skeleton and state
first and its depth after, 16 files at a time, each downloaded and copied straight into $WW_SRC
(nothing kept on local disk beyond the files in flight, in $WW_TMP, default ./ww_tmp). Licence:
non-commercial research only, no redistribution - keep the copy private.
"""
import concurrent.futures as cf, json, os, subprocess, time, urllib.request

TOKEN = os.environ["HF_TOKEN"]
H = {"Authorization": f"Bearer {TOKEN}"}
REPO = "https://huggingface.co/datasets/AlayaLab/WildWorld"
DEST = os.environ["WW_SRC"].rstrip("/")
STORE = os.environ.get("STORAGE_CLI", "solar-storage")
TMP = os.environ.get("WW_TMP", "ww_tmp")


def files():
    d = json.load(urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/api/datasets/AlayaLab/WildWorld", headers=H)))
    return [s["rfilename"] for s in d["siblings"] if s["rfilename"].startswith("data_part")]


def allowed(name):
    try:
        urllib.request.urlopen(urllib.request.Request(f"{REPO}/resolve/main/{name}", headers=H | {"Range": "bytes=0-10"}))
        return True
    except Exception:
        return False


def one(name):
    local = f"{TMP}/{name.replace('/', '__')}"
    try:
        r = subprocess.run(["curl", "-sfL", "--retry", "5", "-H", f"Authorization: Bearer {TOKEN}", "-o", local,
                            f"{REPO}/resolve/main/{name}"], capture_output=True)
        if r.returncode:
            return f"fail {name} curl {r.returncode}"
        subprocess.run([STORE, "cp", local, f"{DEST}/{name}"], check=True, capture_output=True)
        return f"ok {name}"
    finally:
        if os.path.exists(local):
            os.remove(local)


os.makedirs(TMP, exist_ok=True)
names = files()
while not allowed(names[0]):
    print(time.strftime("%T"), "waiting for access approval", flush=True)
    time.sleep(600)
first = [n for n in names if not n.endswith("depth.mkv")]
later = [n for n in names if n.endswith("depth.mkv")]
print(time.strftime("%T"), "access granted;", len(first), "rgb/skeleton/state files then", len(later), "depth files", flush=True)
counts = {"ok": 0, "fail": 0}
with cf.ThreadPoolExecutor(16) as pool:
    for result in pool.map(one, first + later):
        counts[result.split()[0]] += 1
        if result.startswith("fail") or sum(counts.values()) % 200 == 0:
            print(time.strftime("%T"), result, counts, flush=True)
print(time.strftime("%T"), "WILDWORLD DONE", counts, flush=True)
