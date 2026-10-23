"""Tiny job queue: keeps at most P compute jobs alive (counting ones started elsewhere).

usage: nohup python3 queue.py jobs.txt P &
jobs.txt lines: "<threads> <command ...>"  (run from the repo root, logs in logs/)
"""
import subprocess, sys, time, os, re

ROOT = os.path.dirname(os.path.abspath(__file__))
PAT = r'triakis.run_idmrg|torus_dmrg.py'


def running():
    out = subprocess.run(['pgrep', '-f', PAT], capture_output=True, text=True).stdout.split()
    me = str(os.getpid())
    n = 0
    for pid in out:
        try:
            cmd = open(f'/proc/{pid}/cmdline').read().replace('\0', ' ')
        except OSError:
            continue
        if cmd.startswith('python3') and 'queue.py' not in cmd and pid != me:
            n += 1
    return n


def main():
    jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
    P = int(sys.argv[2])
    for line in jobs:
        while running() >= P:
            time.sleep(30)
        T, cmd = line.split(None, 1)
        name = re.sub(r'[^A-Za-z0-9.=_]', '', cmd.replace('python3 -m triakis.run_idmrg', 'idmrg'))[:120]
        env = dict(os.environ, OMP_NUM_THREADS=T, MKL_NUM_THREADS=T, OPENBLAS_NUM_THREADS=T)
        cwd = os.path.join(ROOT, 'verification') if 'torus_dmrg' in cmd else ROOT
        log = open(os.path.join(ROOT, 'logs', name + '.out'), 'w')
        subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
        print(time.strftime('%H:%M:%S'), 'started', cmd, flush=True)
        time.sleep(20)


if __name__ == '__main__':
    main()
