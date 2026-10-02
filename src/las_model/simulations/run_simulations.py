#!/usr/bin/env python
"""Run the simulation scripts that generate the data for the figures.

Each script under simulations/figure_XX/ is run in its own Python process, in folder order,
so the semantics are exactly those of running it by hand. Without arguments every folder is
run except those listed in EXCLUDE; name folders or scripts to run a subset. The data
directory is the one the scripts normally use (ROOT_DIR from the environment or .env), or
the one given with --root-dir.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SIMULATIONS_DIR = Path(__file__).resolve().parent
SRC_DIR = SIMULATIONS_DIR.parents[1]
REPO_DIR = SRC_DIR.parent

# Folders left out of a full run; they can still be named explicitly on the command line
EXCLUDE = {}

# Scripts in a folder run alphabetically unless the folder is listed here. Later entries
# read the outputs of earlier ones.
ORDER = {
    'figure_06': ['cascade_spatial.py', 'cascade_spatial_moranI.py', 'cascade_spatial_analyze.py', 'cascade_time.py'],
}


def is_stub(script):
    """One-line placeholders that point at another figure's script."""
    with open(script, encoding='utf-8') as f:
        return f.readline().lstrip('﻿').startswith('# Stub file')


def all_folders():
    return sorted((p for p in SIMULATIONS_DIR.iterdir() if p.is_dir() and p.name.lower().startswith('figure_')),
                  key=lambda p: p.name.lower())


def scripts_in(folder):
    """The runnable scripts of one folder, in execution order."""
    scripts = [p for p in folder.glob('*.py') if not p.name.startswith(('plot_', '_')) and not is_stub(p)]
    if folder.name in ORDER:
        names = {p.name for p in scripts}
        missing = set(ORDER[folder.name]) ^ names
        if missing:
            sys.exit(f"{folder.name}: ORDER does not match its scripts: {' '.join(sorted(missing))}")
        return [folder / name for name in ORDER[folder.name]]
    return sorted(scripts)


def resolve_targets(targets):
    """Folders, scripts relative to the simulations directory, paths, or bare script names."""
    if not targets:
        return [s for d in all_folders() if d.name not in EXCLUDE for s in scripts_in(d)]
    scripts = []
    for target in targets:
        candidate = SIMULATIONS_DIR / target
        if candidate.is_dir():
            scripts.extend(scripts_in(candidate))
        elif candidate.is_file():
            scripts.append(candidate)
        elif Path(target).is_file():
            scripts.append(Path(target).resolve())
        else:
            matches = [s for d in all_folders() for s in scripts_in(d) if s.name == target]
            if len(matches) == 1:
                scripts.append(matches[0])
            elif not matches:
                sys.exit(f"unknown target: {target}")
            else:
                sys.exit(f"ambiguous target {target}: " + ' '.join(str(m.relative_to(SIMULATIONS_DIR)) for m in matches))
    return list(dict.fromkeys(scripts))


def display_name(script):
    """A script's name relative to the simulations directory, or its path if it lies elsewhere."""
    try:
        return str(script.relative_to(SIMULATIONS_DIR))
    except ValueError:
        return str(script)


def run_script(script, root_dir, log_dir, progress):
    """Run one script in a child process; returns (exit code, seconds).

    Output handling: by default the script's stdout and stderr stream to the terminal, and
    with --log are also copied to a log file. With --progress the script's progress bars are
    drawn on the terminal through an extra file descriptor (PROGRESS_FD, see
    utils/parallel.py), so that with --log everything else can go to the log alone.
    """
    env = dict(os.environ)
    if root_dir is not None:
        env['ROOT_DIR'] = str(root_dir)
    env['PYTHONPATH'] = str(SRC_DIR) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['PYTHONUNBUFFERED'] = '1'
    cmd = [sys.executable, str(script)]
    pass_fds = ()
    if progress and sys.platform != 'win32':
        progress_fd = os.dup(sys.stderr.fileno())      # the child inherits this copy of the terminal
        env['PROGRESS_FD'] = str(progress_fd)
        pass_fds = (progress_fd,)
    t0 = time.time()
    try:
        if log_dir is None:
            code = subprocess.run(cmd, env=env, cwd=REPO_DIR, pass_fds=pass_fds).returncode
        else:
            log_path = log_dir / f'{script.parent.name}_{script.stem}.log'
            with open(log_path, 'w') as log:
                if progress:
                    code = subprocess.run(cmd, env=env, cwd=REPO_DIR, stdout=log, stderr=subprocess.STDOUT, pass_fds=pass_fds).returncode
                else:
                    proc = subprocess.Popen(cmd, env=env, cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    for line in proc.stdout:
                        sys.stdout.write(line)
                        log.write(line)
                    code = proc.wait()
    finally:
        for fd in pass_fds:
            os.close(fd)
    return code, time.time() - t0


def hms(seconds):
    seconds = int(round(seconds))
    return f'{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('targets', nargs='*', metavar='TARGET',
                        help='a folder (figure_02), a script (figure_02/satprod_PprodAsweep.py) or a bare script name; default: all folders except '
                             + ' '.join(sorted(EXCLUDE)))
    parser.add_argument('--root-dir', metavar='DIR', type=Path, help='data directory to write to (default: ROOT_DIR from the environment or .env)')
    parser.add_argument('--dry-run', action='store_true', help='list what would run, in order, and exit')
    parser.add_argument('--stop-on-error', action='store_true', help='stop at the first failing script (default: continue and report)')
    parser.add_argument('--log', metavar='DIR', type=Path, help='also write each script\'s output to DIR/<folder>_<script>.log')
    parser.add_argument('--progress', action='store_true', help='draw a progress bar per script on the terminal; with --log, the scripts\' own output then goes only to the log')
    args = parser.parse_args()

    scripts = resolve_targets(args.targets)

    if args.root_dir is None:
        from dotenv import load_dotenv
        load_dotenv(REPO_DIR / '.env')
        if not os.environ.get('ROOT_DIR'):
            parser.error('no data directory: pass --root-dir or set ROOT_DIR (in the environment or a .env file)')
        root_dir = Path(os.environ['ROOT_DIR'])
    else:
        root_dir = args.root_dir.resolve()
        root_dir.mkdir(parents=True, exist_ok=True)

    print(f'data directory: {root_dir}')
    print(f'{len(scripts)} script(s):')
    for s in scripts:
        print(f'  {display_name(s)}')
    if args.dry_run:
        return
    if args.log:
        args.log.mkdir(parents=True, exist_ok=True)

    results = []
    for s in scripts:
        name = display_name(s)
        print(f'\n===== {name} =====', flush=True)
        code, seconds = run_script(s, args.root_dir and root_dir, args.log, args.progress)
        results.append((name, code, seconds))
        if code != 0 and args.stop_on_error:
            break

    print('\n===== summary =====')
    for name, code, seconds in results:
        print(f'{"ok    " if code == 0 else "FAILED"}  {hms(seconds)}  {name}')
    failed = [name for name, code, _ in results if code != 0]
    print(f'{len(results) - len(failed)} ok, {len(failed)} failed, total {hms(sum(t for _, _, t in results))}')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
