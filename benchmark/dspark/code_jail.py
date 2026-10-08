"""Pure-stdlib MBPP execution: readonly chroot, nobody UID, seccomp, limits.

Requires Linux root/CAP_SYS_CHROOT for setup ONLY. Candidate Python is never
executed as root, has no host mounts/network/process creation/writable files.
This is local benchmark isolation, not a general hostile multi-tenant service.
"""
import argparse
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import time

WRAPPER = r'''
import ctypes, json, os, resource, sys
assert os.getuid() == 65534 and os.geteuid() == 65534
payload = json.loads(sys.stdin.read(1000000))
for key, limit in ((resource.RLIMIT_CPU, 2), (resource.RLIMIT_AS, 268435456),
                   (resource.RLIMIT_FSIZE, 0), (resource.RLIMIT_NOFILE, 32),
                   (resource.RLIMIT_NPROC, 1), (resource.RLIMIT_CORE, 0)):
    resource.setrlimit(key, (limit, limit))
libc = ctypes.CDLL(None, use_errno=True)
assert libc.prctl(38, 1, 0, 0, 0) == 0
sec = ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libseccomp.so.2')
sec.seccomp_init.argtypes = [ctypes.c_uint32]
sec.seccomp_init.restype = ctypes.c_void_p
sec.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
sec.seccomp_syscall_resolve_name.restype = ctypes.c_int
sec.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
sec.seccomp_load.argtypes = [ctypes.c_void_p]
ctx = sec.seccomp_init(0x7fff0000)
assert ctx
for name in ('socket socketpair connect bind listen accept accept4 clone clone3 fork vfork '
             'execve execveat ptrace process_vm_readv process_vm_writev kill tkill tgkill '
             'bpf io_uring_setup userfaultfd mount umount2 chroot unshare setns '
             'open_by_handle_at name_to_handle_at keyctl add_key request_key kexec_load '
             'init_module finit_module delete_module reboot setuid setgid setresuid '
             'setresgid setreuid setregid setgroups').split():
    nr = sec.seccomp_syscall_resolve_name(name.encode())
    if nr >= 0:
        assert sec.seccomp_rule_add(ctx, 0x00050001, nr, 0) == 0
assert sec.seccomp_load(ctx) == 0
namespace = {'__name__': '__candidate__'}
try:
    for statement in payload.get('imports', []):
        exec(compile(statement, '<test-import>', 'exec'), namespace)
    exec(compile(payload['code'], '<candidate>', 'exec'), namespace)
    for statement in payload['tests']:
        exec(compile(statement, '<test>', 'exec'), namespace)
    result = {'passed': True, 'uid': os.getuid()}
except BaseException as exc:
    result = {'passed': False, 'error': type(exc).__name__, 'detail': str(exc)[:1000], 'uid': os.getuid()}
print('\nDSPARK_JAIL_RESULT:' + json.dumps(result), flush=True)
'''


def build(root):
    if root.exists():
        raise ValueError('fresh jail required')
    root.mkdir(parents=True)
    executable = Path(sys.executable).resolve()
    stdlib = Path(sys.base_prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}'
    (root / 'bin').mkdir()
    shutil.copy2(executable, root / 'bin' / 'python3')
    shutil.copytree(stdlib, root / 'lib' / stdlib.name,
        ignore=shutil.ignore_patterns('site-packages', '__pycache__', 'test', 'tests', 'idlelib', 'tkinter', 'ensurepip'))
    libraries = set()
    binaries = [executable] + list((stdlib / 'lib-dynload').glob('*.so'))
    for binary in binaries:
        for line in subprocess.check_output(['ldd', str(binary)], text=True).splitlines():
            for token in line.split():
                if token.startswith('/') and Path(token).is_file():
                    libraries.add(Path(token))
    libraries.add(Path('/usr/lib/x86_64-linux-gnu/libseccomp.so.2'))
    libraries.add(Path('/usr/lib/x86_64-linux-gnu/libffi.so.8'))
    for lib in libraries:
        destination = root / str(lib).lstrip('/')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lib, destination, follow_symlinks=True)
    # Host container disallows getrandom; Python needs readonly random fallback.
    # This single random character device exposes neither storage nor host files.
    (root / 'dev').mkdir()
    os.mknod(root / 'dev/urandom', stat.S_IFCHR | 0o444, os.makedev(1, 9))
    # Everything readonly for nobody; no /proc, storage devices or host mounts.
    for folder, _, files in os.walk(root):
        os.chmod(folder, 0o755)
        for name in files:
            path = Path(folder) / name
            os.chmod(path, 0o755 if path == root / 'bin' / 'python3'
                     or path.name.startswith('ld-linux') else 0o644)
    probe = evaluate(root, 'import math\ndef f(x): return math.sqrt(x)', ['assert f(9) == 3'])
    if not probe.get('passed'):
        raise RuntimeError('jail smoke failed: ' + repr(probe))
    for code in ('import socket; socket.socket()', 'import os; os.fork()',
                 "open('/root/conda-envs/sglang/bin/python', 'rb')", "open('/newfile', 'w')"):
        if evaluate(root, code, []).get('passed'):
            raise RuntimeError('isolation probe unexpectedly passed: ' + code)
    print(json.dumps({'jail': str(root), 'smoke': 'passed', 'isolation_probes': 4}), flush=True)


def evaluate(root, code, tests, imports=(), timeout=4):
    command = ['/usr/sbin/chroot', '--userspec=65534:65534', str(root),
               '/bin/python3', '-s', '-P', '-S', '-B', '-c', WRAPPER]
    payload = json.dumps({'code': code, 'tests': tests, 'imports': list(imports)}).encode()
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, close_fds=True, start_new_session=True, cwd='/',
        env={'PYTHONHOME': '/'})
    captured = bytearray()
    outcome = None
    try:
        process.stdin.write(payload)
        process.stdin.close()
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                if time.monotonic() >= deadline:
                    outcome = 'wall_timeout'
                    break
                for key, _ in selector.select(min(.1, max(0, deadline - time.monotonic()))):
                    chunk = os.read(key.fd, 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    captured.extend(chunk)
                if len(captured) > 65536:
                    outcome = 'output_limit'
                    break
        if outcome:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
        process.stdout.close()
    if outcome:
        return {'passed': False, 'error': outcome}
    text = captured.decode(errors='replace')
    marker = 'DSPARK_JAIL_RESULT:'
    if marker not in text or process.returncode:
        return {'passed': False, 'error': 'child_exit', 'exit_code': process.returncode, 'detail': text[-1000:]}
    return json.loads(text.rsplit(marker, 1)[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', required=True, type=Path)
    build(parser.parse_args().build)
