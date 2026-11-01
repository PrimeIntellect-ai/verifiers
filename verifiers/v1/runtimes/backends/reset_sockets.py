"""Shut down the outbound TCP connections a VM pause left dead; runs inside the box.

A resumed box's processes still hold their pre-pause connections as ESTABLISHED, but the
network state outside the VM is gone, so a reader on one waits until its own timeout
(a chat program's model call: minutes). Shutting each one down turns that wait into an
immediate error its owner retries on a fresh connection. Loopback connections survive a
pause and stay, as do inbound ones (on a listening port) and those of the platform's
guest agents, which carry exec itself. Stdlib only, Linux 5.6+ (pidfd_getfd), run as
root: it borrows each process's socket through its pidfd.

`list` prints the outbound connections, by inode, before a pause; given those inodes, a
reset after the resume shuts down only the ones still open, never a connection opened
since. Without them (someone else paused the box) it shuts down every outbound one."""

import ctypes
import os
import socket
import sys

SYS_PIDFD_OPEN = 434
SYS_PIDFD_GETFD = 438
ESTABLISHED, LISTEN = "01", "0A"
GUEST_AGENTS = {"sandboxd", "guestinit"}


def connections() -> list[tuple[str, str, str, str]]:
    """(local, remote, state, inode) for every TCP socket, in /proc's hex notation."""
    rows = []
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path) as table:
                lines = table.read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            rows.append((fields[1], fields[2], fields[3], fields[9]))
    return rows


def loopback(address: str) -> bool:
    """Whether a /proc hex address (little-endian words) is in 127/8 or ::1."""
    host = address.split(":")[0]
    if host == "00000000000000000000000001000000":
        return True
    if len(host) == 32 and host[:24] == "0000000000000000FFFF0000":
        host = host[24:]  # IPv4-mapped IPv6
    return len(host) == 8 and host[6:] == "7F"


def outbound() -> set[str]:
    """The inodes of the established connections this box opened to another host."""
    rows = connections()
    listening = {
        int(local.split(":")[1], 16) for local, _, state, _ in rows if state == LISTEN
    }
    return {
        inode
        for local, remote, state, inode in rows
        if state == ESTABLISHED
        and not loopback(remote)
        and int(local.split(":")[1], 16) not in listening
    }


def main(args: list[str]) -> None:
    if args == ["list"]:
        print(" ".join(sorted(outbound())))
        return
    dead = outbound()
    if args:
        dead &= set(args)
    libc = ctypes.CDLL(None, use_errno=True)
    me = os.getpid()
    reset = 0
    for pid in filter(str.isdigit, os.listdir("/proc")):
        if int(pid) in (1, me):
            continue
        try:
            with open(f"/proc/{pid}/comm") as comm:
                if comm.read().strip() in GUEST_AGENTS:
                    continue
            fds = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue  # exited meanwhile
        pidfd = -1
        for fd in fds:
            try:
                link = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if not (link.startswith("socket:[") and link[8:-1] in dead):
                continue
            if pidfd < 0:
                pidfd = libc.syscall(SYS_PIDFD_OPEN, int(pid), 0)
                if pidfd < 0:
                    break
            borrowed = libc.syscall(SYS_PIDFD_GETFD, pidfd, int(fd), 0)
            if borrowed < 0:
                continue
            # Shutdown acts on the socket the owner shares, unlike close on our copy.
            with socket.socket(fileno=borrowed) as sock:
                if os.fstat(borrowed).st_ino != int(link[8:-1]):
                    continue  # the owner reused the fd since we read its link
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                    reset += 1
                except OSError:
                    pass
        if pidfd >= 0:
            os.close(pidfd)
    print(reset)


if __name__ == "__main__":
    main(sys.argv[1:])
