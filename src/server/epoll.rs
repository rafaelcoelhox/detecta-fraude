use libc::{
    c_int, epoll_ctl, epoll_event, epoll_pwait2, epoll_wait, timespec, ENOSYS, EPOLLIN, EPOLLOUT,
    EPOLLRDHUP, EPOLL_CTL_MOD,
};

#[repr(C)]
struct EpollParams {
    busy_poll_usecs: u32,
    busy_poll_budget: u16,
    prefer_busy_poll: u8,
    _pad: u8,
}

const fn iow(ty: u32, nr: u32, size: u32) -> libc::c_ulong {
    ((1u32 << 30) | (size << 16) | (ty << 8) | nr) as libc::c_ulong
}
const EPIOCSPARAMS: libc::c_ulong = iow(0x8A, 0x01, std::mem::size_of::<EpollParams>() as u32);

pub(crate) fn configure_busy_poll(epfd: c_int) {
    let usecs = env_u32("EPOLL_BUSY_POLL_US", 50);
    let prefer = env_u32("EPOLL_PREFER_BUSY_POLL", 1) as u8;
    if usecs == 0 && prefer == 0 {
        return;
    }
    let params = EpollParams {
        busy_poll_usecs: usecs,
        busy_poll_budget: env_u32("EPOLL_BUSY_POLL_BUDGET", 8) as u16,
        prefer_busy_poll: prefer,
        _pad: 0,
    };
    unsafe {
        libc::ioctl(epfd, EPIOCSPARAMS as _, &params as *const EpollParams);
    }
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub(crate) fn epoll_spin_us() -> u64 {
    std::env::var("EPOLL_SPIN_US")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50)
}

pub(crate) fn epoll_idle_us() -> u64 {
    std::env::var("EPOLL_IDLE_US")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

pub(crate) fn epoll_timeout_ms() -> c_int {
    std::env::var("EPOLL_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
}

pub(crate) fn wait_idle(
    epfd: c_int,
    events: *mut epoll_event,
    max_events: c_int,
    idle_us: u64,
    fallback_timeout_ms: c_int,
) -> c_int {
    if idle_us == 0 {
        return unsafe { epoll_wait(epfd, events, max_events, fallback_timeout_ms) };
    }

    let timeout = timespec {
        tv_sec: (idle_us / 1_000_000) as _,
        tv_nsec: ((idle_us % 1_000_000) * 1000) as _,
    };
    let n = unsafe { epoll_pwait2(epfd, events, max_events, &timeout, std::ptr::null()) };
    if n < 0 && std::io::Error::last_os_error().raw_os_error() == Some(ENOSYS) {
        return unsafe { epoll_wait(epfd, events, max_events, fallback_timeout_ms) };
    }
    n
}

pub(crate) fn update_interest(epfd: c_int, fd: c_int, want_write: bool) {
    let mut flags = EPOLLIN | EPOLLRDHUP;
    if want_write {
        flags |= EPOLLOUT;
    }
    let mut ev = epoll_event {
        events: flags as u32,
        u64: fd as u64,
    };
    unsafe {
        epoll_ctl(epfd, EPOLL_CTL_MOD, fd, &mut ev);
    }
}
