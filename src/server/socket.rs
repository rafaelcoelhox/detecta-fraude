use libc::{
    c_int, c_void, cmsghdr, iovec, msghdr, recvmsg, EAGAIN, EINTR, F_GETFL, F_SETFL,
    MSG_CMSG_CLOEXEC, MSG_DONTWAIT, O_NONBLOCK, SCM_RIGHTS, SOL_SOCKET,
};
use std::io;
use std::os::raw::c_uchar;
use std::path::Path;

pub(crate) fn set_nonblocking(fd: c_int) -> io::Result<()> {
    unsafe {
        let flags = libc::fcntl(fd, F_GETFL);
        if flags < 0 {
            return Err(io::Error::last_os_error());
        }
        let r = libc::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        if r < 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

pub(crate) fn set_tcp_options(fd: c_int) {
    let on: c_int = 1;
    unsafe {
        libc::setsockopt(
            fd,
            libc::IPPROTO_TCP,
            libc::TCP_NODELAY,
            &on as *const c_int as *const c_void,
            std::mem::size_of::<c_int>() as libc::socklen_t,
        );
        libc::setsockopt(
            fd,
            libc::IPPROTO_TCP,
            libc::TCP_QUICKACK,
            &on as *const c_int as *const c_void,
            std::mem::size_of::<c_int>() as libc::socklen_t,
        );
    }
}

pub(crate) fn recv_fds(uds_fd: c_int) -> io::Result<Option<c_int>> {
    let mut dummy = [0u8; 1];
    let mut iov = iovec {
        iov_base: dummy.as_mut_ptr() as *mut c_void,
        iov_len: dummy.len(),
    };
    const CMSG_SPACE: usize = 24 + 16;
    let mut cmsg_buf = [0u8; CMSG_SPACE];
    let mut mh: msghdr = unsafe { std::mem::zeroed() };
    mh.msg_iov = &mut iov;
    mh.msg_iovlen = 1;
    mh.msg_control = cmsg_buf.as_mut_ptr() as *mut c_void;
    mh.msg_controllen = cmsg_buf.len() as _;

    let n = unsafe { recvmsg(uds_fd, &mut mh, MSG_DONTWAIT | MSG_CMSG_CLOEXEC) };
    if n < 0 {
        let err = io::Error::last_os_error();
        let code = err.raw_os_error().unwrap_or(0);
        if code == EAGAIN || code == libc::EWOULDBLOCK {
            return Err(err);
        }
        if code == EINTR {
            return Ok(None);
        }
        return Err(err);
    }
    if mh.msg_controllen == 0 {
        return Ok(None);
    }
    let cmsg = unsafe { CMSG_FIRSTHDR(&mh) };
    if cmsg.is_null() {
        return Ok(None);
    }
    let level = unsafe { (*cmsg).cmsg_level };
    let ctype = unsafe { (*cmsg).cmsg_type };
    if level != SOL_SOCKET || ctype != SCM_RIGHTS {
        return Ok(None);
    }
    let data = unsafe { CMSG_DATA(cmsg) };
    let fd = unsafe { std::ptr::read_unaligned(data as *const c_int) };
    Ok(Some(fd))
}

#[allow(non_snake_case)]
unsafe fn CMSG_FIRSTHDR(mh: *const msghdr) -> *mut cmsghdr {
    if (*mh).msg_controllen as usize >= std::mem::size_of::<cmsghdr>() {
        (*mh).msg_control as *mut cmsghdr
    } else {
        std::ptr::null_mut()
    }
}

#[allow(non_snake_case)]
unsafe fn CMSG_DATA(cmsg: *const cmsghdr) -> *mut c_uchar {
    (cmsg as *mut u8).add(cmsg_align(std::mem::size_of::<cmsghdr>())) as *mut c_uchar
}

fn cmsg_align(n: usize) -> usize {
    (n + std::mem::size_of::<usize>() - 1) & !(std::mem::size_of::<usize>() - 1)
}

pub fn create_listener(path: &Path) -> io::Result<c_int> {
    let _ = std::fs::remove_file(path);
    let fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC, 0) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
    let bytes = path.as_os_str().as_encoded_bytes();
    if bytes.len() >= addr.sun_path.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "path too long"));
    }
    for (i, b) in bytes.iter().enumerate() {
        addr.sun_path[i] = *b as i8;
    }
    let len = std::mem::size_of::<libc::sa_family_t>() + bytes.len();
    let r = unsafe {
        libc::bind(
            fd,
            &addr as *const libc::sockaddr_un as *const libc::sockaddr,
            len as libc::socklen_t,
        )
    };
    if r < 0 {
        return Err(io::Error::last_os_error());
    }
    if unsafe { libc::listen(fd, 4) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(fd)
}

pub fn accept_lb(listener: c_int) -> io::Result<c_int> {
    let fd = unsafe {
        libc::accept4(
            listener,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            libc::SOCK_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(fd)
}

pub fn close_fd(fd: c_int) {
    unsafe {
        libc::close(fd);
    }
}
