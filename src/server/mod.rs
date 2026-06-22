use crate::index::IndexReader;
use crate::response::Responses;
use libc::{
    c_int, c_void, epoll_create1, epoll_ctl, epoll_event, epoll_wait, EAGAIN, EINTR, EPOLLERR,
    EPOLLHUP, EPOLLIN, EPOLLRDHUP, EPOLL_CTL_ADD, EPOLL_CTL_DEL,
};
use std::io;
use std::mem::MaybeUninit;

mod epoll;
mod http;
mod socket;

pub use socket::{accept_lb, close_fd, create_listener};

use epoll::{
    configure_busy_poll, epoll_idle_us, epoll_spin_us, epoll_timeout_ms, update_interest, wait_idle,
};
use http::{
    flush_write, handle_request, parse_request, write_all_nonblock, ParseStatus, WriteResult,
};
use socket::{recv_fds, set_nonblocking, set_tcp_options};

const CONN_BUF_CAP: usize = 16384;
const WRITE_BUF_CAP: usize = 512;
const MAX_CLIENT_FD: usize = 65_536;
const DEFAULT_CONN_POOL_INIT: usize = 128;

pub struct Server {
    epfd: c_int,
    uds_fd: c_int,
    index: &'static IndexReader,
    responses: &'static Responses,
    conns: Vec<Option<Box<Conn>>>,
    conn_pool: Vec<Box<Conn>>,
    conn_pool_cap: usize,
}

struct Conn {
    in_buf: [u8; CONN_BUF_CAP],
    in_len: usize,
    write_buf: [u8; WRITE_BUF_CAP],
    write_len: usize,
    write_pos: usize,
}

impl Conn {
    fn new() -> Box<Self> {
        Box::new(Conn {
            in_buf: [0; CONN_BUF_CAP],
            in_len: 0,
            write_buf: [0; WRITE_BUF_CAP],
            write_len: 0,
            write_pos: 0,
        })
    }

    #[inline]
    fn reset(&mut self) {
        self.in_len = 0;
        self.write_len = 0;
        self.write_pos = 0;
    }

    #[inline]
    fn consume(&mut self, n: usize) {
        if n == self.in_len {
            self.in_len = 0;
        } else if n > 0 {
            self.in_buf.copy_within(n..self.in_len, 0);
            self.in_len -= n;
        }
    }

    #[inline]
    fn has_pending_write(&self) -> bool {
        self.write_pos < self.write_len
    }
}

impl Server {
    pub fn new(
        uds_fd: c_int,
        index: &'static IndexReader,
        responses: &'static Responses,
    ) -> io::Result<Self> {
        set_nonblocking(uds_fd)?;
        let epfd = unsafe { epoll_create1(libc::EPOLL_CLOEXEC) };
        if epfd < 0 {
            return Err(io::Error::last_os_error());
        }
        configure_busy_poll(epfd);
        let mut s = Server {
            epfd,
            uds_fd,
            index,
            responses,
            conns: {
                let mut slots = Vec::with_capacity(MAX_CLIENT_FD);
                slots.resize_with(MAX_CLIENT_FD, || None);
                slots
            },
            conn_pool: Vec::new(),
            conn_pool_cap: conn_pool_cap(),
        };
        for _ in 0..s.conn_pool_cap.min(DEFAULT_CONN_POOL_INIT) {
            s.conn_pool.push(Conn::new());
        }
        s.register(uds_fd)?;
        Ok(s)
    }

    fn register(&mut self, fd: c_int) -> io::Result<()> {
        let mut ev = epoll_event {
            events: (EPOLLIN | EPOLLRDHUP) as u32,
            u64: fd as u64,
        };
        let r = unsafe { epoll_ctl(self.epfd, EPOLL_CTL_ADD, fd, &mut ev) };
        if r < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    fn deregister(&mut self, fd: c_int) {
        let mut ev = epoll_event { events: 0, u64: 0 };
        unsafe {
            epoll_ctl(self.epfd, EPOLL_CTL_DEL, fd, &mut ev);
        }
    }

    fn slot_mut(&mut self, fd: c_int) -> Option<&mut Box<Conn>> {
        let idx = fd as usize;
        self.conns.get_mut(idx)?.as_mut()
    }

    fn alloc_conn(&mut self) -> Box<Conn> {
        match self.conn_pool.pop() {
            Some(mut c) => {
                c.reset();
                c
            }
            None => Conn::new(),
        }
    }

    fn put_conn(&mut self, fd: c_int, conn: Box<Conn>) -> bool {
        let idx = fd as usize;
        if idx >= self.conns.len() {
            return false;
        }
        self.conns[idx] = Some(conn);
        true
    }

    fn take_conn(&mut self, fd: c_int) -> Option<Box<Conn>> {
        let idx = fd as usize;
        self.conns.get_mut(idx)?.take()
    }

    pub fn run(&mut self) -> io::Result<()> {
        const MAX_EVENTS: usize = 128;
        let mut events: [epoll_event; MAX_EVENTS] = unsafe { MaybeUninit::zeroed().assume_init() };
        let spin = std::time::Duration::from_micros(epoll_spin_us());
        let idle_us = epoll_idle_us();
        let idle_timeout = epoll_timeout_ms();
        loop {
            let mut n = unsafe { epoll_wait(self.epfd, events.as_mut_ptr(), MAX_EVENTS as i32, 0) };
            if n == 0 && !spin.is_zero() {
                let start = std::time::Instant::now();
                while start.elapsed() < spin {
                    n = unsafe { epoll_wait(self.epfd, events.as_mut_ptr(), MAX_EVENTS as i32, 0) };
                    if n != 0 {
                        break;
                    }
                    std::hint::spin_loop();
                }
            }
            if n == 0 {
                n = wait_idle(
                    self.epfd,
                    events.as_mut_ptr(),
                    MAX_EVENTS as i32,
                    idle_us,
                    idle_timeout,
                );
            }
            if n < 0 {
                let err = io::Error::last_os_error();
                if err.raw_os_error() == Some(EINTR) {
                    continue;
                }
                return Err(err);
            }
            for i in 0..n as usize {
                let fd = events[i].u64 as c_int;
                if fd == self.uds_fd {
                    self.handle_uds()?;
                } else {
                    self.handle_conn(fd, events[i].events);
                }
            }
        }
    }

    fn handle_uds(&mut self) -> io::Result<()> {
        loop {
            match recv_fds(self.uds_fd) {
                Ok(Some(fd)) => {
                    if fd < 0 || fd as usize >= MAX_CLIENT_FD {
                        unsafe { libc::close(fd) };
                        continue;
                    }
                    if let Err(_) = set_nonblocking(fd) {
                        unsafe { libc::close(fd) };
                        continue;
                    }
                    set_tcp_options(fd);
                    if self.register(fd).is_err() {
                        unsafe { libc::close(fd) };
                        continue;
                    }
                    let conn = self.alloc_conn();
                    if !self.put_conn(fd, conn) {
                        self.deregister(fd);
                        unsafe { libc::close(fd) };
                        continue;
                    }
                    self.handle_conn(fd, EPOLLIN as u32);
                }
                Ok(None) => return Ok(()),
                Err(e) => {
                    if e.kind() == io::ErrorKind::WouldBlock {
                        return Ok(());
                    }
                    return Err(e);
                }
            }
        }
    }

    fn handle_conn(&mut self, fd: c_int, events: u32) {
        let mut close = false;

        if (events & (EPOLLIN as u32)) != 0 {
            let conn = match self.slot_mut(fd) {
                Some(c) => c,
                None => return,
            };
            let space = CONN_BUF_CAP - conn.in_len;
            if space == 0 {
                close = true;
            } else {
                loop {
                    let n = unsafe {
                        libc::recv(
                            fd,
                            conn.in_buf.as_mut_ptr().add(conn.in_len) as *mut c_void,
                            space,
                            0,
                        )
                    };
                    if n > 0 {
                        conn.in_len += n as usize;
                        break;
                    }
                    if n == 0 {
                        close = true;
                        break;
                    }
                    let err = io::Error::last_os_error();
                    let code = err.raw_os_error().unwrap_or(0);
                    if code == EAGAIN || code == libc::EWOULDBLOCK {
                        break;
                    }
                    if code == EINTR {
                        continue;
                    }
                    close = true;
                    break;
                }
            }
        } else if (events & ((EPOLLERR | EPOLLHUP) as u32)) != 0 {
            close = true;
        }

        if !close {
            close = self.process_requests(fd);
        }
        if close {
            self.close_conn(fd);
        }
    }

    fn close_conn(&mut self, fd: c_int) {
        self.deregister(fd);
        if let Some(mut conn) = self.take_conn(fd) {
            conn.reset();
            if self.conn_pool.len() < self.conn_pool_cap {
                self.conn_pool.push(conn);
            }
        }
        unsafe { libc::close(fd) };
    }

    fn process_requests(&mut self, fd: c_int) -> bool {
        let idx = fd as usize;
        let index = self.index;
        let responses = self.responses;
        loop {
            let conn = match self.conns.get_mut(idx).and_then(|s| s.as_mut()) {
                Some(c) => c,
                None => return true,
            };
            if conn.has_pending_write() {
                if !flush_write(fd, conn) {
                    update_interest(self.epfd, fd, true);
                    return false;
                }
                update_interest(self.epfd, fd, false);
            }

            let parse_result = parse_request(&conn.in_buf[..conn.in_len]);
            let (resp, consumed): (&[u8], usize) = match parse_result {
                ParseStatus::Need => return false,
                ParseStatus::Bad => (&responses.fallback[..], conn.in_len),
                ParseStatus::Got {
                    method,
                    path,
                    body_start,
                    body_end,
                } => {
                    let m_ptr = method.as_ptr();
                    let m_len = method.len();
                    let p_ptr = path.as_ptr();
                    let p_len = path.len();
                    let body = &conn.in_buf[body_start..body_end];
                    let body_ptr = body.as_ptr();
                    let body_len = body.len();
                    let method: &[u8] = unsafe { std::slice::from_raw_parts(m_ptr, m_len) };
                    let path: &[u8] = unsafe { std::slice::from_raw_parts(p_ptr, p_len) };
                    let body: &[u8] = unsafe { std::slice::from_raw_parts(body_ptr, body_len) };
                    let resp = handle_request(method, path, body, index, responses);
                    (resp, body_end)
                }
            };

            conn.consume(consumed);

            match write_all_nonblock(fd, resp) {
                WriteResult::Done => continue,
                WriteResult::Pending(remaining) => {
                    if remaining.len() > WRITE_BUF_CAP {
                        return true;
                    }
                    conn.write_buf[..remaining.len()].copy_from_slice(remaining);
                    conn.write_len = remaining.len();
                    conn.write_pos = 0;
                    update_interest(self.epfd, fd, true);
                    return false;
                }
                WriteResult::Closed => return true,
            }
        }
    }
}

fn conn_pool_cap() -> usize {
    std::env::var("CONN_POOL_CAP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_CONN_POOL_INIT)
}
