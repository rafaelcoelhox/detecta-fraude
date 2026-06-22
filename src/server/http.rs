use super::Conn;
use crate::index::IndexReader;
use crate::parse::parse_payload;
use crate::response::Responses;
use crate::vectorize::vectorize_q;
use libc::{c_int, c_void, EAGAIN, EINTR};
use std::io;

const MAX_REQ_HEAD: usize = 4096;
const MAX_BODY: usize = 4096;

pub(crate) enum ParseStatus<'a> {
    Need,
    Bad,
    Got {
        method: &'a [u8],
        path: &'a [u8],
        body_start: usize,
        body_end: usize,
    },
}

pub(crate) fn parse_request(buf: &[u8]) -> ParseStatus<'_> {
    let head_end = match find_double_crlf(buf) {
        Some(p) => p,
        None => {
            if buf.len() > MAX_REQ_HEAD {
                return ParseStatus::Bad;
            }
            return ParseStatus::Need;
        }
    };

    let line_end = match buf.iter().position(|&b| b == b'\r') {
        Some(p) => p,
        None => return ParseStatus::Bad,
    };
    let line = &buf[..line_end];
    let sp1 = match line.iter().position(|&b| b == b' ') {
        Some(p) => p,
        None => return ParseStatus::Bad,
    };
    let after_method = &line[sp1 + 1..];
    let sp2 = match after_method.iter().position(|&b| b == b' ') {
        Some(p) => p,
        None => return ParseStatus::Bad,
    };
    let method = &line[..sp1];
    let path = &after_method[..sp2];

    let body_start = head_end + 4;

    if method == b"GET" {
        return ParseStatus::Got {
            method,
            path,
            body_start,
            body_end: body_start,
        };
    }

    let cl = match find_content_length(&buf[..head_end]) {
        Some(n) => n,
        None => return ParseStatus::Bad,
    };
    if cl > MAX_BODY {
        return ParseStatus::Bad;
    }
    let body_end = body_start + cl;
    if buf.len() < body_end {
        return ParseStatus::Need;
    }
    ParseStatus::Got {
        method,
        path,
        body_start,
        body_end,
    }
}

#[inline]
fn find_double_crlf(buf: &[u8]) -> Option<usize> {
    if buf.len() < 4 {
        return None;
    }
    let mut i = 3;
    while i < buf.len() {
        if buf[i] == b'\n' && buf[i - 1] == b'\r' && buf[i - 2] == b'\n' && buf[i - 3] == b'\r' {
            return Some(i - 3);
        }
        i += 1;
    }
    None
}

fn find_content_length(headers: &[u8]) -> Option<usize> {
    const NEEDLE: &[u8] = b"content-length:";
    let mut i = 0;
    'outer: while i + NEEDLE.len() <= headers.len() {
        for (k, &c) in NEEDLE.iter().enumerate() {
            let h = headers[i + k];
            let lh = if h.is_ascii_uppercase() { h + 32 } else { h };
            if lh != c {
                i += 1;
                continue 'outer;
            }
        }
        let mut j = i + NEEDLE.len();
        while j < headers.len() && (headers[j] == b' ' || headers[j] == b'\t') {
            j += 1;
        }
        let start = j;
        while j < headers.len() && (b'0'..=b'9').contains(&headers[j]) {
            j += 1;
        }
        if j == start {
            return None;
        }
        let s = std::str::from_utf8(&headers[start..j]).ok()?;
        return s.parse().ok();
    }
    None
}

pub(crate) fn handle_request<'a>(
    method: &[u8],
    path: &[u8],
    body: &[u8],
    index: &IndexReader,
    responses: &'a Responses,
) -> &'a [u8] {
    if method == b"POST" && path == b"/fraud-score" {
        let payload = match parse_payload(body) {
            Ok(p) => p,
            Err(_) => return &responses.fallback,
        };
        let q = vectorize_q(&payload);
        let fraud_count = index.fraud_count(&q);
        return responses.for_count(fraud_count);
    }
    if method == b"GET" && path == b"/ready" {
        return &responses.ready;
    }
    &responses.fallback
}

pub(crate) enum WriteResult<'a> {
    Done,
    Pending(&'a [u8]),
    Closed,
}

#[inline]
pub(crate) fn write_all_nonblock(fd: c_int, mut buf: &[u8]) -> WriteResult<'_> {
    while !buf.is_empty() {
        let n = unsafe {
            libc::send(
                fd,
                buf.as_ptr() as *const c_void,
                buf.len(),
                libc::MSG_NOSIGNAL,
            )
        };
        if n > 0 {
            buf = &buf[n as usize..];
            continue;
        }
        if n == 0 {
            return WriteResult::Closed;
        }
        let err = io::Error::last_os_error();
        let code = err.raw_os_error().unwrap_or(0);
        if code == EAGAIN || code == libc::EWOULDBLOCK {
            return WriteResult::Pending(buf);
        }
        if code == EINTR {
            continue;
        }
        return WriteResult::Closed;
    }
    WriteResult::Done
}

pub(crate) fn flush_write(fd: c_int, conn: &mut Conn) -> bool {
    while conn.write_pos < conn.write_len {
        let remaining = &conn.write_buf[conn.write_pos..conn.write_len];
        let n = unsafe {
            libc::send(
                fd,
                remaining.as_ptr() as *const c_void,
                remaining.len(),
                libc::MSG_NOSIGNAL,
            )
        };
        if n > 0 {
            conn.write_pos += n as usize;
        } else if n == 0 {
            return false;
        } else {
            let err = io::Error::last_os_error();
            let code = err.raw_os_error().unwrap_or(0);
            if code == EAGAIN || code == libc::EWOULDBLOCK {
                return false;
            }
            if code == EINTR {
                continue;
            }
            return false;
        }
    }
    conn.write_len = 0;
    conn.write_pos = 0;
    true
}
