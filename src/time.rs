#[derive(Clone, Copy, Debug)]
pub struct Stamp {
    pub epoch_minutes: i64,
    pub hour: u8,
    pub weekday: u8,
}

#[inline]
fn dig2(b: &[u8], off: usize) -> u32 {
    let d0 = (b[off] - b'0') as u32;
    let d1 = (b[off + 1] - b'0') as u32;
    d0 * 10 + d1
}

#[inline]
fn dig4(b: &[u8], off: usize) -> u32 {
    let d0 = (b[off] - b'0') as u32;
    let d1 = (b[off + 1] - b'0') as u32;
    let d2 = (b[off + 2] - b'0') as u32;
    let d3 = (b[off + 3] - b'0') as u32;
    d0 * 1000 + d1 * 100 + d2 * 10 + d3
}

// Algoritmo Howard Hinnant: days_from_civil. Suporta intervalo amplo, exato.
#[inline]
fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = y - (m <= 2) as i32;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as i64;
    let m = m as i64;
    let d = d as i64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era as i64 * 146097 + doe - 719468
}

#[inline]
pub fn parse_iso8601(buf: &[u8]) -> Option<Stamp> {
    // o dataset oficial respeita o formato.
    if buf.len() < 20 {
        return None;
    }
    let year = dig4(buf, 0) as i32;
    let month = dig2(buf, 5);
    let day = dig2(buf, 8);
    let hour = dig2(buf, 11);
    let minute = dig2(buf, 14);
    let second = dig2(buf, 17);
    if month == 0 || month > 12 || day == 0 || day > 31 || hour > 23 || minute > 59 || second > 60 {
        return None;
    }
    let days = days_from_civil(year, month, day);
    let total_seconds = days * 86_400 + hour as i64 * 3600 + minute as i64 * 60 + second as i64;
    let epoch_minutes = total_seconds.div_euclid(60);
    // 1970-01-01 (days=0) é uma quinta-feira. Na convenção oficial segunda=0,
    // domingo=6, quinta=3. Ajustamos.
    // (days + 3) mod 7 produz 0=segunda quando days=4 (1970-01-05, segunda).
    // days=0 -> (0+3)%7 = 3 = quinta. ok.
    let weekday = ((days.rem_euclid(7) + 3).rem_euclid(7)) as u8;
    Some(Stamp {
        epoch_minutes,
        hour: hour as u8,
        weekday,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic() {
        let s = parse_iso8601(b"2026-03-11T20:23:35Z").unwrap();
        assert_eq!(s.hour, 20);
        // 2026-03-11 é quarta-feira => 2
        assert_eq!(s.weekday, 2);
    }

    #[test]
    fn weekday_known_dates() {
        // 1970-01-05 é segunda.
        assert_eq!(parse_iso8601(b"1970-01-05T00:00:00Z").unwrap().weekday, 0);
        // 1970-01-04 é domingo.
        assert_eq!(parse_iso8601(b"1970-01-04T00:00:00Z").unwrap().weekday, 6);
        // 2026-03-15 é domingo.
        assert_eq!(parse_iso8601(b"2026-03-15T00:00:00Z").unwrap().weekday, 6);
    }

    #[test]
    fn minutes_delta() {
        let a = parse_iso8601(b"2026-03-11T14:58:35Z").unwrap();
        let b = parse_iso8601(b"2026-03-11T20:23:35Z").unwrap();
        assert_eq!(b.epoch_minutes - a.epoch_minutes, 325);
    }
}
