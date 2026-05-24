use crate::time::{parse_iso8601, Stamp};

#[derive(Default, Clone, Copy, Debug)]
pub struct RawPayload<'a> {
    pub amount: f64,
    pub installments: u32,
    pub requested_at: Stamp,
    pub customer_avg_amount: f64,
    pub tx_count_24h: u32,
    pub known_merchants_buf: &'a [u8],
    pub merchant_id: &'a [u8],
    pub merchant_mcc: &'a [u8],
    pub merchant_avg_amount: f64,
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f64,
    pub has_last_tx: bool,
    pub last_tx_stamp: Stamp,
    pub last_tx_km: f64,
}

impl Default for Stamp {
    fn default() -> Self {
        Stamp {
            epoch_minutes: 0,
            hour: 0,
            weekday: 0,
        }
    }
}

#[derive(Debug)]
pub struct ParseError;

#[inline]
fn skip_ws(buf: &[u8], mut i: usize) -> usize {
    while i < buf.len() {
        let c = buf[i];
        if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
            i += 1;
        } else {
            break;
        }
    }
    i
}

#[inline]
fn expect(buf: &[u8], i: usize, c: u8) -> Result<usize, ParseError> {
    let i = skip_ws(buf, i);
    if i >= buf.len() || buf[i] != c {
        return Err(ParseError);
    }
    Ok(i + 1)
}

#[inline]
fn read_string<'a>(buf: &'a [u8], i: usize) -> Result<(&'a [u8], usize), ParseError> {
    let i = skip_ws(buf, i);
    if i >= buf.len() || buf[i] != b'"' {
        return Err(ParseError);
    }
    let start = i + 1;
    let mut j = start;
    while j < buf.len() && buf[j] != b'"' {
        // strings do contrato não contêm escapes
        j += 1;
    }
    if j >= buf.len() {
        return Err(ParseError);
    }
    Ok((&buf[start..j], j + 1))
}

#[inline]
fn read_number_f64(buf: &[u8], i: usize) -> Result<(f64, usize), ParseError> {
    let i = skip_ws(buf, i);
    let start = i;
    let mut j = i;
    if j < buf.len() && (buf[j] == b'-' || buf[j] == b'+') {
        j += 1;
    }
    while j < buf.len() {
        let c = buf[j];
        if (b'0'..=b'9').contains(&c)
            || c == b'.'
            || c == b'e'
            || c == b'E'
            || c == b'-'
            || c == b'+'
        {
            j += 1;
        } else {
            break;
        }
    }
    if j == start {
        return Err(ParseError);
    }
    let s = std::str::from_utf8(&buf[start..j]).map_err(|_| ParseError)?;
    let v: f64 = s.parse().map_err(|_| ParseError)?;
    Ok((v, j))
}

#[inline]
fn read_number_u32(buf: &[u8], i: usize) -> Result<(u32, usize), ParseError> {
    let i = skip_ws(buf, i);
    let start = i;
    let mut j = i;
    while j < buf.len() && (b'0'..=b'9').contains(&buf[j]) {
        j += 1;
    }
    if j == start {
        return Err(ParseError);
    }
    let s = std::str::from_utf8(&buf[start..j]).map_err(|_| ParseError)?;
    let v: u32 = s.parse().map_err(|_| ParseError)?;
    Ok((v, j))
}

#[inline]
fn read_bool(buf: &[u8], i: usize) -> Result<(bool, usize), ParseError> {
    let i = skip_ws(buf, i);
    if i + 4 <= buf.len() && &buf[i..i + 4] == b"true" {
        Ok((true, i + 4))
    } else if i + 5 <= buf.len() && &buf[i..i + 5] == b"false" {
        Ok((false, i + 5))
    } else {
        Err(ParseError)
    }
}

#[inline]
fn is_null(buf: &[u8], i: usize) -> bool {
    let i = skip_ws(buf, i);
    i + 4 <= buf.len() && &buf[i..i + 4] == b"null"
}

// Avança até encontrar o fim do valor atual a partir de uma posição que aponta
// para o início de um valor JSON. Usado para pular campos não interessantes
// (como `id`) sem alocar.
fn skip_value(buf: &[u8], i: usize) -> Result<usize, ParseError> {
    let mut i = skip_ws(buf, i);
    if i >= buf.len() {
        return Err(ParseError);
    }
    match buf[i] {
        b'"' => {
            i += 1;
            while i < buf.len() && buf[i] != b'"' {
                i += 1;
            }
            if i >= buf.len() {
                return Err(ParseError);
            }
            Ok(i + 1)
        }
        b'{' | b'[' => {
            let open = buf[i];
            let close = if open == b'{' { b'}' } else { b']' };
            let mut depth = 1;
            i += 1;
            while i < buf.len() && depth > 0 {
                let c = buf[i];
                if c == b'"' {
                    i += 1;
                    while i < buf.len() && buf[i] != b'"' {
                        i += 1;
                    }
                } else if c == open {
                    depth += 1;
                } else if c == close {
                    depth -= 1;
                }
                i += 1;
            }
            if depth != 0 {
                return Err(ParseError);
            }
            Ok(i)
        }
        b't' => Ok(i + 4),
        b'f' => Ok(i + 5),
        b'n' => Ok(i + 4),
        _ => {
            // número
            while i < buf.len() {
                let c = buf[i];
                if c == b','
                    || c == b'}'
                    || c == b']'
                    || c == b' '
                    || c == b'\n'
                    || c == b'\r'
                    || c == b'\t'
                {
                    break;
                }
                i += 1;
            }
            Ok(i)
        }
    }
}

// Extrai o slice cru do array `known_merchants` (incluindo colchetes) para que
// a busca por merchant_id seja feita com substring match sem alocar.
fn read_array_raw<'a>(buf: &'a [u8], i: usize) -> Result<(&'a [u8], usize), ParseError> {
    let i = skip_ws(buf, i);
    if i >= buf.len() || buf[i] != b'[' {
        return Err(ParseError);
    }
    let start = i;
    let end = skip_value(buf, i)?;
    Ok((&buf[start..end], end))
}

// Caminha por um objeto JSON entre `{` e `}` chamando `on_key(key, value_start)`
// para cada par. O callback retorna a próxima posição depois do valor.
fn for_each_kv<'a, F>(buf: &'a [u8], start: usize, mut on_key: F) -> Result<usize, ParseError>
where
    F: FnMut(&'a [u8], &'a [u8], usize) -> Result<Option<usize>, ParseError>,
{
    let mut i = expect(buf, start, b'{')?;
    loop {
        i = skip_ws(buf, i);
        if i < buf.len() && buf[i] == b'}' {
            return Ok(i + 1);
        }
        let (key, next) = read_string(buf, i)?;
        i = expect(buf, next, b':')?;
        i = skip_ws(buf, i);
        let consumed = on_key(buf, key, i)?;
        i = match consumed {
            Some(n) => n,
            None => skip_value(buf, i)?,
        };
        i = skip_ws(buf, i);
        if i < buf.len() && buf[i] == b',' {
            i += 1;
            continue;
        }
        if i < buf.len() && buf[i] == b'}' {
            return Ok(i + 1);
        }
        return Err(ParseError);
    }
}

pub fn parse_payload(buf: &[u8]) -> Result<RawPayload<'_>, ParseError> {
    let mut p: RawPayload<'_> = RawPayload::default();

    for_each_kv(buf, 0, |buf, key, vstart| match key {
        b"id" => Ok(None),
        b"transaction" => {
            let end = for_each_kv(buf, vstart, |buf, k, v| match k {
                b"amount" => {
                    let (n, e) = read_number_f64(buf, v)?;
                    p.amount = n;
                    Ok(Some(e))
                }
                b"installments" => {
                    let (n, e) = read_number_u32(buf, v)?;
                    p.installments = n;
                    Ok(Some(e))
                }
                b"requested_at" => {
                    let (s, e) = read_string(buf, v)?;
                    p.requested_at = parse_iso8601(s).ok_or(ParseError)?;
                    Ok(Some(e))
                }
                _ => Ok(None),
            })?;
            Ok(Some(end))
        }
        b"customer" => {
            let end = for_each_kv(buf, vstart, |buf, k, v| match k {
                b"avg_amount" => {
                    let (n, e) = read_number_f64(buf, v)?;
                    p.customer_avg_amount = n;
                    Ok(Some(e))
                }
                b"tx_count_24h" => {
                    let (n, e) = read_number_u32(buf, v)?;
                    p.tx_count_24h = n;
                    Ok(Some(e))
                }
                b"known_merchants" => {
                    let (arr, e) = read_array_raw(buf, v)?;
                    p.known_merchants_buf = arr;
                    Ok(Some(e))
                }
                _ => Ok(None),
            })?;
            Ok(Some(end))
        }
        b"merchant" => {
            let end = for_each_kv(buf, vstart, |buf, k, v| match k {
                b"id" => {
                    let (s, e) = read_string(buf, v)?;
                    p.merchant_id = s;
                    Ok(Some(e))
                }
                b"mcc" => {
                    let (s, e) = read_string(buf, v)?;
                    p.merchant_mcc = s;
                    Ok(Some(e))
                }
                b"avg_amount" => {
                    let (n, e) = read_number_f64(buf, v)?;
                    p.merchant_avg_amount = n;
                    Ok(Some(e))
                }
                _ => Ok(None),
            })?;
            Ok(Some(end))
        }
        b"terminal" => {
            let end = for_each_kv(buf, vstart, |buf, k, v| match k {
                b"is_online" => {
                    let (b, e) = read_bool(buf, v)?;
                    p.is_online = b;
                    Ok(Some(e))
                }
                b"card_present" => {
                    let (b, e) = read_bool(buf, v)?;
                    p.card_present = b;
                    Ok(Some(e))
                }
                b"km_from_home" => {
                    let (n, e) = read_number_f64(buf, v)?;
                    p.km_from_home = n;
                    Ok(Some(e))
                }
                _ => Ok(None),
            })?;
            Ok(Some(end))
        }
        b"last_transaction" => {
            if is_null(buf, vstart) {
                p.has_last_tx = false;
                Ok(Some(vstart + 4))
            } else {
                p.has_last_tx = true;
                let end = for_each_kv(buf, vstart, |buf, k, v| match k {
                    b"timestamp" => {
                        let (s, e) = read_string(buf, v)?;
                        p.last_tx_stamp = parse_iso8601(s).ok_or(ParseError)?;
                        Ok(Some(e))
                    }
                    b"km_from_current" => {
                        let (n, e) = read_number_f64(buf, v)?;
                        p.last_tx_km = n;
                        Ok(Some(e))
                    }
                    _ => Ok(None),
                })?;
                Ok(Some(end))
            }
        }
        _ => Ok(None),
    })?;

    Ok(p)
}

// Confere se `merchant_id` aparece como string dentro do slice cru de
// known_merchants. Comparação byte-a-byte com aspas — evita falsos positivos
// por prefix match.
pub fn merchant_in_known(known_raw: &[u8], merchant_id: &[u8]) -> bool {
    if merchant_id.is_empty() {
        return false;
    }
    let needle_len = merchant_id.len();
    let mut i = 0;
    while i < known_raw.len() {
        if known_raw[i] == b'"' {
            let start = i + 1;
            let mut j = start;
            while j < known_raw.len() && known_raw[j] != b'"' {
                j += 1;
            }
            if j - start == needle_len && &known_raw[start..j] == merchant_id {
                return true;
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &[u8] = br#"{
      "id": "tx-3576980410",
      "transaction": { "amount": 384.88, "installments": 3, "requested_at": "2026-03-11T20:23:35Z" },
      "customer": { "avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-009", "MERC-001", "MERC-001"] },
      "merchant": { "id": "MERC-001", "mcc": "5912", "avg_amount": 298.95 },
      "terminal": { "is_online": false, "card_present": true, "km_from_home": 13.7090520965 },
      "last_transaction": { "timestamp": "2026-03-11T14:58:35Z", "km_from_current": 18.8626479774 }
    }"#;

    #[test]
    fn full_payload() {
        let p = parse_payload(EXAMPLE).unwrap();
        assert!((p.amount - 384.88).abs() < 1e-9);
        assert_eq!(p.installments, 3);
        assert!((p.customer_avg_amount - 769.76).abs() < 1e-9);
        assert_eq!(p.tx_count_24h, 3);
        assert_eq!(p.merchant_id, b"MERC-001");
        assert_eq!(p.merchant_mcc, b"5912");
        assert!((p.merchant_avg_amount - 298.95).abs() < 1e-9);
        assert!(!p.is_online);
        assert!(p.card_present);
        assert!((p.km_from_home - 13.7090520965).abs() < 1e-9);
        assert!(p.has_last_tx);
        assert!((p.last_tx_km - 18.8626479774).abs() < 1e-9);
        assert!(merchant_in_known(p.known_merchants_buf, p.merchant_id));
    }

    #[test]
    fn null_last_transaction() {
        let s = br#"{"id":"x","transaction":{"amount":1,"installments":1,"requested_at":"2026-03-11T00:00:00Z"},"customer":{"avg_amount":1,"tx_count_24h":0,"known_merchants":[]},"merchant":{"id":"A","mcc":"5411","avg_amount":1},"terminal":{"is_online":false,"card_present":true,"km_from_home":1},"last_transaction":null}"#;
        let p = parse_payload(s).unwrap();
        assert!(!p.has_last_tx);
    }

    #[test]
    fn merchant_membership_exact() {
        let arr = br#"["MERC-001","MERC-010"]"#;
        assert!(merchant_in_known(arr, b"MERC-001"));
        assert!(merchant_in_known(arr, b"MERC-010"));
        assert!(!merchant_in_known(arr, b"MERC-01"));
        assert!(!merchant_in_known(arr, b"MERC-0"));
        assert!(!merchant_in_known(arr, b"MERC-0100"));
    }
}
