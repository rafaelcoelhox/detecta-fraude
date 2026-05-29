use crate::consts::*;
use crate::parse::{merchant_in_known, RawPayload};
use crate::{quantize, DIM, STORE_DIM};

#[inline]
fn clamp01(v: f64) -> f64 {
    if v < 0.0 {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

#[inline(always)]
fn quantize_clamped(v: f64) -> i16 {
    quantize(clamp01(v))
}

pub fn vectorize_f64(p: &RawPayload<'_>) -> [f64; DIM] {
    let mut v = [0.0f64; DIM];

    v[0] = clamp01(p.amount / MAX_AMOUNT);
    v[1] = clamp01(p.installments as f64 / MAX_INSTALLMENTS);

    let ratio = if p.customer_avg_amount > 0.0 {
        p.amount / p.customer_avg_amount
    } else {
        f64::INFINITY
    };
    v[2] = clamp01(ratio / AMOUNT_VS_AVG_RATIO);

    v[3] = p.requested_at.hour as f64 / 23.0;
    v[4] = p.requested_at.weekday as f64 / 6.0;

    if p.has_last_tx {
        let minutes = (p.requested_at.epoch_minutes - p.last_tx_stamp.epoch_minutes) as f64;
        v[5] = clamp01(minutes / MAX_MINUTES);
        v[6] = clamp01(p.last_tx_km / MAX_KM);
    } else {
        v[5] = -1.0;
        v[6] = -1.0;
    }

    v[7] = clamp01(p.km_from_home / MAX_KM);
    v[8] = clamp01(p.tx_count_24h as f64 / MAX_TX_COUNT_24H);
    v[9] = if p.is_online { 1.0 } else { 0.0 };
    v[10] = if p.card_present { 1.0 } else { 0.0 };
    v[11] = if merchant_in_known(p.known_merchants_buf, p.merchant_id) {
        0.0
    } else {
        1.0
    };
    v[12] = mcc_risk_lookup(p.merchant_mcc);
    v[13] = clamp01(p.merchant_avg_amount / MAX_MERCHANT_AVG_AMOUNT);

    v
}

pub fn vectorize_q(p: &RawPayload<'_>) -> [i16; STORE_DIM] {
    let mut q = [0i16; STORE_DIM];
    q[0] = quantize_clamped(p.amount / MAX_AMOUNT);
    q[1] = quantize_clamped(p.installments as f64 / MAX_INSTALLMENTS);

    let ratio = if p.customer_avg_amount > 0.0 {
        p.amount / p.customer_avg_amount
    } else {
        f64::INFINITY
    };
    q[2] = quantize_clamped(ratio / AMOUNT_VS_AVG_RATIO);

    q[3] = quantize(p.requested_at.hour as f64 / 23.0);
    q[4] = quantize(p.requested_at.weekday as f64 / 6.0);

    if p.has_last_tx {
        let minutes = (p.requested_at.epoch_minutes - p.last_tx_stamp.epoch_minutes) as f64;
        q[5] = quantize_clamped(minutes / MAX_MINUTES);
        q[6] = quantize_clamped(p.last_tx_km / MAX_KM);
    } else {
        q[5] = -(crate::SCALE as i16);
        q[6] = -(crate::SCALE as i16);
    }

    q[7] = quantize_clamped(p.km_from_home / MAX_KM);
    q[8] = quantize_clamped(p.tx_count_24h as f64 / MAX_TX_COUNT_24H);
    q[9] = if p.is_online { crate::SCALE as i16 } else { 0 };
    q[10] = if p.card_present { crate::SCALE as i16 } else { 0 };
    q[11] = if merchant_in_known(p.known_merchants_buf, p.merchant_id) {
        0
    } else {
        crate::SCALE as i16
    };
    q[12] = quantize(mcc_risk_lookup(p.merchant_mcc));
    q[13] = quantize_clamped(p.merchant_avg_amount / MAX_MERCHANT_AVG_AMOUNT);
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_payload;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn legit_example() {
        let s = br#"{
          "id": "tx-1329056812",
          "transaction": { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
          "customer": { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003","MERC-016"] },
          "merchant": { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
          "terminal": { "is_online": false, "card_present": true, "km_from_home": 29.23 },
          "last_transaction": null
        }"#;
        let p = parse_payload(s).unwrap();
        let v = vectorize_f64(&p);
        // Esperado: [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]
        assert!(approx(v[0], 0.004112, 1e-4));
        assert!(approx(v[1], 2.0 / 12.0, 1e-4));
        assert!(approx(v[2], 0.05, 1e-4));
        assert!(approx(v[3], 18.0 / 23.0, 1e-3));
        assert!(approx(v[4], 2.0 / 6.0, 1e-3)); // quarta-feira => 2
        assert_eq!(v[5], -1.0);
        assert_eq!(v[6], -1.0);
        assert!(approx(v[7], 0.02923, 1e-4));
        assert!(approx(v[8], 3.0 / 20.0, 1e-4));
        assert_eq!(v[9], 0.0);
        assert_eq!(v[10], 1.0);
        assert_eq!(v[11], 0.0); // MERC-016 está em known
        assert!(approx(v[12], 0.15, 1e-9));
        assert!(approx(v[13], 60.25 / 10000.0, 1e-9));
    }

    #[test]
    fn fraud_example_unknown_merchant_and_null() {
        let s = br#"{
          "id":"tx-3330991687",
          "transaction":{"amount":9505.97,"installments":10,"requested_at":"2026-03-14T05:15:12Z"},
          "customer":{"avg_amount":81.28,"tx_count_24h":20,"known_merchants":["MERC-008","MERC-007","MERC-005"]},
          "merchant":{"id":"MERC-068","mcc":"7802","avg_amount":54.86},
          "terminal":{"is_online":false,"card_present":true,"km_from_home":952.27},
          "last_transaction":null
        }"#;
        let p = parse_payload(s).unwrap();
        let v = vectorize_f64(&p);
        // Esperado: [0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055]
        assert!(approx(v[0], 0.950597, 1e-4));
        assert!(approx(v[1], 10.0 / 12.0, 1e-4));
        assert_eq!(v[2], 1.0); // clamp
        assert!(approx(v[3], 5.0 / 23.0, 1e-4));
        // 2026-03-14 é sábado => 5
        assert!(approx(v[4], 5.0 / 6.0, 1e-4));
        assert_eq!(v[5], -1.0);
        assert_eq!(v[6], -1.0);
        assert!(approx(v[7], 0.95227, 1e-4));
        assert_eq!(v[8], 1.0);
        assert_eq!(v[9], 0.0);
        assert_eq!(v[10], 1.0);
        assert_eq!(v[11], 1.0); // unknown
        assert!(approx(v[12], 0.75, 1e-9)); // 7802
        assert!(approx(v[13], 0.005486, 1e-9));
    }

    #[test]
    fn unknown_mcc_defaults_to_05() {
        let s = br#"{"id":"x","transaction":{"amount":1,"installments":1,"requested_at":"2026-03-11T00:00:00Z"},"customer":{"avg_amount":1,"tx_count_24h":0,"known_merchants":[]},"merchant":{"id":"A","mcc":"9999","avg_amount":1},"terminal":{"is_online":false,"card_present":true,"km_from_home":1},"last_transaction":null}"#;
        let p = parse_payload(s).unwrap();
        let v = vectorize_f64(&p);
        assert_eq!(v[12], 0.5);
    }

    #[test]
    fn last_tx_minutes_and_km() {
        let s = br#"{"id":"x","transaction":{"amount":1,"installments":1,"requested_at":"2026-03-11T20:23:35Z"},"customer":{"avg_amount":1,"tx_count_24h":0,"known_merchants":[]},"merchant":{"id":"A","mcc":"5411","avg_amount":1},"terminal":{"is_online":false,"card_present":true,"km_from_home":1},"last_transaction":{"timestamp":"2026-03-11T14:58:35Z","km_from_current":18.8626479774}}"#;
        let p = parse_payload(s).unwrap();
        let v = vectorize_f64(&p);
        // delta = 325 minutos -> 325/1440 = 0.22569...
        assert!(approx(v[5], 325.0 / 1440.0, 1e-9));
        assert!(approx(v[6], 18.8626479774 / 1000.0, 1e-9));
    }
}
