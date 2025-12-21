pub fn seconds_to_ms(seconds: f32) -> i64 {
    if !seconds.is_finite() {
        return 0;
    }

    let ms = (seconds * 1_000.0).round().max(0.0);
    ms as i64
}
