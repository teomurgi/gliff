//! Manual smoke test: capture 1 s of the default sink's monitor and play
//! it back locally. Needs a live PipeWire session; not run by the test
//! suite.
fn main() {
    let mut capture = match gliff_audio::Capture::start() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("capture unavailable (ok on a headless machine): {e}");
            return;
        }
    };
    let playback = match gliff_audio::Playback::start() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("playback unavailable (ok on a headless machine): {e}");
            return;
        }
    };
    let mut total = 0usize;
    let t0 = std::time::Instant::now();
    while t0.elapsed() < std::time::Duration::from_secs(2) {
        std::thread::sleep(std::time::Duration::from_millis(25));
        let mut packets = Vec::new();
        capture.drain(&mut packets);
        total += packets.len();
        for p in packets {
            playback.play(p);
        }
    }
    println!("captured+played {total} packets in 2s");
}
