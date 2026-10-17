//! Playback of the remote machine's Opus packets into the local default
//! sink, on a thread running a PipeWire main loop.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use pipewire as pw;
use pw::properties::properties;
use pw::spa;

use crate::{check_format, format_param, init, Error, PacketQueue, Result};

/// Packets queued for the sink: 1.2 s. PipeWire runs this stream's callback
/// about four times a second and takes two ~256 ms buffers each time, so
/// one callback can consume half a second at once; the queue must absorb
/// that plus jitter and the network's own batching, or packets drop on the
/// way in. The network loop never blocks on audio: past the cap the oldest
/// packet is dropped and Opus conceals the gap.
const QUEUE_PACKETS: usize = 60;

/// What the PipeWire thread shares with the owner.
struct Shared {
    queue: PacketQueue,
    failed: Option<Error>,
}

/// Playback into the default sink; owns a thread with an Opus decoder and
/// a PipeWire stream. Dropping stops the thread (see `Capture` for how).
pub struct Playback {
    shared: Arc<Mutex<Shared>>,
    mainloop: crate::LoopHandle,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Playback {
    /// Spawn the playback thread. Fails when PipeWire cannot be reached, so
    /// the session continues video-only.
    pub fn start() -> Result<Playback> {
        init();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("gliff-audio-play".into())
            .spawn(move || run_playback(ready_tx))
            .map_err(|e| Error::ThreadEnded(e.to_string()))?;
        let (mainloop, shared) = ready_rx
            .recv()
            .map_err(|_| Error::ThreadEnded("before setup finished".into()))??;
        Ok(Self {
            shared,
            mainloop,
            thread: Some(handle),
        })
    }

    /// Queue one Opus packet; the packet is dropped when the internal
    /// buffer already holds ~500 ms, so a stalled or slow sink never
    /// backs up the network loop. Reports a stream that has failed since
    /// the last packet (a format mismatch): the sink then plays silence
    /// while the session continues video-only.
    pub fn play(&self, packet: Vec<u8>) {
        let mut shared = self.shared.lock().unwrap();
        if let Some(e) = shared.failed.take() {
            tracing::warn!(error = %e, "audio playback stopped");
        }
        if let Some(dropped) = shared.queue.push(packet) {
            tracing::warn!(dropped, "audio playback queue full; dropped oldest");
        }
    }
}

impl Drop for Playback {
    fn drop(&mut self) {
        crate::quit_loop(&self.mainloop);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn run_playback(ready: std::sync::mpsc::Sender<Result<(crate::LoopHandle, Arc<Mutex<Shared>>)>>) {
    match open_stream() {
        Ok(running) => {
            let _ = ready.send(Ok((running.handle(), running.shared.clone())));
            running.run();
        }
        Err(e) => {
            let _ = ready.send(Err(e));
        }
    }
}

/// The stereo samples waiting to play and where playout is within them.
/// Pure logic, so the resampling is tested without PipeWire.
#[derive(Default)]
struct Playout {
    /// Decoded interleaved samples not yet played.
    pending: VecDeque<f32>,
    /// Playout position in frames within `pending`, fractional: the sink
    /// and the sender run on different clocks, so playback is resampled by
    /// a trim close to 1 rather than dropping or repeating whole packets.
    pos: f64,
}

impl Playout {
    /// The next output frame, advancing by `step` input frames. `refill`
    /// appends the next decoded packet, returning false when none is left;
    /// then `None` comes back and the caller zeroes the rest of the buffer.
    fn next_frame(
        &mut self,
        step: f64,
        mut refill: impl FnMut(&mut VecDeque<f32>) -> bool,
    ) -> Option<(f32, f32)> {
        loop {
            let i = self.pos as usize;
            if self.pending.len() >= 2 * (i + 2) {
                let t = (self.pos - i as f64) as f32;
                let (l0, r0) = (self.pending[2 * i], self.pending[2 * i + 1]);
                let (l1, r1) = (self.pending[2 * i + 2], self.pending[2 * i + 3]);
                self.pos += step;
                let used = self.pos as usize;
                if used > 0 {
                    self.pending.drain(..2 * used);
                    self.pos -= used as f64;
                }
                return Some((l0 + (l1 - l0) * t, r0 + (r1 - r0) * t));
            }
            if !refill(&mut self.pending) {
                return None;
            }
        }
    }

    fn depth_frames(&self) -> f64 {
        (self.pending.len() / 2) as f64
    }
}

/// Decodes queued packets into the playout buffer, on demand.
struct Refiller {
    decoder: opus::Decoder,
    /// Scratch for one decoded frame, kept across callbacks.
    decoded: Vec<f32>,
    shared: Arc<Mutex<Shared>>,
    decode_warned: bool,
}

/// What the process callback needs, moved into the listener's user data.
struct PlaybackState {
    playout: Playout,
    refiller: Refiller,
    /// The negotiated format is not the one we asked for: playing what the
    /// graph sends instead would be noise, so the process callback stops
    /// and the sink keeps its own silence.
    failed: bool,
}

/// Playout depth the rate control aims for: 600 ms, half the queue, which
/// is a little over one callback's double-buffered take.
const TARGET_FRAMES: f64 = 28_800.0;
/// Largest playout-rate change, either way. Two percent is about 35 cents
/// of pitch, too small to hear, and enough to follow a clock that runs a
/// little fast or slow.
const MAX_TRIM: f64 = 0.02;
/// Depth error that earns the full trim: 250 ms.
const FULL_TRIM_FRAMES: f64 = 12_000.0;

/// The playout step for the next callback: above the target depth the
/// stream plays slightly faster, below it slightly slower.
fn trim_step(playout: &Playout, shared: &Mutex<Shared>) -> f64 {
    let queued = shared.lock().unwrap().queue.len();
    let depth = playout.depth_frames() + (queued * crate::FRAME_SAMPLES) as f64;
    let error = ((depth - TARGET_FRAMES) / FULL_TRIM_FRAMES).clamp(-1.0, 1.0);
    1.0 + MAX_TRIM * error
}

impl Refiller {
    /// Decode the next queued packet into `pending`; false when the queue
    /// is empty.
    fn refill(&mut self, pending: &mut VecDeque<f32>) -> bool {
        let Some(packet) = self.shared.lock().unwrap().queue.pop() else {
            return false;
        };
        match self.decoder.decode_float(&packet, &mut self.decoded, false) {
            Ok(samples) => pending.extend(&self.decoded[..samples * 2]),
            Err(e) if !self.decode_warned => {
                self.decode_warned = true;
                tracing::warn!(error = %e, "opus decode failed");
            }
            // A bad packet decodes as nothing; Opus conceals the hole.
            Err(_) => {}
        }
        true
    }
}

struct Running {
    mainloop: pw::main_loop::MainLoopRc,
    shared: Arc<Mutex<Shared>>,
    _stream: pw::stream::StreamRc,
    _listener: pw::stream::StreamListener<PlaybackState>,
}

impl Running {
    /// The raw main loop pointer for the owner's quit call; see
    /// [`crate::LoopHandle`].
    fn handle(&self) -> crate::LoopHandle {
        crate::LoopHandle(self.mainloop.as_raw_ptr())
    }

    fn run(self) {
        self.mainloop.run();
    }
}

fn open_stream() -> Result<Running> {
    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context
        .connect_rc(None)
        .map_err(|e| Error::ThreadEnded(format!("cannot connect to PipeWire: {e}")))?;

    let shared = Arc::new(Mutex::new(Shared {
        queue: PacketQueue::new(QUEUE_PACKETS),
        failed: None,
    }));
    let decoder = opus::Decoder::new(crate::RATE, opus::Channels::Stereo)?;

    let props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Playback",
        *pw::keys::MEDIA_ROLE => "Screen",
        *pw::keys::MEDIA_CLASS => "Stream/Output/Audio",
        *pw::keys::MEDIA_NAME => "gliff",
        *pw::keys::AUDIO_RATE => crate::RATE.to_string(),
        *pw::keys::AUDIO_CHANNELS => crate::CHANNELS.to_string(),
    };
    let stream = pw::stream::StreamRc::new(core, "gliff-audio", props)?;

    let listener = stream
        .add_local_listener_with_user_data(PlaybackState {
            playout: Playout::default(),
            refiller: Refiller {
                decoder,
                decoded: vec![0.0; crate::FRAME_INTERLEAVED],
                shared: shared.clone(),
                decode_warned: false,
            },
            failed: false,
        })
        .param_changed(|_stream, state, id, param| {
            let Some(param) = param else { return };
            if id != spa::param::ParamType::Format.as_raw() {
                return;
            }
            let mut info = spa::param::audio::AudioInfoRaw::new();
            if let Err(e) = check_format(&mut info, param) {
                state.failed = true;
                state.refiller.shared.lock().unwrap().failed = Some(e);
            }
        })
        .process(|stream, state| {
            if state.failed {
                return;
            }
            while let Some(mut buffer) = stream.dequeue_buffer() {
                let datas = buffer.datas_mut();
                if datas.is_empty() {
                    continue;
                }
                let stride = size_of::<f32>() * crate::CHANNELS as usize;
                let max = datas[0].data().map_or(0, |d| d.len());
                // PipeWire pre-sets the chunk size to what this cycle
                // wants: one quantum, not the whole buffer. Filling the
                // buffer's `maxsize` instead would hand the sink a whole
                // 256 ms at once and drain the queue in bursts.
                let want = match datas[0].chunk().size() as usize {
                    0 => max,
                    want => want.min(max) & !(stride - 1),
                };
                let step = trim_step(&state.playout, &state.refiller.shared);
                let mut written = 0usize;
                if let Some(data) = datas[0].data() {
                    while written + stride <= want {
                        let Some((l, r)) = state
                            .playout
                            .next_frame(step, |pending| state.refiller.refill(pending))
                        else {
                            break;
                        };
                        data[written..written + size_of::<f32>()].copy_from_slice(&l.to_le_bytes());
                        data[written + size_of::<f32>()..written + stride]
                            .copy_from_slice(&r.to_le_bytes());
                        written += stride;
                    }
                    // Past the last sample we had, the sink gets silence
                    // rather than whatever the buffer held.
                    data[written..want].fill(0);
                }
                let chunk = datas[0].chunk_mut();
                *chunk.offset_mut() = 0;
                *chunk.stride_mut() = stride as i32;
                *chunk.size_mut() = want as u32;
            }
        })
        .register()?;

    let format = format_param().map_err(|e| Error::ThreadEnded(format!("{e:#}")))?;
    let mut params = [pw::spa::pod::Pod::from_bytes(&format).ok_or(Error::FormatMismatch)?];
    stream.connect(
        spa::utils::Direction::Output,
        None,
        pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
        &mut params,
    )?;

    Ok(Running {
        mainloop,
        shared,
        _stream: stream,
        _listener: listener,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A playout holding `frames` frames of a ramp on the left channel and
    /// a constant right, refilled from nothing.
    fn ramp(frames: usize) -> Playout {
        let mut pending = VecDeque::new();
        for i in 0..frames {
            pending.push_back(i as f32);
            pending.push_back(-1.0);
        }
        Playout { pending, pos: 0.0 }
    }

    #[test]
    fn unit_step_plays_the_samples_in_order() {
        // Interpolation looks one frame ahead, so 4 frames make 3 outputs.
        let mut p = ramp(4);
        let out: Vec<_> = (0..3)
            .map(|_| p.next_frame(1.0, |_| false).unwrap())
            .collect();
        assert_eq!(out, vec![(0.0, -1.0), (1.0, -1.0), (2.0, -1.0)]);
        assert!(p.next_frame(1.0, |_| false).is_none());
    }

    #[test]
    fn a_faster_step_skips_and_a_slower_one_repeats_by_interpolation() {
        // Step 2: every other frame.
        let mut p = ramp(8);
        let fast: Vec<f32> = (0..3)
            .map(|_| p.next_frame(2.0, |_| false).unwrap().0)
            .collect();
        assert_eq!(fast, vec![0.0, 2.0, 4.0]);
        // Step 0.5: each frame in between, interpolated.
        let mut p = ramp(4);
        let slow: Vec<f32> = (0..5)
            .map(|_| p.next_frame(0.5, |_| false).unwrap().0)
            .collect();
        assert_eq!(slow, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn refill_is_asked_only_when_the_playout_runs_short() {
        let mut p = Playout::default();
        let mut packets = 2;
        let mut refills = 0;
        let mut refill = |pending: &mut VecDeque<f32>| {
            if packets == 0 {
                return false;
            }
            packets -= 1;
            refills += 1;
            pending.extend(std::iter::repeat_n(0.5, 2 * crate::FRAME_SAMPLES));
            true
        };
        let mut frames = 0;
        while p.next_frame(1.0, &mut refill).is_some() {
            frames += 1;
        }
        assert_eq!(refills, 2);
        assert_eq!(frames, 2 * crate::FRAME_SAMPLES - 1);
    }

    #[test]
    fn trim_is_bounded_and_points_the_right_way() {
        let shared = Mutex::new(Shared {
            queue: PacketQueue::new(QUEUE_PACKETS),
            failed: None,
        });
        // Empty: far below target, so slower but never more than MAX_TRIM.
        let low = trim_step(&Playout::default(), &shared);
        assert!((1.0 - MAX_TRIM - 1e-12..1.0).contains(&low));
        // Deep: far above target, so faster.
        let mut deep = Playout::default();
        deep.pending
            .extend(std::iter::repeat_n(0.0, 2 * 2 * TARGET_FRAMES as usize));
        let high = trim_step(&deep, &shared);
        assert!((1.0..=1.0 + MAX_TRIM + 1e-12).contains(&high));
    }
}
