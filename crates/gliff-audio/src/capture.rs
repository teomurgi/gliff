//! Capture of the default sink's monitor: what the remote machine plays,
//! encoded as Opus frames for the session to send.

use std::sync::{Arc, Mutex};

use pipewire as pw;
use pw::properties::properties;
use pw::spa;

use crate::{check_format, format_param, init, Error, PacketQueue, Result, FRAME_INTERLEAVED};

/// Encoded packets held for the session loop: one second of audio. A
/// fuller queue means the socket write is stalled, and the oldest is
/// dropped so the sound the client eventually plays is not seconds late.
const QUEUE_PACKETS: usize = 50;
/// The most one `drain` hands over, so one loop wakeup cannot dump a full
/// second of audio into the writer queue.
const DRAIN_MAX: usize = 10;
/// A full sink of Opus frames at the 128 kbit/s ceiling, with room for
/// silence bursts (DTX makes them smaller, never larger).
const MAX_PACKET: usize = 4096;

/// What the PipeWire thread shares with the owner.
struct Shared {
    queue: PacketQueue,
    /// Set by the param or process callbacks when the stream cannot be
    /// used; the session logs it on the next drain and stops asking.
    failed: Option<Error>,
}

/// One capture source per session; owns a thread running a PipeWire main
/// loop. Dropping stops the thread.
pub struct Capture {
    shared: Arc<Mutex<Shared>>,
    mainloop: crate::LoopHandle,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Capture {
    /// Spawn the PipeWire capture thread. Fails when PipeWire cannot be
    /// reached (no daemon, no device), so the session continues video-only.
    pub fn start() -> Result<Capture> {
        init();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("gliff-audio-capture".into())
            .spawn(move || run_capture(ready_tx))
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

    /// Take the next batch of encoded 20 ms Opus frames (usually 0-2).
    /// Must be called promptly (every loop wakeup); the internal queue
    /// holds about a second and drops the oldest packet past that.
    pub fn drain(&mut self, dst: &mut Vec<Vec<u8>>) {
        let mut shared = self.shared.lock().unwrap();
        if let Some(e) = shared.failed.take() {
            tracing::warn!(error = %e, "audio capture stopped");
        }
        shared.queue.drain_into(dst, DRAIN_MAX);
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        crate::quit_loop(&self.mainloop);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The PipeWire thread: connect, stream the sink monitor into the encoder,
/// and run the loop until quit. Setup errors go back to `start`; anything
/// later is flagged in the shared state for the next drain.
fn run_capture(ready: std::sync::mpsc::Sender<Result<(crate::LoopHandle, Arc<Mutex<Shared>>)>>) {
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

/// Everything the process callback needs, moved into the listener's user
/// data.
struct CaptureState {
    encoder: opus::Encoder,
    /// Interleaved samples accumulated toward the next 20 ms frame.
    pending: Vec<f32>,
    shared: Arc<Mutex<Shared>>,
    /// Log the first encode failure only; a broken encoder reports every
    /// frame otherwise.
    encode_warned: bool,
    /// The negotiated format is not the one we asked for. Encoding what the
    /// graph sends instead would put garbage on the wire, so the process
    /// callback stops; the session reads the failure from `shared` on its
    /// next drain.
    failed: bool,
}

/// The main loop plus its stream, wired up and connected.
struct Running {
    mainloop: pw::main_loop::MainLoopRc,
    shared: Arc<Mutex<Shared>>,
    // The stream (which owns its core) and its listener are held so their
    // Drop runs after the loop stops; they are never touched through here.
    _stream: pw::stream::StreamRc,
    _listener: pw::stream::StreamListener<CaptureState>,
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
    let mut encoder = opus::Encoder::new(
        crate::RATE,
        opus::Channels::Stereo,
        opus::Application::Audio,
    )?;
    encoder.set_bitrate(opus::Bitrate::Bits(128_000))?;

    // The sink-monitor hint: PipeWire matches this stream to the monitor
    // of the default sink, so we hear what the machine plays.
    let props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Screen",
        *pw::keys::MEDIA_CLASS => "Stream/Input/Audio",
        *pw::keys::STREAM_CAPTURE_SINK => "true",
        *pw::keys::MEDIA_NAME => "gliff",
        *pw::keys::AUDIO_RATE => crate::RATE.to_string(),
        *pw::keys::AUDIO_CHANNELS => crate::CHANNELS.to_string(),
    };
    let stream = pw::stream::StreamRc::new(core, "gliff-audio", props)?;

    let listener = stream
        .add_local_listener_with_user_data(CaptureState {
            encoder,
            pending: Vec::with_capacity(FRAME_INTERLEAVED * 2),
            shared: shared.clone(),
            encode_warned: false,
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
                state.shared.lock().unwrap().failed = Some(e);
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
                let chunk = datas[0].chunk();
                let (offset, size) = (chunk.offset() as usize, chunk.size() as usize);
                let Some(data) = datas[0].data() else {
                    continue;
                };
                let Some(bytes) = data.get(offset..offset + size) else {
                    continue;
                };
                // F32LE interleaved stereo, so the byte count is a whole
                // number of f32s; a partial trailing sample is dropped.
                state.pending.reserve(bytes.len() / size_of::<f32>());
                for b in bytes.chunks_exact(size_of::<f32>()) {
                    state
                        .pending
                        .push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                }
                while state.pending.len() >= FRAME_INTERLEAVED {
                    let frame: Vec<f32> = state.pending.drain(..FRAME_INTERLEAVED).collect();
                    let mut out = vec![0u8; MAX_PACKET];
                    match state.encoder.encode_float(&frame, &mut out) {
                        Ok(len) => {
                            out.truncate(len);
                            let dropped = state
                                .shared
                                .lock()
                                .unwrap()
                                .queue
                                .push(std::mem::take(&mut out));
                            if let Some(dropped) = dropped {
                                tracing::warn!(dropped, "audio capture queue full; dropped oldest");
                            }
                        }
                        Err(e) if !state.encode_warned => {
                            state.encode_warned = true;
                            tracing::warn!(error = %e, "opus encode failed");
                        }
                        Err(_) => {}
                    }
                }
            }
        })
        .register()?;

    let format = format_param().map_err(|e| Error::ThreadEnded(format!("{e:#}")))?;
    let mut params = [pw::spa::pod::Pod::from_bytes(&format).ok_or(Error::FormatMismatch)?];
    stream.connect(
        spa::utils::Direction::Input,
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
