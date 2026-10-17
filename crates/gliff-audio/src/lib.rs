//! Audio for a gliff session: PipeWire capture of the default sink's
//! monitor on the server, Opus on the wire, playback into the local default
//! sink on the client.
//!
//! Both ends own one thread running a PipeWire main loop; the owning
//! structs (`Capture`, `Playback`) are handles whose `Drop` stops the
//! thread. The PipeWire objects are the plain (non-`Send`) variants,
//! confined to their thread; only packet queues and a weak main loop
//! reference cross the boundary. Everything fails soft: a machine without
//! PipeWire, an audio device or the opus library gets a video-only session
//! with one warning, because `start` returns an error its caller logs and
//! ignores.
//!
//! The format is fixed: 48 kHz stereo F32LE, Opus at 128 kbit/s in 20 ms
//! frames (960 samples per channel). Each side advertises exactly one
//! format, so the graph adapts to us and no resampling is needed; a
//! negotiation that lands on anything else stops the stream instead of
//! sending or playing garbage.

mod capture;
mod playback;

pub use capture::Capture;
pub use playback::Playback;

use anyhow::Context;

/// A raw `pw_main_loop` pointer that may cross to the owner's thread for
/// exactly one call: `pw_main_loop_quit`, which PipeWire documents as
/// allowed from any thread (it sets the loop's flag and wakes its poll
/// through the internal eventfd). The weak `Rc` reference the crate offers
/// is not `Send`, and closing the loop's poll fd leaves `run` spinning on
/// EBADF instead of returning, so this is the smallest working stop.
pub(crate) struct LoopHandle(pub(crate) *mut pipewire::sys::pw_main_loop);

// SAFETY: the pointer is used only for `pw_main_loop_quit` from the
// owner's Drop, while the audio thread keeps the loop alive; the Drop
// then joins that thread before the loop could be destroyed.
unsafe impl Send for LoopHandle {}

pub(crate) fn quit_loop(handle: &LoopHandle) {
    // SAFETY: `handle` points at the audio thread's live main loop (the
    // thread was set up before the handle was handed out, and is joined
    // right after this call), and pw_main_loop_quit is thread-safe.
    unsafe { pipewire::sys::pw_main_loop_quit(handle.0) };
}

/// The one negotiated format, both directions.
pub(crate) const RATE: u32 = 48_000;
pub(crate) const CHANNELS: u32 = 2;
/// One Opus frame: 20 ms at 48 kHz, per channel.
pub(crate) const FRAME_SAMPLES: usize = 960;
/// Samples per Opus frame, both channels interleaved.
pub(crate) const FRAME_INTERLEAVED: usize = FRAME_SAMPLES * CHANNELS as usize;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("PipeWire: {0}")]
    PipeWire(#[from] pipewire::Error),
    #[error("opus: {0}")]
    Opus(#[from] opus::Error),
    #[error("the negotiated format is not 48 kHz stereo F32LE")]
    FormatMismatch,
    #[error("the audio thread ended: {0}")]
    ThreadEnded(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// `pipewire::init` is process-wide; the crate's `OnceLock` inside the
/// function makes it idempotent, so there is nothing to guard here beyond
/// calling it before any other PipeWire use.
pub(crate) fn init() {
    pipewire::init();
}

/// Build the single EnumFormat param both streams advertise.
pub(crate) fn format_param() -> anyhow::Result<Vec<u8>> {
    use libspa::param::audio::{AudioFormat, AudioInfoRaw};
    use libspa::pod;
    let mut info = AudioInfoRaw::new();
    info.set_format(AudioFormat::F32LE);
    info.set_rate(RATE);
    info.set_channels(CHANNELS);
    let bytes = pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pod::Value::Object(pod::Object {
            type_: libspa::sys::SPA_TYPE_OBJECT_Format,
            id: libspa::sys::SPA_PARAM_EnumFormat,
            properties: info.into(),
        }),
    )
    .context("serialise the audio format param")?
    .0
    .into_inner();
    Ok(bytes)
}

/// The negotiated format, checked in `param_changed`: anything but 48 kHz
/// stereo F32LE means the peer side of the graph will not adapt to us, and
/// we would rather run silent than send or play garbage.
pub(crate) fn check_format(
    info: &mut libspa::param::audio::AudioInfoRaw,
    param: &libspa::pod::Pod,
) -> Result<()> {
    use libspa::param::audio::AudioFormat;
    info.parse(param).map_err(|_| Error::FormatMismatch)?;
    if info.format() == AudioFormat::F32LE && info.rate() == RATE && info.channels() == CHANNELS {
        Ok(())
    } else {
        Err(Error::FormatMismatch)
    }
}

/// A bounded queue of Opus packets crossing to or from an audio thread.
/// Full means the consumer is behind (a stalled socket write on the capture
/// side, a stalled PipeWire loop on the playback side); dropping the oldest
/// keeps latency bounded and Opus conceals the gap.
pub(crate) struct PacketQueue {
    packets: std::collections::VecDeque<Vec<u8>>,
    cap: usize,
    /// Packets dropped for a full queue; reported once, on the next push.
    pub(crate) dropped: u64,
}

impl PacketQueue {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            packets: std::collections::VecDeque::new(),
            cap,
            dropped: 0,
        }
    }

    /// Append `packet`, dropping the oldest when full. Returns the total
    /// dropped so far when the push dropped any, so the caller logs once
    /// per burst instead of per packet.
    pub(crate) fn push(&mut self, packet: Vec<u8>) -> Option<u64> {
        let dropped = if self.packets.len() >= self.cap {
            self.packets.pop_front();
            self.dropped += 1;
            Some(self.dropped)
        } else {
            None
        };
        self.packets.push_back(packet);
        dropped
    }

    pub(crate) fn pop(&mut self) -> Option<Vec<u8>> {
        self.packets.pop_front()
    }

    pub(crate) fn len(&self) -> usize {
        self.packets.len()
    }

    /// Move up to `max` packets into `dst`.
    pub(crate) fn drain_into(&mut self, dst: &mut Vec<Vec<u8>>, max: usize) {
        let n = self.packets.len().min(max);
        dst.extend(self.packets.drain(..n));
    }
}

/// The wire framing of one `ServerMsg::Audio` payload: each packet as a
/// u16 LE length followed by its bytes, concatenated. Kept here rather than
/// in gliff-proto because it is an audio-local encoding, not a CBOR one.
pub mod framing {
    /// Serialise Opus packets into one payload buffer.
    pub fn pack(packets: &[Vec<u8>], dst: &mut Vec<u8>) {
        dst.clear();
        for p in packets {
            dst.extend_from_slice(&(p.len() as u16).to_le_bytes());
            dst.extend_from_slice(p);
        }
    }

    /// Split a payload back into packets; the declared count and total
    /// length must match exactly, or the sender and we disagree on the
    /// framing and the session is broken.
    pub fn unpack(data: &[u8], packets: u8) -> anyhow::Result<Vec<Vec<u8>>> {
        let mut out = Vec::with_capacity(packets as usize);
        let mut rest = data;
        for _ in 0..packets {
            anyhow::ensure!(
                rest.len() >= 2,
                "audio payload shorter than its packet count"
            );
            let len = u16::from_le_bytes([rest[0], rest[1]]) as usize;
            rest = &rest[2..];
            anyhow::ensure!(rest.len() >= len, "audio packet overruns the payload");
            out.push(rest[..len].to_vec());
            rest = &rest[len..];
        }
        anyhow::ensure!(rest.is_empty(), "audio payload has trailing bytes");
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_queue_drops_oldest_and_caps_drain() {
        let mut q = PacketQueue::new(2);
        assert!(q.push(vec![1]).is_none());
        assert!(q.push(vec![2]).is_none());
        assert_eq!(q.push(vec![3]), Some(1));
        assert_eq!(q.push(vec![4]), Some(2));
        let mut dst = Vec::new();
        q.drain_into(&mut dst, 1);
        assert_eq!(dst, vec![vec![3]]);
        assert_eq!(q.len(), 1);
        q.drain_into(&mut dst, 10);
        assert_eq!(dst, vec![vec![3], vec![4]]);
        assert_eq!(q.len(), 0);
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn framing_round_trips_and_rejects_bad_lengths() {
        let packets = vec![vec![1, 2, 3], vec![], vec![4; 700]];
        let mut buf = Vec::new();
        framing::pack(&packets, &mut buf);
        assert_eq!(buf.len(), 3 + 700 + 3 * 2);
        assert_eq!(framing::unpack(&buf, 3).unwrap(), packets);
        assert!(framing::unpack(&buf, 2).is_err());
        assert!(framing::unpack(&buf, 4).is_err());
        assert!(framing::unpack(&buf[..buf.len() - 1], 3).is_err());
        assert!(framing::unpack(&[], 0).unwrap().is_empty());
    }
}
