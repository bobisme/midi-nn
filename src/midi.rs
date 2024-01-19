use std::cell::RefCell;

use color_eyre::{
    eyre::{ensure, eyre},
    Result,
};
use midly::{
    num::{u4, u7},
    MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind,
};
use num::Zero;

use crate::{
    notes::{Beats, Note},
    sequence::Event,
};

#[derive(Debug)]
pub struct Midi<'a> {
    smf: Smf<'a>,
    pub notes: Vec<Note>,
    pub deltas: Vec<Beats>,
}

impl<'a> Midi<'a> {
    pub fn notes(&self) -> &Vec<Note> {
        &self.notes
    }
}

fn calc_ticks_per_beat(timing: &Timing, tempo: f32) -> f64 {
    match *timing {
        midly::Timing::Metrical(x) => x.as_int() as f64,
        midly::Timing::Timecode(f, x) => f.as_f32() as f64 * x as f64 / (tempo as f64 / 60.0),
    }
}

fn find_next_off_delta<'a, I>(mut iter: I, key: u7, _channel: Option<u4>) -> u32
where
    I: Iterator<Item = &'a TrackEvent<'a>>,
{
    use MidiMessage::*;

    // let (end_i, _) = iter
    let next_off =
        iter.by_ref()
            .enumerate()
            .find(|(_, TrackEvent { delta: _, kind })| match kind {
                TrackEventKind::Midi {
                    channel: _,
                    message,
                } => match *message {
                    // NoteOn { key: k, vel: _ } if k == key => true,
                    NoteOff { key: k, vel: _ } if k == key => true,
                    _ => false,
                },
                _ => false,
            });
    let Some((end_i, _)) = next_off else {
        return 0;
    };
    let delta = iter
        .by_ref()
        .take(end_i + 1)
        .map(|x| x.delta.as_int())
        .sum();
    if delta <= 1 {
        println!("WARNING: 0-len note: {:?}", key);
    }
    delta
}

pub fn parse(data: &[u8]) -> Result<Midi> {
    let smf = Smf::parse(data)?;
    ensure!(!smf.tracks.len().is_zero(), "no tracks found!");
    // for (i, track) in smf.tracks.iter().enumerate() {
    //     println!("track: {}, events: {}", i, track.len());
    // }
    // get first track with enough events
    let mut track = smf.tracks.get(0).unwrap();
    for i in 1..16 {
        if track.len() > 100 {
            println!("using track {}", i - 1);
            break;
        }
        track = smf.tracks.get(i).unwrap();
    }
    let mut notes = Vec::<Note>::new();
    let mut deltas = Vec::<Beats>::new();
    let tempo = RefCell::new(25.0);
    let ticks_per_beat = || {
        let out = match smf.header.timing {
            midly::Timing::Metrical(x) => x.as_int() as f32,
            midly::Timing::Timecode(f, x) => f.as_f32() * x as f32 / (*tempo.borrow() / 60.0),
        } as f64;
        out
    };
    // let mut ticks_per_beat = match smf.header.timing {
    //     midly::Timing::Metrical(x) => x.as_int() as f32,
    //     midly::Timing::Timecode(f, x) => f.as_f32() * x as f32 / (tempo / 60.0),
    // } as f64;
    let mut cumulative_delta = 0;
    for (i, event) in track.iter().enumerate() {
        cumulative_delta += event.delta.as_int();
        match event.kind {
            TrackEventKind::Midi { channel, message } => {
                use MidiMessage::*;
                match message {
                    NoteOff { key: _, vel: _ } => {
                        // println!("got note off: key={} vel={}", key, vel);
                    }
                    NoteOn { key: _, vel } if vel == 0 => {}
                    NoteOn { key, vel } => {
                        let off = find_next_off_delta(track.iter().skip(i), key, Some(channel));
                        let beats = off as f64 / ticks_per_beat();
                        // if beats <= 0.001 {
                        //     continue;
                        // }
                        notes.push(
                            Note::from(key.as_int())
                                .with_vel(vel.as_int())
                                .with_beats(Beats::from(beats)),
                        );
                        let delta = Beats::from(cumulative_delta as f64 / ticks_per_beat());
                        deltas.push(delta);
                        cumulative_delta = 0;
                    }
                    Aftertouch { key: _, vel: _ } => {}
                    Controller {
                        controller: _,
                        value: _,
                    } => {}
                    ProgramChange { program: _ } => {}
                    ChannelAftertouch { vel: _ } => {}
                    PitchBend { bend: _ } => {}
                }
            }
            TrackEventKind::SysEx(_) => {}
            TrackEventKind::Escape(_) => {}
            TrackEventKind::Meta(MetaMessage::Tempo(t)) => {
                tempo.replace(t.as_int() as f32);
                // ticks_per_beat = match smf.header.timing {
                //     midly::Timing::Metrical(x) => x.as_int() as f32,
                //     midly::Timing::Timecode(f, x) => f.as_f32() * x as f32 / (tempo / 60.0),
                // } as f64;
            }
            TrackEventKind::Meta(_) => {}
        }
    }
    Ok(Midi { smf, notes, deltas })
}

pub struct MidiSequence {
    pub events: Vec<Event>,
    ticks_per_beat: u32,
}

impl MidiSequence {
    /// Get a reference to the midi sequence's ticks per beat.
    pub fn ticks_per_beat(&self) -> u32 {
        self.ticks_per_beat
    }
}

impl IntoIterator for MidiSequence {
    type Item = Event;

    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.events.into_iter()
    }
}

pub struct Parser {
    tempo: RefCell<f32>,
    ticks_per_beat: u32,
}

impl Default for Parser {
    fn default() -> Self {
        Self {
            tempo: RefCell::new(120.0),
            ticks_per_beat: 12,
        }
    }
}

impl Parser {
    pub fn with_ticks_per_beat(&self, t: u32) -> Self {
        Self {
            ticks_per_beat: t,
            tempo: RefCell::new(*self.tempo.borrow()),
        }
    }

    fn ticks_from_delta(&self, delta: impl Into<u32>, timing: &Timing) -> u32 {
        let d = delta.into() as f64;
        let tpb = calc_ticks_per_beat(timing, *self.tempo.borrow());
        let beats = d / tpb;

        // println!("parsed {} midi ticks as {} player ticks", d, out);
        (beats * self.ticks_per_beat as f64).round() as u32
    }

    pub fn parse_seq(&self, data: &[u8], track_i: usize) -> Result<MidiSequence> {
        let smf = Smf::parse(data)?;
        let track = smf
            .tracks
            .get(track_i)
            .ok_or_else(|| eyre!("could not get track {}", track_i))?;

        let tempo = RefCell::new(50.0);

        let things = track
            .iter()
            .enumerate()
            .map(|(i, ev)| match ev.kind {
                TrackEventKind::Meta(MetaMessage::Tempo(t)) => {
                    tempo.replace(t.as_int() as f32);
                    (ev.delta, None)
                }
                TrackEventKind::Midi {
                    channel: ch,
                    message,
                } => {
                    use MidiMessage::*;
                    match message {
                        // NoteOff { key, vel: _ } => (ev.delta, Some(Event::stop(key))),
                        // NoteOn { key, vel } if vel == 0 => (ev.delta, Some(Event::stop(key))),
                        NoteOff { key: _, vel: _ } => (ev.delta, None),
                        NoteOn { key: _, vel } if vel == 0 => (ev.delta, None),
                        NoteOn { key, vel } => {
                            let off = find_next_off_delta(track.iter().skip(i + 1), key, Some(ch));
                            match off {
                                0 => (ev.delta, Some(Event::play(key, vel))),
                                _ => {
                                    let ticks = self.ticks_from_delta(off, &smf.header.timing);
                                    // let ticks = match ticks {
                                    //     0 => 1,
                                    //     _ => ticks,
                                    // };
                                    (ev.delta, Some(Event::play_ticks(key, vel, ticks)))
                                }
                            }
                        }
                        _ => (ev.delta, None),
                    }
                }
                _ => (ev.delta, None),
            })
            // compress sequential waits
            .scan(0u32, |state, (delta, event)| {
                *state += delta.as_int();
                match event {
                    None => Some([None, None]),
                    Some(_ev) => {
                        let d = *state as f64;
                        let tpb = calc_ticks_per_beat(&smf.header.timing, *tempo.borrow());
                        let beats = d / tpb;
                        let ticks = (beats * self.ticks_per_beat as f64).round() as u32;
                        *state = 0;
                        Some([Some(Event::wait(ticks)), event])
                    }
                }
            })
            .flatten()
            .filter_map(|e| match e {
                Some(Event::Wait { ticks }) if ticks == 0 => None,
                _ => e,
            });

        Ok(MidiSequence {
            events: things.collect(),
            ticks_per_beat: self.ticks_per_beat,
        })
    }
}

