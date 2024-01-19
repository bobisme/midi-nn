use std::{collections::HashMap, sync::Once};

use midly::{MetaMessage, MidiMessage, TrackEvent, TrackEventKind};

use crate::{
    midi::Midi,
    notes::{Beats, Note},
};

const VECTOR_SIZE: usize = 64;
const MIN_TEMPO: f32 = 1.0;
const MAX_TEMPO: f32 = 300.0;

pub fn parse_table(data: &[u8]) -> Vec<Vec<f32>> {
    ciborium::from_reader(data).unwrap()
}

static mut EMBED_TABLE: Vec<Vec<f32>> = vec![];
static INIT_EMBED_TABLE: Once = Once::new();

fn embed_table() -> &'static Vec<Vec<f32>> {
    unsafe {
        INIT_EMBED_TABLE.call_once(|| {
            const EMBED_TABLE_DATA: &[u8] = include_bytes!("../embed-table-900-64.cbor");
            EMBED_TABLE = parse_table(EMBED_TABLE_DATA);
        });
        &EMBED_TABLE
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
    Min,
    Low,
    LowMed,
    Med,
    MedHigh,
    High,
    Max,
}

impl From<u8> for Level {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Min,
            x if x <= 21 => Self::Low,
            x if x <= 42 => Self::LowMed,
            x if x <= 63 => Self::Med,
            x if x <= 84 => Self::MedHigh,
            x if x <= 105 => Self::High,
            _ => Self::Max,
        }
    }
}

/// Non-token parameters.
#[derive(Debug, Clone, Copy, Default)]
pub struct Params {
    /// Beats until this event happens (from previous event).
    /// Several notes with a delay of 0 means a chord.
    pub delay: f32,
    /// Velocity
    pub vel: f32,
    /// Tempo. -1 = not set, 0.0..1.0 => 1.0..300.0
    pub tempo: f32,
    /// Sustain. -1 = no change, 0.0 = off, 1.0 = on
    pub sustain: f32,
}

impl Params {
    pub fn to_array(self) -> [f32; 4] {
        [self.delay, self.vel, self.tempo, self.sustain]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Token {
    Unknown,
    Pad,
    Start,
    End,
    Control,
    NoteOn { note: u8 },
    NoteOff { note: u8 },
}

pub struct Vector([f32; VECTOR_SIZE]);

impl Token {
    const NOTE_ON_START: u32 = 5;
    const NOTE_ON_END: u32 = Self::NOTE_ON_START + 127;
    const NOTE_OFF_START: u32 = Self::NOTE_ON_END + 1;
    const NOTE_OFF_END: u32 = Self::NOTE_OFF_START + 127;
    const MAX_INDEX: u32 = Self::NOTE_OFF_END;

    pub const fn index(&self) -> u32 {
        match self {
            Token::Unknown => 0,
            Token::Pad => 1,
            Token::Start => 2,
            Token::End => 3,
            Token::Control => 4,
            Token::NoteOn { note } => Self::NOTE_ON_START + (*note as u32),
            Token::NoteOff { note } => Self::NOTE_OFF_START + (*note as u32),
        }
    }

    pub const fn from_index(index: u32) -> Self {
        match index {
            0 => Token::Unknown,
            1 => Token::Pad,
            2 => Self::Start,
            3 => Self::End,
            4 => Self::Control,
            Self::NOTE_ON_START..=Self::NOTE_ON_END => Self::NoteOn {
                note: (index - Self::NOTE_ON_START) as u8,
            },
            Self::NOTE_OFF_START..=Self::NOTE_OFF_END => Self::NoteOff {
                note: (index - Self::NOTE_OFF_START) as u8,
            },
            _ => Self::Unknown,
        }
    }

    // Maps self.index() to a Vector embedding.
    pub fn embed(&self) -> Vector {
        let mut embedding = [0.0; VECTOR_SIZE];
        embedding[self.index() as usize % VECTOR_SIZE] = 1.0;
        Vector(embedding)
    }
}

// simple compile-time check.
#[allow(clippy::assertions_on_constants)]
const _: () = {
    assert!(Token::NOTE_ON_START == 5);
    assert!(Token::NOTE_ON_END == 132);
    assert!(Token::NOTE_OFF_START == 133);
    assert!(Token::NOTE_OFF_END == 260);
};

pub fn build_rev_map() -> HashMap<u32, Token> {
    let mut rev_map = HashMap::new();
    for i in 0..=Token::MAX_INDEX {
        let token = Token::from_index(i);
        rev_map.insert(i, token);
    }
    rev_map
}

pub fn from_note(note: &Note, delta: &Beats) -> (Token, Params) {
    let token = Token::NoteOn { note: note.note() };
    let params = Params {
        delay: (*delta).into(),
        vel: note.vel() as f32 / 127.0,
        tempo: todo!(),
        sustain: todo!(),
    };
    (token, params)
}
pub fn ticks_per_beat(tempo_bpm: f32, timing: &midly::Timing) -> f32 {
    match timing {
        // Metrical is ticks-per-beat
        midly::Timing::Metrical(x) => x.as_int() as f32,
        // Timecode means tick = 1 / fps / subframe
        &midly::Timing::Timecode(fps, sub) => (fps.as_f32() * sub as f32) * (60.0 / tempo_bpm),
    }
}

pub fn bpm_to_param(beats_per_minute: f32) -> f32 {
    (beats_per_minute - MIN_TEMPO) / MAX_TEMPO
}

pub fn param_to_bpm(param: f32) -> f32 {
    param * MAX_TEMPO + MIN_TEMPO
}

pub fn from_event(
    ev: &TrackEvent,
    tempo_bpm: f32,
    timing: &midly::Timing,
    cumulative_delta: u32,
) -> Option<(Token, Params)> {
    let tpb = ticks_per_beat(tempo_bpm, timing);
    let delta = ev.delta.as_int() + cumulative_delta;
    match ev.kind {
        TrackEventKind::Midi {
            channel: _,
            message,
        } => match message {
            MidiMessage::NoteOff { key, vel } => Some((
                Token::NoteOff { note: key.into() },
                Params {
                    delay: delta as f32 / tpb,
                    vel: vel.as_int() as f32 / 127.0,
                    tempo: -1.0,
                    sustain: -1.0,
                },
            )),
            MidiMessage::NoteOn { key, vel } => Some((
                Token::NoteOn { note: key.into() },
                Params {
                    delay: delta as f32 / tpb,
                    vel: vel.as_int() as f32 / 127.0,
                    tempo: -1.0,
                    sustain: -1.0,
                },
            )),
            MidiMessage::Controller { controller, value } if controller.as_int() == 64 => Some((
                Token::Control,
                Params {
                    delay: delta as f32 / tpb,
                    vel: -1.0,
                    tempo: -1.0,
                    sustain: value.as_int() as f32 / 127.0,
                },
            )),
            _ => None,
        },
        TrackEventKind::Meta(MetaMessage::Tempo(us_per_beat)) => {
            let beats_per_minute = 60_000_000 as f32 / us_per_beat.as_int() as f32;
            Some((
                Token::Control,
                Params {
                    delay: delta as f32 / tpb,
                    vel: -1.0,
                    tempo: (beats_per_minute - MIN_TEMPO) / MAX_TEMPO,
                    sustain: -1.0,
                },
            ))
        }
        TrackEventKind::SysEx(_) => None,
        TrackEventKind::Escape(_) => None,
        TrackEventKind::Meta(_) => None,
    }
}

pub fn embed_note(note: &Note, delta: &Beats) -> Vec<f32> {
    let (tok, params) = from_note(note, delta);
    let table = embed_table();
    let mut emb = table[tok.index() as usize].to_vec();
    emb.extend(params.to_array());
    emb
}

pub struct TokenIter<Iter> {
    iter: Iter,
}

impl<Iter> Iterator for TokenIter<Iter>
where
    Iter: Iterator<Item = (Token, Beats)>,
{
    type Item = Vec<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        let (token, beats) = self.iter.next()?;
        let table = embed_table();
        let mut emb = table[token.index() as usize].to_vec();
        emb.push(beats.into());
        Some(emb)
    }
}

pub struct MidiTokenIter<'m> {
    started: bool,
    ended: bool,
    idx: usize,
    midi: &'m Midi<'m>,
}

impl<'m> MidiTokenIter<'m> {
    pub fn embedded(self) -> TokenIter<Self> {
        TokenIter { iter: self }
    }
}

impl<'m> Iterator for MidiTokenIter<'m> {
    type Item = (Token, Params);

    fn next(&mut self) -> Option<Self::Item> {
        if !self.started {
            self.started = true;
            return Some((Token::Start, Params::default()));
        }
        let notes = self.midi.notes();
        let deltas = &self.midi.deltas;
        if self.started && self.idx < notes.len() {
            let i = self.idx;
            self.idx += 1;
            return Some(from_note(&notes[i], &deltas[i]));
        }
        if !self.ended {
            self.ended = true;
            return Some((Token::End, Params::default()));
        }
        None
    }
}

pub fn tokenize_midi<'m>(midi: &'m Midi) -> MidiTokenIter<'m> {
    MidiTokenIter {
        started: false,
        ended: false,
        idx: 0,
        midi,
    }
}

pub fn embed_midi<'m>(midi: &'m Midi) -> TokenIter<MidiTokenIter<'m>> {
    let inner = MidiTokenIter {
        started: false,
        ended: false,
        idx: 0,
        midi,
    };
    inner.embedded()
}
