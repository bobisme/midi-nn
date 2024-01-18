use std::{collections::HashMap, sync::Once};



use crate::{
    midi::Midi,
    notes::{Beats, Note},
};

const TOKEN_COUNT: usize = 900;
const VECTOR_SIZE: usize = 64;

pub fn parse_table(data: &[u8]) -> Vec<Vec<f64>> {
    ciborium::from_reader(data).unwrap()
}

static mut EMBED_TABLE: Vec<Vec<f64>> = vec![];
static INIT_EMBED_TABLE: Once = Once::new();

fn embed_table() -> &'static Vec<Vec<f64>> {
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
    /// Beats this notes plays for.
    pub duration: f64,
    /// Beats until this note starts (from previous note).
    /// Several notes with a delay of 0 means a chord.
    pub delay: f64,
    /// Velocity
    pub vel: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Token {
    Unknown,
    Pad,
    Start,
    End,
    Note { note: u8 },
}

pub struct Vector([f32; VECTOR_SIZE]);

impl Token {
    pub fn index(&self) -> u32 {
        match self {
            Token::Unknown => 0,
            Token::Pad => 1,
            Token::Start => 2,
            Token::End => 3,
            Token::Note { note } => *note as u32,
        }
    }

    // Maps self.index() to a Vector embedding.
    pub fn embed(&self) -> Vector {
        let mut embedding = [0.0; VECTOR_SIZE];
        embedding[self.index() as usize % VECTOR_SIZE] = 1.0;
        Vector(embedding)
    }
}

pub fn build_rev_map() -> HashMap<u32, Token> {
    let mut rev_map = HashMap::new();
    for i in 0..128 {
        let token = match i {
            0 => Token::Unknown,
            1 => Token::Pad,
            2 => Token::Start,
            3 => Token::End,
            _ => Token::Note { note: i as u8 },
        };
        rev_map.insert(i, token);
    }
    rev_map
}

pub fn from_note(note: &Note, delta: &Beats) -> (Token, Params) {
    let token = Token::Note { note: note.note() };
    let params = Params {
        duration: note.beats().into(),
        delay: (*delta).into(),
        vel: note.vel() as f64 / 127.0,
    };
    (token, params)
}

pub fn embed_note(note: &Note, delta: &Beats) -> Vec<f64> {
    let (tok, params) = from_note(note, delta);
    let table = embed_table();
    let mut emb = table[tok.index() as usize].to_vec();
    emb.push(params.duration);
    emb.push(params.delay);
    emb
}

pub struct TokenIter<Iter> {
    iter: Iter,
}

impl<Iter> Iterator for TokenIter<Iter>
where
    Iter: Iterator<Item = (Token, Beats)>,
{
    type Item = Vec<f64>;

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
