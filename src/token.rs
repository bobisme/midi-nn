use std::{collections::HashMap, sync::Once};

use color_eyre::Result;

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
            x if x <= 25 => Self::Low,
            x if x <= 50 => Self::LowMed,
            x if x <= 75 => Self::Med,
            x if x <= 100 => Self::MedHigh,
            x if x < 127 => Self::High,
            _ => Self::Max,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Token {
    Unknown,
    Pad,
    Start,
    End,
    Note { note: u8, vel: Level },
}

pub struct Vector([f32; VECTOR_SIZE]);

impl Token {
    pub fn index(&self) -> u32 {
        match self {
            Token::Unknown => 0,
            Token::Pad => 1,
            Token::Start => 2,
            Token::End => 3,
            Token::Note { note, vel } => *note as u32 * 7 + *vel as u32 + 3,
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
    for i in 0..(128 * 7 + 4) {
        let token = match i {
            0 => Token::Unknown,
            1 => Token::Pad,
            2 => Token::Start,
            3 => Token::End,
            _ => {
                let note = ((i - 3) / 7) as u8;
                let vel = match (i - 3) % 7 {
                    0 => Level::Min,
                    1 => Level::Low,
                    2 => Level::LowMed,
                    3 => Level::Med,
                    4 => Level::MedHigh,
                    5 => Level::High,
                    6 => Level::Max,
                    _ => Level::Min,
                };
                Token::Note { note, vel }
            }
        };
        rev_map.insert(i, token);
    }
    rev_map
}

pub fn from_note(note: &Note) -> (Token, Beats) {
    let level: Level = note.vel().into();
    let token = Token::Note {
        note: note.note(),
        vel: level,
    };
    (token, note.beats())
}

pub fn embed_note(note: &Note) -> Vec<f64> {
    let (tok, beats) = from_note(note);
    let table = embed_table();
    let mut emb = table[tok.index() as usize].to_vec();
    emb.push(beats.into());
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
    type Item = (Token, Beats);

    fn next(&mut self) -> Option<Self::Item> {
        if !self.started {
            self.started = true;
            return Some((Token::Start, Beats::zero()));
        }
        let notes = self.midi.notes();
        if self.started && self.idx < notes.len() {
            self.idx += 1;
            return Some(from_note(&notes[self.idx - 1]));
        }
        if !self.ended {
            self.ended = true;
            return Some((Token::End, Beats::zero()));
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
