use std::{
    cmp,
    convert::{TryFrom},
    fmt::Display,
    time::Duration,
};

use color_eyre::eyre::{bail, ensure, eyre};
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{digit1, one_of},
    combinator::{map_res, opt, recognize},
    multi::many0,
    sequence::{preceded, separated_pair, tuple},
    IResult,
};

use crate::error::{Result};

#[derive(Clone, Copy, Debug, Default)]
pub struct Beats {
    inner: f64,
}

impl Display for Beats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl Beats {
    pub fn zero() -> Self {
        Self { inner: 0.0 }
    }

    pub fn from_millis(&self, millis_per_beat: impl Into<f64>) -> Duration {
        Duration::from_millis((self.inner * millis_per_beat.into()).trunc() as u64)
    }

    pub fn from_bpm(&self, tempo: impl Into<f64>) -> Duration {
        self.from_millis(60_000.0 / tempo.into())
    }
}

// impl PartialEq for Beats {
//     fn eq(&self, other: &Self) -> bool {
//         (self.inner - other.inner).abs() < 0.001
//     }
// }
impl Eq for Beats {}

impl std::hash::Hash for Beats {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        ((self.inner * 1_000_000f64) as u64).hash(state);
    }
}

impl<T: Into<f64> + Copy> PartialEq<T> for Beats {
    fn eq(&self, other: &T) -> bool {
        (self.inner - (*other).into()).abs() < 0.001
    }
}

impl PartialEq<Beats> for f64 {
    fn eq(&self, other: &Beats) -> bool {
        *other == *self
    }
}

impl PartialEq<Beats> for i64 {
    fn eq(&self, other: &Beats) -> bool {
        *other == *self as f64
    }
}

// impl PartialEq<i64> for Beats {
//     fn eq(&self, other: &i64) -> bool {
//         self.inner == *other as f64
//     }
// }

impl From<f64> for Beats {
    fn from(x: f64) -> Self {
        Beats { inner: x }
    }
}

impl From<i64> for Beats {
    fn from(x: i64) -> Self {
        Beats { inner: x as f64 }
    }
}

impl From<Beats> for f64 {
    fn from(b: Beats) -> Self {
        b.inner
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Note {
    note: u8,
    vel: u8,
    // nanobeats
    beats: Beats,
}

impl Note {
    pub fn transpose(&self, semis: i8) -> Self {
        Self {
            note: cmp::max(self.note as i8 + semis, 0) as u8,
            vel: self.vel,
            beats: self.beats,
        }
    }
}

// impl std::slice::Join<&str> for Note {
//     type Output = String;

//     fn join(slice: &Self, sep: &str) -> Self::Output {
//         slice
//             .iter()
//             .map(|x| x.to_string())
//             .collect::<Vec<String>>()
//             .join(sep)
//     }
// }

impl Display for Note {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "n{}v{}b{}", self.note, self.vel, self.beats)
    }
}

impl Default for Note {
    fn default() -> Self {
        Self {
            note: 60,
            vel: 64,
            beats: 1.into(),
        }
    }
}

pub enum NoteMod {
    Velocity(u8),
    Beats(f64),
}

fn parse_velocity(x: &str) -> IResult<&str, NoteMod> {
    use nom::character::complete::char;
    map_res(preceded(char('v'), digit1), |v: &str| {
        v.parse::<u8>().map(NoteMod::Velocity)
    })(x)
}
fn parse_beats(x: &str) -> IResult<&str, NoteMod> {
    use nom::character::complete::char;
    map_res(
        preceded(
            char('b'),
            alt((
                parse_decimal,
                parse_fraction,
                map_res(digit1, |d: &str| d.parse::<f64>()),
            )),
        ),
        |b: f64| -> Result<NoteMod> { Ok(NoteMod::Beats(b)) },
    )(x)
}
fn parse_decimal(x: &str) -> IResult<&str, f64> {
    map_res(
        recognize(separated_pair(digit1, tag("."), digit1)),
        |res: &str| res.parse::<f64>(),
    )(x)
}

fn parse_fraction(x: &str) -> IResult<&str, f64> {
    use nom::character::complete::char;
    map_res(
        separated_pair(digit1, char('/'), digit1),
        |(a, b): (&str, &str)| -> Result<f64> {
            let a: f64 = a.parse()?;
            let b: f64 = b.parse()?;
            Ok(a / b)
        },
    )(x)
}

type ParseNoteResult<'a> = IResult<&'a str, (char, Option<char>, &'a str, Vec<NoteMod>)>;

fn parse_note(x: &str) -> ParseNoteResult {
    use nom::character::complete::char;
    let num = alt((
        recognize(tuple((char('-'), one_of("12")))),
        recognize(tuple((opt(char('+')), one_of("012345678")))),
    ));
    tuple((
        one_of("abcdefgABCDEFG"),
        opt(one_of("#b^vsfSF")),
        num,
        many0(alt((parse_velocity, parse_beats))),
    ))(x)
}

fn apply_mods(note: &mut Note, mods: Vec<NoteMod>) {
    use NoteMod::*;

    if mods.is_empty() {
        return;
    }
    for m in mods {
        match m {
            Velocity(v) => note.vel = v,
            Beats(b) => note.beats = b.into(),
        }
    }
}

fn note_from_str(x: &str) -> color_eyre::Result<Note> {
    let (_, (note, sf, oct, mods)) =
        parse_note(x).map_err(|e| eyre!("failed to parse note: {}", e))?;
    // let (vel, beats) = handle_velocity_and_beats(vel, beats);
    let out = 24;
    let out: u8 = match note.to_ascii_uppercase() {
        'C' => out,
        'D' => out + 2,
        'E' => out + 4,
        'F' => out + 5,
        'G' => out + 7,
        'A' => out + 9,
        'B' => out + 11,
        _ => bail!("invalid note: {}", note),
    };
    let out = match sf {
        Some('#' | '^' | 's' | 'S') => out + 1,
        Some('b' | 'v' | 'f' | 'F') => out - 1,
        None => out,
        _ => bail!("invalid sharp or flat symbol: {}", sf.unwrap()),
    };
    let octave: i16 = oct.parse()?;
    let out: i16 = out as i16 + (12 * octave);
    ensure!(out <= 127, "g8 (127) is the highest MIDI note");
    // let vel = vel.map_or(Ok(64), |v| v.parse())?;
    // let beats = beats.map_or(Ok(0.1f64), |b| b.parse())?;
    let mut n = Note {
        note: out as u8,
        ..Default::default()
    };
    apply_mods(&mut n, mods);
    Ok(n)
}

impl Note {
    pub fn off(note: u8) -> Self {
        Note {
            note,
            vel: 0,
            beats: 0.into(),
        }
    }
    /// Get a reference to the note's beats.
    pub fn beats(&self) -> Beats {
        self.beats
    }

    /// Set the note's beats.
    pub fn with_beats(&self, beats: impl Into<Beats>) -> Self {
        Note {
            beats: beats.into(),
            ..*self
        }
    }

    /// Get a reference to the note's vel.
    pub fn vel(&self) -> u8 {
        self.vel
    }

    /// Set the note's vel.
    pub fn with_vel(&self, vel: u8) -> Self {
        Self { vel, ..*self }
    }

    /// Get a reference to the note's note.
    pub fn note(&self) -> u8 {
        self.note
    }
}

impl From<u8> for Note {
    fn from(x: u8) -> Self {
        Note {
            note: x,
            ..Default::default()
        }
    }
}

impl TryFrom<&str> for Note {
    type Error = color_eyre::Report;

    fn try_from(value: &str) -> std::result::Result<Self, Self::Error> {
        note_from_str(value)
    }
}

#[cfg(test)]
mod test_notes {
    use super::*;
    use hamcrest2::prelude::*;

    #[derive(Clone, Default, Debug)]
    struct Env {}

    #[test]
    fn test_note_from_str() {
        rspec::run(&rspec::describe("note_from_str", Env::default(), |ctx| {
            ctx.it("parses notes", |_| {
                assert_that!(note_from_str("c3").unwrap().note, eq(60));
            });
            [
                ("parses # as sharp", "#"),
                ("parses ^ as sharp", "^"),
                ("parses s as sharp", "s"),
                ("parses S as sharp", "S"),
            ]
            .iter()
            .for_each(|(name, x)| {
                ctx.it(name, |_| {
                    let note = ["c", x, "3"].join("");
                    let note = note_from_str(&note).unwrap();
                    assert_that!(note.note, eq(61));
                });
            });
            [
                ("parses b as flat", "b"),
                ("parses v as flat", "v"),
                ("parses f as flat", "f"),
                ("parses F as flat", "F"),
            ]
            .iter()
            .for_each(|(name, x)| {
                ctx.it(name, |_| {
                    let note = ["c", x, "3"].join("");
                    assert_that!(note_from_str(&note).unwrap().note, eq(59));
                });
            });

            [
                ("octave -2", -2, 0u8),
                ("octave -1", -1, 12),
                ("octave 0", 0, 24),
                ("octave 1", 1, 36),
                ("octave 2", 2, 48),
                ("octave 3", 3, 60),
                ("octave 4", 4, 72),
                ("octave 5", 5, 84),
                ("octave 6", 6, 96),
                ("octave 7", 7, 108),
                ("octave 8", 8, 120),
            ]
            .iter()
            .for_each(|(name, octave, expected)| {
                ctx.it(name, |_| {
                    let note = format!("c{}", *octave);
                    assert_that!(note_from_str(&note).unwrap().note, eq(*expected));
                });
            });

            ctx.it("doesn't allow octaves > 8", |_ctx| {
                assert_that!(note_from_str("c9"), err());
            });

            ctx.it("doesn't allow octaves < -2", |_ctx| {
                assert_that!(note_from_str("c-3"), err());
            });

            ctx.it("doesn't allow notes higher than g8", |_ctx| {
                assert_that!(note_from_str("g8").unwrap().note, eq(127));
                assert_that!(note_from_str("a8"), err());
            });

            ctx.it("doesn't allow notes higher than c-2", |_ctx| {
                assert_that!(note_from_str("c-2").unwrap().note, eq(0));
                assert_that!(note_from_str("b-3"), err());
            });

            ctx.it("parses optional velocity", |_ctx| {
                let n = note_from_str("c1v100").unwrap();
                assert_that!(n.vel, eq(100));
            });

            ctx.it("parses optional int beats", |_ctx| {
                let n = note_from_str("c1b16").unwrap();
                assert_that!(n.beats, eq(16));
            });

            ctx.it("parses decimal beats", |_ctx| {
                let n = note_from_str("c1b2.5").unwrap();
                assert_that!(n.beats, eq(2.5));
            });
            ctx.it("parses fractional beats", |_ctx| {
                let n = note_from_str("c1b1/2").unwrap();
                assert_that!(n.beats, eq(0.5));
            });

            ctx.it("parses velocity and beats", |_ctx| {
                let n = note_from_str("c1v123b1.5").unwrap();
                assert_that!(n.vel, eq(123));
                assert_that!(n.beats, eq(1.5));
            });

            ctx.it("parses beats and velocity", |_ctx| {
                let n = note_from_str("c1b22v45").unwrap();
                assert_that!(n.beats, eq(22));
                assert_that!(n.vel, eq(45));
            });
        }));
    }
}

