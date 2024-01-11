use std::io::prelude::*;
use std::{cmp, fs::File};

use midly::Smf;

use notes::Note;
use token::{embed_midi, embed_note, tokenize_midi};

use crate::token::from_note;

pub mod error;
pub mod midi;
pub mod notes;
pub mod sequence;
pub mod token;

const MOONLIGHT: &[u8] = include_bytes!("../beethopin.mid");

fn main() -> color_eyre::Result<()> {
    let parsed = midi::parse(MOONLIGHT)?;
    {
        let iter = tokenize_midi(&parsed);
        let tokenized_track: Vec<(u32, f64)> = iter.map(|(t, b)| (t.index(), b.into())).collect();
        ciborium::into_writer(&tokenized_track, File::create("moonlight.tokens")?)?;
    }
    // {
    //     let iter = embed_midi(&parsed);
    //     let embedded_song: Vec<_> = iter.collect();
    //     let mut f = File::create("moonlight.embedded")?;
    //     ciborium::into_writer(&embedded_song, f)?;
    // }
    // for emb in iter {
    //     dbg!(emb);
    // }
    let token_map = token::build_rev_map();
    // dbg!(&token_map);
    // dbg!(token_map.len());
    let f = File::open("sample-0.tokens")?;
    let sample: Vec<(u32, f64)> = ciborium::from_reader(f)?;
    let mut smf = Smf::new(midly::Header {
        format: midly::Format::SingleTrack,
        timing: midly::Timing::Metrical(500.into()),
    });
    let mut track = midly::Track::new();
    for (token_idx, beats) in sample {
        let token = token_map[&token_idx];
        // dbg!(token);
        let mut prev_note = 0u8;
        match token {
            token::Token::Unknown => {}
            token::Token::Pad => {}
            token::Token::Start => {}
            token::Token::End => {}
            token::Token::Note { note, vel } => {
                let vel = match vel {
                    token::Level::Min => 0,
                    token::Level::Low => 25,
                    token::Level::LowMed => 50,
                    token::Level::Med => 64,
                    token::Level::MedHigh => 100,
                    token::Level::High => 120,
                    token::Level::Max => 127,
                };
                let beats = if beats < 0.0 { 0.0 } else { beats };
                let n = Note::from(note).with_vel(vel).with_beats(beats);
                track.push(midly::TrackEvent {
                    delta: 250.into(),
                    kind: midly::TrackEventKind::Midi {
                        channel: 0.into(),
                        message: midly::MidiMessage::NoteOff {
                            key: prev_note.into(),
                            vel: n.vel().into(),
                        },
                    },
                });
                track.push(midly::TrackEvent {
                    delta: 0.into(),
                    kind: midly::TrackEventKind::Midi {
                        channel: 0.into(),
                        message: midly::MidiMessage::NoteOn {
                            key: n.note().into(),
                            vel: n.vel().into(),
                        },
                    },
                });
                prev_note = n.note();
            }
        }
    }
    smf.tracks.push(track);
    smf.save("sample-0.mid")?;
    Ok(())
}
