use std::collections::HashMap;
use std::io::prelude::*;
use std::{cmp, fs::File};

use clap::{Arg, Parser, Subcommand};
use midly::Smf;

use notes::{Beats, Note};
use token::{embed_midi, embed_note, tokenize_midi};

use crate::token::from_note;

pub mod error;
pub mod midi;
pub mod notes;
pub mod sequence;
pub mod token;

const MOONLIGHT: &[u8] = include_bytes!("../beethopin.mid");

#[derive(Parser, Debug, Clone)]
struct Tokenize {
    /// Path to midi file.
    path: String,
    /// Path to output token file.
    out_path: String,
}

#[derive(Parser, Debug, Clone)]
struct Midify {
    /// Path to token file.
    path: String,
    /// Path to output midi file.
    out_path: String,
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    /// Convert a .midi file to .token
    Tokenize(Tokenize),
    /// Convert a .token file to .midi
    Midify(Midify),
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    cmd: Commands,
}

fn beats_to_ticks(ticks_per_beat: u16, beats: impl Into<f64>) -> u32 {
    let b: f64 = beats.into();
    (b * ticks_per_beat as f64) as u32
}

fn main() -> color_eyre::Result<()> {
    let args = Args::parse();
    match &args.cmd {
        Commands::Tokenize(cmd) => {
            println!("reading {}", &cmd.path);
            let midi_data = std::fs::read(&cmd.path)?;
            let parsed = midi::parse(&midi_data)?;
            let iter = tokenize_midi(&parsed);
            let tokenized_track: Vec<(u32, f64, f64)> = iter
                .map(|(t, p)| (t.index(), p.duration, p.delay))
                .collect();
            println!("writing to {}", &cmd.out_path);
            ciborium::into_writer(&tokenized_track, File::create(&cmd.out_path)?)?;
        }
        Commands::Midify(cmd) => {
            let token_map = token::build_rev_map();
            // dbg!(&token_map);
            // dbg!(token_map.len());
            println!("reading {}", &cmd.path);
            let f = File::open(&cmd.path)?;
            let sample: Vec<(u32, f64, f64)> = ciborium::from_reader(f)?;
            let ticks_per_beat = 1000;
            let mut smf = Smf::new(midly::Header {
                format: midly::Format::SingleTrack,
                timing: midly::Timing::Metrical(ticks_per_beat.into()),
            });
            let mut track = midly::Track::new();
            track.push(midly::TrackEvent {
                delta: 0.into(),
                kind: midly::TrackEventKind::Meta(midly::MetaMessage::Tempo(1_000_000.into())),
            });
            let mut current_tick = 0u32;
            let mut cumulative_delta = 0;
            let mut active_notes: HashMap<u8, u32> = HashMap::new();
            for (token_idx, beats, delay) in sample {
                let token = token_map[&token_idx];
                let mut prev_note = 0u8;
                match token {
                    token::Token::Note { note, vel } => {
                        let vel = match vel {
                            token::Level::Min => 0,
                            token::Level::Low => 21,
                            token::Level::LowMed => 42,
                            token::Level::Med => 63,
                            token::Level::MedHigh => 84,
                            token::Level::High => 105,
                            token::Level::Max => 127,
                        };
                        let beats = if beats < 0.0 { 0.0 } else { beats };
                        let delay = if delay < 0.0 { 0.0 } else { delay };
                        let n = Note::from(note).with_vel(vel).with_beats(beats);
                        let mut to_remove = Vec::new();
                        for (&n, &exp) in active_notes.iter() {
                            if current_tick >= exp {
                                to_remove.push(n);
                                track.push(midly::TrackEvent {
                                    delta: 0.into(),
                                    kind: midly::TrackEventKind::Midi {
                                        channel: 0.into(),
                                        message: midly::MidiMessage::NoteOff {
                                            key: prev_note.into(),
                                            vel: 0.into(),
                                        },
                                    },
                                });
                            }
                        }
                        for n in to_remove {
                            active_notes.remove(&n);
                        }
                        let delta = beats_to_ticks(ticks_per_beat, delay);
                        current_tick += delta;
                        track.push(midly::TrackEvent {
                            delta: delta.into(),
                            kind: midly::TrackEventKind::Midi {
                                channel: 0.into(),
                                message: midly::MidiMessage::NoteOn {
                                    key: n.note().into(),
                                    vel: n.vel().into(),
                                },
                            },
                        });
                        active_notes.insert(
                            n.note(),
                            current_tick + beats_to_ticks(ticks_per_beat, beats),
                        );
                        prev_note = n.note();
                    }
                    token::Token::Unknown => {}
                    token::Token::Pad => {}
                    token::Token::Start => {}
                    token::Token::End => {}
                }
            }
            smf.tracks.push(track);
            println!("writing to {}", &cmd.out_path);
            smf.save(&cmd.out_path)?;
        }
    }
    Ok(())
}
