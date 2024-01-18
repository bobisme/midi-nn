use std::collections::HashMap;
use std::fs::File;

use clap::{Parser, Subcommand};
use midly::Smf;

use notes::Note;
use token::tokenize_midi;

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
    /// Transposition in semitones.
    #[arg(long)]
    transpose: Option<i8>,
    /// Path to output token file.
    #[arg(long)]
    out_path: Option<String>,
}

#[derive(Parser, Debug, Clone)]
struct Midify {
    /// Path to token file.
    path: String,
    /// Path to output midi file.
    #[arg(long)]
    out_path: Option<String>,
    #[arg(long, default_value_t = 500_000)]
    us_per_beat: u32,
}

#[derive(Parser, Debug, Clone)]
struct Inspect {
    /// Path to midi file.
    path: String,
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    /// Convert a .midi file to .token
    Tokenize(Tokenize),
    /// Convert a .token file to .midi
    Midify(Midify),
    /// Inspect a midi file.
    Inspect(Inspect),
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

#[derive(Debug, Clone, Copy)]
struct ActiveNote {
    start_tick: u32,
    length_ticks: u32,
}

fn main() -> color_eyre::Result<()> {
    let args = Args::parse();
    match &args.cmd {
        Commands::Tokenize(cmd) => {
            println!("reading {}", &cmd.path);
            let midi_data = std::fs::read(&cmd.path)?;
            let mut parsed = midi::parse(&midi_data)?;
            let transpose = cmd.transpose.unwrap_or(0);
            if transpose != 0 {
                println!("transposing {:+} semitones", transpose);
            }
            let transposed_notes = parsed
                .notes()
                .iter()
                .map(|x| x.transpose(transpose))
                .collect();
            parsed.notes = transposed_notes;
            let iter = tokenize_midi(&parsed);
            let tokenized_track: Vec<(u32, f64, f64, f64)> = iter
                .map(|(t, p)| (t.index(), p.duration, p.delay, p.vel))
                .collect();
            let out_path = match &cmd.out_path {
                Some(p) => p.clone(),
                None => format!("{}.{transpose:+}-semis.tokens", cmd.path),
            };
            println!("writing to {}", out_path);
            ciborium::into_writer(&tokenized_track, File::create(&out_path)?)?;
        }
        Commands::Midify(cmd) => {
            let token_map = token::build_rev_map();
            // dbg!(&token_map);
            // dbg!(token_map.len());
            println!("reading {}", &cmd.path);
            let f = File::open(&cmd.path)?;
            let sample: Vec<(u32, f64, f64, f64)> = ciborium::from_reader(f)?;
            let ticks_per_beat = 1_000;
            let mut smf = Smf::new(midly::Header {
                format: midly::Format::SingleTrack,
                timing: midly::Timing::Metrical(ticks_per_beat.into()),
            });
            let mut track = midly::Track::new();
            track.push(midly::TrackEvent {
                delta: 0.into(),
                kind: midly::TrackEventKind::Meta(midly::MetaMessage::TimeSignature(4, 2, 36, 8)),
            });
            track.push(midly::TrackEvent {
                delta: 0.into(),
                kind: midly::TrackEventKind::Meta(midly::MetaMessage::Tempo(
                    cmd.us_per_beat.into(),
                )),
            });
            let mut current_tick = 0u32;
            let mut active_notes: HashMap<u8, ActiveNote> = HashMap::new();
            for (sample_i, (token_idx, beats, delay, vel)) in sample.into_iter().enumerate() {
                // dbg!(&active_notes);
                // dbg!(current_tick);
                let vel = vel * 1.5;
                let vel = vel.clamp(0.1, 1.0);
                if beats < 0.01 {
                    println!("WARNING: tiny beat in sample {sample_i}");
                }
                let beats = beats.clamp(0.0625, 4.0);
                let delay = delay.clamp(0.0, 4.0);
                let token = token_map[&token_idx];
                let mut prev_note = 0u8;
                match token {
                    token::Token::Note { note } => {
                        let vel = (vel * 127.0) as u8;
                        let n = Note::from(note).with_vel(vel).with_beats(beats);
                        let mut to_remove = Vec::new();
                        for (&n, &active) in active_notes.iter() {
                            if current_tick >= (active.start_tick + active.length_ticks) {
                                to_remove.push(n);
                                track.push(midly::TrackEvent {
                                    delta: 0.into(),
                                    kind: midly::TrackEventKind::Midi {
                                        channel: 0.into(),
                                        message: midly::MidiMessage::NoteOff {
                                            // key: prev_note.into(),
                                            key: n.into(),
                                            vel: 0.into(),
                                        },
                                    },
                                });
                            }
                        }
                        // dbg!(&to_remove);
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
                            ActiveNote {
                                start_tick: current_tick,
                                length_ticks: beats_to_ticks(ticks_per_beat, beats),
                            },
                        );
                        prev_note = n.note();
                    }
                    token::Token::Unknown => {}
                    token::Token::Pad => {}
                    token::Token::Start => {}
                    token::Token::End => {}
                }
            }
            track.push(midly::TrackEvent {
                delta: 0.into(),
                kind: midly::TrackEventKind::Meta(midly::MetaMessage::EndOfTrack),
            });
            smf.tracks.push(track);
            let out_path = match &cmd.out_path {
                Some(p) => p.clone(),
                None => cmd.path.clone() + ".mid",
            };
            println!("writing to {}", &out_path);
            smf.save(&out_path)?;
        }
        Commands::Inspect(cmd) => {
            println!("reading {}", &cmd.path);
            let midi_data = std::fs::read(&cmd.path)?;
            let smf = Smf::parse(&midi_data)?;
            for (i, track) in smf.tracks.into_iter().enumerate() {
                println!("track {i} has {} events", track.len());
                for ev in track {
                    println!("{:?}", ev);
                }
            }
        }
    }
    Ok(())
}
