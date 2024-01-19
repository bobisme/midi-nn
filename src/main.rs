use std::collections::HashMap;
use std::fs::File;

use clap::{Parser, Subcommand};
use midly::{Smf, TrackEvent};

use notes::Note;
use token::tokenize_midi;

use crate::token::Params;

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
struct InspectMidi {
    /// Path to midi file.
    path: String,
}

#[derive(Parser, Debug, Clone)]
struct InspectTokens {
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
    InspectMidi(InspectMidi),
    /// Inspect a tokens file.
    InspectTokens(InspectTokens),
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

fn get_best_track(smf: &midly::Smf) -> usize {
    let (i, _) = smf
        .tracks
        .iter()
        .enumerate()
        .max_by_key(|(i, x)| x.len())
        .unwrap();
    i
}

pub type Encoded = (u32, f32, f32, f32, f32);

fn main() -> color_eyre::Result<()> {
    let args = Args::parse();
    match &args.cmd {
        Commands::Tokenize(cmd) => {
            println!("reading {}", &cmd.path);
            let midi_data = std::fs::read(&cmd.path)?;
            let smf = Smf::parse(&midi_data)?;
            // 500,000 us == 120 BPM
            let mut beats_per_minute = 120.0;
            let timing = smf.header.timing;
            let mut cumulative_delta = 0;
            let transpose = cmd.transpose.unwrap_or(0);
            if transpose != 0 {
                println!("NOT IMPLEMENTED: transposing {:+} semitones", transpose);
            }
            let track_i = get_best_track(&smf);
            let track = &smf.tracks[track_i];
            let mut tokenized_track = Vec::new();
            for ev in track {
                let res = token::from_event(ev, beats_per_minute, &timing, cumulative_delta);
                if res.is_none() {
                    cumulative_delta += ev.delta.as_int();
                    continue;
                }
                cumulative_delta = 0;
                let (tok, p) = res.unwrap();
                if matches!(tok, token::Token::Control) && p.tempo >= 0.0 {
                    beats_per_minute = token::param_to_bpm(p.tempo);
                }
                tokenized_track.push((tok.index(), p.delay, p.vel, p.tempo, p.sustain));
            }
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
            let samples: Vec<Encoded> = ciborium::from_reader(f)?;
            let ticks_per_beat = 1_000;
            let timing = midly::Timing::Metrical(ticks_per_beat.into());
            let mut smf = Smf::new(midly::Header {
                format: midly::Format::SingleTrack,
                timing,
            });
            let mut track = midly::Track::new();
            track.push(midly::TrackEvent {
                delta: 0.into(),
                kind: midly::TrackEventKind::Meta(midly::MetaMessage::TimeSignature(4, 4, 36, 8)),
            });
            track.push(midly::TrackEvent {
                delta: 0.into(),
                kind: midly::TrackEventKind::Meta(midly::MetaMessage::Tempo(
                    cmd.us_per_beat.into(),
                )),
            });
            let mut current_tick = 0u32;
            let mut active_notes: HashMap<u8, ActiveNote> = HashMap::new();
            for (sample_i, (token_idx, delay, vel, tempo, sustain)) in
                samples.into_iter().enumerate()
            {
                // dbg!(&active_notes);
                // dbg!(current_tick);
                let vel = vel * 1.5;
                let vel = vel.clamp(0.1, 1.0);
                let delay = delay.clamp(0.0, 64.0);
                let tempo = tempo.clamp(-1.0, 0.8);
                let sustain = tempo.clamp(-1.0, 1.0);
                let delta = (delay * ticks_per_beat as f32) as u32;
                let token = token_map[&token_idx];
                match token {
                    token::Token::NoteOn { note } => {
                        let vel = (vel * 127.0) as u8;
                        track.push(midly::TrackEvent {
                            delta: delta.into(),
                            kind: midly::TrackEventKind::Midi {
                                channel: 0.into(),
                                message: midly::MidiMessage::NoteOn {
                                    key: note.into(),
                                    vel: vel.into(),
                                },
                            },
                        });
                    }
                    token::Token::NoteOff { note } => {
                        let vel = (vel * 127.0) as u8;
                        track.push(midly::TrackEvent {
                            delta: delta.into(),
                            kind: midly::TrackEventKind::Midi {
                                channel: 0.into(),
                                message: midly::MidiMessage::NoteOff {
                                    key: note.into(),
                                    vel: vel.into(),
                                },
                            },
                        });
                    }
                    token::Token::Unknown => {}
                    token::Token::Pad => {}
                    token::Token::Start => {}
                    token::Token::End => {}
                    token::Token::Control => {
                        if sustain >= 0.0 {
                            track.push(midly::TrackEvent {
                                delta: delta.into(),
                                kind: midly::TrackEventKind::Midi {
                                    channel: 0.into(),
                                    message: midly::MidiMessage::Controller {
                                        controller: 64.into(),
                                        value: ((sustain * 127.0) as u8).into(),
                                    },
                                },
                            });
                        }
                        if tempo >= 0.0 {
                            let bpm = token::param_to_bpm(tempo);
                            let us_per_beat = (60_000_000.0 / bpm) as u32;
                            track.push(midly::TrackEvent {
                                delta: delta.into(),
                                kind: midly::TrackEventKind::Meta(midly::MetaMessage::Tempo(
                                    us_per_beat.into(),
                                )),
                            });
                        }
                    }
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
        Commands::InspectMidi(cmd) => {
            println!("reading {}", &cmd.path);
            let midi_data = std::fs::read(&cmd.path)?;
            let smf = Smf::parse(&midi_data)?;
            println!("HEADER: {:?}", smf.header);
            for (i, track) in smf.tracks.into_iter().enumerate() {
                println!("track {i} has {} events", track.len());
                for ev in track {
                    println!("{:?}", ev);
                }
            }
        }
        Commands::InspectTokens(cmd) => {
            println!("reading {}", &cmd.path);
            let f = File::open(&cmd.path)?;
            let samples: Vec<Encoded> = ciborium::from_reader(f)?;
            for (idx, delay, vel, tempo, sustain) in samples {
                let tok = token::Token::from_index(idx);
                let params = Params {
                    delay,
                    vel,
                    tempo,
                    sustain,
                };
                println!("{:?} {:?}", tok, params);
            }
        }
    }
    Ok(())
}
