use std::fs::File;

use clap::{Parser, Subcommand};
use midly::Smf;

use crate::token::{Params, REV_MAP};

pub mod error;
pub mod midi;
pub mod notes;
pub mod sequence;
pub mod token;

// const MOONLIGHT: &[u8] = include_bytes!("../beethopin.mid");
const TOKEN_VERSION: u16 = 4;

#[derive(Parser, Debug, Clone)]
struct Tokenize {
    /// Paths to midi files.
    #[arg(required = true)]
    paths: Vec<String>,
    /// Transposition in semitones.
    #[arg(long)]
    transpose: Option<i8>,
    /// Path to output token file.
    #[arg(long)]
    out_dir: Option<String>,
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
    /// Convert a .midi file to .tokens
    Tokenize(Tokenize),
    /// Convert a .tokens file to .midi
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

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
struct Header {
    pub version: u16,
    pub token_vocab_size: u16,
}

impl Default for Header {
    fn default() -> Self {
        Self {
            version: TOKEN_VERSION,
            token_vocab_size: token::REV_MAP.len() as u16,
        }
    }
}

fn get_best_track(smf: &midly::Smf) -> usize {
    let (i, _) = smf
        .tracks
        .iter()
        .enumerate()
        .max_by_key(|(_, x)| x.len())
        .unwrap();
    i
}

pub type Encoded = (u32, f32);
fn tokenize(cmd: &Tokenize) -> color_eyre::Result<()> {
    for path in &cmd.paths {
        println!("reading {}", path);
        let midi_data = std::fs::read(path)?;
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
            if res.is_empty() {
                cumulative_delta += ev.delta.as_int();
                continue;
            }
            cumulative_delta = 0;
            for (tok, p) in res {
                if let token::Token::Tempo { bpm } = tok {
                    beats_per_minute = bpm as f32;
                }
                tokenized_track.push((tok.index(), p.vel));
            }
        }
        let out_dir = cmd.out_dir.clone().unwrap_or("tokenized".to_string());
        let out_path = match transpose {
            0 => format!("{out_dir}/{}.tokens", path),
            _ => format!("{out_dir}/{}.{transpose:+}-semis.tokens", path),
        };
        let _ = std::fs::create_dir(out_dir);
        println!("writing to {}", out_path);
        ciborium::into_writer(
            &(Header::default(), &tokenized_track),
            File::create(&out_path)?,
        )?;
    }
    Ok(())
}

fn midify(cmd: &Midify) -> color_eyre::Result<()> {
    // dbg!(&token_map);
    // dbg!(token_map.len());
    println!("reading {}", &cmd.path);
    let f = File::open(&cmd.path)?;
    // let (header, samples): (Header, Vec<Encoded>) = ciborium::from_reader(f)?;
    let (header, samples): (Header, Vec<Encoded>) = ciborium::from_reader(f)?;
    assert!(header.version == TOKEN_VERSION);
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
        kind: midly::TrackEventKind::Meta(midly::MetaMessage::Tempo(cmd.us_per_beat.into())),
    });
    let mut delta = 0;
    for (token_idx, vel) in samples.into_iter() {
        let vel = vel * 1.5;
        let vel = vel.clamp(0.1, 1.0);
        let token = REV_MAP[token_idx as usize];
        match token {
            token::Token::Wait { divisions } => {
                let beats = divisions as f32 / token::BEAT_DIVISIONS as f32;
                delta += (beats * ticks_per_beat as f32) as u32;
            }
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
            token::Token::Sustain { on } => {
                let val = match on {
                    true => 127,
                    false => 0,
                };
                track.push(midly::TrackEvent {
                    delta: delta.into(),
                    kind: midly::TrackEventKind::Midi {
                        channel: 0.into(),
                        message: midly::MidiMessage::Controller {
                            controller: 64.into(),
                            value: val.into(),
                        },
                    },
                });
            }
            token::Token::Tempo { bpm } => {
                let us_per_beat = 60_000_000 / bpm as u32;
                track.push(midly::TrackEvent {
                    delta: delta.into(),
                    kind: midly::TrackEventKind::Meta(midly::MetaMessage::Tempo(
                        us_per_beat.into(),
                    )),
                });
            }
            token::Token::Control => {}
        }
        if !matches!(token, token::Token::Wait { divisions: _ }) {
            delta = 0;
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
    Ok(())
}

fn inspect_midi(cmd: &InspectMidi) -> color_eyre::Result<()> {
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
    Ok(())
}
fn inspect_tokens(cmd: &InspectTokens) -> color_eyre::Result<()> {
    println!("reading {}", &cmd.path);
    let f = File::open(&cmd.path)?;
    let (header, samples): (Header, Vec<Encoded>) = ciborium::from_reader(f)?;
    println!("TOKEN FORMAT {}", header.version);
    for (idx, vel) in samples {
        let tok = token::Token::from_index(idx);
        let params = Params { vel };
        println!("{:?} {:?}", tok, params);
    }
    Ok(())
}

fn main() -> color_eyre::Result<()> {
    let args = Args::parse();
    match &args.cmd {
        Commands::Tokenize(cmd) => tokenize(cmd),
        Commands::Midify(cmd) => midify(cmd),
        Commands::InspectMidi(cmd) => inspect_midi(cmd),
        Commands::InspectTokens(cmd) => inspect_tokens(cmd),
    }
}
