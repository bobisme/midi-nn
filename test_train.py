import torch

from train import load_tracks


def test_load_tracks_adds_track_position():
    tracks = load_tracks(["tokenized/mond_3_format0.mid.tokens"])
    assert len(tracks) == 1
    assert len(tracks[0]) == 15114
    assert len(tracks[0][0]) == 4
    start_track_pos = tracks[0][0][-1]
    mid_track_pos = tracks[0][15114 // 2][-1]
    end_track_pos = tracks[0][-1][-1]
    assert start_track_pos == 0.0
    assert (1.0 - end_track_pos) < 0.0001
    assert (0.5 - mid_track_pos) < 0.0001
