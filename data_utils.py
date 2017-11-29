from music21 import converter, note, chord, stream
import numpy as np
import fnmatch
import os
import scipy.io


def m21_to_piano_roll(m21_stream, quantize_step=4):
    """
    :param stream: Music21 stream of song data.
    :param quantize_step: Denominator indicating what fraction of a quarter note to quantize to. Ex: 4 = sixteenth note.
    :return piano_roll: numpy array of song in piano roll format.
    """
    song = m21_stream.flat.notes.quantize([quantize_step])
    duration = song.duration.quarterLength

    assert duration.is_integer()

    piano_roll = np.zeros([128, int(duration*quantize_step)])

    for offset in np.linspace(0.0, duration, int(duration*quantize_step)+1, endpoint=True):
        notes_at_offset = song.getElementsByOffset(offset, mustBeginInSpan=False, includeElementsThatEndAtStart=False)
        for obj in notes_at_offset:
            start = song.elementOffset(obj)
            if start == offset:
                if type(obj) == note.Note:
                    # Beginning of note
                    piano_roll[obj.pitch.midi][int(offset*quantize_step)] = 2
                elif type(obj) == chord.Chord:
                    for pitch in obj.pitches:
                        piano_roll[pitch.midi][int(offset * quantize_step)] = 2
            else:
                if type(obj) == note.Note:
                    # Continued note
                    piano_roll[obj.pitch.midi][int(offset*quantize_step)] = 1
                elif type(obj) == chord.Chord:
                    for pitch in obj.pitches:
                        piano_roll[pitch.midi][int(offset * quantize_step)] = 1

    return piano_roll

def piano_roll_to_m21(piano_roll, quantize_step=4):
    """
    :param piano_roll: Piano roll representation of song.
    :param quantize_step: Denominator indicating what fraction of a quarter note to quantize to. Ex: 4 = sixteenth note.
    :return m21_stream: Music21 stream of notes from the piano roll.
    """
    m21_stream = stream.Stream()
    for midi_val, note_axis in enumerate(piano_roll):
        in_note = False
        note_start = None
        for i, val in enumerate(note_axis):
            curr_offset = float(i) / quantize_step
            if val == 2:
                if in_note:
                    # Finish last note
                    note_to_add = note.Note(midi_val, quarterLength=(curr_offset - note_start))
                    note_to_add.offset = note_start
                    m21_stream.insert(note_to_add)

                    # Reset start
                    note_start = curr_offset
                else:
                    in_note = True
                    note_start = curr_offset
            elif val == 0:
                if in_note:
                    # Finish last note
                    note_to_add = note.Note(midi_val, quarterLength=(curr_offset - note_start))
                    note_to_add.offset = note_start
                    m21_stream.insert(note_to_add)

                    in_note = False
        if in_note:
            note_to_add = note.Note(midi_val, quarterLength=(curr_offset - note_start))
            note_to_add.offset = note_start
            m21_stream.insert(note_to_add)
    return m21_stream


def get_filepaths(dirname='data', ext='krn'):
    matches = []
    for root, dirnames, filenames in os.walk(dirname):
        for filename in fnmatch.filter(filenames, '*.'+ext):
            matches.append(os.path.join(root, filename))
    print '\n', 'Found %d kern files.' % (len(matches),), '\n'
    return matches


def serialize_songs(data_dir='data/np_data', fname='all_data'):
    """
    Read a directory of .krn files and convert to piano roll representation.
    Write results to a .npz file.

    :param data_dir: location where the .npz file should be written.
    :param fname: name of .npz file
    :return:
    """
    if os.path.exists(os.path.join(data_dir, fname)):
        os.remove(os.path.join(data_dir, fname))
    songs = {}
    for i, fp in enumerate(get_filepaths()):
        print 'Processing file %d, %s' % (i+1, fp)
        try:
            m21_stream = converter.parse(fp, format='humdrum')
        except Exception:
            print 'Parse failed, skipping.'
            continue
        try:
            pr = m21_to_piano_roll(m21_stream)
        except Exception:
            print 'Conversion to piano roll failed, skipping.'
            continue
        songs[fp] = pr
    np.savez(os.path.join(data_dir, fname), **songs)


def clean_filename(fname):
    """
    Return a sanitized filename that works as a valid MATLAB field identifier
    :param fname:
    :return:
    """
    return (fname
            .replace('-', '')
            .replace('_', '')
            .replace('.', '')
            .replace(',', '')
            .replace('/', '_'))


def write_matfile_from_npz(np_file='data/np_data/all_data.npz', matfile='data/mat/all_data.mat'):
    data = np.load(np_file)
    data_dict = {clean_filename(name): data[name] for name in data.files}
    scipy.io.savemat(matfile, mdict=data_dict)

if __name__ == '__main__':
    pass

