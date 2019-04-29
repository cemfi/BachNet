import torch
import matplotlib.pyplot as plt
import numpy as np
from music21 import pitch, stream, note, tie, key, meter, clef



class Synthesizer:
    def __init__(self):
        self.current_ts = None
        self.current_ks = None

        self.last_s_note_midi = None
        self.last_a_note_midi = None
        self.last_t_note_midi = None
        self.last_b_note_midi = None

        self.last_s_note = None
        self.last_a_note = None
        self.last_t_note = None
        self.last_b_note = None

    def synthesizeFromArray(self, arr, show=True):

        s = stream.Score(id='mainScore')
        p0 = stream.Part(id='sopran')
        p1 = stream.Part(id='alt')
        p2 = stream.Part(id='tenor')
        p3 = stream.Part(id='bass')

        counter = 0

        for chordFound in arr:
            # print(chordFound)
            soprano_note_midi = np.argsort(chordFound[194:256])[-1] + 30  # ende plus 2, damit pause und continues drin
            alto_note_midi = np.argsort(chordFound[132:194])[-1] + 30
            tenor_note_midi = np.argsort(chordFound[70:132])[-1] + 30
            bass_note_midi = np.argsort(chordFound[8:70])[-1] + 30
            time_sig = self._get_time_sig(chordFound[275:278])
            key_sig = self._get_key_sig(np.argsort(chordFound[263:275])[-1])  # arg is number of sharps
            first_note = bool(chordFound[0]) and bool(chordFound[256])

            if counter == 0:
                m_s = stream.Measure()
                m_a = stream.Measure()
                m_t = stream.Measure()
                m_b = stream.Measure()
                m_s.append(clef.TrebleClef())
                m_a.append(clef.TrebleClef())
                m_t.append(clef.BassClef())
                m_b.append(clef.BassClef())
            elif first_note:
                p0.append(m_s)
                p1.append(m_a)
                p2.append(m_t)
                p3.append(m_b)
                m_s = stream.Measure()
                m_a = stream.Measure()
                m_t = stream.Measure()
                m_b = stream.Measure()

            if self.current_ks != key_sig:
                m_s.append(key_sig)
                m_a.append(key_sig)
                m_t.append(key_sig)
                m_b.append(key_sig)
                self.current_ks = key_sig

            if self.current_ts == None or (self.current_ts.denominator != time_sig.denominator and self.current_ts.numerator != time_sig.numerator):
                m_s.append(time_sig)
                m_a.append(time_sig)
                m_t.append(time_sig)
                m_b.append(time_sig)
                self.current_ts = time_sig

            duration = 0.5

            soprano_note, self.last_s_note_midi = self._make_element_from_midi(soprano_note_midi, self.last_s_note, duration, self.last_s_note_midi)
            alto_note, self.last_a_note_midi = self._make_element_from_midi(alto_note_midi, self.last_a_note, duration, self.last_a_note_midi)
            tenor_note, self.last_t_note_midi = self._make_element_from_midi(tenor_note_midi, self.last_t_note, duration, self.last_t_note_midi)
            bass_note, self.last_b_note_midi = self._make_element_from_midi(bass_note_midi, self.last_b_note, duration, self.last_b_note_midi)

            soprano_note.lyric = str(counter)
            counter += 1
            m_s.append(soprano_note)
            m_a.append(alto_note)
            m_t.append(tenor_note)
            m_b.append(bass_note)

            self.last_s_note = soprano_note
            self.last_a_note = alto_note
            self.last_t_note = tenor_note
            self.last_b_note = bass_note

        p0.append(m_s)
        p1.append(m_a)
        p2.append(m_t)
        p3.append(m_b)

        p0 = p0.stripTies(retainContainers=True)
        s.append(p0)
        p1 = p1.stripTies(retainContainers=True)
        s.append(p1)
        p2 = p2.stripTies(retainContainers=True)
        s.append(p2)
        p3 = p3.stripTies(retainContainers=True)
        s.append(p3)

        #s.show("text")

        if show:
            s.show()

        return s

    def _make_element_from_midi(self, midi, last_note, duration, last_note_midi):
        if midi == 90: # if rest is encoded
            element = note.Rest()
        elif midi == 91:   # if continue is encoded
            # first beat (no note before)
            if last_note_midi is None:
                print("Warning: continue on first beat")
                midi = 90
                element = note.Rest()

            # continue in cases where a note has been encountered before
            else:
                midi = last_note_midi
                if midi == 90:  # continue after rest
                    element = note.Rest()
                else:  # most common, "normal" continue-case: continue after note
                    element = note.Note(midi)
                last_note.tie = tie.Tie('start')
                element.tie = tie.Tie('stop')

        # most common, "normal" overall-case: note not continue/rest but really a note
        else:
            element = note.Note(midi)
        element.duration.quarterLength = duration
        return element, midi


    def _get_key_sig(self, number_of_sharps):
        if number_of_sharps > 6:
            number_of_sharps = number_of_sharps - 12
        return key.KeySignature(number_of_sharps)


    def _get_time_sig(self, ts_bits):
        if ts_bits[0] == 1:
            return meter.TimeSignature('4/4')
        elif ts_bits[1] == 1:
            return meter.TimeSignature('3/4')
        elif ts_bits[2] == 1:
            return  meter.TimeSignature('3/2')
        else:
            raise Exception("no valid TS found")

#s = Synthesizer()
#fileI = np.load("/Users/alexanderleemhuis/Informatik/PY/PRJ/bach git/BachNet/chordDataSQ/debugo.csv")
#fileI = np.genfromtxt("/Users/alexanderleemhuis/Informatik/PY/PRJ/bach git/BachNet/data/bwv101.7.mxl_2_-3debugi.csv")
#s.synthesizeFromArray(fileI.transpose())