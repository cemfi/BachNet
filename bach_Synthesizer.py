import torch
import matplotlib.pyplot as plt
import numpy as np
from music21 import chord, stream, note, tie, key, meter



class Synthesizer:
    def __init__(self):
        print("Synth")

    def synthesizeFromArray(self, arr, show=True):

        s = stream.Score(id='mainScore')
        p0 = stream.Part(id='sopran')
        p1 = stream.Part(id='alt')
        p2 = stream.Part(id='tenor')
        p3 = stream.Part(id='bass')

        last_s_note = None
        last_a_note = None
        last_t_note = None
        last_b_note = None

        counter = 0

        first_note = True

        for chordFound in arr:
            soprano_note_midi = np.argsort(chordFound[194:256])[-1] + 30  ##ende plus 2, damit pause und continues drin
            alto_note_midi = np.argsort(chordFound[132:194])[-1] + 30
            tenor_note_midi = np.argsort(chordFound[70:132])[-1] + 30
            bass_note_midi = np.argsort(chordFound[8:70])[-1] + 30
            key_sharps = np.argsort(chordFound[263:275])[-1]
            chord_is_one = bool(chordFound[0]) and bool(chordFound[256])
            print(chord_is_one)

            if False:# first_note or chord_is_one:
                m_s = stream.Measure()
                m_a = stream.Measure()
                m_t = stream.Measure()
                m_b = stream.Measure()
                if first_note:
                    p0.append(m_s)
                    p1.append(m_a)
                    p2.append(m_t)
                    p3.append(m_b)

            duration = 0.5

            soprano_note, soprano_note_midi, p0 = self._make_element_from_midi(soprano_note_midi, p0, duration, last_s_note)
            alto_note, alto_note_midi, p1 = self._make_element_from_midi(alto_note_midi, p1, duration, last_a_note)
            tenor_note, tenor_note_midi, p2 = self._make_element_from_midi(tenor_note_midi, p2, duration, last_t_note)
            bass_note, bass_note_midi, p3 = self._make_element_from_midi(bass_note_midi, p3, duration, last_b_note)

            soprano_note.lyric = str(counter)
            counter += 1
            #m_s.append(soprano_note)
            p0.append(soprano_note)
            #m_a.append(alto_note)
            p1.append(alto_note)
            #m_t.append(tenor_note)
            p2.append(tenor_note)
            #m_b.append(bass_note)
            p3.append(bass_note)

            last_s_note = soprano_note_midi
            last_a_note = alto_note_midi
            last_t_note = tenor_note_midi
            last_b_note = bass_note_midi

        #m = stream.Measure()
        #m.insert(0, meter.TimeSignature('3/4'))
        #key_s = key.KeySignature(key_sharps)
        #m.insert(0, key_s)
        #s.insert(0, m)

        p0 = p0.stripTies()
        s.insert(0, p0)
        p1 = p1.stripTies()
        s.insert(0, p1)
        p2 = p2.stripTies()
        s.insert(0, p2)
        p3 = p3.stripTies()#wtf
        #p3 = p3.stripTies(retainContainers=True)
        s.insert(0, p3)

        s.show("text")


        if show:
            s.show()

        return s

    def _make_element_from_midi(self, midi, stream, duration, last_note):
        if midi == 90:
            element = note.Rest()
        elif midi == 91:
            if last_note is None: #first beat
                print("Warning: continue on first beat Soprano")
                midi = 90
                element = note.Rest()
            else:
                midi = last_note
                if midi == 90:
                    element = note.Rest()
                else:
                    element = note.Note(midi)
                stream.getElementAtOrBefore(stream.highestOffset).tie = tie.Tie('start')
                element.tie = tie.Tie('stop')
        else:
            element = note.Note(midi)
        element.duration.quarterLength = duration
        return element, midi, stream

#s = Synthesizer()
#fileI = np.genfromtxt("test/val/bwv1.6.mxl0/o.csv", delimiter=',', dtype=np.single)
#s.synthesizeFromArray(fileI.transpose())