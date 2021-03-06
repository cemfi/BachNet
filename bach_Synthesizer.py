import torch
import copy

import matplotlib.pyplot as plt
import numpy as np
from music21 import analysis, stream, note, tie, key, meter, clef, layout, bar, expressions



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

    def synthesizeFromArray(self, arr, show=True, ties=True, piano_reduction=False):

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
            time_sig = self._get_time_sig(chordFound[275:279])
            key_sig = self._get_key_sig(np.argsort(chordFound[263:275])[-1])  # arg is number of sharps
            is_fermata = chordFound[260]
            first_note = bool(chordFound[0]) and bool(chordFound[256])

            # if you're at the very start create new measures and clefs
            if counter == 0:
                m_s = stream.Measure()
                m_a = stream.Measure()
                m_t = stream.Measure()
                m_b = stream.Measure()
                m_s.append(clef.TrebleClef())
                m_a.append(clef.TrebleClef())
                m_t.append(clef.BassClef())
                m_b.append(clef.BassClef())
            # if you're at a beat one append old measures and create new ones
            elif first_note:
                p0.append(m_s)
                p1.append(m_a)
                p2.append(m_t)
                p3.append(m_b)
                m_s = stream.Measure()
                m_a = stream.Measure()
                m_t = stream.Measure()
                m_b = stream.Measure()

            # if the current key sig is not the one that was found in the chord change to new one. self for time sig
            if self.current_ks != key_sig:
                m_s.append(key_sig)
                m_a.append(key_sig)
                m_t.append(key_sig)
                m_b.append(key_sig)
                self.current_ks = key_sig

            if self.current_ts is None or (self.current_ts.denominator != time_sig.denominator and self.current_ts.numerator != time_sig.numerator):
                m_s.append(time_sig)
                m_a.append(time_sig)
                m_t.append(time_sig)
                m_b.append(time_sig)
                self.current_ts = time_sig

            # duration is always a quaver, could be changed later
            duration = 0.5

            soprano_note, self.last_s_note_midi = self._make_element_from_midi(soprano_note_midi, self.last_s_note, duration, self.last_s_note_midi, is_fermata)
            alto_note, self.last_a_note_midi = self._make_element_from_midi(alto_note_midi, self.last_a_note, duration, self.last_a_note_midi, is_fermata)
            tenor_note, self.last_t_note_midi = self._make_element_from_midi(tenor_note_midi, self.last_t_note, duration, self.last_t_note_midi, is_fermata)
            bass_note, self.last_b_note_midi = self._make_element_from_midi(bass_note_midi, self.last_b_note, duration, self.last_b_note_midi, is_fermata)

            # for debugging and visualization purposes
            if not piano_reduction:
                soprano_note.lyric = str(counter)
                alto_note.lyric = "0"
            counter += 1

            # append notes to measures
            m_s.append(soprano_note)
            m_a.append(alto_note)
            m_t.append(tenor_note)
            m_b.append(bass_note)

            # save last notes to compare later
            self.last_s_note = soprano_note
            self.last_a_note = alto_note
            self.last_t_note = tenor_note
            self.last_b_note = bass_note

        # when finished finally append last measures
        p0.append(m_s)
        p1.append(m_a)
        p2.append(m_t)
        p3.append(m_b)

        # make it beautiful
        if ties:
            p0 = p0.stripTies(retainContainers=True)
            p1 = p1.stripTies(retainContainers=True)
            p2 = p2.stripTies(retainContainers=True)
            p3 = p3.stripTies(retainContainers=True)
        s.append(p0)
        s.append(p1)
        s.append(p2)
        s.append(p3)

        if piano_reduction:
            pr = stream.Score()
            pr.append(copy.deepcopy(p0))
            pr.append(copy.deepcopy(p1))
            prPart = pr.partsToVoices(voiceAllocation=2).parts[0]
            prPart.id = "piano_right"

            pl = stream.Score()
            pl.append(copy.deepcopy(p2))
            pl.append(copy.deepcopy(p3))
            plPart = pl.partsToVoices(voiceAllocation=2).parts[0]
            plPart.id = "piano_left"

            s.append(prPart)
            s.append(plPart)

            staff_group_piano = layout.StaffGroup([prPart, plPart], name = 'Reduction', abbreviation = 'Rd.', symbol = 'brace')
            s.insert(0, staff_group_piano)

        staff_group = layout.StaffGroup([p0, p1, p2, p3], symbol = 'bracket')
        staff_group.barTogether = 'yes'
        s.insert(0, staff_group)

        s.finalBarline = bar.Barline('final')

        # TODO: doesn't work!
        #p0 = self._simplify_enharmonics_in_stream(p0)
        #p1 = self._simplify_enharmonics_in_stream(p1)
        #p2 = self._simplify_enharmonics_in_stream(p2)
        #p3 = self._simplify_enharmonics_in_stream(p3)

        #s.show("text") # debugging

        if show:
            s.show()

        return s

    def _make_element_from_midi(self, midi, last_note, duration, last_note_midi, hasNoteFermata):
        if midi == 90: # if rest is encoded
            element = note.Rest()
            if hasNoteFermata:
                element.expressions.append(expressions.Fermata())
        elif midi == 91:   # if continue is encoded
            # first beat (no note before)
            if last_note_midi is None:
                print("Warning: continue on first beat")
                midi = 90
                element = note.Rest()
                if hasNoteFermata:
                    element.expressions.append(expressions.Fermata())

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
            if hasNoteFermata:
                element.expressions.append(expressions.Fermata())
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
        elif ts_bits[3] == 1:
            return  meter.TimeSignature('12/8')
        else:
            raise Exception("no valid TS found")

    def _simplify_enharmonics_in_stream(self, st):
        print(st)
        pitch_list_for_accidental_simplify = []
        for measure in st:
            print(measure)
            for el in measure:
                print(el)
                if hasattr(el, 'pitch'):
                    pitch_list_for_accidental_simplify.append(el.pitch)

        print(pitch_list_for_accidental_simplify)

        es = analysis.enharmonics.EnharmonicSimplifier(pitch_list_for_accidental_simplify)
        print(es.bestPitches())
        r = es.bestPitches()

        print(es.bestPitches())

        elcount = 0
        for measure in st:
            for el in measure:
                if hasattr(el, 'pitch'):
                    el.pitch = pitch_list_for_accidental_simplify[elcount]
                    elcount += 1

#s = Synthesizer()
#fileI = np.load("/Users/alexanderleemhuis/Informatik/PY/PRJ/bach git/BachNet/chordDataSQ/debugo.csv")
#fileI = np.genfromtxt("/Users/alexanderleemhuis/Informatik/PY/PRJ/bach git/BachNet/data/bwv101.7.mxl_2_-3debugi.csv")
#s.synthesizeFromArray(fileI.transpose())