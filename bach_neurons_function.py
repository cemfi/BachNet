import torch

import numpy as np

from music21 import harmony, interval, analysis, roman, chord
import matplotlib.pyplot as plt


class NeuronsFunctionComparator:
    def __init__(self, numberNeurons):
        self.counter = 0
        self.all_neurons = np.empty((numberNeurons, 0))
        self.all_template = np.empty((0,0))

    def analyze(self, score, maxChords):   # until now: everything major mode!
        activation_template = self._analyse_function(score, maxChords)
        self.all_template = np.append(self.all_template, activation_template)
        return activation_template

    def store_neurons(self, neurons):   # until now: everything major mode!
        self.all_neurons = np.append(self.all_neurons, neurons.detach().numpy(), axis=1)

    def find_correlations(self):
        neuronCounter = 0
        for neuron in self.all_neurons:
            neuron -= np.min(neuron)   # shift to min zero
            neuron /= np.max(neuron)   # scale to max one

            #print(neuron.shape)
            #print(self.all_template.shape)

            cor = self._correlation(neuron, self.all_template)
            if cor > 0.1:
                print("--")
                print(neuronCounter)
                print(cor)
            neuronCounter += 1

    def _analyse_function(self, score, maxChords):
        activationTemplate = np.zeros(maxChords)

        ks1 = score.flat.keySignature.asKey()
        ks2 = score.flat.keySignature.asKey("minor")

        self.counter *= 1
        cr = analysis.reduceChords.ChordReducer()
        score = score.chordify()
        newS = cr.reduceMeasureToNChords(score, maxChords, weightAlgorithm=cr.qlbsmpConsonance, trimBelow=0.3).flat
        # newS.show('text')

        dom_candidate_positions = []
        dom_candidate_root = None
        for semiquarter in range(maxChords):
            for e in newS.getElementsByOffset(semiquarter / 2):
                if type(e) == chord.Chord:
                    e.removeRedundantPitchNames(inPlace=True)
                    e.simplifyEnharmonics(inPlace=True)
                    print(e)
                    print(harmony.chordSymbolFigureFromChord(e, True))
                    if len(dom_candidate_positions) > 0:
                        new_root = e.root()
                        new_root.octave = None

                        does_5_1 = interval.notesToChromatic(last_root, new_root).semitones == 5 or interval.notesToChromatic(last_root, new_root).semitones == -7
                        does_5_6 = interval.notesToChromatic(last_root, new_root).semitones == 2 or interval.notesToChromatic(last_root, new_root).semitones == 1

                        if new_root == dom_candidate_root:
                            dom_candidate_positions.append(semiquarter)
                        elif does_5_1 or does_5_6:
                            print("Dominante eingelöst")
                            for dom_pos in dom_candidate_positions:
                                activationTemplate[dom_pos] = 1
                            dom_candidate_positions = []
                            dom_candidate_root = None
                        else:
                            dom_candidate_positions = []
                            dom_candidate_root = None

                    f = roman.romanNumeralFromChord(e, ks1)
                    if "V" in f.figure:
                        activationTemplate[semiquarter] = 1
                    f = roman.romanNumeralFromChord(e, ks2)
                    if "V" in f.figure:
                        activationTemplate[semiquarter] = 1
                    if "dominant seventh" in e.commonName:
                        activationTemplate[semiquarter] = 1
                    if e.canBeDominantV():
                        dom_candidate_positions.append(semiquarter)
                        last_root = e.root()
                        last_root.octave = None

            # no element found (doesn't fire on first beat because of timesig, ..
            if len(newS.getElementsByOffset(semiquarter / 2)) == 0:
                activationTemplate[semiquarter] = activationTemplate[semiquarter - 1]
                if len(dom_candidate_positions) > 0:
                    dom_candidate_positions.append(semiquarter)
        if np.sum(activationTemplate) == 0:
            print("nothing matches criterion")
        return activationTemplate

    def _correlation(self, neuron, template):
        corMatr = np.corrcoef(neuron, template)
        return corMatr[0, 1]
