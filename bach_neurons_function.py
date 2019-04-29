import torch
import itertools
import operator

from scipy.signal import argrelextrema
import numpy as np

from music21 import harmony, key, analysis, roman, chord
import matplotlib.pyplot as plt



class NeuronsFunctionComparator:
    def __init__(self):
        self.counter = 0
        self.findings = []

    def compare(self, score, neurons, maxChords, mode):   # until now: everything major mode!
        if mode == "chord-type":
            activationTemplate = self._analyse_chord_type(score, maxChords)
        elif mode == "function":
            activationTemplate = self._analyse_function(score, maxChords)


        self._find_neuron(activationTemplate, neurons, maxChords, 5.5)#, binarize=0.9)


    def _find_neuron(self, template, neurons, maxChords, threshHold, binarize=None):
        neuronCounter = 0
        for neuron in neurons:
            neuron -= torch.min(neuron)   # shift to min zero
            neuron /= torch.max(neuron)   # scale to max one
            #print(neuron.detach().numpy())
            #print(np.sum(abs((neuron.detach().numpy()+activationTemplate))))
            #if np.sum(abs((neuron.detach().numpy()+activationTemplate)-1)) < 4:
            #    print(neuronCounter)

            #neuron = neuron * neuron

            if binarize is not None:
                neuron = (neuron > binarize).float() * 1

            neuron = neuron.detach().numpy()

            #plt.plot(neuron)
            #plt.show()

            #maxs = argrelextrema(neuron, np.greater)
            #neuron = np.zeros(maxChords)
            #for max in maxs:
            #    neuron[max] = 1

            #plt.plot(neuron)
            #plt.show()

            #step_counter = 0
            #for neuStep in neuron:
            #    if neuStep > 0.5 and template[step_counter] == 1.0:
            #        self.findings.append(neuronCounter)
                #else:
                #    print("wrong")
            #    step_counter += 1

            if np.sum(abs((neuron-template))) < threshHold:
                print(neuronCounter)
                self.findings.append(neuronCounter)
            if len(self.findings) != 0:
                print(self.most_common(self.findings))
            neuronCounter += 1

    def _analyse_chord_type(self, score, maxChords):
        activationTemplate = np.zeros(maxChords)

        self.counter *= 1
        cr = analysis.reduceChords.ChordReducer()
        score = score.chordify()
        newS = cr.reduceMeasureToNChords(score, maxChords, weightAlgorithm=cr.qlbsmpConsonance, trimBelow=0.3).flat
        # newS.show('text')

        for semiquarter in range(maxChords):
            for e in newS.getElementsByOffset(semiquarter / 2):
                if type(e) == chord.Chord:
                    # chord_type = harmony.chordSymbolFigureFromChord(e, True)[1]
                    # chord_type = e.quality
                    # if "diminished" in chord_type:
                    #    activationTemplate[semiquarter] = 1
                    if "dominant seventh" in e.commonName:
                        activationTemplate[semiquarter] = 1
            if len(newS.getElementsByOffset(semiquarter / 2)) == 0:  # no element found (doesn't fire on first beat because of timesig, ..
                activationTemplate[semiquarter] = activationTemplate[semiquarter - 1]
        return activationTemplate


    def _analyse_function(self, score, maxChords):
        activationTemplate = np.zeros(maxChords)

        ks1 = score.flat.keySignature.asKey()
        ks2 = score.flat.keySignature.asKey("minor")
        #print(ks)

        self.counter *= 1
        cr = analysis.reduceChords.ChordReducer()
        score = score.chordify()
        newS = cr.reduceMeasureToNChords(score, maxChords, weightAlgorithm=cr.qlbsmpConsonance, trimBelow=0.3).flat
        # newS.show('text')

        for semiquarter in range(maxChords):
            # print(semiquarter)
            for e in newS.getElementsByOffset(semiquarter / 2):
                if type(e) == chord.Chord:
                    f = roman.romanNumeralFromChord(e, ks1)
                    if "I6" in f:
                        activationTemplate[semiquarter] = 1
                    f = roman.romanNumeralFromChord(e, ks2)
                    if "I6" in f:
                        activationTemplate[semiquarter] = 1
            if len(newS.getElementsByOffset(
                    semiquarter / 2)) == 0:  # no element found (doesn't fire on first beat because of timesig, ..
                activationTemplate[semiquarter] = activationTemplate[semiquarter - 1]
        return activationTemplate

    def most_common(self, L):
        # get an iterable of (item, iterable) pairs
        SL = sorted((x, i) for i, x in enumerate(L))
        # print 'SL:', SL
        groups = itertools.groupby(SL, key=operator.itemgetter(0))

        # auxiliary function to get "quality" for an item
        def _auxfun(g):
            item, iterable = g
            count = 0
            min_index = len(L)
            for _, where in iterable:
                count += 1
                min_index = min(min_index, where)
            print('item %r, count %r, minind %r' % (item, count, min_index))
            return count, -min_index

        # pick the highest-count/earliest item
        return max(groups, key=_auxfun)[0]