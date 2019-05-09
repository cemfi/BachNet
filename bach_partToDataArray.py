from music21 import repeat, note, key, meter, expressions
import os
import numpy as np
from tqdm import tqdm


# structure of output data:
# noteslot = offset + note - 30 for kontra f# = 0, f3 = 59 (6 slots am rande frei für dataaug, nutzbar also: 6 - 53)

# 0 - 7         quaterposition in bar
# 8 - 67        bass note (5 octaves!, 60 vals)      - EXAMPLE: C1 im Bass -> 8 + MIDI60 - 30 = slot 38 
# 68            bass pause                                                    8 + MIDI90 - 30 = slot 68 would equal pause
# 69            bass continue                                                 8 + MIDI91 - 30 = slot 69 would equal continue
# 70 - 129      tenor note
# 130           tenor pause
# 131           tenor continue
# 132 - 191     alto note
# 192           alto pause
# 193           alto continue
# 194 - 253     soprano note
# 254           soprano pause
# 255           soprano continue
# 256           semiquater 1
# 257           semiquater 2
# 258           semiquater 3
# 259           semiquater 4
# 260           (fermata yes)
# 261           start
# 262           stop
# 263 - 274     0 sharp - 11 sharps 
# 275           4/4
# 276           3/4
# 277           3/2
# 278           12/8

class PartConverter:
    def __init__(self):
        self.skipFile = False
        self.timeSigs = {'4/4': 0, '3/4': 0, '3/2': 0, '12/8': 0}

    def convertToDataArray(self, score, title="", inputOnly=False):  # input wether a score which constists of one part only or a metadataelement from music21
        self.exceptionEntry = None

        dataI = None
        dataO = None

        try:
            if not inputOnly:
                score_parsed = score.parse()
                soprano = score_parsed['Soprano']
                # soprano = self._extend_repeats(soprano)
                alto = score_parsed['Alto']
                # alto = self._extend_repeats(alto)
                tenor = score_parsed['Tenor']
                # tenor = self._extend_repeats(tenor)
                bass = score_parsed['Bass']
                # bass = self._extend_repeats(bass)
            else:
                soprano = score  # just take the score object
        except:
            self.exceptionEntry = title + ": not all voices found"
            print("skip: ", self.exceptionEntry)
            return None, None, True, self.exceptionEntry

        # one could ignore orchestral works
        if not inputOnly:
            if score.metadata.numberOfParts > 4:
                pass

        duration = soprano.duration.quarterLength

        if not inputOnly:
            dataO = self._createEmptyZeros()  # starttoken, deleted later
        dataI = self._createEmptyZeros()  # starttoken, deleted later

        lastSNote = None  # noteElement from beat before
        sopranoNoteElement = None  # noteElement from  current beat
        if not inputOnly:
            lastANote = None
            lastTNote = None
            lastBNote = None
            altoNoteElement = None
            tenorNoteElement = None
            bassNoteElement = None

        currentKey = None  # = number of sharps (F major = 11)
        currentTimeSig = None  # 4/4, 3/4, 3/2, 12/8 = 0, 1, 2, 3

        sQasQuater = 0
        while sQasQuater <= duration - 0.25:  # semiquavers measured in quarters (so dur * 4 is also in quarters)

            hasNoteFermata = False
            if not inputOnly:
                dataOThisBeat = self._createEmptyZeros()
            dataIThisBeat = self._createEmptyZeros()

            for e in soprano.flat.getElementsByOffset(sQasQuater):
                if type(e) == key.Key or type(e) == key.KeySignature:
                    if e.sharps < 0:
                        currentKey = e.sharps + 12
                    else:
                        currentKey = e.sharps
                elif type(e) == meter.TimeSignature:
                    if e.numerator == 4 and e.denominator == 4:
                        currentTimeSig = 0
                    elif e.numerator == 3 and e.denominator == 4:
                        currentTimeSig = 1
                    elif e.numerator == 3 and e.denominator == 2:
                        currentTimeSig = 2
                    elif e.numerator == 12 and e.denominator == 8:
                        currentTimeSig = 3
                    else:
                        print(e)
                        raise Exception("Carefull: TimeSignature not bach-like")

            sopranoChanged = False  # sopranoChanged needed for tied notes later
            sopranoNoteElement = self._changeCurrentElement(sopranoNoteElement, soprano, sQasQuater)
            if not inputOnly:
                altoChanged = False
                tenorChanged = False
                bassChanged = False
                altoNoteElement = self._changeCurrentElement(altoNoteElement, alto, sQasQuater)
                tenorNoteElement = self._changeCurrentElement(tenorNoteElement, tenor, sQasQuater)
                bassNoteElement = self._changeCurrentElement(bassNoteElement, bass, sQasQuater)

            try:
                beat = soprano.beatAndMeasureFromOffset(sQasQuater)[0]
            except Exception as e:
                break  # in some files (bwv119.9.mxl?) there seems to be a duration-bug

            if beat > 8:
                raise Exception("measure longer than 8 quaters")

            if sQasQuater != 0.:  # except for fist run
                # part has changed if the last element and the current element are at different position
                if lastSNote.getOffsetInHierarchy(soprano) != sopranoNoteElement.getOffsetInHierarchy(soprano):
                    sopranoChanged = True
                if not inputOnly:
                    if lastANote.getOffsetInHierarchy(alto) != altoNoteElement.getOffsetInHierarchy(alto):
                        altoChanged = True
                    if lastTNote.getOffsetInHierarchy(tenor) != tenorNoteElement.getOffsetInHierarchy(tenor):
                        tenorChanged = True
                    if lastBNote.getOffsetInHierarchy(bass) != bassNoteElement.getOffsetInHierarchy(bass):
                        bassChanged = True
            else:
                sopranoChanged = True
                if not inputOnly:
                    altoChanged = True
                    tenorChanged = True
                    bassChanged = True

            # get Note at quater in stream (title for log)
            sopranoNote, self.skipFile = self._getMidiNote(soprano, sQasQuater, title)
            if not inputOnly:
                altoNote, self.skipFile = self._getMidiNote(alto, sQasQuater, title)
                tenorNote, self.skipFile = self._getMidiNote(tenor, sQasQuater, title)
                bassNote, self.skipFile = self._getMidiNote(bass, sQasQuater, title)

            # midi 90 = break, midi 91 = continue
            if not sopranoChanged:
                sopranoNote = 91
            if not inputOnly:
                if not bassChanged:
                    bassNote = 91
                if not tenorChanged:
                    tenorNote = 91
                if not altoChanged:
                    altoNote = 91

            for e in sopranoNoteElement.expressions:
                if type(e) == expressions.Fermata:
                    hasNoteFermata = True

            if not inputOnly:
                dataOThisBeat[8 + bassNote - 30] = 1  # where in array notes start + note - 30
                dataOThisBeat[70 + tenorNote - 30] = 1  # (transposing 30 notes -> has to be redone when synthesizing!)
                dataOThisBeat[132 + altoNote - 30] = 1
                dataO = np.append(dataO, dataOThisBeat, axis=1)

            dataIThisBeat[0 + int(beat // 1) - 1] = 1  # 1, 2, 3, 4, 5, 6, 7, 8
            dataIThisBeat[256] = ((beat - int(beat)) == 0)  # every start of 4 semiquavers
            dataIThisBeat[257] = ((beat - int(beat)) == 0.25)
            dataIThisBeat[258] = ((beat - int(beat)) == 0.5)
            dataIThisBeat[259] = ((beat - int(beat)) == 0.75)
            dataIThisBeat[260] = hasNoteFermata


            # soprano notes
            dataIThisBeat[194 + sopranoNote - 30] = 1
            self._applyTimeSig(dataIThisBeat, currentTimeSig)
            self._applyKey(dataIThisBeat, currentKey)
            dataI = np.append(dataI, dataIThisBeat, axis=1)

            sQasQuater += 0.5  # next quaver

            # save last note for next iteration
            lastSNote = sopranoNoteElement
            if not inputOnly:
                lastANote = altoNoteElement
                lastTNote = tenorNoteElement
                lastBNote = bassNoteElement

            if self.skipFile:
                break

        if not inputOnly:
            dataO = dataO[:, 1:]  # delete starttoken
        dataI = dataI[:, 1:]  # delete starttoken

        self._flagStartStop(dataI)  # set start/stopflags

        self.adjustTimeSig(currentTimeSig)

        if dataI is None:
            raise Exception()
        return dataI, dataO, self.skipFile, self.exceptionEntry

    def _applyTimeSig(self, data, currentTimeSig):
        if currentTimeSig != None:
            data[275 + currentTimeSig] = 1

    def _applyKey(self, data, currentKey):
        if currentKey != None:
            data[263 + currentKey] = 1

    def _changeCurrentElement(self, currentEl, stream, quater):
        #print("-------")
        thisNote = currentEl  # because if no note found, the current element stays the same
        noteAlreadyFound = False
        for e in stream.flat.getElementsByOffset(quater):
            if noteAlreadyFound and (type(e) == note.Note or type(e) == note.Rest):
                print("Divisi at quater ", quater)
                if thisNote.duration.quarterLength == 0.0:
                    print("ignore grace notes")
                    thisNote = e
            if type(e) == note.Note:
                thisNote = e
                noteAlreadyFound = True
                #print(e)
            elif type(e) == note.Rest:
                thisNote = e
                noteAlreadyFound = True
                #print(e)
        return thisNote

    def _createEmptyZeros(self):
        return np.zeros([279, 1], dtype=np.float32)

    def _getMidiNote(self, stream, quater, title):
        el = stream.flat.getElementAtOrBefore(quater, [note.Note, note.Rest])
        if type(el) == note.Note:
            sopranoNote = int(el.pitch.ps)
            return sopranoNote, False  # False -> don't skip
        elif type(el) == note.Rest:
            sopranoNote = 90
            return sopranoNote, False
        else:
            self.exceptionEntry = title + ": unknown object found: " + str(type(el))
            print("skip: ", self.exceptionEntry)
            return 0, True

    def _flagStartStop(self, data):
        stillOnFirst = True
        c = 0
        for dataStep in data.transpose():
            if c == 0:  # first step -> startflag on
                dataStep[261] = 1
            elif stillOnFirst and dataStep[194 + 91 - 30] == 1:  # if stillonfirst and voice sontinues -> startflag on
                dataStep[261] = 1
            else:
                stillOnFirst = False
            c += 1
        c = 0
        encounteredNoteFromEnd = False  # 71(! here) 91(False) 91(False) 91(False)   <-
        for dataStep in data.transpose()[::-1]:  # starting at the end
            if c == 0:
                dataStep[262] = 1  # last element -> stopflag on
            if not encounteredNoteFromEnd:  # only if before or at last note (not 91)
                if dataStep[194 + 91 - 30] == 1:  # if continue -> stopflag on
                    dataStep[262] = 1
                else:
                    encounteredNoteFromEnd = True  # if not continue: noteFound! on, but now we're on a note
                    dataStep[262] = 1
            c += 1

    def getTimeSigs(self):
        return self.timeSigs

    def adjustTimeSig(self, ts):
        if ts == 0:
            self.timeSigs['4/4'] += 1
        if ts == 1:
            self.timeSigs['3/4'] += 1
        if ts == 2:
            self.timeSigs['3/2'] += 1
        if ts == 3:
            self.timeSigs['12/8'] += 1

    def _extend_repeats(self, score_org):
        e = repeat.Expander(score_org)
        e.repeatBarsAreCoherent()
        s2 = e.process()
        if score_org != s2:
            score_org.show('text')
            s2.show('text')
        return s2
