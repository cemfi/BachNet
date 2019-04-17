import os
import datetime

from music21 import corpus, note
import numpy as np
from tqdm import tqdm

from bach_partToDataArray import PartConverter


class DataDownloader:
    def __init__(self, pathFolder, transpositions=[0], overwrite=True):
        print("Downloader instantiated")
        self.transpositionsDataAug = transpositions
        self.overwrite = overwrite
        self.pathFolder = pathFolder
        self.exceptionLog = []

    def download(self, valPercent=5, piecesMax=100000):
        print("Now Downloading Files. Searching Corpus.")
        # cb = corpus.search('palestrina')
        cb = corpus.search('bach/bwv')
        c = 0
        cTrain = 0
        cValid = 0
        counterSkipped = 0
        print(len(cb), " files found.")
        print("Splitting Test/Val: ", round((len(cb) * ((100 - valPercent) / 100))), "/", round((len(cb) * (valPercent / 100))))  # stimmt das?
        pc = PartConverter()
        for x in tqdm(cb):
            title = x.metadata.title
            # x.show()
            # x.show("text")
            if title == "bwv248.64-6.mxl":
                continue
            # bwv248.64-6.mxl #invisible break (doesn't matter cause other instruments play)

            if c <= (len(cb) * (valPercent / 100)):
                self.datafolderPrefix = self.pathFolder + "/valid/"
            else:
                self.datafolderPrefix = self.pathFolder + "/train/"

            tqdm.write(title)

            dataI, dataO, skipFile, exceptionEntry = pc.convertToDataArray(x, title)
            timeSigs = pc.getTimeSigs()
            if exceptionEntry is not None:
                self.exceptionLog.append(exceptionEntry)
                counterSkipped += 1
                continue

            datafolderPiece = self.datafolderPrefix + x.metadata.title

            for tp in self.transpositionsDataAug:
                nameConcat = datafolderPiece + "_" + str(c) + "_" + str(tp)
                if skipFile or ((os.path.exists(nameConcat)) and not self.overwrite):
                    break
                else:
                    os.makedirs(nameConcat, exist_ok=True)
                    dataOtoFile = np.array(dataO, copy=True)
                    dataItoFile = np.array(dataI, copy=True)
                    dataOtoFile[(14 + tp):(61 + tp), :] = dataO[14:61, :]
                    dataOtoFile[(76 + tp):(123 + tp), :] = dataO[76:123, :]
                    dataOtoFile[(138 + tp):(185 + tp), :] = dataO[138:185, :]
                    dataOtoFile[(200 + tp):(247 + tp), :] = dataO[200:247, :]
                    dataItoFile[(200 + tp):(247 + tp), :] = dataI[200:247, :]
                    fname = nameConcat + "/" + "o.npy"
                    np.save(fname, dataOtoFile)
                    fname = nameConcat + "/" + "i.npy"
                    np.save(fname, dataItoFile)
                    # if title == "bwv119.9.mxl":     #duration bug
                    if c == 1:  # debug
                        np.savetxt((self.pathFolder) + "/debugo.csv", dataOtoFile, fmt='%d')
                        np.savetxt((self.pathFolder) + "/debugi.csv", dataItoFile, fmt='%d')

            if self.datafolderPrefix == self.pathFolder + "/train/":
                cTrain += 1
            elif self.datafolderPrefix == self.pathFolder + "/valid/":
                cValid += 1
            c += 1
            if c >= piecesMax:
                break
        with open((self.pathFolder + "/log.txt"), "a") as myfile:
            myfile.write(str(datetime.datetime.now()))
            myfile.write("\n")
            myfile.write("\n".join(self.exceptionLog))
            myfile.write("\n")
            myfile.write(("skipped: " + str(counterSkipped)))
            myfile.write("\n")
            myfile.write(("processed: " + str(c)))
            myfile.write("\n")
            myfile.write(("processed train: " + str(cTrain)))
            myfile.write("\n")
            myfile.write(("processed valid: " + str(cValid)))
            myfile.write("\n")
            myfile.write(("valid percentage:" + str(cValid / c)))
            myfile.write("\n")
            myfile.write(("number of 4/4:" + str(timeSigs['4/4'])))
            myfile.write("\n")
            myfile.write(("number of 3/4:" + str(timeSigs['3/4'])))
            myfile.write("\n")
            myfile.write(("number of 3/2:" + str(timeSigs['3/2'])))
