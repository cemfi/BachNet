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
            # bwv248.64-6.mxl #invisible break

            if c <= (len(cb) * (valPercent / 100)):
                self.datafolderPrefix = self.pathFolder + "/valid/"
            else:
                self.datafolderPrefix = self.pathFolder + "/train/"

            tqdm.write(title)

            dataI, dataO, skipFile, exceptionEntry = pc.convertToDataArray(x, title)
            timeSigs = pc.getTimeSigs()
            sharps = pc.getSharps()
            found16ss = pc.get16s()
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
                    data_out_to_file = np.array(dataO, copy=True)
                    data_in_to_file = np.array(dataI, copy=True)
                    # reset before transposition is applied
                    data_out_to_file[8:67, :] = np.zeros((59, len(data_out_to_file[0, :])))
                    data_out_to_file[70:129, :] = np.zeros((59, len(data_out_to_file[0, :])))
                    data_out_to_file[132:191, :] = np.zeros((59, len(data_out_to_file[0, :])))
                    data_in_to_file[194:253, :] = np.zeros((59, len(data_in_to_file[0, :])))
                    # copy moved music
                    data_out_to_file[(14 + tp):(61 + tp), :] = dataO[14:61, :]
                    data_out_to_file[(76 + tp):(123 + tp), :] = dataO[76:123, :]
                    data_out_to_file[(138 + tp):(185 + tp), :] = dataO[138:185, :]
                    data_in_to_file[(200 + tp):(247 + tp), :] = dataI[200:247, :]

                    keyShift = (tp * 7) % 12  # C -> E is tp=4 so plus 4 sharps

                    data_in_to_file[263:275,:] = np.roll(dataI[263:275,:], keyShift, axis=0)


                    fname = nameConcat + "/" + "o.npy"
                    np.save(fname, data_out_to_file)
                    fname = nameConcat + "/" + "i.npy"
                    np.save(fname, data_in_to_file)
                    # if title == "bwv119.9.mxl":     #duration bug
                    # if title == "bwv119.9.mxl":     #duration bug
                    if c == 2:  # random number for debug
                        np.savetxt(((self.pathFolder) + "/" + title + "_" + str(c) + "_" + str(tp) + "debugo.csv"), data_out_to_file, fmt='%d')
                        np.savetxt(((self.pathFolder) + "/" + title + "_" + str(c) + "_" + str(tp) + "debugi.csv"), data_in_to_file, fmt='%d')

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
            myfile.write("\n")
            myfile.write(("number of 12/8:" + str(timeSigs['12/8'])))
            myfile.write("\n")
            myfile.write("\n")
            myfile.write(("16s" + str(found16ss)))
            myfile.write("\n")
            myfile.write(("sharps" + str(sharps)))
