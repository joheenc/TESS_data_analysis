import phasma2 as phasma
import astropy.units as u
import matplotlib.pyplot as plt
import numpy
import sys
import csv

with open('data.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    tics = []
    taus = []
    periods = []
    durations = []
    disps = []
    sectors = []
    for row in readCSV:
        tics.append(row[0])
        taus.append(row[1])
        periods.append(row[2])
        durations.append(row[3])
        disps.append(row[4])
        sectors.append(row[5:])
csvfile.close()

for i in range(len(tics)):
    try:
        print("Examining TIC ID " + str(tics[i]) + "(" + str(i) + "/" + str(len(tics)) + ")")
        object_of_interest = phasma.Tess(int(tics[i]), float(periods[i]) * u.day,
             float(durations[i])*u.hour, float(taus[i]), sectors[i])
        phase, flux, flux_err = (object_of_interest.phase,
                                 object_of_interest.flux,
                                 object_of_interest.flux_err)

        plt.figure()
        plt.scatter(phase, flux, alpha=0.7, s=1, color='gray')
        # plt.errorbar(phase, flux, flux_err, fmt='o', alpha=0.5, color='black')
        savepath = "curves_gray/" + str(tics[i]) + ".png"
        print("saving to " + savepath + "\n")
        plt.savefig(savepath)
    except Exception as e:
        print(e)
