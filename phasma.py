from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import os
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.utils.data import download_file
import astropy.units as u

import urllib.request
from bs4 import BeautifulSoup
import requests
import subprocess
from operator import itemgetter
from scipy.interpolate import interp1d
import scipy.stats as stats
from scipy.special import erfcinv
import scipy.optimize as optimize


__all__ = ["Tess", "Kepler"]


class Phasecurve(object):
    """
    Returns a phase curve detrended by phasma.
    Parent class for Tess and Kepler.
    """
    @u.quantity_input(period=u.day, transit_duration=u.hr)
    def __init__(self, period, transit_duration, transit_epoch,
                 transit_duration_buff=1.0, remove_fits=False,
                 plot_clean_lc=False, plot_raw_lc=False, transit_at_0=True,
                 cleaning_window=False, save=True, filename=False,
                 mask_primary=False, mask_secondary=False, nphasebins=500,
                 offset_correction=False, medianbin=False):
        """

        Parameters
        ----------
        period : `~astropy.units.Quantity`
            Orbital period
        transit_duration : `~astropy.units.Quantity`
            Duration of the transit, i.e. t_14
        transit_epoch : float or int
            Time of transit (defined different for Kepler and TESS, see their
            class docstrings for clarification)
        transit_duration_buff : float or int, optional
            Coefficient on the transit duration, e.g. for a 10% uncertainty
            on the transit duration, you would want to set
            transit_duration_buff = 1.1. If you want to double the
            transit_duration, set transit_duration_buff = 2, etc.
            Default is 1.0.
        remove_fits : bool, optional
            Set to True if you want to remove the downloaded raw light curve
            fits files. This is recommended to save disk space if you don't
            plan on running phasma multiple times for the same object.
            Default is False.
        plot_clean_lc : bool, optional
            Plots the light curve with outliers removed. Default is False.
        plot_raw_lc : bool, optional
            Plots the raw light curve for all quarters provided.
        transit_at_0 : bool, optional
            Set to False to define the phase such that the primary transit
            occurs at phase = +/- 0.5. Default is such that the primary occurs
            at phase = 0.0.
        cleaning_window : bool, float, or int, optional
            For defining a custom moving median window (in units of days) for
            outlier removal. The default is either one hundreth of the transit
            duration or 10 * cadence, whichever is larger.
        save : bool, optional
            Set to False to prevent from automatically saving the phase curve
            to a file. Default is True.
        filename : str, optional
            The path for saving the phase curve to. Default is the current
            directory with the filename "phasecurve.csv"
        mask_primary : bool, optional
            Set to True to remove the transit from the analysis. Note: values
            surrounding the transit in the final phase curvewill be affected
            by the moving median. Default is False.
        mask_secondary : bool, optional
            Set to False to keep the secondary eclipse in the analysis.
            Note: values surrounding the secondary eclipse in the final
            phase curve will be affected by the moving median.
        nphasebins : int or None, optional
            Number of data points in the final binned phase curve.
            Set to None to return all data points (excluding outliers).
            Default is 500.
        offset_correction : bool, optional
            NEEDs WORK. Finds the DC offsets which minimizes the cost function
            defined in Jansen & Kipping 2018 (see Figure 5 and Section
            2.7 for a more detailed explanation). Default is False.
        """
        self.period = period
        self.transit_duration = transit_duration
        self.transit_epoch = transit_epoch
        self.transit_duration_buff = transit_duration_buff
        self.remove_fits = remove_fits
        self.plot_clean_lc = plot_clean_lc
        self.plot_raw_lc = plot_raw_lc
        self.transit_at_0 = transit_at_0
        self.cleaning_window = cleaning_window
        self.mask_primary = mask_primary
        self.mask_secondary = mask_secondary
        self.nphasebins = nphasebins
        self.offset_correction = offset_correction
        self.medianbin = medianbin

    def write(self, directory=None, filename=False):
        """Writes the phase curve to a csv file."""
        if not filename:
            filename = directory + '/phasecurve.csv'

        print("Writing the phase curve to " + filename + " ...")
        with open(filename, 'w') as w:
            for i, j, m in zip(self.phase,
                               self.flux,
                               self.flux_err):
                w.write(str(i) + ',' +
                        str(j) + ',' +
                        str(m) + '\n')
        return

    def plot(self, show=True, save=False, file_format='png', bin=False,
             alpha=0.5):
        """Plots the phase curve. NEEDS REVIEW"""
        phase = self.phase
        flux = self.flux
        flux_err = self.flux_err

        if bin:
            phase, flux, flux_err = _bin(self.nphasebins, phase,
                                         flux, flux_err)

        plt.figure(figsize=(16, 5))
        plt.errorbar(phase, flux, yerr=flux_err,
                     fmt='o', alpha=alpha, color='black')
        plt.ylabel('ppm')
        plt.xlabel('phase')
        plt.xlim(-0.5, 0.5)

        if show:
            plt.show()

        if save:
            plt.savefig('phasma_phasecurves/' + self.tic + '.csv')

        return

    def _plot_raw_lc(self, show=True, save=False, file_format='png'):
        """Plots the raw light curve. (NEEDS REVIEW)"""
        plt.figure(figsize=(16, 5))
        plt.scatter(self._raw_time, self._raw_flux,
                    color='black')
        if show:
            plt.show()

        if save:
            plt.savefig('rawlc_' + self.tic + '.' + file_format)

    def _locate_gaps(self, t):
        """Gets the indices where the significant gaps start and end.
        Significant gap is defined to be > 0.1 * period"""
        true_gap_starts = []
        true_gap_ends = []
        for i in range(len(t)-1):
            if t[i+1] - t[i] > 0.1 * self.period.value:
                true_gap_starts += [i]
                true_gap_ends += [i+1]

        return true_gap_starts, true_gap_ends

    def _split_lc_at_gap(self, true_gap_starts, true_gap_ends):
        """Splits the light curve into separate lists where significant
        gaps occur"""
        t = self._raw_time
        flux = self._raw_flux
        flux_err = self._raw_flux_err
        period = self.period.to(u.day).value
        cadence = self.cadence.to(u.day).value

        if len(true_gap_starts) == 0:
            print("No data gaps to split at, continuing...")
            split_time = t
            split_flux = flux
            split_flux_err = flux_err

        else:

            split_time = []
            split_flux = []
            split_flux_err = []

            baseline_too_short = 0

            if len(true_gap_starts) == 1:
                if not len(t[:true_gap_starts[0] + 1]) < 2 * period / cadence:
                    split_time += [list(t[:true_gap_starts[0] + 1])]
                    split_flux += [list(flux[:true_gap_starts[0] + 1])]
                    split_flux_err += [list(flux_err[:true_gap_starts[0] + 1])]

                else:
                    baseline_too_short += 1

                if not len(t[true_gap_ends[0]:]) < 2 * period / cadence:
                    split_time += [list(t[true_gap_ends[0]:])]
                    split_flux += [list(flux[true_gap_ends[0]:])]
                    split_flux_err += [list(flux_err[true_gap_ends[0]:])]

                else:
                    baseline_too_short += 1

            elif true_gap_starts[0] != 0:

                split_time = []
                split_flux = []
                split_flux_err = []

                if not len(t[:true_gap_starts[0] + 1]) < 2 * period / cadence:
                    split_time += [list(t[:true_gap_starts[0] + 1])]
                    split_flux += [list(flux[:true_gap_starts[0] + 1])]
                    split_flux_err += list([flux_err[:true_gap_starts[0] + 1]])
                else:
                    baseline_too_short += 1

                for i in range(len(true_gap_starts)-1):
                    if not len(t[true_gap_ends[i]:true_gap_starts[i+1]]
                               ) < 2 * period / cadence:
                        split_time += [list(t[
                            true_gap_ends[i]:true_gap_starts[i+1]])]
                        split_flux += [list(flux[
                            true_gap_ends[i]:true_gap_starts[i+1]])]
                        split_flux_err += [list(flux_err[
                            true_gap_ends[i]:true_gap_starts[i+1]])]
                    else:
                        baseline_too_short += 1

                if not len(t[true_gap_ends[-1]:]) < 2 * period / cadence:
                    split_time += [list(t[true_gap_ends[-1]:])]
                    split_flux += [list(flux[true_gap_ends[-1]:])]
                    split_flux_err += [list(flux_err[true_gap_ends[-1]:])]
                else:
                    baseline_too_short += 1

            else:
                split_time = []
                split_flux = []
                split_flux_err = []
                for i in range(len(true_gap_starts) - 1):
                    if not len(t[true_gap_ends[i]:true_gap_starts[i+1]]
                               ) < 2 * period / cadence:
                        split_time += [list(t[
                            true_gap_ends[i]:true_gap_starts[i+1]])]
                        split_flux += [list(flux[
                            true_gap_ends[i]:true_gap_starts[i+1]])]
                        split_flux += [list(flux_err[
                            true_gap_ends[i]:true_gap_starts[i+1]])]
                    else:
                        baseline_too_short += 1

            if baseline_too_short > 0:
                print("After splitting the data at significant gaps, "
                      "the baseline was shorter than twice the length of "
                      "the period on " + str(baseline_too_short) +
                      " occasions.")

        return split_time, split_flux, split_flux_err

    def _phase(self, time):
        """Returns the phases corresponding to a given time array."""
        return ((time - self.transit_epoch) / self.period.value) % 1 - 0.5

    def _fold(self, phase, time, flux, flux_err):
        """Folds flux on the period given and returns the
        resulting phase curve sorted by the phase"""
        transverse_data = np.vstack([phase, time, flux, flux_err]).T

        # sort the data by phase
        (sorted_phase,
         sorted_t,
         sorted_flux,
         sorted_flux_err) = np.array(sorted(transverse_data,
                                            key=itemgetter(0))).T

        return sorted_phase, sorted_t, sorted_flux, sorted_flux_err

    def _clean(self, time, flux, flux_err):
        """Applies a moving median function and discards outliers
        defined by flux > 1.4826 * MAD * sqrt(2) * erfcinv(1/n)"""
        transit_phase = (self.transit_duration *
                         self.transit_duration_buff /
                         self.period / 2).to(u.Unit('')).value

        if not self.cleaning_window:
            trans_window = (self.transit_duration_buff *
                            self.transit_duration / 100
                            ).to(u.day).value
            min_window = 11 * self.cadence.to(u.day).value
            self.cleaning_window = (np.maximum(trans_window, min_window) /
                                    self.period.to(u.day).value)

        (trimmed_t,
         trimmed_flux,
         trimmed_flux_err,
         moving_med_func) = _moving_median(time, flux, flux_err,
                                           self.cleaning_window)

        # get the residuals
        res = ((trimmed_flux - moving_med_func(trimmed_t)) /
               trimmed_flux_err)

        # remove outliers
        outlier_cutoff = (1.4826 * stats.median_absolute_deviation(res) *
                          np.sqrt(2) * erfcinv(1 / len(res)))
        outliers = abs(res) > outlier_cutoff

        trimmed_t = trimmed_t[~outliers]
        trimmed_flux = trimmed_flux[~outliers]
        trimmed_flux_err = trimmed_flux_err[~outliers]

        return trimmed_t, trimmed_flux, trimmed_flux_err

    def _wrap(self):
        """Applies the cleaning and detrending."""

        if self.plot_clean_lc or self.plot_raw_lc:
            plt.figure(figsize=(16, 5))
            plt.scatter(self._raw_time, self._raw_flux, color='black')
            if self.plot_raw_lc and not self.plot_clean_lc:
                plt.show()

        # split the data by saving to temporary fits files
        true_gap_starts, true_gap_ends = self._locate_gaps(self._raw_time)
        (split_time,
         split_flux,
         split_flux_err) = self._split_lc_at_gap(true_gap_starts,
                                                 true_gap_ends)

        if self.offset_correction:
            # create empty arrays to store data
            pji = np.zeros((len(split_time), self.nphasebins))
            tji = np.zeros((len(split_time), self.nphasebins))
            fji = np.zeros((len(split_time), self.nphasebins))
            wji = np.zeros((len(split_time), self.nphasebins))

        time_all = np.array([])
        flux_all = np.array([])
        flux_err_all = np.array([])

        # loop through data separated by gaps
        for continuous in range(len(split_time)):

            # remove outliers from the semi sectors
            (clean_t,
             clean_flux,
             clean_flux_err) = self._clean(np.array(split_time[continuous]),
                                           np.array(split_flux[continuous]),
                                           np.array(split_flux_err[continuous])
                                           )

            # second pass at outlier removal
            (clean_t,
             clean_flux,
             clean_flux_err) = self._clean(clean_t, clean_flux,
                                           clean_flux_err)

            if self.plot_clean_lc:
                plt.scatter(clean_t, clean_flux, color='red')

            # define the phase corresponding to the cleaned fluxes
            phase = self._phase(clean_t)

            transit_phase = (self.transit_duration *
                             self.transit_duration_buff /
                             self.period / 2).to(u.Unit('')).value

            # remove transit from light curve if called for
            if self.mask_primary:
                in_transit = ((phase <= -0.5 + transit_phase) +
                              (phase >= 0.5 - transit_phase))
                clean_flux[in_transit] = np.nan
                clean_flux_err[in_transit] = np.nan

            # remove secondary, if called for
            if self.mask_secondary:
                in_occultation = ((phase <= transit_phase) &
                                  (phase >= - transit_phase))
                clean_flux[in_occultation] = np.nan
                clean_flux_err[in_occultation] = np.nan

            # apply phasma to cleaned data
            (phasma_t,
             phasma_flux,
             phasma_flux_err) = _phasma_detrend(self.period.to(u.day).value,
                                                clean_t,
                                                clean_flux,
                                                clean_flux_err)

            phasma_p = self._phase(phasma_t)
            p, t, f, ferr = self._fold(phasma_p,
                                       phasma_t,
                                       phasma_flux,
                                       phasma_flux_err)

            # combine semisectors and sectors
            time_all = np.append(time_all, phasma_t)
            flux_all = np.append(flux_all, phasma_flux)
            flux_err_all = np.append(flux_err_all, phasma_flux_err)

            if self.offset_correction:

                # fold the light curve and sort into a phase curve
                phasma_p = self._phase(phasma_t)
                p, t, f, ferr = self._fold(phasma_p,
                                           phasma_t,
                                           phasma_flux,
                                           phasma_flux_err)

                # put the transit at phase = 0 if called for
                if self.transit_at_0:
                    p, f, ferr = _redefine_phase(p, f, ferr)

                # bin the phase curve
                (bin_phase,
                 bin_flux,
                 bin_flux_err) = _bin(self.nphasebins, p, f, ferr)

                # put the binned phase curve into one big array
                tji[continuous] = bin_time
                pji[continuous] = bin_phase
                fji[continuous] = bin_flux
                wji[continuous] = 1 / (bin_flux_err ** 2)

        if self.plot_clean_lc:
            plt.show()

        if self.offset_correction:
            # correct for the arbitrary offset created by
            # the moving median filter
            phase, flux, flux_err = _offset_correction(pji, fji, wji)
            return phase, np.mean(tji, axis=0), flux, flux_err

        phase_all = self._phase(time_all)

        phase, time, flux, flux_err = self._fold(phase_all, time_all,
                                                 flux_all, flux_err_all)

        if self.transit_at_0:
            phase, flux, flux_err = _redefine_phase(phase, flux, flux_err)

        if self.nphasebins is not None:
            return _bin(self.nphasebins, phase, flux, flux_err,
                        median=self.medianbin)
        else:
            return phase, flux, flux_err


class Tess(Phasecurve):
    """
    Returns the phase curve of an object of interest observed by TESS.
    """
    @u.quantity_input(period=u.day, transit_duration=u.hr)
    def __init__(self, tic, period, transit_duration, transit_epoch, sectors,
                 remove_curl=False, transit_duration_buff=1.0,
                 remove_fits=False, plot_clean_lc=False, plot_raw_lc=False,
                 transit_at_0=True, cleaning_window=False, save=True,
                 filename=False, mask_primary=False, mask_secondary=False,
                 nphasebins=500, offset_correction=False, medianbin=False):
        """

        Parameters
        ----------
        tic : int or str
            The TESS Input Catalog (TIC) ID of the object
        period : `~astropy.units.Quantity`
            Orbital period
        transit_duration : `~astropy.units.Quantity`
            Duration of the transit, i.e. t_14
        transit_epoch : float or int
            Time of transit in BJD - 2457000
        sectors : list or tuple
            Sector(s) of interest
        remove_curl : bool, optional
            Set to True to delete the curl files downloaded from MAST.
            This is recommended to save disk space if you don't plan on
            running phasma multiple times for the same object.
            Default is False.
        transit_duration_buff : float or int, optional
            Coefficient on the transit duration, e.g. for a 10% uncertainty
            on the transit duration, you would want to set
            transit_duration_buff = 1.1. If you want to double the
            transit_duration, set transit_duration_buff = 2, etc.
            Default is 1.0.
        remove_fits : bool, optional
            Set to True if you want to remove the downloaded raw light curve
            fits files. This is recommended to save disk space if you don't
            plan on running phasma multiple times for the same object.
            Default is False.
        plot_clean_lc : bool, optional
            Plots the light curve with outliers removed. Default is False.
        plot_raw_lc : bool, optional
            Plots the raw light curve for all quarters provided.
        transit_at_0 : bool, optional
            Set to False to define the phase such that the primary transit
            occurs at phase = +/- 0.5. Default is such that the primary occurs
            at phase = 0.0.
        cleaning_window : bool, float, or int, optional
            For defining a custom moving median window (in units of days) for
            outlier removal. The default is either one hundreth of the transit
            duration or 10 * cadence, whichever is larger.
        save : bool, optional
            Set to False to prevent from automatically saving the phase curve
            to a file. Default is True.
        filename : str, optional
            The path for saving the phase curve to. Default is the current
            directory with the filename "phasecurve.csv"
        mask_primary : bool, optional
            Set to True to remove the transit from the analysis. Note: values
            surrounding the transit in the final phase curvewill be affected
            by the moving median. Default is False.
        mask_secondary : bool, optional
            Set to False to keep the secondary eclipse in the analysis.
            Note: values surrounding the secondary eclipse in the final
            phase curve will be affected by the moving median.
        nphasebins : int or None, optional
            Number of data points in the final binned phase curve.
            Set to None to return all data points (excluding outliers).
            Default is 500.
        offset_correction : bool, optional
            (NEEDS REVIEW)
            Finds the DC offsets which minimizes the cost function
            defined in Jansen & Kipping 2018 (see Figure 5 and Section
            2.7 for a more detailed explanation). Default is False.
        """
        super().__init__(period, transit_duration, transit_epoch,
                         transit_duration_buff=transit_duration_buff,
                         remove_fits=remove_fits, plot_clean_lc=plot_clean_lc,
                         plot_raw_lc=plot_raw_lc, transit_at_0=transit_at_0,
                         cleaning_window=cleaning_window, save=save,
                         filename=filename, mask_primary=mask_primary,
                         mask_secondary=mask_secondary, nphasebins=nphasebins,
                         offset_correction=offset_correction,
                         medianbin=medianbin)

        # make a directory for this target if it doesn't aready exist
        self.tic_dir = 'fitsfiles'
        if not os.path.exists(self.tic_dir):
            os.makedirs(self.tic_dir)

        self.tic = str(tic)
        self.sectors = sectors
        self.remove_curl = remove_curl

        (self._raw_time,
         self._raw_flux,
         self._raw_flux_err) = self._get_raw_lightcurve()

        # self.cadence = stats.mode(np.diff(self._raw_time)).mode[0] * u.day
        #
        # (self.phase,
        #  self.flux,
        #  self.flux_err) = self._wrap()

        if save:
            self.write(directory="phasecurves_csv/", filename="phasecurves_csv/"+self.tic+'.csv')

    def _get_raw_lightcurve(self):
        """Downloads the TESS light curves from MAST and puts it all
        into one big array"""

        self.actual_sectors = []

        time = np.array([])
        flux = np.array([])
        flux_err = np.array([])
        for sector in self.sectors:

            for file in os.listdir(self.tic_dir+'/'):
                try:
                    if str(int(file[24:40])) == self.tic:
                        fits_file = file
                except:
                    continue
                        # print(fits_file)
                # # download the curl file for each sector if not in directory
                # curl_sh_path = '../curlscripts/tesscurl_sector_' + str(sector) + '_lc.sh'
                # if not os.path.isfile(curl_sh_path):
                #     print("Downloading the light curve curl file for sector " +
                #           str(sector) + "...")
                #     urllib.request.urlretrieve('https://archive.stsci.edu/' +
                #                                'missions/tess/download_' +
                #                                'scripts/sector/tesscurl_' +
                #                                'sector_' + str(sector) +
                #                                '_lc.sh', curl_sh_path)
                #
                # with open(curl_sh_path) as curl_sh:
                #     array_of_curls = np.array(curl_sh.read().splitlines())
                #
                #     # search for this toi's curl
                #     toi_curls = [curl for curl in array_of_curls
                #                  if self.tic in curl]
                #
                #     # download the fits files if not in directory
                #     mast_url = ('https://mast.stsci.edu/api/v0.1/Download/' +
                #                 'file/?uri=mast:TESS/product/')
                #     for curl in toi_curls:
                #         fits_file = curl[16:71]
                #         if not os.path.isfile(self.tic_dir + '/' + fits_file):
                #             print('Downloading the fits files for TIC ' +
                #                   self.tic + " in sector " + str(sector) +
                #                   "... ")
                #             urllib.request.urlretrieve(mast_url + fits_file,
                #                                        self.tic_dir + '/' +
                #                                        fits_file)
                #
                # # delete the curl files to save space
                # if self.remove_curl:
                #     os.remove(curl_sh_path)

                # unpack the fits file
            raw_time, raw_flux, raw_flux_err = _unpack_fits(self.tic_dir +
                                                            '/' +
                                                            fits_file,
                                                            'QUALITY')
            print(self.tic_dir + '/' + fits_file)
            time = np.append(time, raw_time)
            flux = np.append(flux, raw_flux)
            flux_err = np.append(flux_err, raw_flux_err)

            # delete the fits file to save space
            if self.remove_fits:
                os.remove(fits_file)

            # FIX
            self.actual_sectors += [sector]

        return time, flux, flux_err


class Kepler(Phasecurve):
    @u.quantity_input(period=u.day, transit_duration=u.hr)
    def __init__(self, kic, period, transit_duration, transit_epoch,
                 cadence='lc', transit_duration_buff=1.0,
                 remove_fits=False, plot_clean_lc=False,
                 plot_raw_lc=False, transit_at_0=True,
                 cleaning_window=False, save=True, filename=False,
                 mask_primary=False, mask_secondary=False, nphasebins=500,
                 offset_correction=False, medianbin=False):
        """
        Returns the phase curve of an object of interest observed by TESS.

        Parameters
        ----------
        period : `~astropy.units.Quantity`
            Orbital period
        transit_duration : `~astropy.units.Quantity`
            Duration of the transit, i.e. t_14
        transit_epoch : float or int
            Time of transit in BJD - 2454833
        cadence : {'lc', 'sc'}
          The temporal cadence of the data. 'lc' for long cadence (30 min),
          'sc' for short cadence (1 min). Default is 'lc'.
        transit_duration_buff : float or int, optional
            Coefficient on the transit duration, e.g. for a 10% uncertainty
            on the transit duration, you would want to set
            transit_duration_buff = 1.1. If you want to double the
            transit_duration, set transit_duration_buff = 2, etc.
            Default is 1.0.
        remove_fits : bool, optional
            Set to True if you want to remove the downloaded raw light curve
            fits files. This is recommended to save disk space if you don't
            plan on running phasma multiple times for the same object.
            Default is False.
        plot_clean_lc : bool, optional
            Plots the light curve with outliers removed. Default is False.
        plot_raw_lc : bool, optional
            Plots the raw light curve for all quarters provided.
        transit_at_0 : bool, optional
            Set to False to define the phase such that the primary transit
            occurs at phase = +/- 0.5. Default is such that the primary occurs
            at phase = 0.0.
        cleaning_window : bool, float, or int, optional
            For defining a custom moving median window (in units of days) for
            outlier removal. The default is either one hundreth of the transit
            duration or 10 * cadence, whichever is larger.
        save : bool, optional
            Set to False to prevent from automatically saving the phase curve
            to a file. Default is True.
        filename : str, optional
            The path for saving the phase curve to. Default is the current
            directory with the filename "phasecurve.csv"
        mask_primary : bool, optional
            Set to True to remove the transit from the analysis. Note: values
            surrounding the transit in the final phase curvewill be affected
            by the moving median. Default is False.
        mask_secondary : bool, optional
            Set to False to keep the secondary eclipse in the analysis.
            Note: values surrounding the secondary eclipse in the final
            phase curve will be affected by the moving median.
        nphasebins : int or None, optional
            Number of data points in the final binned phase curve.
            Set to None to return all data points (excluding outliers).
            Default is 500.
        offset_correction : bool, optional
            (NEEDS REVIEW)
            Finds the DC offsets which minimizes the cost function
            defined in Jansen & Kipping 2018 (see Figure 5 and Section
            2.7 for a more detailed explanation). Default is False.
        """
        super().__init__(period, transit_duration, transit_epoch,
                         transit_duration_buff=transit_duration_buff,
                         remove_fits=remove_fits, plot_clean_lc=plot_clean_lc,
                         plot_raw_lc=plot_raw_lc, transit_at_0=transit_at_0,
                         cleaning_window=cleaning_window, save=save,
                         filename=filename, mask_primary=mask_primary,
                         mask_secondary=mask_secondary, nphasebins=nphasebins,
                         offset_correction=offset_correction,
                         medianbin=medianbin)

        # make a directory for this target if it doesn't aready exist
        self.kic_dir = './' + str(kic)
        if not os.path.exists(self.kic_dir):
            os.makedirs(self.kic_dir)

        self.kic = str(kic)

        self.cadence_str = cadence[0] + 'l' + cadence[1]

        (self._raw_time,
         self._raw_flux,
         self._raw_flux_err) = self._get_raw_lightcurve()

        self.cadence = stats.mode(np.diff(self._raw_time)).mode[0] * u.day

        (self.phase,
         self.flux,
         self.flux_err) = self._wrap()

        if save:
            self.write(directory=self.kic_dir, filename=filename)

    def _get_raw_lightcurve(self):
        """Downloads the Kepler light curves from MAST and puts
        it all into one big array"""

        time = np.array([])
        flux = np.array([])
        flux_err = np.array([])

        mast_url = 'http://archive.stsci.edu/pub/kepler/lightcurves//'
        # correct for zero padding in KIC ID
        nzero = 9 - len(self.kic)
        kic_long = nzero * "0" + str(self.kic)
        kic_url = mast_url + kic_long[:4] + '/' + kic_long + '/'
        url_content = requests.get(kic_url).text
        soup = BeautifulSoup(url_content, 'html.parser')

        # check for 404 error
        if '404' in str(soup.find_all('title')):
            print('ERROR: The requested URL ' + kic_url +
                  ' was not found on this server')
            exit()

        fits_files = [node.get('href')
                      for node in soup.find_all('a')
                      if node.get('href').endswith('fits')
                      and self.cadence_str in node.get('href')]

        for fits_file in fits_files:

            # only download the fits files if not already in directory
            if not os.path.isfile(self.kic_dir + '/' + fits_file):
                print("Downloading the fits files " + fits_file +
                      " for KIC " + self.kic)
                urllib.request.urlretrieve(kic_url + fits_file,
                                           self.kic_dir + '/' + fits_file)

            # unpack the fits file
            raw_time, raw_flux, raw_flux_err = _unpack_fits(self.kic_dir +
                                                            '/' + fits_file,
                                                            'SAP_QUALITY')
            time = np.append(time, raw_time)
            flux = np.append(flux, raw_flux)
            flux_err = np.append(flux_err, raw_flux_err)

            # delete the fits file to save space
            if self.remove_fits:
                os.remove(fits_file)

        return time, flux, flux_err


def _unpack_fits(fits_path, quality_str):
    """Removes "bad data" and normalizes the flux by the median"""
    open_fits = fits.open(fits_path)
    fits_data = open_fits[1].data
    raw_time = fits_data.field('TIME')
    raw_flux = fits_data.field('PDCSAP_FLUX')
    raw_flux_err = fits_data.field('PDCSAP_FLUX_ERR')
    data_quality = fits_data.field(quality_str)

    # remove flagged data
    good_data = (data_quality == 0) & (~np.isnan(raw_flux))

    raw_time = raw_time[good_data]
    norm_flux = raw_flux[good_data] / np.median(raw_flux[good_data])
    norm_flux_err = raw_flux_err[good_data] / np.median(raw_flux[good_data])

    return raw_time, norm_flux, norm_flux_err


def _moving_median(x, y, y_err, window_size):
    """Returns a moving median function and data arrays equal
    in size to the moving median function"""

    moving_med_x = np.array([])  # x in middle of bins
    moving_med_y = np.array([])
    i = 0
    while x[i] + window_size <= x[-1]:
        in_window = (x >= x[i]) & (x < x[i] + window_size)
        moving_med_x = np.append(moving_med_x, np.nanmedian(x[in_window]))
        moving_med_y = np.append(moving_med_y, np.nanmedian(y[in_window]))
        i += 1

    moving_med_func = interp1d(moving_med_x, moving_med_y)

    trim = (x >= moving_med_x[0]) & (x <= moving_med_x[-1])
    return x[trim], y[trim], y_err[trim], moving_med_func


def _phasma_detrend(P, time, flux, flux_err):
    """Applies phasma (i.e. moving median with window = period)"""
    window_size = P
    (trimmed_t,
     trimmed_flux,
     trimmed_flux_err,
     moving_med_func) = _moving_median(time, flux, flux_err, window_size)
    return (trimmed_t,
            (trimmed_flux / moving_med_func(trimmed_t) - 1) * 1e6,
            trimmed_flux_err * 1e6)


def _redefine_phase(phase, flux, flux_err):
    """Manipulate arrays such that transit occurs at phase = 0"""
    first_half = phase < 0
    last_half = phase >= 0

    new_phase = np.append(phase[last_half] - 0.5,
                          phase[first_half] + 0.5)
    new_flux = np.append(flux[last_half], flux[first_half])
    new_flux_err = np.append(flux_err[last_half],
                             flux_err[first_half])

    return new_phase, new_flux, new_flux_err


def _bin(nbins, x, flux, flux_err, median=False):
    """Bin data into nbins number of bins"""
    bin_start = x[0]
    bin_end = x[-1]
    bin_edges, binsize = np.linspace(bin_start, bin_end, nbins,
                                     retstep=True)
    binned_x = np.linspace(bin_start + binsize / 2,
                               bin_end - binsize / 2,
                               nbins)
    bin_indices = np.digitize(x, bin_edges) - 1

    binned_flux = np.array([])
    binned_error = np.array([])
    for i in range(max(bin_indices) + 1):
        bin = bin_indices == i

        flux_to_bin = flux[bin]
        notnan = ~np.isnan(flux_to_bin)
        flux_to_bin = flux_to_bin[notnan]

        flux_err_to_bin = flux_err[bin][notnan]

        if len(flux_to_bin) == 1:
            binned_flux = np.append(binned_flux, flux_to_bin)
            binned_error = np.append(binned_error, flux_err_to_bin)

        elif len(flux_to_bin) > 1:
            N = len(flux_to_bin)
            weights = 1 / (flux_err_to_bin ** 2)
            V1 = np.nansum(weights)
            V2 = np.nansum(weights ** 2)

            weighted_mean = np.nansum(flux_to_bin * weights) / V1

            sample_variance = (np.nansum(weights *
                                         (flux_to_bin - weighted_mean) ** 2) /
                               (V1 - V2 / V1))
            stdev = np.sqrt(sample_variance) / np.sqrt(N - 1)

            if median:
                import robustats
                binned_flux = np.append(binned_flux,
                                        robustats.weighted_median(flux_to_bin,
                                                                  weights))
                binned_error = np.append(binned_error, stdev *
                                         np.sqrt(N * np.pi / (2 * (N - 1))))
            else:
                binned_flux = np.append(binned_flux, weighted_mean)
                binned_error = np.append(binned_error, stdev)

        else:
            binned_flux = np.append(binned_flux, np.array([np.nan]))
            binned_error = np.append(binned_error, np.array([np.nan]))

    return binned_x, binned_flux, binned_error


def _offset_correction(phases, fluxes, weights):
    """Finds the DC offsets which minimizes the cost function
    defined in Jansen & Kipping 2018 (see Figure 5 and Section
    2.7 for a more detailed explanation)"""

    def _weighted_avg(y, w):
        """
        Parameters
        ----------
        y : 2d array, data values
        w : 2d array, weights associated with y
        """
        V1 = np.nansum(w, axis=0)
        V2 = np.nansum(w ** 2, axis=0)
        mustar = np.nansum(y * w, axis=0) / V1
        sample_variance = (np.nansum(w * (y - mustar) ** 2, axis=0) /
                           (V1 - V2 / V1))

        # only take number of nonzero bin values into account
        nonzeros = np.ones(y.shape)
        nonzeros[np.isnan(y)] = 0
        stdev = (np.sqrt(sample_variance) /
                 np.sqrt(np.sum(nonzeros, axis=0) - 1))

        return mustar, stdev

    def _cost_function(offset_i, offset, i, fji, wji):
        """
        Parameters
        ----------
        offset_i :
            the (arbitrary??) offset from the weighted mean
            (i.e. the flux binned across semisectors). Each semisector
            has a unique value of offset_i shared across all points of
            phase.
        fji :
            binned flux at jth phase for the ith semisector
        wji :
            weight (i.e. 1 / std^2 ) at jth phase for the ith semisector
        """
        # scipy optimization flattens the array for some reason, so reshape it
        offset[i] = np.zeros(len(offset.T)) + offset_i

        mu_j, stdev_j = _weighted_avg(fji - offset, wji)
        chisqr_j = np.nansum((fji - offset - mu_j) ** 2 * wji, axis=0)

        return np.nansum(chisqr_j)

    theta = np.zeros(fluxes.shape)

    for row in range(len(fluxes)):
        best_row_offset = optimize.fmin(_cost_function, theta[row][0],
                                        args=(theta, row, fluxes, weights),
                                        disp=False)
        theta[row] = best_row_offset

    mean_phase = np.nanmean(phases, axis=0)
    corrected_flux, corrected_flux_err = _weighted_avg(fluxes - theta, weights)

    return mean_phase, corrected_flux, corrected_flux_err
