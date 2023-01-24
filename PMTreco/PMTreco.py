import midas.file_reader
from datetime import datetime
import numpy as np
import cygno as cy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
import os

from cygno import s3, cmd


import useful_functions as uf

class PMTreco:
    def __init__(self, run_number, path = './tmp/', tag = 'LNGS',
                 cloud = True, verbose = False,
                 ped = False):

        if not isinstance(run_number, int):
            self.run_list = run_number
        else: self.run_list = [run_number] 
        
        
        self.path    = path
        self.tag     = tag
        self.cloud   = cloud
        self.verbose = verbose
        
        
        self.cam_exp  = 0.3 ### implement the readout from midas file
        
        if not ped: ### implement auto-pedestal "detection"
            self.WFs      = self.get_waveform()
            
            self.pic_num  = self.getPicNum()
            self.WF_num   = self.getWFNum()
            self.ev_rate  = self.getEventRate()

            self.thresh   = 0.30 #mV
        
     
    def getmid(self, run):
        fname = s3.mid_file(run, tag=self.tag, cloud=self.cloud, verbose=self.verbose)
        filetmp = cmd.cache_file(fname, cachedir=self.path, verbose=self.verbose)
        return midas.file_reader.MidasFile(filetmp)
    
        
    def daq_dgz2array(self, bank, header, min_channels = 0, max_channels = 32, verbose=False):
    
        ''' Custom function to acquire waveform from a .mid/.mid.gz file. The bank must be passed
            with the relative header, interested channels could be passed. Return an ndarray with all
            the waveform divided by trigger and then by channel.
        '''

        data_offset = 0               # the offset to read the bank.data of the midas file
        number_triggers= header[0]
        number_channels= header[1]
        number_samples = header[2]

        waveforms = np.zeros([number_triggers, max_channels, number_samples])  # creation of the array to return
                                                                                # with the desired dimension    

        for itrigger in range(number_triggers):
            for ichannels in range(number_channels):

                if (min_channels<= ichannels < max_channels):  # acquisition of the selected channels
                    waveforms[itrigger][ichannels] = (bank.data[data_offset:data_offset+number_samples])

                data_offset += number_samples       # adding the number of samples to acquire the next waveform


        if verbose:
            print(waveforms, number_triggers, number_channels)

        return waveforms
    
    def get_waveform(self, outsize = 20, max_channels=32, view=False, truncate=False, verbose=False):
        """
        help function which extracts waveforms from midas file 
        with given output size (# of total events).
        """
        
        wfs = []
        for r in self.run_list:
            
            print(r)
            if(os.path.exists(self.path+'WFs_{:05d}.npy'.format(r))):
                wfs.append(np.load(self.path+'WFs_{:05d}.npy'.format(r), allow_pickle=True))
            else:
            
            
                mfile = self.getmid(r)
                count = 0
                waveform = []
                dgtz_header = []
                tttag = []
                for event in tqdm(mfile):
                    if truncate and count==outsize:
                        print("I'll stop here, bye!")
                        break
                    if event.header.is_midas_internal_event():
                        if verbose: print("Saw a special event")
                        continue
                    if event.header.event_id!=1:
                        if verbose: print("not 1", count, event.header.serial_number)
                        continue
                    bank_names = ", ".join(b.name for b in event.banks.values())
                    for bank_name, bank in event.banks.items():
                        if bank_name=='DGH0': # PMTs waveform !!! the interesting loop is this !!!
                            header = cy.daq_dgz2header(bank)
                            tot_header = np.array(bank.data)
                            
                            N  = tot_header[4]
                            i1 = np.where(tot_header == 1720)[0][0]
                            
                            reduced_head = tot_header[(i1-N):(i1)]
                            
                            # this is the way we store the data: waveform has 32 channels,
                            # but the interesting ones are 0 (aka the trigger) and 1,2,3,4 (aka the 4 PMTs)
                            wf= self.daq_dgz2array(event.banks['DIG0'], header, max_channels = max_channels)
                            count+=1
                            waveform.append(wf)
                            dgtz_header.append(tot_header)
                            tttag.append(reduced_head)


                wfs_tmp = np.asarray(waveform, dtype = object)
                dgh_tmp = np.asarray(dgtz_header, dtype = object)
                ttt_tmp = np.asarray(tttag, dtype = object)
                
                np.save(self.path+'WFs_{:05d}.npy'.format(r), wfs_tmp, allow_pickle=True, fix_imports=False)
                np.save(self.path+'DGH_{:05d}.npy'.format(r), dgh_tmp, allow_pickle=True, fix_imports=False)
                np.save(self.path+'TTT_{:05d}.npy'.format(r), ttt_tmp, allow_pickle=True, fix_imports=False)
                wfs.append(wfs_tmp)
        
        return wfs
    
    
    def getPicNum(self):
        picnums = []
        for i in range(len(self.run_list)):
            picnums.append(self.WFs[i].shape[0])
        return picnums
    
    def getWFNum(self):
        tot_evs = np.zeros(len(self.run_list))
        for j in range(len(self.run_list)):
            for i in range(self.WFs[j].shape[0]):
                tot_evs[j] = tot_evs[j] + self.WFs[j][i].shape[0]
        return tot_evs
    
    def getEventRate(self):
        return self.WF_num / (self.cam_exp+0.18) / self.pic_num
    
    # Functions that could be useful for the analysis
    
#    def setThreshold(self, th):
#        self.thresh = th
#
#
#    def getPedestal(self, pic_index, event_index, channel, run = 99999, pmax = 200):
#        return np.mean(self.WFs[run][pic_index][event_index][channel][:pmax])
#
#    def getIntegral(self, pic_index, event_index, channel, run = 99999, int_min = 200, int_max = 400):
#        ped = self.getPedestal(pic_index, event_index, channel, run = run)
#        yyyy = - (self.WFs[run][pic_index][event_index][channel]-ped)
#        return np.sum(yyyy[int_min:int_max]) * 4/3 / 50 / 4096 # momentarily hard coded
#
#
#    def plot_single_WF(self, pic_index, event_index, channels, save = False, run = 99999):
#        for ch in channels:
#            ped = self.getPedestal(pic_index, event_index, ch, run = run)
#            yyyy = (self.WFs[run][pic_index][event_index][ch]-ped)/4096 # hard coded???
#            plt.plot(np.arange(0, len(yyyy), 1), yyyy, label = 'CH{:02d}'.format(ch))
#
#        plt.xlabel('DGTZ sample')
#        plt.ylabel('Waveform [V]')
#        plt.legend()
#        if save: plt.savefig(path+'plot_WF_{0:06d}_{1:06d}_{2:06d}.jpg'.format(run, pic_index, event_index))
#        plt.show()
#
#
#
#    def getAllIntegrals(self, channel):
#        ints = np.zeros(int(np.sum(self.WF_num)))
#        count = 0
#        for r in range(len(self.run_list)):
#            for i in range(self.pic_num[r]):
#                for j in range(len(self.WFs[r][i])):
#                    ints[count] = self.getIntegral(i, j, channel, r)
#                    count = count + 1
#        return ints
#
#
#
#
#    def getSelection(self, selection = 'all', channels = [1,2,3,4],
#                     threshold      = 0.030, #mV
#                     threshold_long = 0.015, #mV
#                     min_len        =    15, #sample
#                     max_len        = 10**8, #sample
#                     cosmic_len     =    80, #sample
#                     majority       =     2  #channels
#                    ):
#
#        if len(channels)>4:
#            raise Exception("More than 4 channels for the selection.")
#
#        selected = []
#
#        for r in range(len(self.run_list)):
#            for i in range(self.pic_num[r]):
#                for j in range(len(self.WFs[r][i])):
#                    over_threshold = 0
#                    Fe     = False
#                    too_long = False
#                    Cosmic =  False
#                    positives = 0
#                    for ch in channels:
#                        ped = self.getPedestal(i, j, ch, run = r)
#                        yyy = (self.WFs[r][i][j][ch] - ped) / 4096 # hard coded???
#                        LLL = np.where(yyy < - threshold)[0]
#
#                        if np.sum(-yyy[200:400])>0: positives += 1
#
#                        #print(len(LLL))
#                        if(len(LLL)>min_len):
#                            over_threshold += 1
#                            if selection == 'Fe' or selection == 'all':
#                                Fe = True
#                                #print('ciao')
#
#                        LLL2 = np.where(yyy < -threshold_long)[0]
#                        #print(len(LLL2))
#                        if selection == 'Fe' or selection == 'all':
#                            if(len(LLL2) > max_len):
#                                #print("esatto")
#                                too_long = True
#
#                        if selection == 'cosmics' or selection == 'all':
#                            if(len(LLL2)>cosmic_len):
#                                #print("Eh si")
#                                Cosmic = True
#
#
#                    #print(over_threshold, Fe)
#                    #print(positives)
#                    #positives = 4
#                    sel = -2
#                    if(over_threshold >= majority ):
#                        sel = -1
#                        if Fe and not Cosmic and not too_long and positives == 4:
#                            #print(i, j, positives, r)
#                            sel = 0
#                        elif Cosmic and not Fe and positives == 4:
#                            sel = 1
#                        elif Cosmic and Fe and not too_long and positives == 4:
#                            sel = 2
#
#                    selected.append(np.array([i, j, sel, r]))
#
#        return np.array(selected)
#
#
#
#    def plotIntegral(self, channel, selection = '', bins = 50, p_range = [-0.05, 0.2], save = False):
#
#        ints = self.getAllIntegrals(channel)
#        if selection != '':
#            sel = self.getSelection(selection = selection)
#
#
#        plt.hist(ints, bins = bins, range = p_range, label = 'CH{:02d}'.format(channel), alpha = 0.4)
#
#        if selection == 'Fe' or selection == 'all':
#            #print(ints[np.where(sel.T[2] == 0)][:10])
#            plt.hist(ints[np.where(sel.T[2] == 0)], bins = bins,
#                     range = p_range, label = 'CH{:02d} - Fe'.format(channel), alpha = 0.4)
#
#        if selection == 'Cosmic' or selection == 'all':
#            plt.hist(ints[np.where(sel.T[2] == 1)], bins = bins,
#                     range = p_range, label = 'CH{:02d} - Cosmic'.format(channel), alpha = 0.4)
#        if selection == 'all' :
#            plt.hist(ints[np.where(sel.T[2] == 2)], bins = bins,
#                     range = p_range, label = 'CH{:02d} - Fe & Cosmic'.format(channel), alpha = 0.4)
#
#
#
#        plt.title('Histogram of the integral charge of channel CH{:02d}'.format(channel))
#        plt.xlabel('Charge [nC]')
#        plt.ylabel('Events')
#        plt.legend()
#        if save: plt.savefig('hist_int_CH{:02d}.jpg'.format(channel))
#        plt.show()
#
#
#    def plotSumPMTs(self, channels = [1,2,3,4], selection = '', bins = 50, p_range = [-0.05, 0.4], save = False):
#
#        if selection != '':
#                sel = self.getSelection(selection = selection)
#
#
#        all_ints = 0
#        for channel in channels:
#            ints = self.getAllIntegrals(channel)
#            if selection == 'Fe':
#                ints  = ints[np.where(sel.T[2] == 0)] #* gs[channel-1]
#            else:
#                raise Exception("Selection different from Fe still to be implemented")
#
#
#            all_ints = all_ints + ints
#
#
#
#        plt.hist(all_ints, bins = bins, range = p_range, alpha = 0.4)
#
#        plt.title('Histogram of the integral charge of the sum of the PMT signals')
#        plt.xlabel('Charge [nC]')
#        plt.ylabel('Events')
#        if save: plt.savefig('hist_allints.jpg'.format(channel))
#        plt.show()
#
#
#        return all_ints
#
#
#    def getFePeak(self, channels = [1,2,3,4], selection = '',
#                  bkgfile = './bkg_template_15mV.npy', bins = 50, p_range = [-0.05, 0.4], plot = False):
#
#
#
#
#        if not os.path.isfile(bkgfile):
#            raise Exception('Unable to load bkg template for this analysis. Missing file.')
#
#        if selection != '':
#                sel = self.getSelection(selection = selection)
#
#
#        all_ints = 0
#        for channel in channels:
#            ints = self.getAllIntegrals(channel)
#            if selection == 'Fe':
#                ints  = ints[np.where(sel.T[2] == 0)] #* gs[channel-1]
#            else:
#                raise Exception("Selection different from Fe still to be implemented")
#
#
#            all_ints = all_ints + ints
#
#
#        bkg_all = np.load(bkgfile)
#
#        factor = len(all_ints[np.where(all_ints > 0.4)])/len(bkg_all[np.where(bkg_all >0.4)])
#
#        y,   x = np.histogram(all_ints, bins = 50, range = [0, 0.8])
#        yb, xb = np.histogram(bkg_all , bins = 50, range = [0, 0.8])
#
#
#        def gaus(x, A, mu, sigma):
#            return A * np.exp(-(x-mu)**2 / sigma**2 /2)
#
#        pars, sigmas = curve_fit(gaus, xb[1:], y - yb *factor)
#
#        if plot:
#            plt.figure(figsize =(8,6))
#            plt.step(x[1:] , y, where = 'pre', color = 'C0', label = 'all')
#            plt.step(xb[1:], yb * factor, where = 'pre', color = 'C1', label = 'bkg template')
#            plt.step(xb[1:], y - yb * factor, where = 'pre', color = 'C2', label = 'all - bkg_template')
#            xline = np.linspace(0, 0.8, 100)
#            yline = gaus(xline, pars[0], pars[1], pars[2])
#
#            plt.plot(xline, yline, color = 'C3', label = 'gaussian fit to the diff' )
#            string = 'Peak = {0:.3f} +- {1:.3f}'.format(pars[1], pars[2])
#            plt.text(0.24, np.max(y/3), string )
#            plt.legend()
#            plt.show()
#
#        return pars[1], pars[2], np.sqrt(sigmas[1][1])
#
#
