#!/usr/bin/env python    
# -*- coding:utf-8 -*-
'''
            STEREO:
                Waldemar Jankowski - waldemar.jankowski95@gmail.com
                Rafał Bysiek - bysiekrafal@gmail.com
                Jakub Pasula - deczu22@icloud.com
            PLL:
                Sebastian Jamrożek - sebastian.jamrozek@poczta.pl
            TABLICA COS:
                Anna Latocha - anquee@gmail.com
            RDS:
                Sebastian Jamrożek
                Waldemar Jankowski
                Rafał Bysiek
                Jakub Pasula
            OPTYMALIZACJA:
                Waldemar Jankowski
                Sebastian Jamrożek
            KOMENTARZ 
                      - Tablica cosinusów jest niewykorzystywana, ponieważ działanie funkcji
                        cos z modułu math jest około 6 razy szybsze
                      - Bity z RDSu zapisuje do pliku, który jest tworzony w folderze, w którym
                        aktualnie znajduje się użytkownik w konsoli podczas uruchamiania programu.
                      - Gdyby zamiast bitów w pliku rds.txt pojawiały się 'krzaki', należy usunąć
                        plik i spróbować uruchomić program ponownie, gdyby to nie pomogło, należy
                        otworzyć plik innym notatnikiem niż systemowy (Windows), dla przykładu
                        Notepad++ lub jakim kolwiek innym środowiskiem (NETBeans, CodeBlocks itp.). 
                      
'''
import sys

import time as t
import time as time
import threading
import types
import numpy as np
import sounddevice as sd    # speaker
import os
import math
import cmath
import rtlsdr as rtlsdr
import matplotlib.pyplot as plt
import scipy.io             # save as *.mat file

from queuelib import queue
from Queue import Queue
from Queue import Empty
from time import sleep
from scipy import signal
from pynput import keyboard


# ===================================================================== #
# =========== \/ P W S Z   T A R N Ó W   S T E R E O \/ =============== #
# ===================================================================== #
class CosTable:
    numResults = 0.0
    tableRange = 0.0
    step = 0;
    table = []

    
    def __init__(self, numResults, tableRange):
        self.numResults = numResults
        self.tableRange = tableRange
        self.generateTable()
        
    def generateTable(self):
        for index in range(0, self.numResults):
            value = (self.tableRange/self.numResults)*index

            result = math.cos(value)
            self.table.append(result)

        self.step = self.numResults/self.tableRange

    def cos(self, radian):
        index = radian * self.step

        if round(index) > int(index):
            return (self.table[int(index % len(self.table))] + self.table[int(index % len(self.table))])/2
        return self.table[int(index % len(self.table))]

ct = CosTable(5000, 2*math.pi)
# ===================================================================== #
# ==================== /\ P W S Z   T A R N Ó W /\ ==================== #
# ===================================================================== #

# stereo, radio decoder
class pyFM:
    # fc - carrier frequency in MHz
    def __init__(self, fc, verbose=False):
        
        self.fc = fc
        self.verbose = verbose
        self.rfFs = 256000
        self.hybridFs = 256000
        self.audioFs = self.hybridFs/8  # 32000 kHz
        self.demodLastVal = np.array([0])
        self.channels = 2
        self.rds = ""
        self.pilotFreq = 19000
        self.tapsBP = signal.firwin(199, [self.pilotFreq - 2000, self.pilotFreq + 2000], nyq=self.hybridFs * 0.5, pass_zero=False, window='hamming', scale=False)
        # N=199, fc=16870 Hz -> first notch in 19000 Hz
        self.tapsLP = signal.firwin(199, self.pilotFreq-2130, nyq=self.hybridFs * 0.5, pass_zero=True, window='hamming', scale=False) #Fir Filrer desingne w/ Window methon (Window=Hamming)
        self.filterDelayVal = signal.lfilter_zi(self.tapsLP,1) 
        self.pllFreq = 2 * 3.14159265359 * self.pilotFreq / self.hybridFs
        self.pllAlpha = 0.01
        self.pllBeta = self.pllAlpha * self.pllAlpha / 4
        self.pllLastTheta = 0
        self.speakerIsRunning = False
        # ===================================================================== #
        # =========== \/ P W S Z   T A R N Ó W   S T E R E O \/ =============== #
        # ===================================================================== #

        # do PLL
        self.c38 = []
        self.pllTheta = []
        for i in range(self.hybridFs):
          self.c38.append(0)
          self.pllTheta.append(0)
        self.ctf2 = [(19000-2000)/float(self.hybridFs)/2, (19000+2000)/float(self.hybridFs)/2]
        self.ctf2_array = np.asarray(self.ctf2)
        self.hBP19 = signal.firwin(199,self.ctf2_array, nyq=self.hybridFs * 0.5, pass_zero=False, window='hamming', scale=False)
        # ----

        # do RDS
        self.p57 = []
        self.c57 = []
        self.c1 = []
        for i in range(self.hybridFs):
          self.c57.append(0)
          self.c1.append(0)
        self.fsymb = self.pilotFreq/16
        self.frds = 3*self.pilotFreq
        self.df2 = 1650
        self.ctf3 = [(self.frds-self.df2)/float(self.hybridFs)/2, (self.hybridFs+self.df2)/float(self.hybridFs)/2]
        self.ctf3_array = np.asarray(self.ctf3)
        self.hBP57 = signal.firwin(199, self.ctf3_array, nyq=self.hybridFs * 0.5, pass_zero=False, window='hamming', scale=False)

        self.Tsymb = 1./self.fsymb
        self.Nsymb = self.rfFs
        self.Nsymb4 = int(math.floor(4*self.Nsymb))
        self.df = float(self.rfFs)/float(self.Nsymb4)
        self.f = []
        step = 0
        while step < 2/self.Tsymb:
            self.f.append(step)
            step += self.df

        Nf = len(self.f)
        H = np.zeros((self.Nsymb4))
        H[:Nf] = np.cos(np.multiply(np.pi, self.f) * self.Tsymb /4)
        H[:-1 - (Nf - 1): -1] = H[1:Nf]

        self.H = H
        self.hpsf = np.fft.fftshift(np.fft.ifft(self.H));
        maxvalue = max(self.hpsf)
        for i in range (0, len(self.hpsf) - 1):
            self.hpsf[i] = self.hpsf[i]/maxvalue
            
        vgen = []
        for i in range (0, self.Nsymb4):
            vgen.append(i)

        self.phasePSH = np.angle(np.exp(np.dot(np.multiply(-1j * 2 * np.pi * self.fsymb/self.hybridFs, vgen), self.hpsf.real.T)));

        self.N = 25600                    #Len of t and carrier38 array
        self.dt = 1/float(self.hybridFs)  #sampling period
        self.ppic = 2*cmath.pi*38000      #used later. Def here cause of optimalization issue
        self.t = []                       #empty t array
        self.K = 8
        for index in range(0,self.N):     #create array with time between samples
            self.t.append(self.dt*index)
        # bez PLL
        self.carrier38 = []               #wykorzystywane bez PLL 
        for index in range(0,self.N):
            self.carrier38.append(math.cos(self.ppic*self.t[index]))
        self.ctf = [(38000-12500)/float(self.hybridFs)/2,(38000+12500)/float(self.hybridFs)/2] #differs from matlab cause of function construction
        self.ctf_array = np.asarray(self.ctf)                                                  #converts out ctf to numpy array                                                                         
        self.hBP38 = signal.firwin(199,self.ctf_array, nyq=self.hybridFs * 0.5, pass_zero=False, window='hamming', scale=False)     #designs FIR Bandpass filter with hamming window
        
	# ===================================================================== #
	# ==================== /\ P W S Z   T A R N Ó W /\ ==================== #
	# ===================================================================== #
        # only for testing: recording (10 seconds)
        self.rawIQ = np.zeros(self.rfFs*10, dtype=np.complex)
        self.rawIQiter = 0
        self.data = np.zeros(10240, dtype=np.complex)

    def printHelloWorld(self):
        print('FM radio in Python:\n')
        print('Credits: ToDo\n')
        print('Usage:\n\tkeyboard left/right:\ttune +/-10 kHz\n\tkeyboard up/down:\ttune +-100 kHz\n\tkeyboard ESC:\t\tEXIT (kill)\n')
        
    # create threads and start FM decoding
    def start(self):
        self.queueRfIQ = Queue()
        self.queueAudio = Queue()
        self.queueCommandsSpeaker = Queue()

        self.threads = []
        self.threads.append( threading.Thread(target=self.receiver, args=(self.queueRfIQ, )) )
        self.threads.append( threading.Thread(target=self.dataProcessing, args=(self.queueRfIQ, self.queueAudio, )) )
        self.threads.append( threading.Thread(target=self.speaker, args=(self.queueAudio, self.queueCommandsSpeaker, )) )
        self.threads.append( threading.Thread(target=self.keyListener))

        for th in self.threads:
            th.start()
        
        self.queueCommandsSpeaker.put('start')

    def keyListener(self):
        def on_keyPress(key):
            if key == keyboard.Key.right:
                self.setFc(self.fc+5e4)        # tune to +10kHz
            elif key == keyboard.Key.left:
                self.setFc(self.fc-5e4)        # tune to +10kHz
            elif key == keyboard.Key.up:
                self.setGain('up')
            elif key == keyboard.Key.down:
                self.setGain('down')
            elif key == keyboard.Key.esc:
                os._exit(0)
                
        with keyboard.Listener( on_press=on_keyPress ) as listener:
            listener.join()
   
    def stop(self):
        # send "finish" signal to threads and join
        for th in self.threads:
            th.join()

    def setFc(self, freq):
        # change self.fc dynamically
        self.fc = freq                  # TODO connect to receiver()
        print('\t\tchange fc to: {0} MHz'.format(self.fc/1e6))
        self.sdr.set_center_freq(self.fc)

    def setGain(self, gain):
        if gain=='up':
            if self.gainIndex < len(self.gains)-1:
                self.gainIndex += 1
        elif gain=='down':
            if self.gainIndex > 0:
                self.gainIndex -= 1
        else:
            print('unrecognized up/down gain command')

        self.sdr.set_gain(self.gains[self.gainIndex]/10)
        # print('gain index{0}, len{1}:'.format(str(self.gainIndex), str(len(self.gains))))
        print('current gain {0} set to {1}({2}): '.format(str(self.sdr.get_gain()), str(self.gains[self.gainIndex]), str(self.gainIndex)))



    def receiver(self, rfIQqueue ):
        def rtl_callback(samples, rtlsdr_obj):
            rfIQqueue.put(samples)
            self.data = samples[0:10240]
            
        self.sdr = rtlsdr.RtlSdr()
        self.sdr.sample_rate = self.rfFs
        self.sdr.center_freq = self.fc
        self.sdr.gain = 'auto'
        # self.sdr.gain = 43.9
        self.gains = self.sdr.get_gains()
        self.gainIndex = 0
        self.sdr.read_samples_async(rtl_callback, self.rfFs/10)    # around 100ms of data buffer
        
        # wait for self.finished signal
        sleep(500)


    # assume rfIQ at sampling rate 256 kHz
    def dataProcessing(self, rfIQqueue, audioqueue):
        while True:
            tstart = time.clock()
            rfIQ = rfIQqueue.get()
             #fm demodulation (complex->real)
            hybrid = self.demodulate(rfIQ)
            # L+R reconstruction
            audioLpR = self.filtAndDecimate(hybrid)
	    #print ("================================================")
	    #print (len(audioLpR))
	    #print ("================================================")
            #audioL = audioLpR
            #audioR = audioLpR
            # ===================================================================== #
            # =========== \/ P W S Z   T A R N Ó W   S T E R E O \/ =============== #
            # ===================================================================== #
            audioLmR = self.filtAndDecimateStereo(hybrid)
            audioL = []     #Create empty array for L and R channel 
            audioR = []     
            for index in range(0,len(audioLpR)):    
                audioL.append((audioLpR[index]+audioLmR[int(index/self.K)]))    #append to L array  Element from LpR array + every index/K element of LmR array
                audioR.append((audioLpR[index]-audioLmR[int(index/self.K)]))    #append to R array Element from LPR array - every index/K element of LmR array
            #
            #
            audioqueue.put(np.transpose([audioL,audioR]))   #Insert transposed audioL and audioR into audioqueue for speaker function
            rdsbits = self.decodeRDS(hybrid)
            self.writeRDSToFile(rdsbits)
            # ===================================================================== #
            # ==================== /\ P W S Z   T A R N Ó W /\ ==================== #
            # ===================================================================== #
            tstop = time.clock()
            print "Data processing time = " + str((tstop-tstart)) + "\n\n";
            #print (' Data processing time = ',tstop-tstart)



    def speaker(self, audioqueue, commands):
        self.local_buffer = np.ndarray(shape=(0,2))
        buffLen = self.audioFs/10
        def callback(outdata, frames, time, status):
            if not commands.empty():
                command = commands.get()
                if command == "pause":
                    self.speakerIsRunning = False
                elif command == "start":
                    self.speakerIsRunning = True
                elif command == "stop":
                    self.speakerIsRunning = False
                    raise sd.CallbackStop
            if not self.speakerIsRunning:
                outdata[:] = np.array([[0.0, 0.0] for x in range(buffLen)])
            elif self.speakerIsRunning:
                data = np.zeros(shape=(0,2))
                try:
                    if len(self.local_buffer) >= buffLen:
                        data = self.local_buffer[:buffLen]
                        self.local_buffer = np.delete(self.local_buffer, np.s_[:buffLen], axis=0)
                    else:
                        data = np.append(self.local_buffer, audioqueue.get_nowait(), axis=0)
                        while len(data) < buffLen:
                            data = np.append(data, audioqueue.get_nowait(), axis=0)
                        self.local_buffer = np.ndarray(shape=(0,2))
                    data_len = len(data)
                    if data_len > buffLen:
                        self.local_buffer = np.append(self.local_buffer, data[buffLen:,:], axis=0)
                        data = np.delete(data, np.s_[buffLen:], axis=0)
                        
                except Empty:
                    print >> sys.stderr, 'Buffer is empty'
                if len(data) < len(outdata):
                    outdata[:len(data)] = data
                    for i in range(len(outdata) - len(data)):
                        outdata[len(data)+i] = [0.0, 0.0]
                else:
                    outdata[:] = data
            
        with sd.OutputStream(samplerate=self.audioFs,
                            blocksize=self.audioFs/10, # 100ms
                            channels=self.channels,
                            dtype=np.float32,
                            callback=callback) as ss:
                            
            while (ss.active):
                sd.wait()
                sleep(1)
				
    def demodulate(self, rfIQ):
        rfIQ = np.concatenate((self.demodLastVal, rfIQ))
        self.demodLastVal = rfIQ[-1:]
        rfDemod = np.angle(rfIQ[1:] * np.conj(rfIQ[:-1]))
        return rfDemod/(2*math.pi)*self.hybridFs;

    def filtAndDecimate(self, hybrid):
        subsampled, self.filterDelayVal = signal.lfilter(self.tapsLP, 1, hybrid, zi=self.filterDelayVal)   # tapsLP - wagi, 1 - who knows, hybrid, sygnał pobrany ze sticka, zi = delay
        return subsampled[::self.hybridFs/self.audioFs]/100000
    # ===================================================================== #
    # =========== \/ P W S Z   T A R N Ó W   S T E R E O \/ =============== #
    # ===================================================================== #
    
    def phaseLockedLoop(self, y):
        tstart = t.clock()
        self.pllTheta = np.zeros(len(y)/self.K)
        self.pllTheta[0] = self.pllLastTheta
        for n in range(0, len(y), 8):
            pherr = np.sin(self.pllTheta[n/self.K]) * (-y[n])
            self.pllTheta[n/self.K] = self.pllTheta[n/self.K] + self.pllFreq - self.pllAlpha*pherr
            self.pllFreq += self.pllBeta * pherr
        self.c38 = np.cos(2*self.pllTheta[0:len(self.pllTheta) - 1])
        self.c57 = np.cos(3*self.pllTheta[0:len(self.pllTheta) - 1])
        self.pllLastTheta = self.pllTheta[-1]
        tstop = t.clock()
        print (' PLL time = ',tstop-tstart)
        return self.c38
    
    #Gettin  L-R Signal
    def filtAndDecimateStereo(self, hybrid):
        stereo, self.filterDelaVal = signal.lfilter(self.hBP38, 1, hybrid, zi=self.filterDelayVal)
        stereo_down = []
        # Z PLL
        self.pilotFilter = signal.lfilter(self.hBP19, 1, hybrid)
        self.carrierShift38 = self.phaseLockedLoop(self.pilotFilter)
        #stereo_down = np.multiply(stereo, self.c38);
        for index in range(0,self.N/self.K):
            stereo_down.append(stereo[index*self.K]*self.c38[index/self.K])
        stereo_downLP = signal.lfilter(self.tapsLP,1,stereo_down)
        stereo_lowfreq = []
        for index in range (0, len(stereo_downLP)):
            stereo_lowfreq.append(stereo_downLP[index])
        
        stereo_lf_array = np.asarray(stereo_lowfreq)
        return stereo_lf_array[::int(self.hybridFs/self.audioFs)]/100000
 
    def decodeRDS(self, hybrid):
        tstartd = time.clock()
        d, self.filterDelaVal = signal.lfilter(self.hBP57, 1, hybrid, zi=self.filterDelayVal)
        for index in range(0,self.N/self.K):
            d[index] = (d[index*self.K]*self.c57[index/self.K])
        carrier = 57000.0
        bitrate = 1187.5
        bitsteps = int(round(self.rfFs / bitrate))
        wlen = int(1.5 * self.rfFs / bitrate)
        w = np.zeros(wlen)
        for i in xrange(wlen):
            t = (i - 0.5 * (wlen - 1)) * 4.0 * bitrate / self.rfFs
            if abs(abs(t) - 0.5) < 1.0e-4:
                w[i] = 0.25 * np.pi - 0.25 * np.pi * (abs(t) - 0.5)
            else:
                w[i] = np.cos(np.pi * t) / (1 - 4.0 * t * t)
        w /= np.sum(w ** 2)
        demod_phase = 0.0
        prev_a1 = 0.0
        prevb = np.array([])
        pos = 0
        bits = []
        levels = []
        if not isinstance(d, types.GeneratorType):
            d = [d]
        for b in d:
            n = len(b)
            ps = np.arange(n) * (carrier / float(self.rfFs)) + demod_phase
            dem = (3.14 * b) * np.exp(-2j * np.pi * ps)
            demod_phase = (demod_phase + n * carrier / float(self.rfFs)) % 1.0
            self.p57 = dem
            prevb = np.concatenate((prevb[pos:], dem))
            self.c57 = prevb
            pos = 0
            # Detekcja bitów
            while pos + bitsteps + wlen < len(prevb):
                a1 = np.sum(prevb[pos:pos + wlen] * w)
                a2 = np.sum(prevb[pos + bitsteps // 2:pos + wlen + bitsteps // 2] * w)
                a3 = np.sum(prevb[pos + bitsteps // 4:pos + wlen + bitsteps // 4] * w)
                sym = a1.real * prev_a1.real + a1.imag * prev_a1.imag
                prev_a1 = a1
                if sym < 0:
                    bits.append(1)
                else:
                    bits.append(0)
                
                a1a2 = a1.real * a2.real + a1.imag * a2.imag
                a1a3 = a1.real * a3.real + a1.imag * a3.imag
                levels.append(-a1a2)
                if a1a2 >= 0:
                    pos += 5 * bitsteps // 8
                elif a1a3 > -0.02 * a1a2:
                    pos += (102 * bitsteps) // 100
                elif a1a3 > -0.01 * a1a2:
                    pos += (101 * bitsteps) // 100
                elif a1a3 < 0.02 * a1a2:
                    pos += (98 * bitsteps) // 100
                elif a1a3 < 0.01 * a1a2:
                    pos += (99 * bitsteps) // 100
                else:
                    pos += bitsteps
            
        tstopd = time.clock()
        print (' RDS decoding time = ',tstopd-tstartd)
        return (bits)

    def writeRDSToFile(self, rdsbits):
        f = open('rds.txt','ab')
        #print(rdsbits)
        bits = []
        for i in range(0,len(rdsbits)):
            bits.append(int(rdsbits[i]));
        np.savetxt(f, bits, '%d', newline='')
        f.close

    # ===================================================================== #
    # ==================== /\ P W S Z   T A R N Ó W /\ ==================== #
    # ===================================================================== #


# main program 
if __name__ == '__main__':
    fm = pyFM(95.4e6)
    #fm = pyFM(98.1e6)
    #fm = pyFM(104.1e6)
    fm.printHelloWorld()
    fm.start()

    #plot figures:
    #plt.ion()
    #while True:
    #    plt.pause(0.2)
    #    plt.clf()
    #    f, psd = signal.welch(fm.data, 256e3, nperseg=1024)
    #    plt.plot(f, 10*np.log10(np.abs(psd)))
    #    plt.show()
    fm.stop()
