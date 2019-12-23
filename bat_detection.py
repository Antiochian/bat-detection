# -*- coding: utf-8 -*- python3
"""
Created on Tue Dec 17 12:12:40 2019

@author: Henry Griffiths / Antiochian

Hiya,

This is a program I pulled together in the space of 48 hours to act as a
prelimiary audio analysis program to run on Audiomoth hardware.

I think it works, and the theory seems to make sense to me, but it definitely
needs a lot of testing and some serious tweaking as I didn't really have any
good field data to run through it to give it a proper workout.

This should all be in my notes, but I'll reiterate why this program is needed
again anyway.

Baically the problem is that if we are recording data constantly, we run out of
storage space on the SD cards far before we run out of battery or anything like
that. But of course 99.9% of that audio data is useless background noise we dont
care about, so we would like a way to filter that out and only bother saving the
audio we ACTUALLY care about. There is plenty of pre-made commercial software that
does this, but its all too bulky and complex to fit on a tiny audiomoth, so we
need our own, ultralightweight solution.

And lucky for us, the Audiomoth comes with a built in Goertzel filter which I 
think we can leverage to create just such a preliminary triggering system.

HOW THE SYSTEM WORKS:

Beforehand, decide upon two frequency bands for
i) background noise
ii) bat noise
And also pick the bandwith range for both

For example, you might take background as 7.5-8.5kHz, bat noise at 47.5-48.5kHz
(bandwidths of 1000Hz). These values will have to be tweaked experimentally.

PROGRAM TAKES AS INPUT: 512ms or 1024ms long 256kHz audio file from microphone

Split file up into 500/1000 256-sample sections

Consider 5 of these sections at a time, with a sliding "buffer"
Feed each 5-section buffer into a Goertzel algorithm to perform a discrete
fourier transform and get out the magnitude of each target frequency band

Note that since all we care about is the magnitude, we can take some shortcuts
in our implementation of the Goertzel algorithm:
    https://en.wikipedia.org/wiki/Goertzel_algorithm#Power-spectrum_terms

We take the ratio of bat_magnitude/noise_magnitude for each 5-section buffer,
take the MEDIAN score, and if its more than a predefined threshold 
(experimentally determined) then we immediately stop all analysis and save the
entire 512/1024ms audio recording to the SD card.

If the ratio is NOT more than this threshold (no bat detected), then the buffer slides along 1,
and we continue until either the threshold is triggered or we finish the audio
file.

A similar method was used for cicada detection in 2016, as I recall.
"""
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile as wav
from statistics import median
import time

#these are the default values for the various tweakable constants
bat_center = 48000 #estimated strongest frequency of bat call
noise_center = 8000 #estimated strongest frequency of background noise
THRESHOLD = 400
STOP_ON_DETECT = False #False = always check whole data sample. True = break off execution the moment you get a positive

freq_range = 250 #half the bandwidth of frequency search (eg: window = (250, 750) )

config_dict = {}
config_dict['bat_center'] = bat_center
config_dict['noise_center'] = noise_center
config_dict['THRESHOLD'] = THRESHOLD
config_dict['STOP_ON_DETECT'] = STOP_ON_DETECT
config_dict['freq_range'] = freq_range

def get_targeting_constants():
    #i am aware this function is pointless, but its a vestigal limb from an earlier namespace implementation
    bat_center = config_dict['bat_center'] 
    noise_center = config_dict['noise_center'] 
    freq_range = config_dict['freq_range'] #range observed
    return bat_center,noise_center,freq_range

def get_constants():
    THRESHOLD = config_dict['THRESHOLD'] # HIGHLY experimental
    STOP_ON_DETECT = config_dict['STOP_ON_DETECT']
    #these constants never change
    SLIDING_BUFFER_SIZE = 5
    TARGET_SAMPLE_NUMBER = 128000
    NUMBER_OF_DATA_WINDOWS = 500

    
    return SLIDING_BUFFER_SIZE,TARGET_SAMPLE_NUMBER,NUMBER_OF_DATA_WINDOWS,THRESHOLD,STOP_ON_DETECT
    
def goertzel(samples, sample_rate, *freqs):
    """
    Implementation of the Goertzel algorithm
    `samples` is a windowed one-dimensional signal originally sampled at `sample_rate`.
    *freqs is a list of windows to search in, for example ( (1000,2000) , (4500,7500) )
    
    Function cant search frequencies that are more than ~half the sample rate,
    for aliasing reasons (google "Nyquist Condition" if you want to know more)
    
    The function returns 2 arrays, one containing the actual frequencies calculated,
    the second the coefficients `(real part, imag part, POWER)` for each of those frequencies.
    We only care about the power in this case so some of this algorithm is a waste of time
    """
    window_size = len(samples)
    f_step = sample_rate / float(window_size)
    f_step_normalized = 1.0 / window_size

    # Calculate all the DFT bins we have to compute to include frequencies
    # in `freqs`.
    bins = set()
    for f_range in freqs:
        f_start, f_end = f_range
        k_start = int(math.floor(f_start / f_step))
        k_end = int(math.ceil(f_end / f_step))

        if k_end > window_size - 1: raise ValueError('frequency out of range %s' % k_end)
        bins = bins.union(range(k_start, k_end))

    # For all the bins, calculate the DFT term
    n_range = range(0, window_size)
    freqs = []
    results = []
    for k in bins:

        # Bin frequency and coefficients for the computation
        f = k * f_step_normalized
        w_real = 2.0 * math.cos(2.0 * math.pi * f)
        w_imag = math.sin(2.0 * math.pi * f)

        # Doing the calculation on the whole sample
        d1, d2 = 0.0, 0.0
        for n in n_range:
            y  = samples[n] + w_real * d1 - d2
            d2, d1 = d1, y

        # Storing results `(real part, imag part, power)`
        results.append((
            0.5 * w_real * d1 - d2, w_imag * d1,
            d2**2 + d1**2 - w_real * d1 * d2)
        )
        freqs.append(f * sample_rate)
    return freqs, results

# ---- MAIN PROGRAM --- #
def sliding_buffer_analysis(filename,slowed=False): #high_ratio_1 at 44.1kHz
    bat_center,noise_center,freq_range = get_targeting_constants()

    SAMPLE_RATE, data = wav.read(filename)
    #sometimes the test data i find online is measured at 250kHz, but is slowed to 25khz so its human audible
    #the boolean "slowed" just corrects that
    if slowed:
        SAMPLE_RATE *= 10
    FREQUENCY_WINDOWS = [(noise_center - freq_range,noise_center + freq_range) , (bat_center - freq_range,bat_center + freq_range)] #BAT WINDOW NEEDS ADJUSTING
    
    if analysis_128k(SAMPLE_RATE, data, FREQUENCY_WINDOWS):
        print("-------------")
        print("Bat detected!")
        print("-------------")  
    return
       
def analysis_128k(SAMPLE_RATE, data, FREQUENCY_WINDOWS): #split into 500 256-sample windows
    t0 = time.time() #start the clock
    BAT_WINDOW = FREQUENCY_WINDOWS[1]
    BACKGROUND_WINDOW = FREQUENCY_WINDOWS[0]
    SLIDING_BUFFER_SIZE,TARGET_SAMPLE_NUMBER,NUMBER_OF_DATA_WINDOWS,THRESHOLD,STOP_ON_DETECT = get_constants()
    
    #This is just to cope with the inconsistency of online data, data coming from the audiomoth
    #should always have the exact right number of samples and this wont be needed
    if len(data) > TARGET_SAMPLE_NUMBER: #crop data if its too much
        data = data[0:TARGET_SAMPLE_NUMBER]
    elif len(data) < TARGET_SAMPLE_NUMBER: #pad data if its too little
        data = list(data) + [0]*(TARGET_SAMPLE_NUMBER-len(data))
        
    buffered_data = [] #array containing 500 256-sample data windows
    for w in range(NUMBER_OF_DATA_WINDOWS):
        slice_size = int(TARGET_SAMPLE_NUMBER/NUMBER_OF_DATA_WINDOWS)
        index = w*slice_size
        buffered_data.append(data[index:index+slice_size])
    
    #feed first 5 windows into Goertzel filter and continue (until you run out of windows or detect a bat)
    i0 = 0 #start window
    j0 = SLIDING_BUFFER_SIZE #close window
    ratios = []
    while True:
        if i0%25 == 0:
            percent = i0*5//25
            print("\r Working...",percent,"%",end="")
        data_slice = [y for x in buffered_data[i0:j0] for y in x] 
        #extract only magnitudes:
        background_magnitude , goertzel_magnitude, freq_bin = goertzel_compare(SAMPLE_RATE, data_slice, BAT_WINDOW, BACKGROUND_WINDOW)
        bg_med = median(background_magnitude)
        if bg_med == 0:
            bg_med = 1
        RATIO = median(goertzel_magnitude)/bg_med
        ratios.append(RATIO)
        
        if RATIO > THRESHOLD and STOP_ON_DETECT == True:
            
            plot_hit(goertzel_magnitude, background_magnitude,data_slice, data, freq_bin, ratios, THRESHOLD)
            
            print("\nTime taken:", round(time.time()-t0,5),"seconds")
            print("\tWindow trigger #: ",i0)
            print("Average ratio:",round(sum(ratios)/len(ratios),5))
            print("Max ratio:", round(max(ratios),5))
            print("Threshold ratio:",THRESHOLD)
            return True
        i0 += 1
        j0 += 1
        if j0 > NUMBER_OF_DATA_WINDOWS:

            plot_hit(goertzel_magnitude, background_magnitude,data_slice, data, freq_bin, ratios, THRESHOLD)
            
            print("\nTime taken:", round(time.time()-t0,5),"seconds")
            print("Average ratio:",round(sum(ratios)/len(ratios),5))
            print("Max ratio:", round(max(ratios),5))
            print("Threshold ratio:",THRESHOLD)
            return False
        
def plot_hit(goertzel_magnitude, background_magnitude, data_slice, data, freq_bin, ratios, THRESHOLD):
    #function to plot some data for testing/demonstration purposes, completely useless
    #for the actual thing
    print("\n")
    ymax = max(max(goertzel_magnitude),max(background_magnitude))

    plt.subplot(2,2,1)
    plt.title('Background Noise in last instant')
    plt.plot(freq_bin[0], background_magnitude,'o')    
    plt.ylim(0, ymax)
    
    plt.subplot(2,2,2)
    plt.title('Bat Noise in last instant')
    plt.plot(freq_bin[1], goertzel_magnitude,'o')
    plt.ylim(-0.1, ymax)
    
    plt.subplot(2,2,3)
    plt.title('Raw Waveform Slice')
    xaxis = np.arange(start=0, stop=100, step=100/len(data))
    plt.plot(xaxis,data)
    if len(ratios) != 496:
        plt.vlines(100*len(ratios)/496, min(data),max(data), linestyles='dashed',zorder = 3)
    
    plt.subplot(2,2,4)
    plt.title('Ratio of Bat/Background')
    xaxis = np.arange(start=0, stop=500, step=100/500)
    plt.plot(xaxis[0:len(ratios)],ratios)
    plt.hlines(THRESHOLD,0,100,linestyles='dashed',zorder = 3)
    ymax = max(ratios)
    plt.ylim(0,ymax)

        
    plt.tight_layout()    
    plt.show()
    return      

def goertzel_compare(SAMPLE_RATE, data_slice, BAT_WINDOW, BACKGROUND_WINDOW):
    #inputs two data windows and gets the powers for each
    
    #needs to return the goertzel responses for each window
    bat_freq, bat_results = goertzel(data_slice, SAMPLE_RATE, BAT_WINDOW)
    background_freq, background_results = goertzel(data_slice, SAMPLE_RATE, BACKGROUND_WINDOW)
    
    goertzel_magnitude = []
    background_magnitude = []
    for i in bat_results:
        goertzel_magnitude.append(i[2])
    for j in background_results:
        background_magnitude.append(j[2])
    return background_magnitude, goertzel_magnitude, [background_freq, bat_freq]


#Basic GUI for testing/demonstration purposes only
#this really shouldnt be using global variables so casually, but its ok for a testbed
    
slowed = False
print("\n------------------------------------")
print("// Henry's Prototype Bat Detector //")
print("-------------2019-------------------")
print("\n Select an option:")
print("\t 0 - Cancel")
print(" ------ bats:")
print("\t 1 - Natterer's Bat 1 (250kHz)")
print("\t 2 - Natterer's Bat 2 (250kHz)")
print("\t 3 - Common Pipistrelle 1 (250kHz)")
print("\t 4 - Nyctalus leisleri 1 (250kHz)")
print("\t 5 - Barbastella barbastellus 1 (250kHz)")
print(" ------ not bats:")
print("\t 6 - Busy Restaurant Interior (250kHz)")
print("\t 7 - Clapping (250kHz)")
print("\t 8 - Human whistle (250kHz)")

choice = int(input("\n >: "))
if choice == 0:
    filename = None
elif choice == 1:
    filename = 'positives/Myotis_nattereri_1_o.wav'
    bat_center = 48000
    THRESHOLD = 400
elif choice == 2:
    filename = 'positives/Myotis_nattereri_2_o.wav'
    bat_center = 48000
    THRESHOLD = 200
elif choice == 3:
    filename = 'positives/Pipistrellus_pipistrellus_4.wav'
    bat_center = 50000
    THRESHOLD = 400
    slowed = True
elif choice == 4:
    filename = 'positives/Nyctalus_leisleri_1_o.wav'
    bat_center = 24000
    freq_range = 500
    THRESHOLD = 280000
elif choice == 5:
    filename = 'positives/Barbastella_barbastellus_1_o.wav'
    bat_center = 32000
    THRESHOLD = 1000
elif choice == 6:
    filename = 'negatives/5DF8D02A_GBK.WAV'
elif choice == 7:
    filename = 'negatives/5DF93502_clapping.WAV'
elif choice == 8:
    filename = 'negatives/5DF937EF_whistle.WAV'
else:
    print("Input not recognised. Exiting...")
    filename = None
if filename == None:
    print("cancelling...")
else:
    #update constants
    config_dict = {}
    config_dict['bat_center'] = bat_center
    config_dict['noise_center'] = noise_center
    config_dict['THRESHOLD'] = THRESHOLD
    config_dict['STOP_ON_DETECT'] = STOP_ON_DETECT
    config_dict['freq_range'] = freq_range

    stop_choice = input("Stop on trigger? (Y/N): ")
    if stop_choice == "Y" or stop_choice == "y":
        STOP_ON_DETECT = True
        config_dict['STOP_ON_DETECT'] = STOP_ON_DETECT
        config_dict['STOP_ON_DETECT'] = True
    else:
        STOP_ON_DETECT = False
        config_dict['STOP_ON_DETECT'] = False
    print("Searching for frequencies ~ ",bat_center,"Hz")
    sliding_buffer_analysis(filename,slowed)
