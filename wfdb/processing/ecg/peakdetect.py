import numpy as np
import scipy.signal as scisig


class PanTompkins(object):
    """
    Class for implementing the Pan-Tompkins
    qrs detection algorithm.

    Works on static signals. In future update,
    will work on streaming ecg signal.
    """
    def __init__(self, sig=None, fs=None, streamsig=None):
        self.sig = sig
        self.fs = fs

        self.livesig = livesig
        
        if sig is not None:
            self.siglen = len(sig)

    def detect_qrs_static(self):
        """
        Detect all the qrs locations in the static
        signal
        """

        # Resample the signal to 200Hz if necessary
        self.resample()    

        # Bandpass filter the signal
        self.sig_F = self.bandpass()

        # Calculate the moving wave integration signal
        self.sig_I = self.mwi()

        # Align the filtered and integrated signal with the original
        self.alignsignals()
        

        # Initialize parameters via the two learning phases
        self.learnparams()

        # Loop through every index and detect qrs locations
        for i in range(self.siglen):
            pass

        # Go through each index and detect qrs complexes.
    for i in range(siglen):
        # Determine whether the current index is a peak
        # for each signal
        is_peak_F = ispeak(sig_F, siglen, i, 20)
        is_peak_I = ispeak(sig_I, siglen, i, 20)
        
        # Whether the current index is a signal peak or noise peak
        is_sigpeak_F = False
        is_noisepeak_F = False
        is_sigpeak_I = False
        is_noisepeak_I = False
        
        # If peaks are detected, classify them as signal or noise
        # for their respective channels
        
        if is_peak_F:
            # Satisfied signal peak criteria for the channel
            # but not necessarily overall
            if sig_F[i] > thresh_F:
                is_sigpeak_F = True
            # Did not satisfy signal peak criteria.
            # Label as noise peak
            else:
                is_noisepeak_F = True
                
        if is_peak_I:
            # The
            if sig_I[i] > thresh_I:
                is_peak_sig_I = True
            else:
         
        # Check for double signal peak coincidence and at least >200ms (40 samples samples)
        # since the previous r peak
        is_sigpeak = is_sigpeak_F and is_sigpeak_I and ()
        
        # Found a signal peak!
        if is_sigpeak:
            
            # BUT WAIT, THERE'S MORE! It could be a T-Wave ...
            
            # When an rr interval < 360ms (72 samples), it is checked 
            # to determine whether it is a T-Wave
            if i - prev_r_ind < 360:
                
            
            # Update running parameters
            
            sigpeak_I = 0.875*sigpeak_I + 0.125*sig_I[i]
            sigpeak_F = 0.875*sigpeak_I + 0.125*sig_I[i]
             
            last_r_ind = i
            rr_limitavg
            rr_limitavg
            
        # Not a signal peak. Update running parameters
        # if any other peaks are detected
        elif is_peak_F:
            
        
        last_r_distance = i - last_r_ind
        
        if last_r_distance > 
    
    
    qrs = np.zeros(10)
    
    # Convert the peak indices back to the original fs if necessary 
    if fs!=200:
        qrs = qrs*fs/200
    qrs = qrs.astype('int64')
    
    
    return qrs



    
    def resample(self):
        if self.fs != 200:
            self.sig = scisig.resample(self.sig, int(self.siglen*200/fs))
    
    # Bandpass filter the signal from 5-15Hz
    def bandpass(self, plotsteps):
        # 15Hz Low Pass Filter
        a_low = [1, -2, 1]
        b_low = np.concatenate(([1], np.zeros(4), [-2], np.zeros(5), [1]))
        sig_low = scisig.lfilter(b_low, a_low, self.sig)
        
        # 5Hz High Pass Filter - passband gain = 32, delay = 16 samples
        a_high = [1,-1]
        b_high = np.concatenate(([-1/32], np.zeros(15), [1, -1], np.zeros(14), [1/32]))
        self.sig_F = scisig.lfilter(b_high, a_high, sig_low)
        
        if plotsteps:
            plt.plot(sig_low)
            plt.plot(self.sig_F)
            plt.legend(['After LP', 'After LP+HP'])
            plt.show()

    # Compute the moving wave integration waveform from the filtered signal
    def mwi(sig, plotsteps):
        # Compute 5 point derivative
        a_deriv = [1]
        b_deriv = [1/4, 1/8, 0, -1/8, -1/4]
        sig_F_deriv = scisig.lfilter(b_deriv, a_deriv, self.sig_F)
        
        # Square the derivative
        sig_F_deriv = np.square(sig_F_deriv)
        
        # Perform moving window integration - 150ms (ie. 30 samples wide for 200Hz)
        a_mwi = [1]
        b_mwi = 30*[1/30]
        
        self.sig_I = scisig.lfilter(b_mwi, a_mwi, sig_F_deriv)
        
        if plotsteps:
            plt.plot(sig_deriv)
            plt.plot(self.sig_I)
            plt.legend(['deriv', 'mwi'])
            plt.show()
    
    # Align the filtered and integrated signal with the original
    def alignsignals(self):
        self.sig_F = self.sig_F

        self.sig_I = self.sig_I


    def learnparams(self):
        """
        Initialize parameters using the start of the waveforms
        during the two learning phases described.
        
        "Learning phase 1 requires about 2s to initialize
        detection thresholds based upon signal and noise peaks
        detected during the learning process.
        
        Learning phase two requires two heartbeats to initialize
        RR-interval average and RR-interval limit values. 
        
        The subsequent detection phase does the recognition process
        and produces a pulse for each QRS complex"
        
        This code is not detailed in the Pan-Tompkins
        paper. The PT algorithm requires a threshold to
        categorize peaks as signal or noise, but the
        threshold is calculated from noise and signal 
        peaks. There is a circular dependency when 
        none of the fields are initialized. Therefore this learning phase will detect signal peaks using a
        different method, and estimate the threshold using those peaks.
        
        This function works as follows:
        - Try to find at least 2 signal peaks (qrs complexes) in the
          first N seconds of both signals using simple low order 
          moments. Signal peaks are only defined when the same index is
          determined to be a peak in both signals. Start with N==2.
          If fewer than 2 signal peaks are detected, increment N and
          try again.
        - Using the classified estimated peaks, threshold1 is estimated as
          based on the steady state estimate equation: thres = 0.75*noisepeak + 0.25*sigpeak
          using the mean of the noisepeaks and signalpeaks instead of the
          running value.
        - Using the estimated peak locations, the rr parameters are set.
        
        """
        
        # The sample radius when looking for local maxima
        radius = 20
        # The signal start duration to use for learning
        learntime = 2
        
        while :
            wavelearn_F = filtsig[:200*learntime]
            wavelearn_I = mwi[:200*learntime]
            
            # Find peaks in the signals
            peakinds_F = findpeaks_radius(wavelearn_F, radius)
            peakinds_I = findpeaks_radius(wavelearn_I, radius)
            peaks_F = wavelearn_F[peakinds_F]
            peaks_I = wavelearn_I[peakinds_I]
        
            # Classify signal and noise peaks.
            # This is the main tricky part
            
            # Align peaks to minimum value and set to unit variance
            peaks_F = (peaks_F - min(peaks_F)) / np.std(peaks_F)
            peaks_I = (peaks_I - min(peaks_I)) / np.std(peaks_I)
            sigpeakinds_F = np.where(peaks_F) >= 1.4
            sigpeakinds_I = np.where(peaks_I) >= 1.4
            
            # Final signal peak when both signals agree
            sigpeakinds = np.intersect1d(sigpeaks_F, sigpeaks_I)
            # Noise peaks are the remainders
            noisepeakinds_F = np.setdiff1d(peakinds_F, sigpeakinds)
            noisepeakinds_I = np.setdiff1d(peakinds_I, sigpeakinds)
            
            if len(sigpeakinds)>1:
                break
            
            # Need to detect at least 2 peaks
            learntime = learntime + 1
        
        # Found at least 2 peaks. Use them to set parameters.
        
        # Set running peak estimates to first values
        sigpeak_F = wavelearn_F[sigpeakinds[0]]
        sigpeak_I = wavelearn_I[sigpeakinds[0]]
        noisepeak_F = wavelearn_F[noisepeakinds_F[0]]
        noisepeak_I = wavelearn_I[noisepeakinds_I[0]]
        
        # Use all signal and noise peaks in learning window to estimate threshold
        # Based on steady state equation: thres = 0.75*noisepeak + 0.25*sigpeak
        thres_F = 0.75*np.mean(wavelearn_F[noisepeakinds_F]) + 0.25*np.mean(wavelearn_F[sigpeakinds_F])
        thres_I = 0.75*np.mean(wavelearn_I[noisepeakinds_I]) + 0.25*np.mean(wavelearn_I[sigpeakinds_I])
        # Alternatively, could skip all of that and do something very simple like thresh_F =  max(filtsig[:400])/3
        
        # Set the r-r history using the first r-r interval
        # The most recent 8 rr intervals
        rr_history_unbound = [wavelearn_F[sigpeakinds[1]]-wavelearn_F[sigpeakinds[0]]]*8
        # The most recent 8 rr intervals that fall within the acceptable low and high rr interval limits
        rr_history_bound = [wavelearn_I[sigpeakinds[1]]-wavelearn_I[sigpeakinds[0]]]*8
        
        rr_average_unbound = np.mean(rr_history_unbound)
        rr_average_bound = np.mean(rr_history_bound)
        
        # Wait... what is rr_average_unbound for then?
        rr_low_limit = 0.92*rr_average_bound
        rr_high_limit = 1.16*rr_average_bound
        rr_missed_limit =  1.66*rr_average_bound
        
        return thresh_F, thresh_I, sigpeak_F, sigpeak_I,
               noisepeak_F, noisepeak_I, rr_freeavg_F,
               r_freeavg_I, rr_limitavg_F, rr_limitavg_I







# Determine whether the signal contains a peak at index ind.
# Check if it is the max value amoung samples ind-radius to ind+radius
def ispeak_radius(sig, siglen, ind, radius):
    if sig[ind] == max(sig[max(0,i-radius):min(siglen, i+radius)]):
        return True
    else:
        return False

# Find all peaks in a signal. Simple algorithm which marks a
# peak if the <radius> samples on its left and right are
# all not bigger than it.
def findpeaks_radius(sig, radius):
    
    siglen = len(sig)
    peaklocs = []
    
    # Pad samples at start and end
    sig = np.concatenate((np.ones(radius)*sig[0],sig, np.ones(radius)*sig[-1]))
    
    i=radius
    while i<siglen+radius:
        if sig[i] == max(sig[i-radius:i+radius]):
            peaklocs.append(i)
            i=i+radius
        else:
            i=i+1
        
    peaklocs = np.array(peaklocs)-radius
    return peaklocs

