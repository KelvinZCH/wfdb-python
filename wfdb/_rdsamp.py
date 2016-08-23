import numpy as np
import re
import os
import sys
import requests
from ConfigParser import ConfigParser
# from distutils.sysconfig import get_python_lib

def checkrecordfiles(recordname, filedirectory):
    """Check a local directory along with the database cache directory specified in 
        'config.ini' for all necessary files required to read a WFDB record. 
        Calls pbdownload.dlrecordfiles to download any missing files into the database 
        cache directory. Returns the base record name if all files were present, or a 
        full path record name specifying where the downloaded files are to be read, 
        and a list of files downloaded.

    *If you wish to directly download files for a record, it highly recommended to call 
    'pbdownload.dlrecordfiles' directly. This is a helper function for rdsamp which 
    tries to parse the 'recordname' input to deduce whether it contains a local directory, 
    physiobank database, or both. Its usage format is different and more complex than 
    that of 'dlrecordfiles'.

    Usage: readrecordname, downloadedfiles = checkrecordfiles(recordname, filedirectory)

    Input arguments:
    - recordname (required): The name of the WFDB record to be read 
      (without any file extensions). Can be prepended with a local directory, or a 
      physiobank subdirectory (or both if the relative local directory exists and 
      takes the same name as the physiobank subdirectory). eg: recordname=mitdb/100
    - filedirectory (required): The local directory to check for the files required to 
      read the record before checking the database cache directory. If the 'recordname' 
      argument is prepended with a directory, this function will assume that it is a 
      local directory and prepend that to this 'filedirectory' argument and check the 
      resulting directory instead.

    Output arguments:
    - readrecordname: The record name prepended with the path the files are to be read from.
    - downloadedfiles:  The list of files downloaded from PhysioBank.
    """

    # Base directory to store downloaded physiobank files
    config = loadconfig('wfdb.config')
    dbcachedir = config.get('pbdownload','dbcachedir')
    basedir, baserecname = os.path.split(recordname)

    # At this point we do not know whether basedir is a local directory, a
    # physiobank directory, or both!

    if not basedir:  # if there is no base directory then: 1. The files must be located 
        # in the current working directory.
        # 2. There will be no target files to download. So just directly
        # return. If files are missing we cannot download them anyway.
        return recordname, []

    # If this is reached, basedir is defined. Check if there is a directory
    # called 'basedir':
    # the 'basedir' directory exists. Check it for files.
    if os.path.isdir(basedir):
        # It is possible that basedir is also a physiobank database. Therefore
        # if any files are missing, ,try to download files  assuming basedir is
        # the physiobank database directory. If it turns out that basedir is
        # not a pb database, an error will be triggered. The record would not
        # be readable without the missing file(s) anyway.

        downloaddir = os.path.join(dbcachedir, basedir)

        # The basedir directory is missing the header file.
        if not os.path.isfile(os.path.join(basedir, baserecname + ".hea")):
            # If invalid pb database, function would exit.
            dledfiles = dlrecordfiles(recordname, downloaddir)
            # Files downloaded, confirmed valid pb database.
            return os.path.join(downloaddir, baserecname), dledfiles

        # Header is present in basedir
        fields = readheader(recordname)

        if fields[
                "nseg"] == 1:  # Single segment. Check for all the required dat files
            for f in fields["filename"]:
                # Missing a dat file. Download in db cache dir.
                if not os.path.isfile(os.path.join(basedir, f)):
                    dledfiles = dlrecordfiles(recordname, downloaddir)
                    return os.path.join(downloaddir, baserecname), dledfiles
        else:  # Multi segment. Check for all segment headers and their dat files
            for segment in fields["filename"]:
                if segment != '~':
                    if not os.path.isfile(
                        os.path.join(
                            basedir,
                            segment +
                            ".hea")):  # Missing a segment header
                        dledfiles = dlrecordfiles(recordname, downloaddir)
                        return os.path.join(
                            downloaddir, baserecname), dledfiles
                    segfields = readheader(os.path.join(basedir, segment))
                    for f in segfields["filename"]:
                        if f != '~':
                            if not os.path.isfile(
                                os.path.join(
                                    basedir,
                                    f)):  # Missing a segment's dat file
                                dledfiles = dlrecordfiles(
                                    recordname, downloaddir)
                                return os.path.join(
                                    downloaddir, baserecname), dledfiles

        # All files were already present in the 'basedir' directory.
        return recordname, []

    else:  # there is no 'basedir' directory. Therefore basedir must be a 
           # physiobank database directory. check the current working directory for files. 
           # If any are missing, check the cache directory for files and download missing 
           # files from physiobank.

        pbdir = basedir  # physiobank directory
        downloaddir = os.path.join(dbcachedir, pbdir)

        if not os.path.isfile(baserecname + ".hea"):
            dledfiles = dlrecordfiles(recordname, downloaddir)
            return os.path.join(downloaddir, baserecname), dledfiles

        # Header is present in current working dir.
        fields = readheader(baserecname)

        if fields[
                "nseg"] == 1:  # Single segment. Check for all the required dat files
            for f in fields["filename"]:
                # Missing a dat file. Download in db cache dir.
                if not os.path.isfile(f):
                    dledfiles = dlrecordfiles(recordname, downloaddir)
                    return os.path.join(downloaddir, baserecname), dledfiles
        else:  # Multi segment. Check for all segment headers and their dat files
            for segment in fields["filename"]:
                if segment != '~':
                    if not os.path.isfile(
                        os.path.join(
                            targetdir,
                            segment +
                            ".hea")):  # Missing a segment header
                        dledfiles = dlrecordfiles(recordname, downloaddir)
                        return os.path.join(
                            downloaddir, baserecname), dledfiles
                    segfields = readheader(os.path.join(targetdir, segment))
                    for f in segfields["filename"]:
                        if f != '~':
                            if not os.path.isfile(
                                os.path.join(
                                    targetdir,
                                    f)):  # Missing a segment's dat file
                                dledfiles = dlrecordfiles(
                                    recordname, downloaddir)
                                return os.path.join(
                                    downloaddir, baserecname), dledfiles

        # All files are present in current directory. Return base record name
        # and no dled files.
        return baserecname, []


def dlrecordfiles(pbrecname, targetdir):
    """Check a specified local directory for all necessary files required to read a Physiobank 
       record, and download any missing files into the same directory. Returns a list of files 
       downloaded, or exits with error if an invalid Physiobank record is specified.

    Usage: dledfiles = dlrecordfiles(pbrecname, targetdir)

    Input arguments:
    - pbrecname (required): The name of the MIT format Physiobank record to be read, prepended 
      with the Physiobank subdirectory the file is contain in (without any file extensions). 
      eg. pbrecname=prcp/12726 to download files http://physionet.org/physiobank/database/prcp/12726.hea 
      and 12727.dat
    - targetdir (required): The local directory to check for files required to read the record, 
      in which missing files are also downloaded.

    Output arguments:
    - dledfiles:  The list of files downloaded from PhysioBank.

    """
    physioneturl = "http://physionet.org/physiobank/database/"
    pbdir, baserecname = os.path.split(pbrecname)
    print('Downloading missing file(s) into directory: ', targetdir)

    if not os.path.isdir(
            targetdir):  # Make the target directory if it doesn't already exist
        os.makedirs(targetdir)
        madetargetdir = 1
    else:
        madetargetdir = 0
    dledfiles = []  # List of downloaded files

    # For any missing file, check if the input physiobank record name is
    # valid, ie whether the file exists on physionet. Download if valid, exit
    # if invalid.

    if not os.path.isfile(os.path.join(targetdir, baserecname + ".hea")):
        # Not calling dlorexit here. Extra instruction of removing the faulty
        # created directory.
        try:
            remotefile = physioneturl + pbrecname + ".hea"
            targetfile = os.path.join(targetdir, baserecname + ".hea")
            r = requests.get(remotefile)
            with open(targetfile, "w") as text_file:
                text_file.write(r.text)
            dledfiles.append(targetfile)
        except HTTPError:
            if madetargetdir:
                # Remove the recently created faulty directory.
                os.rmdir(targetdir)
            sys.exit(
                "Attempted to download invalid target file: {}".format(remotefile))

    fields = readheader(os.path.join(targetdir, baserecname))

    # Even if the header file exists, it could have been downloaded prior.
    # Need to check validity of link if ANY file is missing.
    if fields["nseg"] == 1:  # Single segment. Check for all the required dat files
        for f in fields["filename"]:
            if not os.path.isfile(
                os.path.join(
                    targetdir,
                    f)):  # Missing a dat file
                dledfiles = dlorexit(
                    physioneturl + pbdir + "/" + f,
                    os.path.join(
                        targetdir,
                        f),
                    dledfiles)

    else:  # Multi segment. Check for all segment headers and their dat files
        for segment in fields["filename"]:
            if segment != '~':
                if not os.path.isfile(
                    os.path.join(
                        targetdir,
                        segment +
                        ".hea")):  # Missing a segment header
                    dledfiles = dlorexit(
                        physioneturl +
                        pbdir +
                        "/" +
                        segment +
                        ".hea",
                        os.path.join(
                            targetdir,
                            segment +
                            ".hea"),
                        dledfiles)
                segfields = readheader(os.path.join(targetdir, segment))
                for f in segfields["filename"]:
                    if f != '~':
                        if not os.path.isfile(
                            os.path.join(
                                targetdir,
                                f)):  # Missing a segment's dat file
                            dledfiles = dlorexit(
                                physioneturl + pbdir + "/" + f,
                                os.path.join(
                                    targetdir,
                                    f),
                                dledfiles)

    print('Download complete')
    return dledfiles  # downloaded files


# Helper function for dlrecordfiles. Download the file from the specified
# 'url' as the 'filename', or exit with warning.
def dlorexit(url, filename, dledfiles):
    try:
        r = requests.get(url)
        with open(filename, "w") as text_file:
            text_file.write(r.text)
        dledfiles.append(filename)
        return dledfiles
    except HTTPError:
        sys.exit("Attempted to download invalid target file: " + url)


# Download files required to read a wfdb annotation.
def dlannfiles():
    return dledfiles


# Download all the records in a physiobank database.
def dlPBdatabase(database, targetdir):
    return dledfiles


def readheader(recordname):  # For reading signal headers

    # To do: Allow exponential input format for some fields

    # Output dictionary
    fields = {
        'nseg': [],
        'nsig': [],
        'fs': [],
        'nsamp': [],
        'basetime': [],
        'basedate': [],
        'filename': [],
        'fmt': [],
        'sampsperframe': [],
        'skew': [],
        'byteoffset': [],
        'gain': [],
        'units': [],
        'baseline': [],
        'initvalue': [],
        'signame': [],
        'nsampseg': [],
        'comments': []}
    # filename stores file names for both multi and single segment headers.
    # nsampseg is only for multi-segment

    commentlines = []  # Store comments
    headerlines = []  # Store record line followed by the signal lines if any

    # RECORD LINE fields (o means optional, delimiter is space or tab unless specified):
    # record name, nsegments (o, delim=/), nsignals, fs (o), counter freq (o, delim=/, needs fs),
    # base counter (o, delim=(), needs counter freq), nsamples (o, needs fs), base time (o),
    # base date (o needs base time).

    # Regexp object for record line
    rxRECORD = re.compile(
        ''.join(
            [
                "(?P<name>[\w]+)/?(?P<nseg>\d*)[ \t]+",
                "(?P<nsig>\d+)[ \t]*",
                "(?P<fs>\d*\.?\d*)/*(?P<counterfs>\d*\.?\d*)\(?(?P<basecounter>\d*\.?\d*)\)?[ \t]*",
                "(?P<nsamples>\d*)[ \t]*",
                "(?P<basetime>\d*:?\d{,2}:?\d{,2}\.?\d*)[ \t]*",
                "(?P<basedate>\d{,2}/?\d{,2}/?\d{,4})"]))
    # Watch out for potential floats: fs (and also exponent notation...),
    # counterfs, basecounter

    # SIGNAL LINE fields (o means optional, delimiter is space or tab unless specified):
    # file name, format, samplesperframe(o, delim=x), skew(o, delim=:), byteoffset(o,delim=+),
    # ADCgain(o), baseline(o, delim=(), requires ADCgain), units(o, delim=/, requires baseline),
    # ADCres(o, requires ADCgain), ADCzero(o, requires ADCres), initialvalue(o, requires ADCzero),
    # checksum(o, requires initialvalue), blocksize(o, requires checksum),
    # signame(o, requires block)

    # Regexp object for signal lines. Consider flexible filenames, and also ~
    rxSIGNAL = re.compile(
        ''.join(
            [
                "(?P<filename>[\w]*\.?[\w]*~?)[ \t]+(?P<format>\d+)x?"
                "(?P<sampsperframe>\d*):?(?P<skew>\d*)\+?(?P<byteoffset>\d*)[ \t]*",
                "(?P<ADCgain>-?\d*\.?\d*e?[\+-]?\d*)\(?(?P<baseline>-?\d*)\)?/?(?P<units>[\w\^/-]*)[ \t]*",
                "(?P<ADCres>\d*)[ \t]*(?P<ADCzero>-?\d*)[ \t]*(?P<initialvalue>-?\d*)[ \t]*",
                "(?P<checksum>-?\d*)[ \t]*(?P<blocksize>\d*)[ \t]*(?P<signame>[\S]*)"]))

    # Units characters: letters, numbers, /, ^, -,
    # Watch out for potentially negative fields: baseline, ADCzero, initialvalue, checksum,
    # Watch out for potential float: ADCgain.

    # Split comment and non-comment lines
    with open(recordname + ".hea", 'r') as fp:
        for line in fp:
            line = line.strip()
            if line.startswith('#'):  # comment line
                commentlines.append(line)
            elif line:  # Non-empty non-comment line = header line.
                ci = line.find('#')
                if ci > 0:
                    headerlines.append(line[:ci])  # header line
                    # comment on same line as header line
                    commentlines.append(line[ci:])
                else:
                    headerlines.append(line)

    # Get record line parameters
    (_, nseg, nsig, fs, counterfs, basecounter, nsamp,
     basetime, basedate) = rxRECORD.findall(headerlines[0])[0]

    # These fields are either mandatory or set to defaults.
    if not nseg:
        nseg = '1'
    if not fs:
        fs = '250'
    # fs = float(fs)
    fields['nseg'] = int(nseg)
    fields['fs'] = float(fs)
    fields['nsig'] = int(nsig)

    # These fields might by empty
    if nsamp:
        fields['nsamp'] = int(nsamp)
    fields['basetime'] = basetime
    fields['basedate'] = basedate

    # Signal or Segment line paramters
    # Multi segment header - Process segment spec lines in current master
    # header.
    if int(nseg) > 1:
        for i in range(0, int(nseg)):
            (filename, nsampseg) = re.findall(
                '(?P<filename>\w*~?)[ \t]+(?P<nsampseg>\d+)', headerlines[i + 1])[0]
            # filename might be ~ for null segment.
            fields["filename"].append(filename)
            # number of samples for the segment
            fields["nsampseg"].append(int(nsampseg))
    # Single segment header - Process signal spec lines in current regular
    # header.
    else:
        for i in range(0, int(nsig)):  # will not run if nsignals=0
            # get signal line parameters
            (filename,
             fmt,
             sampsperframe,
             skew,
             byteoffset,
             adcgain,
             baseline,
             units,
             adcres,
             adczero,
             initvalue,
             checksum,
             blocksize,
             signame) = rxSIGNAL.findall(headerlines[i + 1])[0]

            # Setting defaults
            if not sampsperframe:
                # Setting strings here so we can always convert strings case
                # below.
                sampsperframe = '1'
            if not skew:
                skew = '0'
            if not byteoffset:
                byteoffset = '0'
            if not adcgain:
                adcgain = '200'
            if not baseline:
                if not adczero:
                    baseline = '0'
                else:
                    baseline = adczero  # missing baseline actually takes adczero value if present
            if not units:
                units = 'mV'
            if not initvalue:
                initvalue = '0'
            if not signame:
                signame = "ch" + str(i + 1)
            if not initvalue:
                initvalue = '0'

            fields["filename"].append(filename)
            fields["fmt"].append(fmt)
            fields["sampsperframe"].append(int(sampsperframe))
            fields["skew"].append(int(skew))
            fields['byteoffset'].append(int(byteoffset))
            fields["gain"].append(float(adcgain))
            fields["baseline"].append(int(baseline))
            fields["units"].append(units)
            fields["initvalue"].append(int(initvalue))
            fields["signame"].append(signame)

    if commentlines:
        for comment in commentlines:
            fields["comments"].append(comment.strip('\s#'))

    return fields


# Get samples from a WFDB binary file
def readdat(
        filename,
        fmt,
        byteoffset,
        sampfrom,
        sampto,
        nsig,
        siglen,
        sampsperframe,
        skew):
    # nsig defines whole file, not selected channels. siglen refers to signal 
    # length of whole file, not selected duration.
    # Recall that channel selection is not performed in this function but in
    # rdsamp.

    tsampsperframe = sum(sampsperframe)  # Total number of samples per frame

    # Figure out the starting byte to read the dat file from. Special formats
    # store samples in specific byte blocks.
    startbyte = int(sampfrom * tsampsperframe *
                    bytespersample[fmt]) + int(byteoffset)
    floorsamp = 0
    # Point the the file pointer to the start of a block of 3 or 4 and keep
    # track of how many samples to discard after reading.
    if fmt == '212':
        floorsamp = (startbyte - byteoffset) % 3  # Extra samples to read
        # Now the byte pointed to is the first of a byte triplet storing 2
        # samples. It happens that the extra samples match the extra bytes for
        # fmt 212
        startbyte = startbyte - floorsamp
    elif (fmt == '310') | (fmt == '311'):
        floorsamp = (startbyte - byteoffset) % 4
        # Now the byte pointed to is the first of a byte quartet storing 3
        # samples.
        startbyte = startbyte - floorsamp

    fp = open(filename, 'rb')

    fp.seek(startbyte)  # Point to the starting sample
    # Read the dat file into np array and reshape.
    sig, nbytesread = processwfdbbytes(
        fp, fmt, sampto - sampfrom, nsig, sampsperframe, floorsamp)

    # Shift the samples in the channels with skew if any
    if max(skew) > 0:
        # Array of samples to fill in the final samples of the skewed channels.
        extrasig = np.empty([max(skew), nsig])
        extrasig.fill(wfdbInvalids[fmt])

        # Load the extra samples if the end of the file hasn't been reached.
        if siglen - (sampto - sampfrom):
            startbyte = startbyte + nbytesread
            # Point the the file pointer to the start of a block of 3 or 4 and
            # keep track of how many samples to discard after reading. For
            # regular formats the file pointer is already at the correct
            # location.
            if fmt == '212':
                # Extra samples to read
                floorsamp = (startbyte - byteoffset) % 3
                # Now the byte pointed to is the first of a byte triplet
                # storing 2 samples. It happens that the extra samples match
                # the extra bytes for fmt 212
                startbyte = startbyte - floorsamp
            elif (fmt == '310') | (fmt == '311'):
                floorsamp = (startbyte - byteoffset) % 4
                # Now the byte pointed to is the first of a byte quartet
                # storing 3 samples.
                startbyte = startbyte - floorsamp
            startbyte = startbyte
            fp.seek(startbyte)
            # The length of extra signals to be loaded
            extraloadlen = min(siglen - (sampto - sampfrom), max(skew))
            nsampextra = extraloadlen * tsampsperframe
            extraloadedsig = processwfdbbytes(
                fp,
                fmt,
                extraloadlen,
                nsig,
                sampsperframe,
                floorsamp)[0]  # Array of extra loaded samples
            # Fill in the extra loaded samples
            extrasig[:extraloadedsig.shape[0], :] = extraloadedsig

        # Fill in the skewed channels with the appropriate values.
        for ch in range(0, nsig):
            if skew[ch] > 0:
                sig[:-skew[ch], ch] = sig[skew[ch]:, ch]
                sig[-skew[ch]:, ch] = extrasig[:skew[ch], ch]

    fp.close()

    return sig


# Read data from a wfdb dat file and process it to obtain digital samples.
# Returns the signal and the number of bytes read.
def processwfdbbytes(fp, fmt, siglen, nsig, sampsperframe, floorsamp=0):
    # siglen refers to the length of the signal to be read. Different from siglen input argument for readdat.
    # floorsamp is the extra sample index used to read special formats.

    tsampsperframe = sum(sampsperframe)  # Total number of samples per frame.
    # Total number of signal samples to be collected (including discarded ones)
    nsamp = siglen * tsampsperframe + floorsamp

    # Reading the dat file into np array and reshaping. Formats 212, 310, and 311 need special processing.
    # Note that for these formats with multi samples/frame, have to convert
    # bytes to samples before returning average frame values.
    if fmt == '212':
        # The number of bytes needed to be loaded given the number of samples
        # needed
        nbytesload = int(np.ceil((nsamp) * 1.5))
        sigbytes = np.fromfile(
            fp,
            dtype=np.dtype(
                datatypes[fmt]),
            count=nbytesload).astype('uint')  # Loaded as unsigned 1 byte blocks

        if tsampsperframe == nsig:  # No extra samples/frame
            # Turn the bytes into actual samples.
            sig = np.zeros(nsamp)  # 1d array of actual samples
            # One sample pair is stored in one byte triplet.
            sig[0::2] = sigbytes[0::3] + 256 * \
                np.bitwise_and(sigbytes[1::3], 0x0f)  # Even numbered samples
            if len(sig > 1):
                # Odd numbered samples
                sig[1::2] = sigbytes[2::3] + 256 * \
                    np.bitwise_and(sigbytes[1::3] >> 4, 0x0f)
            if floorsamp:  # Remove extra sample read
                sig = sig[floorsamp:]
            # Reshape into final array of samples
            sig = sig.reshape(siglen, nsig)
            sig = sig.astype(int)
            # Loaded values as unsigned. Convert to 2's complement form: values
            # > 2^11-1 are negative.
            sig[sig > 2047] -= 4096
        else:  # At least one channel has multiple samples per frame. All extra samples are discarded.
            # Turn the bytes into actual samples.
            sigall = np.zeros(nsamp)  # 1d array of actual samples
            sigall[0::2] = sigbytes[0::3] + 256 * \
                np.bitwise_and(sigbytes[1::3], 0x0f)  # Even numbered samples

            if len(sigall) > 1:
                # Odd numbered samples
                sigall[1::2] = sigbytes[2::3] + 256 * \
                    np.bitwise_and(sigbytes[1::3] >> 4, 0x0f)
            if floorsamp:  # Remove extra sample read
                sigall = sigall[floorsamp:]
            # Convert to int64 to be able to hold -ve values
            sigall = sigall.astype('int')
            # Loaded values as unsigned. Convert to 2's complement form: values
            # > 2^11-1 are negative.
            sigall[sigall > 2047] -= 4096
            # Give the average sample in each frame for each channel
            sig = np.zeros([siglen, nsig])
            for ch in range(0, nsig):
                if sampsperframe[ch] == 1:
                    sig[:, ch] = sigall[
                        sum(([0] + sampsperframe)[:ch + 1])::tsampsperframe]
                else:
                    for frame in range(0, sampsperframe[ch]):
                        sig[:, ch] += sigall[sum(([0] + sampsperframe)
                                                 [:ch + 1]) + frame::tsampsperframe]
            sig = (sig / sampsperframe).astype('int')

    elif fmt == '310':  # Three 10 bit samples packed into 4 bytes with 2 bits discarded

        # The number of bytes needed to be loaded given the number of samples
        # needed
        nbytesload = int(((nsamp) + 2) / 3.) * 4
        if (nsamp - 1) % 3 == 0:
            nbytesload -= 2
        sigbytes = np.fromfile(
            fp,
            dtype=np.dtype(
                datatypes[fmt]),
            count=nbytesload).astype('uint')  # Loaded as unsigned 1 byte blocks
        if tsampsperframe == nsig:  # No extra samples/frame
            # Turn the bytes into actual samples.
            # 1d array of actual samples. Fill the individual triplets.
            sig = np.zeros(nsamp)

            sig[0::3] = (sigbytes[0::4] >> 1)[0:len(sig[0::3])] + 128 * \
                np.bitwise_and(sigbytes[1::4], 0x07)[0:len(sig[0::3])]
            if len(sig > 1):
                sig[1::3] = (sigbytes[2::4] >> 1)[0:len(sig[1::3])] + 128 * \
                    np.bitwise_and(sigbytes[3::4], 0x07)[0:len(sig[1::3])]
            if len(sig > 2):
                sig[2::3] = np.bitwise_and((sigbytes[1::4] >> 3), 0x1f)[0:len(
                    sig[2::3])] + 32 * np.bitwise_and(sigbytes[3::4] >> 3, 0x1f)[0:len(sig[2::3])]
            # First signal is 7 msb of first byte and 3 lsb of second byte.
            # Second signal is 7 msb of third byte and 3 lsb of forth byte
            # Third signal is 5 msb of second byte and 5 msb of forth byte

            if floorsamp:  # Remove extra sample read
                sig = sig[floorsamp:]
            # Reshape into final array of samples
            sig = sig.reshape(siglen, nsig)
            # Convert to int64 to be able to hold -ve values
            sig = sig.astype('int')
            # Loaded values as unsigned. Convert to 2's complement form: values
            # > 2^9-1 are negative.
            sig[sig > 511] -= 1024

        else:  # At least one channel has multiple samples per frame. All extra samples are discarded.
            # Turn the bytes into actual samples.
            # 1d array of actual samples. Fill the individual triplets.
            sigall = np.zeros(nsamp)
            sigall[0::3] = (sigbytes[0::4] >> 1)[0:len(
                sigall[0::3])] + 128 * np.bitwise_and(sigbytes[1::4], 0x07)[0:len(sigall[0::3])]
            if len(sigall > 1):
                sigall[1::3] = (sigbytes[2::4] >> 1)[0:len(
                    sigall[1::3])] + 128 * np.bitwise_and(sigbytes[3::4], 0x07)[0:len(sigall[1::3])]
            if len(sigall > 2):
                sigall[2::3] = np.bitwise_and((sigbytes[1::4] >> 3), 0x1f)[0:len(
                    sigall[2::3])] + 32 * np.bitwise_and(sigbytes[3::4] >> 3, 0x1f)[0:len(sigall[2::3])]
            if floorsamp:  # Remove extra sample read
                sigall = sigall[floorsamp:]
            # Convert to int64 to be able to hold -ve values
            sigall = sigall.astype('int')
            # Loaded values as unsigned. Convert to 2's complement form: values
            # > 2^9-1 are negative.
            sigall[sigall > 511] -= 1024

            # Give the average sample in each frame for each channel
            sig = np.zeros([siglen, nsig])
            for ch in range(0, nsig):
                if sampsperframe[ch] == 1:
                    sig[:, ch] = sigall[
                        sum(([0] + sampsperframe)[:ch + 1])::tsampsperframe]
                else:
                    for frame in range(0, sampsperframe[ch]):
                        sig[:, ch] += sigall[sum(([0] + sampsperframe)
                                                 [:ch + 1]) + frame::tsampsperframe]
            sig = (sig / sampsperframe).astype('int')

    elif fmt == '311':  # Three 10 bit samples packed into 4 bytes with 2 bits discarded
        nbytesload = int((nsamp - 1) / 3.) + nsamp + 1
        sigbytes = np.fromfile(
            fp,
            dtype=np.dtype(
                datatypes[fmt]),
            count=nbytesload).astype('uint')  # Loaded as unsigned 1 byte blocks

        if tsampsperframe == nsig:  # No extra samples/frame
            # Turn the bytes into actual samples.
            # 1d array of actual samples. Fill the individual triplets.
            sig = np.zeros(nsamp)

            sig[0::3] = sigbytes[0::4][
                0:len(sig[0::3])] + 256 * np.bitwise_and(sigbytes[1::4], 0x03)[0:len(sig[0::3])]
            if len(sig > 1):
                sig[1::3] = (sigbytes[1::4] >> 2)[0:len(sig[1::3])] + 64 * \
                    np.bitwise_and(sigbytes[2::4], 0x0f)[0:len(sig[1::3])]
            if len(sig > 2):
                sig[2::3] = (sigbytes[2::4] >> 4)[0:len(sig[2::3])] + 16 * \
                    np.bitwise_and(sigbytes[3::4], 0x7f)[0:len(sig[2::3])]
            # First signal is first byte and 2 lsb of second byte.
            # Second signal is 6 msb of second byte and 4 lsb of third byte
            # Third signal is 4 msb of third byte and 6 msb of forth byte
            if floorsamp:  # Remove extra sample read
                sig = sig[floorsamp:]
            # Reshape into final array of samples
            sig = sig.reshape(siglen, nsig)
            # Convert to int64 to be able to hold -ve values
            sig = sig.astype('int')
            # Loaded values as unsigned. Convert to 2's complement form: values
            # > 2^9-1 are negative.
            sig[sig > 511] -= 1024

        else:  # At least one channel has multiple samples per frame. All extra samples are discarded.
            # Turn the bytes into actual samples.
            # 1d array of actual samples. Fill the individual triplets.
            sigall = np.zeros(nsamp)
            sigall[
                0::3] = sigbytes[
                0::4][
                0:len(
                    sigall[
                        0::3])] + 256 * np.bitwise_and(
                sigbytes[
                    1::4], 0x03)[
                0:len(
                    sigall[
                        0::3])]
            if len(sigall > 1):
                sigall[1::3] = (sigbytes[1::4] >> 2)[0:len(
                    sigall[1::3])] + 64 * np.bitwise_and(sigbytes[2::4], 0x0f)[0:len(sigall[1::3])]
            if len(sigall > 2):
                sigall[2::3] = (sigbytes[2::4] >> 4)[0:len(
                    sigall[2::3])] + 16 * np.bitwise_and(sigbytes[3::4], 0x7f)[0:len(sigall[2::3])]
            if floorsamp:  # Remove extra sample read
                sigall = sigall[floorsamp:]
            # Convert to int64 to be able to hold -ve values
            sigall = sigall.astype('int')
            # Loaded values as unsigned. Convert to 2's complement form: values
            # > 2^9-1 are negative.
            sigall[sigall > 511] -= 1024
            # Give the average sample in each frame for each channel
            sig = np.zeros([siglen, nsig])
            for ch in range(0, nsig):
                if sampsperframe[ch] == 1:
                    sig[:, ch] = sigall[
                        sum(([0] + sampsperframe)[:ch + 1])::tsampsperframe]
                else:
                    for frame in range(0, sampsperframe[ch]):
                        sig[:, ch] += sigall[sum(([0] + sampsperframe)
                                                 [:ch + 1]) + frame::tsampsperframe]
            sig = (sig / sampsperframe).astype('int')

    else:  # Simple format signals that can be loaded as they are stored.

        if tsampsperframe == nsig:  # No extra samples/frame
            sig = np.fromfile(fp, dtype=np.dtype(datatypes[fmt]), count=nsamp)
            sig = sig.reshape(siglen, nsig).astype('int')
        else:  # At least one channel has multiple samples per frame. Extra samples are discarded.
            sigall = np.fromfile(fp,
                                 dtype=np.dtype(datatypes[fmt]),
                                 count=nsamp)  # All samples loaded
            # Keep the first sample in each frame for each channel
            sig = np.empty([siglen, nsig])
            for ch in range(0, nsig):
                if sampsperframe[ch] == 1:
                    sig[:, ch] = sigall[
                        sum(([0] + sampsperframe)[:ch + 1])::tsampsperframe]
                else:
                    for frame in range(0, sampsperframe[ch]):
                        sig[:, ch] += sigall[sum(([0] + sampsperframe)
                                                 [:ch + 1]) + frame::tsampsperframe]
            sig = (sig / sampsperframe).astype('int')
        # Correct byte offset format data
        if fmt == '80':
            sig = sig - 128
        elif fmt == '160':
            sig = sig - 32768
        nbytesload = nsamp * bytespersample[fmt]

    return sig, nbytesload

# Bytes required to hold each sample (including wasted space) for
# different wfdb formats
bytespersample = {'8': 1, '16': 2, '24': 3, '32': 4, '61': 2,
                  '80': 1, '160': 2, '212': 1.5, '310': 4 / 3., '311': 4 / 3.}

# Values that correspond to NAN (Format 8 has no NANs)
wfdbInvalids = {
    '16': -32768,
    '24': -8388608,
    '32': -2147483648,
    '61': -32768,
    '80': -128,
    '160': -32768,
    '212': -2048,
    '310': -512,
    '311': -512}

# Data type objects for each format to load. Doesn't directly correspond
# for final 3 formats.
datatypes = {'8': '<i1', '16': '<i2', '24': '<i3', '32': '<i4',
             '61': '>i2', '80': '<u1', '160': '<u2',
             '212': '<u1', '310': '<u1', '311': '<u1'}

# Channel dependent field items that need to be rearranged if input
# channels is not default.
arrangefields = [
    "filename",
    "fmt",
    "sampsperframe",
    "skew",
    "byteoffset",
    "gain",
    "units",
    "baseline",
    "initvalue",
    "signame"]

def loadconfig(fn):
    """
    Search for a configuration file. Load the first version found.
    """
    config = ConfigParser()
    for loc in [os.curdir,os.path.expanduser("~"),os.path.dirname(__file__)]:
        try: 
            with open(os.path.join(loc,fn)) as source:
                config.readfp(source)
        except IOError:
            pass
    return config

def rdsamp(
        recordname,
        sampfrom=0,
        sampto=[],
        channels=[],
        physical=1,
        stacksegments=1):
    """Read a WFDB record and return the signal as a numpy array and the metadata as a dictionary.

    Usage:
    sig, fields = rdsamp(recordname, sampfrom, sampto, channels, physical, stacksegments)

    Input arguments:
    - recordname (required): The name of the WFDB record to be read (without any file extensions).
    - sampfrom (default=0): The starting sample number to read for each channel.
    - sampto (default=length of entire signal): The final sample number to read for each channel.
    - channels (default=all channels): Indices specifying the channel to be returned.
    - physical (default=1): Flag that specifies whether to return signals in physical (1) or digital (0) units.
    - stacksegments (default=1): Flag used only for multi-segment files. Specifies whether to return the signal as a single stacked/concatenated numpy array (1) or as a list of one numpy array for each segment (0).


    Output variables:
    - sig: An nxm numpy array where n is the signal length and m is the number of channels. 
      If the input record is a multi-segment record, depending on the input stacksegments flag, 
      sig will either be a single stacked/concatenated numpy array (1) or a list of one numpy 
      array for each segment (0). For empty segments, stacked format will contain Nan values, 
      and non-stacked format will contain a single integer specifying the length of the empty segment.
    - fields: A dictionary of metadata about the record extracted or deduced from the header/signal file. 
      If the input record is a multi-segment record, the output argument will be a list of dictionaries:
              : The first list element will be a dictionary of metadata about the master header.
              : If the record is in variable layout format, the next list element will be a dictionary 
                of metadata about the layout specification header.
              : The last list element will be a list of dictionaries of metadata for each segment. 
                For empty segments, the dictionary will be replaced by a single string: 'Empty Segment'
    """

    filestoremove = []
    config = loadconfig('wfdb.config')

    if config.get('pbdownload','getpbfiles'):  # Flag specifying whether to allow downloading from physiobank
        recordname, dledfiles = checkrecordfiles(recordname, os.getcwd())
        if not int(config.get('pbdownload','keepdledfiles')):  # Flag specifying whether to keep downloaded physiobank files
            filestoremove = dledfiles

    fields = readheader(recordname)  # Get the info from the header file

    if fields["nsig"] == 0:
        sys.exit("This record has no signals. Use rdann to read annotations")
    if sampfrom < 0:
        sys.exit("sampfrom must be non-negative")
    dirname, baserecordname = os.path.split(recordname)

    if fields["nseg"] == 1:  # single segment file
        if (len(set(fields["filename"])) ==
                1):  # single dat (or binary) file in the segment
            # Signal length was not specified in the header, calculate it from
            # the file size.
            if not fields["nsamp"]:
                filesize = os.path.getsize(
                    os.path.join(dirname, fields["filename"][0]))
                fields["nsamp"] = int(
                    (filesize -
                     fields["byteoffset"][0]) /
                    fields["nsig"] /
                    bytespersample[
                        fields["fmt"][0]])
            if sampto:
                if sampto > fields["nsamp"]:
                    sys.exit(
                        "sampto exceeds length of record: ",
                        fields["nsamp"])
            else:
                sampto = fields["nsamp"]

            sig = readdat(
                os.path.join(
                    dirname,
                    fields["filename"][0]),
                fields["fmt"][0],
                fields["byteoffset"][0],
                sampfrom,
                sampto,
                fields["nsig"],
                fields["nsamp"],
                fields["sampsperframe"],
                fields["skew"])

            if channels:  # User input channels
                # If input channels are not equal to the full set of ordered
                # channels, remove the non-included ones and reorganize the
                # remaining ones that require it.
                if channels != list(range(0, fields["nsig"])):
                    sig = sig[:, channels]

                    # Rearrange the channel dependent fields
                    for fielditem in arrangefields:
                        fields[fielditem] = [fields[fielditem][ch]
                                             for ch in channels]
                    fields["nsig"] = len(channels)  # Number of signals.

            if (physical == 1) & (fields["fmt"] != '8'):
                # Insert nans/invalid samples.
                sig = sig.astype(float)
                sig[sig == wfdbInvalids[fields["fmt"][0]]] = np.nan
                sig = np.subtract(sig, np.array(
                    [float(i) for i in fields["baseline"]]))
                sig = np.divide(sig, np.array([fields["gain"]]))

        else:  # Multiple dat files in the segment. Read different dats and merge them channel wise.

            if not channels:  # Default all channels
                channels = list(range(0, fields["nsig"]))

            # Get info to arrange channels correctly
            filenames = []  # The unique dat files to open,
            filechannels = {}  # All the channels that each dat file contains
            # The channels for each dat file to be returned and the
            # corresponding target output channels
            filechannelsout = {}
            for ch in list(range(0, fields["nsig"])):
                sigfile = fields["filename"][ch]
                # Correspond every channel to a dat file
                if sigfile not in filechannels:
                    filechannels[sigfile] = [ch]
                else:
                    filechannels[sigfile].append(ch)
                # Get only relevant files/channels to load
                if ch in channels:
                    if sigfile not in filenames:  # Newly encountered dat file
                        filenames.append(sigfile)
                        filechannelsout[sigfile] = [
                            [filechannels[sigfile].index(ch), channels.index(ch)]]
                    else:
                        filechannelsout[sigfile].append(
                            [filechannels[sigfile].index(ch), channels.index(ch)])

            # Allocate final output array
            # header doesn't have signal length. Figure it out from the first
            # dat to read.
            if not fields["nsamp"]:
                filesize = os.path.getsize(
                    os.path.join(dirname, fields["filename"][0]))
                fields["nsamp"] = int((filesize -
                                       fields["byteoffset"][channels[0]]) /
                                      len(filechannels[filenames[0]]) /
                                      bytespersample[fields["fmt"][channels[0]]])

            if not sampto:  # if sampto field is empty
                sampto = fields["nsamp"]
            if physical == 0:
                sig = np.empty([sampto - sampfrom, len(channels)], dtype='int')
            else:
                sig = np.empty([sampto - sampfrom, len(channels)])

            # So we do need to rearrange the fields below for the final output
            # 'fields' item and even for the physical conversion below. The
            # rearranged fields will only contain the fields of the selected
            # channels in their specified order. But before we do this, we need
            # to extract sampsperframe and skew for the individual dat files
            filesampsperframe = {}  # The sampsperframe for each channel in each file
            fileskew = {}  # The skew for each channel in each file
            for sigfile in filenames:
                filesampsperframe[sigfile] = [fields["sampsperframe"][
                    datchan] for datchan in filechannels[sigfile]]
                fileskew[sigfile] = [fields["skew"][datchan]
                                     for datchan in filechannels[sigfile]]

            # Rearrange the fields given the input channel dependent fields
            for fielditem in arrangefields:
                fields[fielditem] = [fields[fielditem][ch] for ch in channels]
            fields["nsig"] = len(channels)  # Number of signals.

            # Read signal files one at a time and store the relevant channels
            for sigfile in filenames:
                sig[:,
                    [outchan[1] for outchan in filechannelsout[sigfile]]] = readdat(os.path.join(dirname,
                                                                                                 sigfile),
                                                                                    fields["fmt"][fields["filename"].index(sigfile)],
                                                                                    fields["byteoffset"][fields["filename"].index(sigfile)],
                                                                                    sampfrom,
                                                                                    sampto,
                                                                                    len(filechannels[sigfile]),
                                                                                    fields["nsamp"],
                                                                                    filesampsperframe[sigfile],
                                                                                    fileskew[sigfile])[:,
                                                                                                       [datchan[0] for datchan in filechannelsout[sigfile]]]

            if (physical == 1) & (fields["fmt"] != '8'):
                # Insert nans/invalid samples.
                sig = sig.astype(float)
                for ch in range(0, fields["nsig"]):
                    sig[sig[:, ch] == wfdbInvalids[
                        fields["fmt"][ch]], ch] = np.nan
                sig = np.subtract(sig, np.array(
                    [float(b) for b in fields["baseline"]]))
                sig = np.divide(sig, np.array([fields["gain"]]))

    # Multi-segment file. Preprocess and recursively call rdsamp on single
    # segments.
    else:

        # Determine if this record is fixed or variable layout. startseg is the
        # first signal segment.
        if fields["nsampseg"][
                0] == 0:  # variable layout - first segment is layout specification file
            startseg = 1
            # Store the layout header info.
            layoutfields = readheader(
                os.path.join(
                    dirname,
                    fields["filename"][0]))
        else:  # fixed layout - no layout specification file.
            startseg = 0

        # Determine the segments and samples that have to be read based on
        # sampfrom and sampto
        # Cumulative sum of segment lengths
        cumsumlengths = list(np.cumsum(fields["nsampseg"][startseg:]))

        if not sampto:
            sampto = cumsumlengths[len(cumsumlengths) - 1]

        if sampto > cumsumlengths[
                len(cumsumlengths) -
                1]:  # Error check sampto
            sys.exit(
                "sampto exceeds length of record: ",
                cumsumlengths[
                    len(cumsumlengths) - 1])

        # First segment
        readsegs = [[sampfrom < cs for cs in cumsumlengths].index(True)]
        if sampto == cumsumlengths[len(cumsumlengths) - 1]:
            readsegs.append(len(cumsumlengths) - 1)  # Final segment
        else:
            readsegs.append([sampto < cs for cs in cumsumlengths].index(True))
        if readsegs[1] == readsegs[0]:  # Only one segment to read
            readsegs = [readsegs[0]]
            # The sampfrom and sampto for each segment
            readsamps = [[sampfrom, sampto]]
        else:
            readsegs = list(
                range(
                    readsegs[0],
                    readsegs[1] +
                    1))  # Expand into list
            # The sampfrom and sampto for each segment. Most are [0, nsampseg]
            readsamps = [[0, fields["nsampseg"][s + startseg]]
                         for s in readsegs]
            # Starting sample for first segment
            readsamps[0][0] = sampfrom - ([0] + cumsumlengths)[readsegs[0]]
            readsamps[len(readsamps) - 1][1] = sampto - ([0] + cumsumlengths)[
                readsegs[len(readsamps) - 1]]  # End sample for last segment
        # Done obtaining segments to read and samples to read in each segment.

        # Preprocess/preallocate according to chosen output format
        if not channels:
            channels = list(range(0, fields["nsig"]))  # All channels
        if stacksegments == 1:  # Output a single concatenated numpy array
            # Figure out the dimensions of the entire signal
            if sampto:  # User inputs sampto
                nsamp = sampto
            elif fields["nsamp"]:  # master header has number of samples
                nsamp = fields["nsamp"]
            else:  # Have to figure out length by adding up all segment lengths
                nsamp = sum(fields["nsampseg"][startseg:])
            nsamp = nsamp - sampfrom
            if physical == 0:
                sig = np.empty((nsamp, len(channels)), dtype='int')
            else:
                sig = np.empty((nsamp, len(channels)), dtype='float')

            indstart = 0  # The overall signal index of the start of the current segment for 
                          # filling in the large np array
        else:  # Output a list of numpy arrays
            # Empty list for storing segment signals.
            sig = [None] * len(readsegs)
        # List for storing segment fields.
        segmentfields = [None] * len(readsegs)

        # Read and store segments one at a time.
        # i is the segment number. It accounts for the layout record if exists
        # and skips past it.
        for i in [r + startseg for r in readsegs]:
            segrecordname = fields["filename"][i]

            # Work out the relative channels to return from each segment
            if startseg == 0:  # Fixed layout signal. Channels for record are always same.
                segchannels = channels
            else:  # Variable layout signal. Work out which channels from the segment to load if any.
                if segrecordname != '~':
                    sfields = readheader(os.path.join(dirname, segrecordname))
                    wantsignals = [layoutfields["signame"][c]
                                   for c in channels]  # Signal names of wanted channels
                    segchannels = []  # The channels wanted that are contained in the segment
                    returninds = []  # 1 and 0 marking channels of the numpy array to be filled by 
                                     # the returned segment channels
                    for ws in wantsignals:
                        if ws in sfields["signame"]:
                            segchannels.append(sfields["signame"].index(ws))
                            returninds.append(1)
                        else:
                            returninds.append(0)
                    returninds = np.array(returninds)
                    # The channels of the overall array that the segment
                    # doesn't contain
                    emptyinds = np.where(returninds == 0)[0]
                    # The channels of the overall array that the segment
                    # contains
                    returninds = np.where(returninds == 1)[0]
                else:
                    segchannels = []

            if stacksegments == 0:  # Return list of np arrays
                # Empty segment or wrong channels. Store indicator and segment
                # length.
                if (segrecordname == '~') | (not segchannels):
                    # sig[i-startseg-readsegs[0]]=fields["nsampseg"][i] # store
                    # the segment length
                    sig[i - startseg - readsegs[0]] = readsamps[i - startseg - \
                        readsegs[0]][1] - readsamps[i - startseg - readsegs[0]][0]
                    segmentfields[i - startseg - readsegs[0]] = "Empty Segment"

                else:  # Non-empty segment that contains wanted channels. Read its signal and header fields
                    sig[i -
                        startseg -
                        readsegs[0]], segmentfields[i -
                                                    startseg -
                                                    readsegs[0]] = rdsamp(recordname=os.path.join(dirname, 
                                                        segrecordname), physical=physical, sampfrom=readsamps[i -
                                                                                                                                                                 startseg -
                                                                                                                                                                 readsegs[0]][0], sampto=readsamps[i -
                                                                                                                                                                                                   startseg -
                                                                                                                                                                                                   readsegs[0]][1], channels=segchannels)

            else:  # Return single stacked np array of all (selected) channels
                indend = indstart + readsamps[i - startseg - readsegs[0]][1] - readsamps[
                    i - startseg - readsegs[0]][0]  # end index of the large array for this segment
                if (segrecordname == '~') | (
                        not segchannels):  # Empty segment or no wanted channels: fill in invalids
                    if physical == 0:
                        sig[indstart:indend, :] = sig[
                            indstart:indend, :] = -2147483648
                    else:
                        sig[indstart:indend, :] = np.nan
                    segmentfields[i - startseg - readsegs[0]] = "Empty Segment"
                else:  # Non-empty segment - Get samples
                    if startseg == 1:  # Variable layout format. Load data then rearrange channels.
                        sig[indstart:indend, returninds], segmentfields[i -
                                                                        startseg -
                                                                        readsegs[0]] = rdsamp(recordname=os.path.join(dirname, segrecordname), physical=physical, sampfrom=readsamps[i -
                                                                                                                                                                                     startseg -
                                                                                                                                                                                     readsegs[0]][0], sampto=readsamps[i -
                                                                                                                                                                                                                       startseg -
                                                                                                                                                                                                                       readsegs[0]][1], channels=segchannels)  # Load all the wanted channels that the segment contains
                        if physical == 0:  # Fill the rest with invalids
                            sig[indstart:indend, emptyinds] = -2147483648
                        else:
                            sig[indstart:indend, emptyinds] = np.nan

                        # Expand the channel dependent fields to match the
                        # overall layout. Remove this following block if you
                        # wish to keep the returned channels' fields without
                        # any 'no channel' placeholders between.
                        expandedfields = dict.copy(
                            segmentfields[i - startseg - readsegs[0]])
                        for fielditem in arrangefields:
                            expandedfields[fielditem] = [
                                'No Channel'] * len(channels)
                            for c in range(0, len(returninds)):
                                expandedfields[fielditem][
                                    returninds[c]] = segmentfields[
                                    i - startseg - readsegs[0]][fielditem][c]
                            segmentfields[
                                i - startseg - readsegs[0]][fielditem] = expandedfields[fielditem]
                        # Keep fields['nsig'] as the value of returned channels
                        # from the segments.

                    else:  # Fixed layout - channels already arranged
                        sig[
                            indstart:indend, :], segmentfields[
                            i - startseg] = rdsamp(
                            recordname=os.path.join(
                                dirname, segrecordname), physical=physical, sampfrom=readsamps[
                                i - startseg - readsegs[0]][0], sampto=readsamps[
                                i - startseg - readsegs[0]][1], channels=segchannels)
                indstart = indend  # Update the start index for filling in the next part of the array

        # Done reading all segments

        # Return a list of fields. First element is the master, next is layout
        # if any, last is a list of all the segment fields.
        if startseg == 1:  # Variable layout format
            fields = [fields, layoutfields, segmentfields]
        else:  # Fixed layout format.
            fields = [fields, segmentfields]

    if filestoremove:
        for fr in filestoremove:
            os.remove(fr)

    return (sig, fields)


if __name__ == '__main__':
    rdsamp(sys.argv)