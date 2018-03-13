'''

GIRMOS toy model for sensitivity calculations.
The code assumes the source is a point source with a single emission line.
It uses realistic sky emission and absorption spectra as well as spectrograph parameters.
GIRMOS parameters include AO EE, optical throughput, spectral resolution, detector noise, and telescope+canopus emissivity.

Author: Suresh Sivanandam
Version: 03052018

'''
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.constants as const
import scipy.ndimage.filters as flt
import scipy.special as special
import sys
from scipy.optimize import curve_fit

plt.close('all')
plt.ion()

c = const.c
h = const.h 
k = const.k
pi = const.pi 

# Instrument Modes
## Change parameters here:
# Image Low-res mode = 0 (0.1" resolution), Image High-res mode = 1 (0.05" resolution)
imgmode = 0
# Spec Low-res mode = 0 (R=3000), High-res mode = 1 (R=9000)
specmode = 0

# Galaxy Parameters
## Change parameters here:
zgal = 1.5
emissionline = 656.3 #H_alpha (nm)
velocitywidth = 30  # km/s
lineflux = 5e-20 # W/m^2 (1e18 erg/s/cm^2 = 1e21 W/m^2)
spatialscale = 0.05 if imgmode > 0 else 0.1 #arcsec

# Science Exposure Parameters
## Change parameters here:
# 3 hour on-source integration (individual exposures 6 minutes long)
# Code assumes ABBA sky subtraction, so a 3-hour on-source integration requires 6 hours of observations
Texp = 6 * 60 #s
NumExp = 30
ReadNoise = 5 #e
DarkCurrent = 0.02 #e/s

def Gaussian(wave,dwave,mean,fwhm,flux):
    sigma = fwhm/2.355

    gauss = np.exp(-0.5*((wave-mean)/sigma)**2)/(sigma*math.sqrt(2*pi))    

    return(flux*gauss*dwave)

def GaussianFit(wave,mean,fwhm,flux):
    sigma = fwhm/2.355

    gauss = np.exp(-0.5*((wave-mean)/sigma)**2)/(sigma*math.sqrt(2*pi))    

    return(flux*gauss)

def Planck(wave, temp):
    """
    Input: Wavelength values (m)
    Returns: B_nu(wave,temp), the Planck distribution function in SI units
    Units: ph/s/arcsec^2/nm/m^2
    """
    
    numer = 2.*h*(c)**2/wave**5
    denom = np.exp(h*c/(k*temp*wave))-1.

    SIPlanck = numer/denom
    
    PhotonE = h*c/wave
    # 1 sr = 4.25e10 arcsec^2
    # 1 m = 1e9 nm
    convPlanck = SIPlanck/(4.25e10*1e9*PhotonE)
    
    return(convPlanck)

def GenerateSpectra(inputWave,inputFlux,smoothing,start,stop,sampling):
    outputWave = np.arange(start,stop,sampling)    
    
    smoothedFlux = flt.gaussian_filter1d(inputFlux,smoothing) 
    
    outputFlux = np.interp(outputWave,inputWave,smoothedFlux)
    
    return(outputWave, outputFlux)

def CalculateNoiseSpectrum(inputWave,flux,noise):

    NoiseSpectrum = np.zeros(len(inputWave))
    
    for i in range(len(inputWave)):
        NoiseSpectrum[i]  = np.random.normal(flux[i],noise[i])

    return(NoiseSpectrum)    

def PSF(radii,strehl,innerscale, outerscale):
    airy = (special.j1(pi*radii/(2*innerscale))/(pi*radii/(2*innerscale)))**2
    lorentz = 1/(1+(radii/outerscale)**2)

    dr = radii[1]-radii[0]
    normalize = np.sum(airy*2*pi*radii*dr)

    airy *= strehl/normalize    
    print('Airy: %f' % np.sum(airy*2*pi*radii*dr))    
    
    lorentz *= (1-strehl)
    print('Lorentz: %f' % np.sum(lorentz*2*pi*radii*dr))
    
    return(airy+lorentz)
        
# Gemini Sky Background (ph/s/arcsec^2/nm/m^2)
SkyBkg = np.loadtxt('cp_skybg_zm_76_15_ph_NIR.dat')
# Gemini Sky Transmission (PWV 7.6mm, Airmass 1.5)
SkyTrans = np.loadtxt('cptrans_zm_76_15_NIR.dat')

# Low-res, and high-res spectroscopy definitions
lowr = 3000
highr = 9000

# Telescope Parameters
D = 7.9 #m
Dobs = 1.2 #m

# Some telescope/instrument parameter assumptions
TelTrans = (0.96)**4
CanopusTrans = (0.96)**7

TelArea = math.pi*((D/2.0)**2-(Dobs/2)**2)

# Instrument Parameters
SliceWidth = 0.025 if imgmode > 0 else 0.05 #arcsec
PixelSize = 0.025 if imgmode > 0 else 0.05 #arcsec

Omega = SliceWidth*PixelSize

QE = 0.85
SpecTrans = 0.45
Emissivity = 0.4
StrehlEfficiency = 0.3 if imgmode > 0 else 0.5

# J-band calculation, smoothing is 9 pixels (R ~ 2900 @ 1.2um)
# H-band calculation, smoothing is 11 pixels (R ~ 3100 @ 1.6 um)
# K-band calculation, smoothing is 15 pixels (R ~ 3000 @ 2.1 um)

# J-band Calculation
R = highr if specmode > 0 else lowr
Jsampling = 1200/(2.0*R)
smoothing = 3 if specmode > 0 else 9

Jmin = 1050
Jmax = 1330

outputJWave, outputJFlux = GenerateSpectra(SkyBkg[:,0],SkyBkg[:,1],smoothing,Jmin,Jmax,Jsampling)
outputJWave, outputJTrans = GenerateSpectra(SkyTrans[:,0]*1000,SkyTrans[:,1],smoothing,Jmin,Jmax,Jsampling)
outputJFlux *= Jsampling


# H-band Calculation
R = highr if specmode > 0 else lowr
Hsampling = 1600/(2.0*R)
smoothing = 3.5 if specmode > 0 else 11

Hmin = 1490
Hmax = 1780

outputHWave, outputHFlux = GenerateSpectra(SkyBkg[:,0],SkyBkg[:,1]+Emissivity*Planck(SkyBkg[:,0]/1e9,280),smoothing,Hmin,Hmax,Hsampling)
outputHWave, outputHTrans = GenerateSpectra(SkyTrans[:,0]*1000,SkyTrans[:,1],smoothing,Hmin,Hmax,Hsampling)
outputHFlux *= Hsampling


# K-band Calculation
R = highr if specmode > 0 else lowr
Ksampling = 2100/(2.0*R)
smoothing = 4 if specmode > 0 else 15

Kmin = 2030
Kmax = 2370

outputKWave, outputKFlux = GenerateSpectra(SkyBkg[:,0],SkyBkg[:,1]+Emissivity*Planck(SkyBkg[:,0]/1e9,280),smoothing,Kmin,Kmax,Ksampling)
outputKWave, outputKTrans = GenerateSpectra(SkyTrans[:,0]*1000,SkyTrans[:,1],smoothing,Kmin,Kmax,Ksampling)
outputKFlux *= Ksampling

# Number of spaxels (should always be 4)
Spaxels = spatialscale**2/Omega if imgmode > 0 else spatialscale**2/Omega
print("Size of Spatial Resolution Element: %f\"" % spatialscale)
print("Spaxel Size: %f\"" % SliceWidth)
print("Number of Spaxels Required: %d" % Spaxels)
print("Spectral Resolution (R): %d" % R)

deltalambda = emissionline * (velocitywidth)/(const.c/1000.0)
actualWavelength = emissionline*(1+zgal)
actualWidth = deltalambda*(1+zgal)

if((actualWavelength<Jmax)&(actualWavelength>Jmin)):
    outputWave = outputJWave
    outputFlux = outputJFlux
    outputTrans = outputJTrans
    sampling = Jsampling
    print("J-band Spectroscopy")
    print("Full Band Npix Required: %d" % len(outputJWave))
elif((actualWavelength<Hmax)&(actualWavelength>Hmin)):
    outputWave = outputHWave
    outputFlux = outputHFlux
    outputTrans = outputHTrans
    sampling = Hsampling
    print("H-band Spectroscopy")
    print("Full Band Npix Required: %d" % len(outputHWave))
elif((actualWavelength<Kmax)&(actualWavelength>Kmin)):
    outputWave = outputKWave
    outputFlux = outputKFlux
    outputTrans = outputKTrans
    sampling = Ksampling
    print("K-band Spectroscopy")
    print("Full Band Npix Required: %d" % len(outputKWave))
else:
    print("Spec Line Out of Range!")
    print("Actual Wavelength: %f" % actualWavelength)
    sys.exit()
    
    
plt.figure()
plt.plot(outputWave,outputFlux)
plt.xlabel('Wavelength (nm)')  
plt.ylabel('Sky+Thermal Flux (ph s$^{-1}$ m$^{-2}$ arcsec$^{-2}$)')
plt.savefig('skyflux.png',dpi=300)
plt.figure()
plt.plot(outputWave,outputTrans)
plt.xlabel('Wavelength (nm)')  
plt.ylabel('Sky Transmission')
plt.savefig('skytrans.png',dpi=300)

# Generate galaxy spectrum
finalWidth = np.sqrt(actualWidth**2+(2*sampling)**2)
galaxySpecUnSmoothed = Gaussian(outputWave,sampling,actualWavelength,finalWidth,lineflux)/(h*c/(outputWave*1e-9))
# Two pixel FWHM spectral smoothing within spectrograph
galaxySpec = flt.gaussian_filter1d(galaxySpecUnSmoothed,2.0/2.355) 

TotalReadNoise = math.sqrt(ReadNoise**2 * NumExp + DarkCurrent*Texp*NumExp)
print("Total Integration Time: %f hrs" % (Texp*NumExp/3600.0))

FinalGalaxySpec = StrehlEfficiency*galaxySpec*outputTrans*TelTrans*CanopusTrans*SpecTrans*QE*TelArea*Texp*NumExp
FinalSkySpec = outputFlux*TelTrans*CanopusTrans*SpecTrans*QE*TelArea*Texp*NumExp*Omega*Spaxels

plt.figure()
p1, = plt.plot(outputWave,FinalGalaxySpec/NumExp)
p2, = plt.plot(outputWave,FinalSkySpec/NumExp)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Flux per Exposure (e$^{-}$)')
plt.yscale('log')
plt.ylim([0.1,10000])
plt.legend([p1,p2],['Source', 'Sky'])

#Factor of 2 in read noise and dark current because a spectral element is two pixels across
TotalNoise = np.sqrt(StrehlEfficiency*FinalGalaxySpec + 2*(FinalSkySpec + 2*Spaxels*ReadNoise**2 * NumExp + 2*Spaxels*DarkCurrent*Texp*NumExp))

SNR = FinalGalaxySpec/TotalNoise

#plt.figure()
#plt.plot(outputWave,SNR)
#plt.xlabel('Wavelength (nm)')
#plt.ylabel('SNR')

plt.figure()
FinalSpec = CalculateNoiseSpectrum(outputWave,StrehlEfficiency*FinalGalaxySpec,TotalNoise)/outputTrans
plt.plot(outputWave,FinalSpec,drawstyle="steps")
#plt.plot(outputWave,FinalSpec,'.')
plt.xlim([actualWavelength-25,actualWavelength+25])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Sky Subtracted Flux (e$^-$)')

# Fit gaussian to galaxy spectrum
g = np.where((outputWave<actualWavelength+25)&(outputWave>actualWavelength-25))
popt,pcov = curve_fit(GaussianFit,outputWave[g],FinalSpec[g],p0=[actualWavelength,0.5,500])
perr = np.sqrt(np.diag(pcov))
fitValue = GaussianFit(outputWave,*popt)
plt.plot(outputWave,fitValue,'r')
plt.ylim([-300,max(fitValue*1.1)])
plt.savefig('galaxyspec.png',dpi=300)

print("Fit Results:", popt)
print("Fit Errors", perr)
print("SNR", popt[2]/perr[2])
print("Z", popt[0]/656.3-1)
print("Sigma", np.sqrt(popt[1]**2-(2*sampling)**2)/popt[0]*c/1000.0)


## Some code for checking EE for a given PSF
#plt.figure()
#rad = np.arange(0.0001,2,0.001)

#dr = rad[1]-rad[0]
#psf = PSF(rad,StrehlEfficiency,0.042/2.0,0.25)
#print(np.sum(psf*2*pi*rad*dr))

##g = np.where(rad<0.1)
##plt.plot(rad,PSF(rad,0.05,0.05,0.4))
##print(np.sum(psf[g]*2*pi*rad[g]*dr))

#ee = np.zeros(len(psf))

#ee[0] = psf[0]*2*pi*rad[0]*dr

#for i in range(1,len(ee)):
#    ee[i] = ee[i-1]+psf[i]*2*pi*rad[i]*dr

#plt.plot(rad,ee)


