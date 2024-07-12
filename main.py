import os
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
from scipy.signal import find_peaks, savgol_filter
from astropy.io import fits
from astropy.timeseries import LombScargle, BoxLeastSquares
from lightkurve import search_lightcurve
from astroquery.mast import Observations
from scipy.fft import fft

# Setting seaborn style for all matplotlib plots
sns.set(style='whitegrid')

%matplotlib inline

# Defining the target and mission parameters
target_name = "TIC 260004324"  # TIC ID 
sector = 1  # sector number for the given TIC ID

# Function to download and process FITS file from a URL
def download_and_process_fits(url):
    try:
        print(f"Downloading data from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Ensure the request has been successful
        with fits.open(BytesIO(response.content)) as hdul:
            hdul.info()  # Print the structure of the FITS file
            data = hdul[1].data  # Access the data from the 2nd HDU
            print(data.columns.names)  # Print the names of the columns
            time = data['TIME']
            flux = data['PDCSAP_FLUX']
            valid_mask = np.isfinite(time) & np.isfinite(flux)
            time = time[valid_mask]
            flux = flux[valid_mask]
            if len(time) == 0 or len(flux) == 0:
                raise ValueError("No valid data points have been found after cleaning.")
            return {'time': time, 'flux': flux}
    except requests.HTTPError as e:
        print(f"HTTP Error processing URL {url}: {e}")
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
    return None
# Analyzing light curve using Lomb-Scargle periodogram, BLS, and FT
def analyze_light_curve(time, flux):
    # Removing NaNs and infinite values from the data
    valid_mask = np.isfinite(time) & np.isfinite(flux)
    time = time[valid_mask]
    flux = flux[valid_mask]
    
    # Ensuring the window_length is suitable for the dataset size
    window_length = min(len(time), 101)
    if window_length % 2 == 0:
        window_length -= 1  # Ensure window_length is not even
    
    try:
        flux_filtered = savgol_filter(flux, window_length=window_length, polyorder=2)
    except np.linalg.LinAlgError:
        print("Warning: SVD did not converge, skipping Savitzky-Golay filter.")
        flux_filtered = flux
    
    # Plot the raw and filtered light curves
    plt.figure(figsize=(10, 6))
    plt.plot(time, flux, 'o', markersize=2, label='Raw')
    plt.plot(time, flux_filtered, '-', label='Filtered')
    plt.xlabel("Time [days]")
    plt.ylabel("Flux [e-/s]")
    plt.legend()
    plt.title("Light Curve")
    plt.show()
    
    # Lomb-Scargle Periodogram
    frequency, power = LombScargle(time, flux_filtered).autopower()
    plt.figure(figsize=(10, 6))
    plt.plot(1 / frequency, power)
    plt.xlabel("Period [days]")
    plt.ylabel("Power")
    plt.title("Lomb-Scargle Periodogram")
    plt.show()
    
    peaks, _ = find_peaks(power, height=0.01)
    significant_periods = 1 / frequency[peaks]
    print("Significant periods (days):", significant_periods)
    
    if len(significant_periods) > 0:
        best_period = significant_periods[0]
        plt.figure(figsize=(10, 6))
        phase = (time % best_period) / best_period
        plt.plot(phase, flux_filtered, 'o', markersize=2)
        plt.xlabel("Phase")
        plt.ylabel("Flux [e-/s]")
        plt.title(f"Folded Light Curve at Period = {best_period:.4f} days")
        plt.show()
    
    # Box Least Squares (BLS) Analysis
    model = BoxLeastSquares(time, flux_filtered)
    periods = np.linspace(0.1, 1, 10000)
    duration = min(0.1, 0.1 * np.min(periods))  # Ensure that the duration is not longer than the minimum period
    result = model.power(periods, duration)
    plt.figure(figsize=(10, 6))
    plt.plot(result.period, result.power)
    plt.xlabel("Period [days]")
    plt.ylabel("Power")
    plt.title("BLS Periodogram")
    plt.show()
    
    best_bls_period = result.period[np.argmax(result.power)]
    print(f"The best BLS period is: {best_bls_period} days")
    
    # Fourier Transform Analysis
    ft_result = fft(flux_filtered)
    freq = np.fft.fftfreq(len(time), np.median(np.diff(time)))
    plt.figure(figsize=(10, 6))
    plt.plot(freq, np.abs(ft_result))
    plt.xlabel("Frequency [1/days]")
    plt.ylabel("Amplitude")
    plt.title("Fourier Transform")
    plt.show()
    
    return significant_periods, best_bls_period

 
#Searching for lightcurves
search_result = search_lightcurve(target_name, mission='TESS', sector=sector)
if search_result:
    lcs = search_result.download_all()
    data = [{'time': lc.time.value, 'flux': lc.flux.value} for lc in lcs]
    
    if len(data) < 5:
        raise ValueError(f"Not enough data to perform analysis. Only {len(data)} valid datasets found. Ensure at least 5 samples are available.")
    
    print(f"Collected {len(data)} valid datasets")
    for i, dataset in enumerate(data):
        print(f"Dataset {i + 1}: Time length = {len(dataset['time'])}, Flux length = {len(dataset['flux'])}")
    
    for dataset in data:
        significant_periods, best_bls_period = analyze_light_curve(dataset['time'], dataset['flux'])
else:
    print("Cannot find lightcurves for the specified target and sector.")

#Downloading full frame images
obs_table = Observations.query_criteria(
    obs_collection="TESS",
    dataproduct_type=["IMAGE"],
    target_name=target_name,
    sector=sector
)

if obs_table:
    if len(obs_table) > 0:
        products = Observations.get_product_list(obs_table)
        ffi_files = Observations.filter_products(products, productSubGroupDescription=["FFIC"])
        
        if not os.path.exists('data'):
            os.makedirs('data')
        
        for ffi in ffi_files:
            url = ffi['dataURL']
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                filepath = os.path.join('data', ffi['productFilename'])
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Successfully downloaded {ffi['productFilename']}")
            else:
                print(f"Failed to download {ffi['productFilename']}")
        print("Downloaded all requested FFIs!")
    else:
        print("No FFIs found for the specified target and sector!")
else:
    print("Results cannot be found for the specified target and sector. Please check the target name and sector number!")
