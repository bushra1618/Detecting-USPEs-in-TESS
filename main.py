import os
import urllib
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from astropy.io import fits
from astropy.timeseries import LombScargle, BoxLeastSquares
from lightkurve import search_lightcurve
from astroquery.mast import Observations
from scipy.fft import fft

# Defining the target and mission parameters
target_name = "TIC 25155310"  # TIC ID with known data
sector = 1  # sector number for the given TIC ID

# Function to download and process FITS file from a URL
def download_and_process_fits(url):
    try:
        print(f"Downloading data from {url}")
        with urllib.request.urlopen(url) as response:
            with fits.open(BytesIO(response.read())) as hdul:
                # Print the structure of the FITS file for debugging
                hdul.info()

                # Let's assume that the light curve data is in HDU 1 (typical for TESS data)
                data = hdul[1].data

                # Print column names to verify correct data access
                print(data.columns.names)

                # Extracting time and flux
                time = data['TIME']
                flux = data['PDCSAP_FLUX']  # Using PDCSAP_FLUX

                # Remove NaNs and invalid values
                valid_mask = np.isfinite(time) & np.isfinite(flux)
                time = time[valid_mask]
                flux = flux[valid_mask]

                if len(time) == 0 or len(flux) == 0:
                    raise ValueError("No valid data points found after cleaning.")

                return {'time': time, 'flux': flux}
    except urllib.error.HTTPError as e:
        print(f"HTTP Error processing URL {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return None

# Analyzing light curve using Lomb-Scargle periodogram, BLS, and FT
def analyze_light_curve(time, flux):
    # Remove trends using Savitzky-Golay filter
    flux_filtered = savgol_filter(flux, window_length=101, polyorder=2)

    # Plotting the raw and filtered light curves
    plt.figure()
    plt.plot(time, flux, 'o', markersize=2, label='Raw')
    plt.plot(time, flux_filtered, '-', label='Filtered')
    plt.xlabel("Time [days]")
    plt.ylabel("Flux [e-/s]")
    plt.legend()
    plt.title("Light Curve")
    plt.show()

    # Performing periodogram analysis using Lomb-Scargle
    frequency, power = LombScargle(time, flux_filtered).autopower()
    plt.figure()
    plt.plot(1 / frequency, power)
    plt.xlabel("Period [days]")
    plt.ylabel("Power")
    plt.title("Lomb-Scargle Periodogram")
    plt.show()

    # Identifying the most significant periods
    peaks, _ = find_peaks(power, height=0.01)
    significant_periods = 1 / frequency[peaks]

    # Print the significant periods
    print("Significant periods (days):", significant_periods)

    # Plot the light curve folded at the most significant period
    if len(significant_periods) > 0:
        best_period = significant_periods[0]
        plt.figure()
        phase = (time % best_period) / best_period
        plt.plot(phase, flux_filtered, 'o', markersize=2)
        plt.xlabel("Phase")
        plt.ylabel("Flux [e-/s]")
        plt.title(f"Folded Light Curve at Period = {best_period:.4f} days")
        plt.show()

    # Perform Box Least Squares (BLS) analysis
    model = BoxLeastSquares(time, flux_filtered)
    periods = np.linspace(0.1, 1, 10000)  # always search periods between 0.1 and 1 days
    result = model.power(periods, 0.2)
    plt.figure()
    plt.plot(result.period, result.power)
    plt.xlabel("Period [days]")
    plt.ylabel("Power")
    plt.title("BLS Periodogram")
    plt.show()

    best_bls_period = result.period[np.argmax(result.power)]
    print(f"Best BLS period: {best_bls_period} days")

    # Perform Fourier Transform (FT) analysis
    ft_result = fft(flux_filtered)
    freq = np.fft.fftfreq(len(time), np.median(np.diff(time)))
    plt.figure()
    plt.plot(freq, np.abs(ft_result))
    plt.xlabel("Frequency [1/days]")
    plt.ylabel("Amplitude")
    plt.title("Fourier Transform")
    plt.show()

    return significant_periods, best_bls_period

# Searching for light curve data using Lightkurve
search_result = search_lightcurve(target_name, mission='TESS', sector=sector)

if search_result:
    # Download the light curve files
    lcs = search_result.download_all()

    # Collect data from Lightkurve search
    data = [{'time': lc.time.value, 'flux': lc.flux.value} for lc in lcs]

    # Ensure there is enough data
    if len(data) < 5:
        raise ValueError(f"Not enough data to perform analysis. Only {len(data)} valid datasets found. Ensure at least 5 samples are available.")

    print(f"Collected {len(data)} valid datasets")

    # (Optional) display data for verification
    for i, dataset in enumerate(data):
        print(f"Dataset {i + 1}: Time length = {len(dataset['time'])}, Flux length = {len(dataset['flux'])}")

    # Analyze each light curve
    for dataset in data:
        significant_periods, best_bls_period = analyze_light_curve(dataset['time'], dataset['flux'])

    # Download full-frame images (FFIs)
    obs_table = Observations.query_criteria(obs_collection="TESS", dataproduct_type=["IMAGE"], target_name=target_name, sector=sector)

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
            else:
                print(f"Failed to download! {ffi['productFilename']}")

        print("Downloaded all requested FFIs!")
    else:
        print("No FFIs found for the specified target and sector.")
else:
    print("Results cannot be found for the specified target and sector. Please check the target name and sector number.")
