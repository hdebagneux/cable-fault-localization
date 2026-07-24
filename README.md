# Research project EDF Cable Fault Modelling Git Repository README

## Overview

The Git Repository that you have been given access to includes our source code for the 2026 Research project concluded with EDF. 

The Git Repository can be cloned into your working repository with the following line:
```bash
git clone https://gitlab-student.centralesupelec.fr/lina.skik/edf_cablefault.git
```

---

## Git Repository Content

- **`README.md`
    This is the file that details how to use the Git repository.

- **`app.py`
    This is the file that allows us to run the interface

- **`components.py`
    This is one of the files necessary for the display of the interface welcome page.

- **`fdr_page.py`
    This is the file that allows us to run the frequency domain section of the interface, and link it with the frequency domain code.

- **`fdr_to_distance.py`
    This is the file that details all of the code for the frequency domain fault analysis, with its incorporated graphs, and is the employed for the interface.

- **`first_order.py`
    This is the file that contains the code for the first order approximation in the Time Domain. This is our initial approximation which we still wanted to include as it was what allowed us to develop the more complex second order code, and is still interesting in and of itself.

- **`second_order.py`
    This is the file that contains the code for the second order approximation in the Time Domain

- **`second_order_streamlit.py`
    This is the file that extracts code from the second order file and is then later used for the Time Domain section of the interface.

- **`styles.py`
    This is another one of the files necessary for the display, especially the fonts and look of the interface.

- **`theme.py`
    This is another file necessary for the interface, especially the colours employed.

- **`time_domain.py`
    This is the file that is responsible for the Time Domain section of the interface, extracting the useful results from the second order codes and displaying them as desired in the interface.

---

### `app.py`

This script codes the Graphical User Interface, which uses data from previous python scripts to allow engineers/technicians/anyone curious about learning more around cable faults to visualise results and to simulate certain cases they might be interested in.
When the user interface is launched, a welcome page with great you with two options, and a description. These two option are to either run simulations in the Time Domain, or in the Frequency domain.
If you click on Time Domain, you will be able to select the type of input signal and its parameters: maximal frequency of the signal f_max, its amplitude in Volts, and the Points Per Wavelength (a measure of spatial resolution only necessary for the Gaussian signal). Default values already give interesting results and we recommend using the Gaussian signal in the Time Domain as it is easier to visualise its propagation. A preview of the input signal will be shown for the user.
You will then be able to select the length of your cable in meters, and once again we recommend going with the default length as much longer lengths may cause the simulation to take time to load. You can then select the region of the fault, and which parameter of RLCG should have a value different from its default. The fault multiplier then allows you to select how large this fault is. For inductance and capacitance faults, a factor of 10 gives interesting results, while for the other a factor of 10000 is necessary before any interesting results arise.
Once this is all parametrised the run simulation button can be clicked to observe the signal propagating through the wire, and see the detected voltage at the starting end of the wire. A final Simulated Fault Summary is available at the bottom of the page, summarising what was detected by our code's calculations.
If instead you click on Frequency Domain, you can instantly select your cable length, source impedance and load impedance, which are set at 50 Ohms as those are the values at which they match the characteristic impedance of a typical coaxial cable such as the one we employed.
The start and end frequency of the frequency samples you will run can be selected, as well as the number of frequency samples. The option to select the number of faults is available, with their position, length, modified parameter, and factor of modification.
The run simulation button can then be clicked to show the results of your simulation. Three graphs are available to display the fault detection, as well as a fault Simulation Summary at the bottom of the page just like in the Time Domain. First of all, panel 1, is the frequency-domain response plot of |S11| versus frequency for the healthy cable (green), the faulty cable (red), and their complex difference |ΔS11| (blue). The deviations between the healthy and faulty curves form the raw signature from which fault location is extracted.
Then, panel 2, corresponding to the distance-domain reflectogram, applies an inverse Fourier transform to ΔS11, converting that signature into a normalized reflection strength plotted against distance from the source. Detected fault positions are marked with triangles and dashed lines, true fault extents are shaded in orange, and two resolution bars indicate the theoretical and effective spatial resolution under the chosen windowing function.
Finally, panel 3: Ground-Truth Impedance Change shows the exact characteristic-impedance variation ΔZc(x) along the cable as defined by the simulation parameters. Because this value is only findable in a simulated environment, we display it here specifically so you can cross-check the peak positions in Panel 2 against the true fault locations, giving a direct visual measure of the technique's accuracy.

To run this Graphical User Interface, streamlit must be installed, which can be done with the following code:

```bash
pip install streamlit
```

To run:
```bash
python -m streamlit run app.py
```

