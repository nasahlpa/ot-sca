# Getting Started

## Prerequisites

### Hardware Requirements

To perform side-channel analysis (SCA) for [OpenTitan](https://github.com/lowRISC/OpenTitan) using the infrastructure provided in this repository, the following hardware equipment is required:

* [ChipWhisperer CW310 FPGA Board with K410T](https://rtfm.newae.com/Targets/CW310%20Bergen%20Board//)

  This is the target board.
  The Xilinx Kintex-7 FPGA is used to implement the full Earl Grey top level of OpenTitan.
  Note that there are different versions of the board available of which the board with the bigger FPGA device (K410T) is required.

* [ChipWhisperer-Husky CW1190 Capture Board](https://github.com/newaetech/ChipWhisperer-Husky)

  This is the capture or scope board.
  It is used to capture power traces of OpenTitan implemented on the target board.
  The capture board comes with SMA and 20-pin ChipWhisperer cable required to connect to the target board.

The following, alternative hardware equipment is no longer supported:

* [ChipWhisperer CW305-A100 FPGA Board](https://rtfm.newae.com/Targets/CW305%20Artix%20FPGA/)

  This is a smaller target board than the CW310.

  **Note:** The Xilinx Artix-7 A100T FPGA on the CW305-A100 is considerably smaller than the Kintex-7 K410T FPGA on the CW310 and does not offer sufficient resources for implementing the full Earl Grey top level of OpenTitan.
  However, it is suitable for implementing the English Breakfast top level, a substantially reduced version of OpenTitan.
  As a result, the CW305-A100 is suitable for performing SCA of AES but not KMAC or OTBN.
  Note that also for this board there exist different versions of which only the board with the bigger FPGA device (XC7A100T-2FTG256) is supported.

  **Support:** Older versions of the repo support this target.

### Hardware Setup

To setup the hardware, first connect the target and capture board together.
You don't want to accidentally short something out with the conductive SMA connector.

#### CW310

As shown in the photo below,
1. Connect the `SHUNTLOW AMPLIFIED` output of the CW310 (topmost SMA) to the CW-Husky `Pos` input.
1. Connect the `ChipWhisperer Connector` (`J14`) on the CW310 to the `ChipWhisperer 20-Pin Connector` on the capture board.
   On the CW-Husky this is the connector on the *SIDE* not the connector on the *FRONT*.

![](img/cw310_cwhusky.jpg)

The CW310 will need a power supply, the default power supply uses a DC barrel connector to supply 12V.
If you use this power supply all switches should be in the default position.
To this end, make sure to:
1. Set the `SW2` switch (to the right of the barrel connector) up to the `5V Regulator` option.
1. Set the switch below the barrel connector to the right towards the `Barrel` option.
1. Set the `Control Power` switch (bottom left corner, `SW7`) to the right.
1. Ensure the `Tgt Power` switch (above the fan, `S1`) is set to the right towards the `Auto` option.

Then,
1. Plug the DC power adapter into the barrel jack (`J11`) in the top left corner of the board.
1. Use a USB-C cable to connect your PC with the `USB-C Data` connector (`J8`) in the lower left corner on the board.
1. Connect the CW-Husky to your PC via USB-C cable.

You should see the blue "Status" LED on the CW310 blinking, along with several green power LEDs.
The "USB-C Power" led may be red as there is no USB-C PD source.
The CW-Husky should also have a green blinking status LED at this point.
If LEDs are solid it may mean the device has not enumerated, which might require additional setup (see UDEV Rules below).

### UDEV Rules

You might need to setup the following `udev` rules to gain access to the two USB devices.
To do so, open the file `/etc/udev/rules.d/90-lowrisc.rules` or create it if does not yet exist and add the following content to it:

```
# CW310 (Kintex Target)
SUBSYSTEM=="usb", ATTRS{idVendor}=="2b3e", ATTRS{idProduct}=="c310", MODE="0666" SYMLINK+="opentitan/cw_310"

# CW-Husky
SUBSYSTEM=="usb", ATTRS{idVendor}=="2b3e", ATTRS{idProduct}=="ace5", MODE="0666" SYMLINK+="opentitan/cw_husky"
```

To activate the rules, type
```console
$ sudo udevadm control --reload
```
and then disconnect and reconnect the devices.

With the optional `SYMLINK` attribute, a stable symbolic link is created, helping to identify the USB devices more easily.

For more details on how to set up `udev` rules, see the corresponding section in the [OpenTitan documentation](https://docs.opentitan.org/doc/ug/install_instructions/#xilinx-vivado).

### Software Requirements

Software required by this repository can be directly installed on your computer.
We recommend using Ubuntu 22.04 or 24.04, other Linux versions might work as well.

#### Python Virtual Environment

To avoid potential dependency conflicts (in particular with the main OpenTitan repository which may be using a different version of ChipWhisperer) it is strongly recommended to setup a virtual Python environment for this repository.

To this end, type
```console
$ python3 -m venv .venv
```
to create a new virtual environment.

From this point on, whenever you want to work with this repository, type
```console
$ source .venv/bin/activate
```
to enable the virtual environment previously initialized.


#### Python Dependencies

This repository has a couple of Python dependencies to be installed through `pip`.
It is recommended to first install the latest version of `pip` and `setuptools` using
```console
python3 -m pip install -U pip "setuptools<66.0.0"
```
You can then run
```console
$ pip install -r python-requirements.txt
```
to install all the required Python dependencies.


#### ChipWhisperer Dependencies

ChipWhisperer itself is installed as a Python dependency through `python_requirements.txt`.

However, it has some non-Python dependencies related to USB.
Please see [this page](https://chipwhisperer.readthedocs.io/en/latest/linux-install.html#required-packages) to install the specific `apt` packages required by ChipWhisperer.

**Notes:**
- We recommend to use Python 3.10. Later versions might also work.
- CW-Husky requires ChipWhisperer 5.7.0 or later.
  The default `python_requirements.txt` will install a version supporting CW-Husky.

#### Git Large File Storage (LFS)

This project uses Git LFS for storing binaries like example target FPGA bitstreams on a remote server.
The repository itself just contains diff-able text pointers to the binaries. It is recommended to install the `git-lfs` tool for transparently accessing the binaries.
Alternatively, they can be downloaded manually from GitHub.

You can run
```console
$ sudo apt install git-lfs
```
to install the `git-lfs` tool on your Ubuntu machine.

If you've cloned the `ot-sca` repo before installing Git LFS, run
```console
$ git-lfs pull
```
to get the binaries.

### Generating OpenTitan Binaries and Bitstreams

Instead of using the example binaries and bitstreams provided by this repository via Git LFS, you can regenerate them from the [OpenTitan](https://github.com/lowRISC/OpenTitan) repository.
This is needed if you want to develop your own, new SCA and FI tests.
For more information on the buildsteps, refer to the following [guide](./building_fpga_bitstreams_binaries.md).

## Side-Channel Analysis

All the capture scripts for performing SCA on OpenTitan are available in the `capture/` directory.
There exist separate capture scripts and configuration files for each target (e.g., AES, HMAC, ...).

### Capturing Power Traces for AES on CW310 with Masking Disabled

The following guideline shows how to capture AES power traces on the CW310.
The first step is to configure the experiment using the corresponding config file:
```
capture/configs/aes_sca_cw310.yaml
```

This config file allows you to configure the following settings:
- target
  - Specify the bitstream and the binary to load.
  - You also can provide a dedicated `port:` that is used for the UART communication with the CW310 board.
- husky
  - If you are using the ChipWhisperer Husky to capture power traces, you can specify the sampling rate and the scope gain.
  - Moreover, you also can specify the number of clock cycles you want to capture and the offset before or after the trigger.
- waverunner
  - If you are using a LeCroy oscilloscope instead, you additional can provide the IP address of the scope.
- capture
  - Configure which scope you want to use.
  - `num_segments` allows you to configure the number of batches for the batch mode.
  - `num_traces` is the number of traces the capture script records.
  - `trace_threshold` configures the number of traces that are stored in RAM before they are flushed to the disk.
- test
  - Configure which test you want to run.
  - Allows you to also enable / disable certain settings, like masking, dummy instructions and much more.

For the example in this guideline, change the following parameters:
- `num_traces: 10000`
- `which_test: aes_fvsr_key`
- `lfsr_seed: 0` - this will disable masking

Then run the script:
```console
$ cd capture
$ ./capture_aes.py -c configs/aes_sca_cw310.yaml -p aes_capture
INFO:root:Initializing target cw310 ...
Initializing target cw310 ...
Connecting and loading FPGA... Done!
Initializing PLL1
Programming 256 bytes at address 0x00000000.
...
Programming 232 bytes at address 0x00028700.
INFO:root:Initializing scope husky with a sampling rate of             200000000...
Initializing scope husky with a sampling rate of             200000000...
(ChipWhisperer Scope WARNING|File ChipWhispererHuskyClock.py:378) ADC frequency must be between 1MHz and 300000000.0MHz - ADC mul has been adjusted to 2
(ChipWhisperer Scope WARNING|File ChipWhispererHuskyClock.py:409) PLL unlocked after updating frequencies
(ChipWhisperer Scope WARNING|File ChipWhispererHuskyClock.py:410) Target clock has dropped for a moment. You may need to reset your target
Connected to ChipWhisperer (num_samples: 1200, num_samples_actual: 1200, num_segments_actual: 1)
INFO:root:Setting up capture aes_fvsr_key ...
Setting up capture aes_fvsr_key ...
Capturing: 100%|████████████████████| 10000/10000 [00:41<00:00, 240.02 traces/s]
INFO:root:Created plot with 100 traces: /home/nasahlpa/projects/ot_sca_doc_update/ot-sca/capture/aes_capture.html
Created plot with 100 traces: /home/nasahlpa/projects/ot_sca_doc_update/ot-sca/capture/aes_capture.html
```

Once the power traces have been collected, a picture similar to the following should be shown in your browser.

![](img/aes_capture_traces.png)

Note the input data can range from `0` to `4095` (this is because ADC resolution on CW-Husky is 12 bits).
If the output value is exceeding those limits you are clipping and losing data.
But if the range is too small you are not using the full dynamic range of the ADC.
You can tune the `scope_gain` setting in the `.yaml` configuration file.
Note that boards have some natural variation, and changes such as the clock frequency, core voltage, and device utilization (FPGA build) will all affect the safe maximum gain setting.

### TVLA Analysis for the Captured AES Power Traces on CW310

As we have disabled masking in the example before, check if a TVLA test can detect the expected leakage.
For this, modify the following config file:
```
analysis/configs/aes_sca_cw310.yaml
```
as following:
- `project_file: ../capture/aes_capture`
- `test_type: "GENERAL_KEY"`
- `trace_threshold: 10000`
then run the TVLA script:
```console
$ cd ../analysis
$ ./tvla.py --cfg-file configs/tvla_cfg_aes_specific_byte0_rnd0.yaml run-tvla
2025-09-26 10:59:10 INFO: Processing Step 1/1: Trace 0 - 9999
2025-09-26 10:59:10 INFO: Converting Traces
2025-09-26 10:59:10 INFO: Will use samples from 0 to 1200
2025-09-26 10:59:10 INFO: Filtering Traces
2025-09-26 10:59:10 INFO: Will use 8931 traces (89.3%)
2025-09-26 10:59:10 INFO: Computing Leakage
2025-09-26 10:59:10 INFO: Building Histograms
2025-09-26 10:59:11 INFO: Computing T-test Statistics
2025-09-26 10:59:12 INFO: Updating T-test Statistics vs. Number of Traces
2025-09-26 10:59:12 INFO: Leakage above threshold identified in the following order(s) marked with X
2025-09-26 10:59:12 INFO: Couldn't compute statistics for order(s) marked with O:
Order 1: X
Order 2: X

2025-09-26 10:59:12 INFO: Plotting Figures to tmp/figures
2025-09-26 10:59:12 INFO: Plotting T-test Statistics vs. Time.
```
To view the TVLA plot, open:
```
analysis/tmp/figures/aes_fixed_vs_random.png
```

![](img/aes_tvla_no_masks.png)

### Capturing Power Traces for AES on CW310 with Masking Enabled

To verify that masking for the AES works, switch back to the `capture/` directory and modify:
```
capture/configs/aes_sca_cw310.yaml
```
as following:
- `lfsr_seed: 0xdeadbeef`

Now, again run the capture script:
```console
$ ./capture_aes.py -c configs/aes_sca_cw310.yaml -p aes_capture_masks_on
```

Switch to the `analysis/` directory and modify:
```
analysis/configs/aes_sca_cw310.yaml
```
as following:
- `project_file: ../capture/aes_capture_masks_on`


After again running the TVLA script:
```console
$ ./tvla.py --cfg-file configs/tvla_cfg_aes_specific_byte0_rnd0.yaml run-tvla
```
No leakage is visible in the TVLA plot:

![](img/aes_tvla_with_masks.png)

### Trace Preprocessing

The `analysis/trace_preprocessing.py` script allows the user to preprocess traces that were
captured with the OpenTitan trace library. This script provides two functionalities:
- Trace filtering: Remove traces that contain samples over the tolerable deviation from average.
- Trace aligning: Uses a trace window to align all traces according to a reference trace.

The filtering can be enabled with the `-f` command line argument and the tolerable deviation
from average with `-s`.
To turn on the trace aligning mechanism, use the `-a` flag. The reference trace is calculated
from the mean of `-n` traces and the window can be specified with `-lw` and `-hw`.
When operating with larger databases, the `-c` parameter can be used to specifiy the number
of processes used for aligning the traces and the `-m` parameter can be used to specify the
maximum amount of traces that are kept in memory per process.
An example is shown below:

```console
$ ./trace_preprocessing.py -i db_in.db -o db_out.db -f -s 7.5 -a -p 5 -n 1000 -lw 21000 -hw 23000 -ms 20 -m 1000 -c 4
```

## Fault Injection

OT-SCA currently supports voltage glitching and EMFI using the ChipWhisperer Husky and ChipShouter combined with the ChipShover XYZ table.
If you do not have a fault injection setup yet, you can try out running OT-SCA in the demo mode.

### Ibex Fault Injection in the Demo Mode

To perform a fault injection attack in the demo mode, i.e., without any FI gear attached, follow the following steps:

```console
$ cp ci/cfg/ci_ibex_fi_vcc_dummy_cw310.yaml fault_injection/configs/ibex_fi_vcc_dummy_cw310.yaml
```
Modify the following paramter in the copied file:
- `port: "/dev/ttyACM_CW310_1`
to point to the UART port of the CW310 board.

```console
$ cd fault_injection
$ ./fi_ibex.py -c configs/ibex_fi_vcc_dummy_cw310.yaml -p ibex_fi
INFO:root:Initializing target cw310 ...
Initializing target cw310 ...
Connecting and loading FPGA... Done!
Initializing PLL1
Programming 256 bytes at address 0x00000000.
...
Injecting:   0%|                         | 0/100 [00:00<?, ? different faults/s]
...
Injecting: 100%|███████████████| 100/100 [00:06<00:00, 15.62 different faults/s]
INFO:root:100 faults, 0(0.0%) successful, 100(100.0%) expected, and 0(0.0%) no response.
100 faults, 0(0.0%) successful, 100(100.0%) expected, and 0(0.0%) no response.
INFO:root:Created plot.
Created plot.
INFO:root:Created plot: /home/nasahlpa/projects/ot_sca_doc_update/ot-sca/fault_injection/ibex_fi.html
Created plot: /home/nasahlpa/projects/ot_sca_doc_update/ot-sca/fault_injection/ibex_fi.html
```

This should produce the following output file:

![](img/fi_ibex.png)

The orange dots indicate that the expected response was retrieved from OpenTitan.
This is expected as in the demo mode no real fault was injected.

### Fault Injection using EMFI

To inject real faults using the ChipShouter EMFI station, please refer to the following [guide](./chipshouter.md).

# Troubleshooting

## Unreachable Husky Scope
If a connection with CW-Husky cannot be established, e.g., due to a failed
firmware update, first try to [erase](https://rtfm.newae.com/Capture/ChipWhisperer-Husky/#erase-pins)
its firmware by shorting ```SJ1``` on the board. If the scope is still
unreachable but the USB connection is detected by Linux, load the firmware
manually:
```python
import chipwhisperer as cw
programmer = cw.SAMFWLoader(scope=None)
programmer.program('/dev/ttyACM0', hardware_type='cwhusky')
```
Replace ```/dev/ttyACM0``` with the corresponding device.
