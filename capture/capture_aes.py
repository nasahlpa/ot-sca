#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
import binascii
import random
import signal
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import chipwhisperer as cw
import numpy as np
import yaml
from Crypto.Cipher import AES
from project_library.trace_library import Metadata, Trace, TraceLibrary
from scopes.scope import Scope, ScopeConfig
from tqdm import tqdm

sys.path.append("../")
from target.cw_fpga import CWFPGA  # noqa: E402
from util import plot  # noqa: E402


def abort_handler_during_loop(this_project, CWLib, sig, frame):
    # Handler for ctrl-c keyboard interrupts
    if this_project is not None:
        print("\nHandling keyboard interrupt")
        if CWLib:
            this_project.close(save=True)
        else:
            this_project.flush_to_disk()

    sys.exit(0)


if __name__ == '__main__':
    now = datetime.now()
    cfg_file = sys.argv[1]
    # Load configuration from file
    with open(cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    # Determine trace library
    CWLib = True
    OTTraceLib = False
    if cfg['capture'].get('trace_db') == 'ot_trace_library':
        CWLib = False
        OTTraceLib = True

    if CWLib:
        # Create ChipWhisperer project for storage of traces and metadata ------
        project = cw.create_project(cfg["capture"]["project_name"], overwrite=True)
    else:
        # Create Trace Database-------------------------------------------------
        project = TraceLibrary(cfg["capture"]["project_name"],
                               cfg["capture"]["trace_threshold"],
                               wave_datatype=np.uint16,
                               overwrite=True)

    # Init target
    cw_target = CWFPGA(
        bitstream = cfg["cwfpga"]["fpga_bitstream"],
        force_programming = cfg["cwfpga"]["force_program_bitstream"],
        firmware = cfg["cwfpga"]["fw_bin"],
        pll_frequency = cfg["cwfpga"]["pll_frequency"],
        baudrate = cfg["cwfpga"]["baudrate"],
        output_len = cfg["cwfpga"]["output_len_bytes"],
    )

    # Init scope
    scope_type = cfg["capture"]["scope_select"]
    scope_cfg = ScopeConfig(
        scope_type = scope_type,
        acqu_channel = cfg[scope_type].get("channel"),
        ip = cfg[scope_type].get("waverunner_ip"),
        num_cycles = cfg[scope_type].get("num_cycles"),
        num_samples = cfg[scope_type].get("num_samples"),
        offset_cycles = cfg[scope_type].get("offset_cycles"),
        offset_samples = cfg[scope_type].get("sample_offset"),
        target_clk_mult = cfg[scope_type].get("target_clk_mult"),
        num_segments = cfg[scope_type]["num_segments"],
        sparsing = cfg[scope_type].get("sparsing"),
        scope_gain = cfg[scope_type].get("scope_gain"),
        pll_frequency = cfg["cwfpga"]["pll_frequency"],
    )
    scope = Scope(scope_cfg)

    # Preparation of Key and plaintext generation ------------------------------

    # Determine which test from configuration
    NUM_SEGMENTS = cfg[scope_type]["num_segments"]
    TEST_FVSR_KEY_RND_PLAINTEXT = False
    TEST_FIXED_KEY_RND_PLAINTEXT = False
    if cfg["test"]["which_test"] == "aes_fvsr_key_random_plaintext":
        TEST_FVSR_KEY_RND_PLAINTEXT = True
        if NUM_SEGMENTS > 1:
            print("ERROR: aes_fvsr_key_random_plaintext only supported "
                  "with num_segments > 1, i.e. batch mode")
    elif cfg["test"]["which_test"] == "aes_fixed_key_random_plaintext":
        TEST_FIXED_KEY_RND_PLAINTEXT = True
        # This test uses the fixed key. It generates random texts through AES
        # encryption using a generation key for single mode and uses ciphertexts
        # as next text input in batch mode.

    # Load fixed key
    key_fixed = bytearray(cfg["test"]["key_fixed"])
    print(f'Using key: {binascii.b2a_hex(bytes(key_fixed))}')
    key = key_fixed
    cw_target.target.simpleserial_write("k", key)

    # Seed the target's PRNGs for initial key masking, and additionally turn off masking when '0'
    cw_target.target.simpleserial_write("l", cfg["test"]["lfsr_seed"].to_bytes(4, "little"))

    if TEST_FIXED_KEY_RND_PLAINTEXT:
        # Cipher to compute expected responses
        cipher = AES.new(bytes(key), AES.MODE_ECB)

        # Load fixed text as first text
        text = bytearray(cfg["test"]["text_fixed"])

        # Prepare generation of new texts/keys by encryption using key_for_generation
        key_for_gen = bytearray(cfg["test"]["key_for_gen"])
        cipher_gen = AES.new(bytes(key_for_gen), AES.MODE_ECB)

        if NUM_SEGMENTS > 1:
            # Set initial plaintext for batch mode
            cw_target.target.simpleserial_write("i", text)

    if TEST_FVSR_KEY_RND_PLAINTEXT:
        # Load seed into host-side PRNG (random Python module, Mersenne twister)
        random.seed(cfg["test"]["batch_prng_seed"])
        # Load seed into OT (also Mersenne twister)
        cw_target.target.simpleserial_write(
            "s",
            cfg["test"]["batch_prng_seed"].to_bytes(4, "little")
        )

        # First trace uses fixed key
        sample_fixed = 1
        # Generate plaintexts and keys for first batch
        cw_target.target.simpleserial_write("g", NUM_SEGMENTS.to_bytes(4, "little"))

    # Main loop for measurements with progress bar -----------------------------

    # Register ctrl-c handler to store traces on abort
    signal.signal(signal.SIGINT, partial(abort_handler_during_loop, project, CWLib))
    remaining_num_traces = cfg["capture"]["num_traces"]
    with tqdm(total=remaining_num_traces, desc="Capturing", ncols=80, unit=" traces") as pbar:
        while remaining_num_traces > 0:

            # Note: Capture performance tested Oct. 2023:
            # Husy with 1200 samples per trace: 50 it/s
            # WaveRunner with 1200 - 50000 samples per trace: ~30 it/s
            #   +10% by setting 'Performance' to 'Analysis' in 'Utilies->Preferences' in GUI
            # WaveRunner batchmode (6k samples, 100 segmets, 1 GHz): ~150 it/s

            # Arm scope --------------------------------------------------------
            scope.arm()

            # Trigger execution(s) ---------------------------------------------
            if NUM_SEGMENTS > 1:
                # Perform batch encryptions
                if TEST_FIXED_KEY_RND_PLAINTEXT:
                    # Execute and generate next text as ciphertext
                    cw_target.target.simpleserial_write("a", NUM_SEGMENTS.to_bytes(4, "little"))
                if TEST_FVSR_KEY_RND_PLAINTEXT:
                    # Execute and generate next keys and plaintexts
                    cw_target.target.simpleserial_write("f", NUM_SEGMENTS.to_bytes(4, "little"))

            else:  # single encryption
                # Load text and trigger execution
                # First iteration uses initial text, new texts are generated below
                cw_target.target.simpleserial_write('p', text)

            # Capture trace(s) -------------------------------------------------
            waves = scope.capture_and_transfer_waves()
            assert waves.shape[0] == NUM_SEGMENTS

            # Storing traces ---------------------------------------------------

            # Loop to compute keys, texts and ciphertexts for each trace and store them.
            for i in range(NUM_SEGMENTS):

                # Compute text, key, ciphertext
                if TEST_FIXED_KEY_RND_PLAINTEXT:
                    ciphertext = bytearray(cipher.encrypt(bytes(text)))

                if TEST_FVSR_KEY_RND_PLAINTEXT:
                    if sample_fixed:
                        # Use fixed_key as this key
                        key = np.asarray(key_fixed)
                    else:
                        # Generate this key from PRNG
                        key = bytearray(cfg["test"]["key_len_bytes"])
                        for ii in range(0, cfg["test"]["key_len_bytes"]):
                            key[ii] = random.randint(0, 255)
                    # Always generate this plaintext from PRNG (including very first one)
                    text = bytearray(16)
                    for ii in range(0, 16):
                        text[ii] = random.randint(0, 255)
                    # Compute ciphertext for this key and plaintext
                    cipher = AES.new(bytes(key), AES.MODE_ECB)
                    ciphertext = bytearray(cipher.encrypt(bytes(text)))
                    # Determine if next iteration uses fixed_key
                    sample_fixed = text[0] & 0x1

                # Sanity check retrieved data (wave)
                assert len(waves[i, :]) >= 1
                # Add trace.
                if OTTraceLib:
                    # OT Trace Library
                    trace = Trace(wave=waves[i, :].tobytes(),
                                  plaintext=text,
                                  ciphertext=ciphertext,
                                  key=key)
                    project.write_to_buffer(trace)
                else:
                    # CW Trace
                    trace = cw.Trace(waves[i, :], text, ciphertext, key)
                    # Append trace Library trace to database
                    # TODO Also use uint16 as dtype for 8 bit WaveRunner for
                    # tvla processing
                    project.traces.append(trace, dtype=np.uint16)

                if TEST_FIXED_KEY_RND_PLAINTEXT:
                    # Use ciphertext as next text, first text is the initial one
                    text = ciphertext

            # Get (last) ciphertext from device and verify ---------------------
            if TEST_FIXED_KEY_RND_PLAINTEXT:
                compare_len = cw_target.target.output_len
            if TEST_FVSR_KEY_RND_PLAINTEXT:
                compare_len = 4
            response = cw_target.target.simpleserial_read('r', compare_len, ack=False)
            if binascii.b2a_hex(response) != binascii.b2a_hex(ciphertext[0:compare_len]):
                raise RuntimeError(f'Bad ciphertext: {response} != {ciphertext}.')

            # Update the loop variable and the progress bar --------------------
            remaining_num_traces -= NUM_SEGMENTS
            pbar.update(NUM_SEGMENTS)

    # Create and show test plot ------------------------------------------------
    # Use this plot to check for clipping and adjust gain appropriately
    if CWLib:
        proj_waves = project.waves
    else:
        proj_waves = project.get_waves()

    if cfg["capture"]["show_plot"]:
        plot.save_plot_to_file(proj_waves, None, cfg["capture"]["plot_traces"],
                               cfg["capture"]["trace_image_filename"], add_mean_stddev=True)
        print(f'Created plot with {cfg["capture"]["plot_traces"]} traces: '
              f'{Path(cfg["capture"]["trace_image_filename"]).resolve()}')

    if CWLib:
        project.settingsDict['datetime'] = now
        project.settingsDict['cfg'] = cfg
        project.settingsDict['sample_rate'] = scope_cfg.num_samples
        project.save()
    else:
        metadata = Metadata(config =cfg_file,
                            datetime=now,
                            bitstream_path=cfg["cwfpga"]["fpga_bitstream"],
                            binary_path=cfg["cwfpga"]["fw_bin"],
                            offset=scope_cfg.offset_samples,
                            sample_rate=scope_cfg.num_samples,
                            scope_gain=scope_cfg.scope_gain
                            )
        project.write_metadata(metadata)
        project.flush_to_disk()