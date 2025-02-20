#!/usr/bin/env python3
# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import binascii
import logging
import random
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from Crypto.Cipher import AES
from project_library.project import ProjectConfig, SCAProject
from scopes.scope import (Scope, ScopeConfig, convert_num_cycles,
                          convert_offset_cycles, determine_sampling_rate)
from tqdm import tqdm

import util.helpers as helpers
from target.communication.sca_aes_gcm_commands import OTAESGCM
from target.communication.sca_prng_commands import OTPRNG
from target.communication.sca_trigger_commands import OTTRIGGER
from target.targets import Target, TargetConfig
from util import check_version
from util import data_generator as dg
from util import plot

"""AES-GCM SCA capture script.

Captures power traces during AES-GCM operations.

Typical usage:
>>> ./capture_aes_gcm.py -c configs/aes_gcm_sca_cw310.yaml -p projects/aes_gcm_sca_capture
"""


logger = logging.getLogger()


def abort_handler_during_loop(this_project, sig, frame):
    """ Abort capture and store traces.

    Args:
        this_project: Project instance.
    """
    if this_project is not None:
        logger.info("\nHandling keyboard interrupt")
        this_project.close(save=True)
    sys.exit(0)


@dataclass
class CaptureConfig:
    """ Configuration class for the current capture.
    """
    capture_mode: str
    batch_mode: bool
    num_traces: int
    num_segments: int
    output_len: int
    ptx_static: bytearray
    aad_static: bytearray
    key_fixed: bytearray
    iv_fixed: bytearray
    key_len_bytes: int
    text_len_bytes: int
    iv_len_bytes: int
    ptx_blocks: int
    ptx_last_block_len_bytes: int
    aad_blocks: int
    aad_last_block_len_bytes: int
    triggers: list[bool]
    trigger_block: int
    protocol: str
    port: Optional[str] = "None"


def setup(cfg: dict, project: Path):
    """ Setup target, scope, and project.

    Args:
        cfg: The configuration for the current experiment.
        project: The path for the project file.

    Returns:
        The target, scope, and project.
    """
    # Calculate pll_frequency of the target.
    # target_freq = pll_frequency * target_clk_mult
    # target_clk_mult is a hardcoded constant in the FPGA bitstream.
    cfg["target"]["pll_frequency"] = cfg["target"]["target_freq"] / cfg["target"]["target_clk_mult"]

    # Create target config & setup target.
    logger.info(f"Initializing target {cfg['target']['target_type']} ...")
    target_cfg = TargetConfig(
        target_type = cfg["target"]["target_type"],
        fw_bin = cfg["target"]["fw_bin"],
        protocol = cfg["target"]["protocol"],
        pll_frequency = cfg["target"]["pll_frequency"],
        bitstream = cfg["target"].get("fpga_bitstream"),
        force_program_bitstream = cfg["target"].get("force_program_bitstream"),
        baudrate = cfg["target"].get("baudrate"),
        port = cfg["target"].get("port"),
        output_len = cfg["target"].get("output_len_bytes"),
        usb_serial = cfg["target"].get("usb_serial"),
        husky_serial = cfg["husky"].get("usb_serial")
    )
    target = Target(target_cfg)

    # Init scope.
    scope_type = cfg["capture"]["scope_select"]

    # Will determine sampling rate (for Husky only), if not given in cfg.
    cfg[scope_type]["sampling_rate"] = determine_sampling_rate(cfg, scope_type)
    # Will convert number of cycles into number of samples if they are not given in cfg.
    cfg[scope_type]["num_samples"] = convert_num_cycles(cfg, scope_type)
    # Will convert offset in cycles into offset in samples, if they are not given in cfg.
    cfg[scope_type]["offset_samples"] = convert_offset_cycles(cfg, scope_type)

    logger.info(f"Initializing scope {scope_type} with a sampling rate of {cfg[scope_type]['sampling_rate']}...")  # noqa: E501

    # Determine if we are in batch mode or not.
    batch = False
    if "batch" in cfg["test"]["which_test"]:
        batch = True

    # Create scope config & setup scope.
    scope_cfg = ScopeConfig(
        scope_type = scope_type,
        batch_mode = batch,
        bit = cfg[scope_type].get("bit"),
        acqu_channel = cfg[scope_type].get("channel"),
        ip = cfg[scope_type].get("waverunner_ip"),
        num_samples = cfg[scope_type]["num_samples"],
        offset_samples = cfg[scope_type]["offset_samples"],
        sampling_rate = cfg[scope_type].get("sampling_rate"),
        num_segments = cfg[scope_type].get("num_segments"),
        sparsing = cfg[scope_type].get("sparsing"),
        scope_gain = cfg[scope_type].get("scope_gain"),
        pll_frequency = cfg["target"]["pll_frequency"],
        scope_sn = cfg[scope_type].get("usb_serial"),
    )
    scope = Scope(scope_cfg)

    # Init project.
    project_cfg = ProjectConfig(type = cfg["capture"]["trace_db"],
                                path = project,
                                wave_dtype = np.uint16,
                                overwrite = True,
                                trace_threshold = cfg["capture"].get("trace_threshold")
                                )
    project = SCAProject(project_cfg)
    project.create_project()

    return target, scope, project


def establish_communication(target, capture_cfg: CaptureConfig):
    """ Establish communication with the target device.

    Args:
        target: The OT target.
        capture_cfg: The capture config.

    Returns:
        ot_aes: The communication interface to the AES SCA application.
        ot_prng: The communication interface to the PRNG SCA application.
        ot_trig: The communication interface to the SCA trigger.
    """
    # Create communication interface to OT AES.
    ot_aes_gcm = OTAESGCM(target=target, protocol=capture_cfg.protocol)

    # Create communication interface to OT PRNG.
    ot_prng = OTPRNG(target=target, protocol=capture_cfg.protocol)

    # Create communication interface to SCA trigger.
    ot_trig = OTTRIGGER(target=target, protocol=capture_cfg.protocol)

    return ot_aes_gcm, ot_prng, ot_trig


def configure_cipher(cfg, capture_cfg, ot_aes_gcm, ot_prng):
    """ Configure the AES cipher.

    Establish communication with the AES cipher and configure the seed.

    Args:
        cfg: The project config.
        capture_cfg: The capture config.
        ot_aes_gcm: The communication interface to the AES-GCM SCA application.
        ot_prng: The communication interface to the PRNG SCA application.
    Returns:
        device_id: The ID of the target device.
    """
    # Check if we want to run AES SCA for FPGA or discrete. On the FPGA, we
    # can use functionality helping us to capture cleaner traces.
    fpga_mode_bit = 0
    if "cw" in cfg["target"]["target_type"]:
        fpga_mode_bit = 1
    # Initialize AES on the target.
    device_id = ot_aes_gcm.init(1)

    # Configure PRNGs.
    # Seed the software LFSR used for initial key masking and additionally
    # turning off the masking when '0'.
    ot_aes_gcm.seed_lfsr(cfg["test"]["lfsr_seed"].to_bytes(4, "little"))

    # Seed the PRNG used for generating keys and plaintexts in batch mode.
    if capture_cfg.batch_mode:
        # Seed host's PRNG.
        random.seed(cfg["test"]["batch_prng_seed"])
        # Seed the target's PRNG.
        ot_prng.seed_prng(cfg["test"]["batch_prng_seed"].to_bytes(4, "little"))

    return device_id


def generate_next_data(sample_fixed, mode, key_fixed, key_len, iv_fixed, iv_len,
                       ptx_fixed, ptx_blocks, ptx_last_block_len, aad_fixed, aad_blocks,
                       aad_last_block_len):
    """ Generate next cipher material for the encryption.

    This function derives the next IV and key for the next encryption.

    Args:
        sample_fixed: Use fixed key or new key.
        mode: The mode of the capture.
        key_fixed: The fixed key.
        key_len: The key length.
        iv_fixed: The fixed IV.
        iv_len: The IV length.
        ptx_fixed: The fixed PTX.
        ptx_blocks: The number of PTX blocks.
        ptx_last_block_len: The length in bytes of the last block.
        aad_fixed: The fixed AAD.
        aad_blocks: The number of AAD blocks.
        aad_last_block_len: The length in bytes of the last block.
    Returns:
        iv: The next IV.
        key: The next key.
        ptx: The next PTX.
        aad: The next AAD.
    """
    if mode == "aes_gcm_fvsr_batch_iv_key":
        # Generate FvsR key or IV.
        if sample_fixed:
            iv = iv_fixed[0:iv_len]
            key = key_fixed[0:key_len]
        else:
            # Generate random IV.
            iv = []
            for i in range(0, 16):
                iv.append(random.randint(0, 255))
            iv = iv[0:iv_len]
            # Generate random key.
            key = []
            for i in range(0, 16):
                key.append(random.randint(0, 255))
            key = key[0:key_len]
        # Returned fixed AAD and PTX.
        aad = aad_fixed
        ptx = ptx_fixed
    elif mode == "aes_gcm_fvsr_batch_ptx_aad":
        if sample_fixed:
            ptx = ptx_fixed
            aad = aad_fixed
        else:
            # Generate random PTX.
            ptx = []
            for i in range(0, ptx_blocks):
                ptx_block = []
                valid_bytes = 16
                for j in range(0, valid_bytes):
                    ptx_block.append(random.randint(0, 255))
                if i == ptx_blocks - 1:
                    valid_bytes = ptx_last_block_len
                ptx.append(ptx_block[0:valid_bytes])
            # Generate random AAD.
            aad = []
            for i in range(0, aad_blocks):
                aad_block = []
                valid_bytes = 16
                for j in range(0, valid_bytes):
                    aad_block.append(random.randint(0, 255))
                if i == aad_blocks - 1:
                    valid_bytes = aad_last_block_len
                aad.append(aad_block[0:valid_bytes])
         # Returned fixed IV and key.
        iv = iv_fixed
        key = key_fixed
    else:
        print("Mode not supported")


    return iv, key, ptx, aad


def calculate_ref_tag(key, iv, aad, aad_blocks, ptx, ptx_blocks):
    """ Calculates the expected tag.

    This function derives the next IV and key for the next encryption.

    Args:
        key: The fixed key.
        iv: The fixed IV.
        aad: The static AAD. Will stay the same for fixed and random set.
        aad_blocks: The number of AAD blocks.
        ptx: The static PTX. Will stay the same for fixed and random set.
        ptx_blocks: The number of PTX blocks.
    Returns:
        tag: The calculated tag output.
    """
    cipher = AES.new(bytes(key), AES.MODE_GCM, nonce=bytes(iv))
    for i in range(0, aad_blocks):
        cipher.update(bytes(aad[i]))
    ptx_combined = []
    for i in range(0, ptx_blocks):
        ptx_combined += ptx[i]
    ctx, tag = cipher.encrypt_and_digest(bytes(ptx_combined))
    return tag


def check_tag(ot_aes_gcm, expected_tag):
    """ Compares the received accumulated tag with the generated one.

    The accumulated tag is the XOR accumulated tag.

    Args:
        ot_aes_gcm: The OpenTitan AES-GCM communication interface.
        expected_tag: The pre-computed tag.
    """
    actual_tag = ot_aes_gcm.read_tag()
    assert expected_tag == actual_tag, (
        f"Incorrect tag!\n"
        f"actual: {actual_tag}\n"
        f"expected: {expected_tag}"
    )


def capture(scope: Scope, ot_aes_gcm: OTAESGCM, capture_cfg: CaptureConfig,
            project: SCAProject, target: Target):
    """ Capture power consumption during AES encryption.

    Supports six different capture types:
    * aes_random: Fixed key, random plaintext.
    * aes_random_batch: Fixed key, random plaintext in batch mode.
    * aes_fvsr_key: Fixed vs. random key.
    * aes_fvsr_key_batch: Fixed vs. random key batch.
    * aes_fvsr_data: Fixed vs. random data.
    * aes_fvsr_data_batch: Fixed vs. random data batch.

    Args:
        scope: The scope class representing a scope (Husky or WaveRunner).
        ot_aes_gcm: The OpenTitan AES-GCM communication interface.
        capture_cfg: The configuration of the capture.
        project: The SCA project.
        target: The OpenTitan target.
    """
    # Optimization for CW trace library.
    num_segments_storage = 1

    # Register ctrl-c handler to store traces on abort.
    signal.signal(signal.SIGINT, partial(abort_handler_during_loop, project))
    # Main capture with progress bar.
    remaining_num_traces = capture_cfg.num_traces
    with tqdm(total=remaining_num_traces, desc="Capturing", ncols=80, unit=" traces") as pbar:
        while remaining_num_traces > 0:
            # Arm the scope.
            scope.arm()

            if capture_cfg.capture_mode == "aes_gcm_fvsr_batch_iv_key":
                # Start FvsR on the device.
                ot_aes_gcm.fvsr_batch_iv_key(capture_cfg.num_segments, capture_cfg.triggers,
                                             capture_cfg.trigger_block, capture_cfg.iv_len_bytes,
                                             capture_cfg.iv_fixed, capture_cfg.key_len_bytes,
                                             capture_cfg.key_fixed, capture_cfg.aad_blocks,
                                             capture_cfg.ptx_blocks, capture_cfg.aad_static,
                                             capture_cfg.aad_last_block_len_bytes,
                                             capture_cfg.ptx_static,
                                             capture_cfg.ptx_last_block_len_bytes)
            elif capture_cfg.capture_mode == "aes_gcm_fvsr_batch_ptx_aad":
                # Start FvsR on the device.
                ot_aes_gcm.fvsr_batch_ptx_aad(capture_cfg.num_segments, capture_cfg.triggers,
                                              capture_cfg.trigger_block, capture_cfg.iv_len_bytes,
                                              capture_cfg.iv_fixed, capture_cfg.key_len_bytes,
                                              capture_cfg.key_fixed, capture_cfg.aad_blocks,
                                              capture_cfg.ptx_blocks, capture_cfg.aad_static,
                                              capture_cfg.aad_last_block_len_bytes,
                                              capture_cfg.ptx_static,
                                              capture_cfg.ptx_last_block_len_bytes)
            else:
                print("Mode not supported")

            # Capture traces.
            waves = scope.capture_and_transfer_waves(target)
            assert waves.shape[0] == capture_cfg.num_segments

            # Generate reference crypto material and store traces.
            sample_fixed = 1
            tag_accumulated = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for i in range(capture_cfg.num_segments):
                # Sanity check retrieved data (wave).
                assert len(waves[i, :]) >= 1
                # Determine IV, key, PTX, and AAD for this round.
                iv, key, ptx, aad = generate_next_data(sample_fixed,
                                                       capture_cfg.capture_mode,
                                                       capture_cfg.key_fixed,
                                                       capture_cfg.key_len_bytes,
                                                       capture_cfg.iv_fixed,
                                                       capture_cfg.iv_len_bytes,
                                                       capture_cfg.ptx_static,
                                                       capture_cfg.ptx_blocks,
                                                       capture_cfg.ptx_last_block_len_bytes,
                                                       capture_cfg.aad_static,
                                                       capture_cfg.aad_blocks,
                                                       capture_cfg.aad_last_block_len_bytes)
                # Next iteration: fixed or random?
                sample_fixed = random.getrandbits(32) & 0x1
                # Calculate the expected tag.
                tag = calculate_ref_tag(key, iv, aad, capture_cfg.aad_blocks,
                                        ptx, capture_cfg.ptx_blocks)
                # Convert bytes into array.
                tag_array = [x for x in tag]
                # Accumulate the tag.
                for j in range(0, 16):
                    tag_accumulated[j] = tag_accumulated[j] ^ tag_array[j]
                # Store trace into database.
                project.append_trace(wave = waves[i, :],
                                     plaintext = bytearray(iv),
                                     ciphertext = bytearray(tag),
                                     key = bytearray(key))

            # Compare received accumulated tag with the generated.
            check_tag(ot_aes_gcm, tag_accumulated)

            # Memory allocation optimization for CW trace library.
            num_segments_storage = project.optimize_capture(num_segments_storage)

            # Update the loop variable and the progress bar.
            remaining_num_traces -= capture_cfg.num_segments
            pbar.update(capture_cfg.num_segments)


def print_plot(project: SCAProject, config: dict, file: Path) -> None:
    """ Print plot of traces.

    Printing the plot helps to adjust the scope gain and check for clipping.

    Args:
        project: The project containing the traces.
        config: The capture configuration.
        file: The output file path.
    """
    if config["capture"]["show_plot"]:
        plot.save_plot_to_file(project.get_waves(0, config["capture"]["plot_traces"]),
                               set_indices = None,
                               num_traces = config["capture"]["plot_traces"],
                               outfile = file,
                               add_mean_stddev=True)
        logger.info(f'Created plot with {config["capture"]["plot_traces"]} traces: '
                    f'{Path(str(file) + ".html").resolve()}')


def main(argv=None):
    # Configure the logger.
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logger.addHandler(console)

    # Parse the provided arguments.
    args = helpers.parse_arguments(argv)

    # Check the ChipWhisperer version.
    check_version.check_cw("5.7.0")

    # Load configuration from file.
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Determine the capture mode and configure the current capture.
    mode = cfg["test"]["which_test"]

    # Setup the target, scope and project.
    target, scope, project = setup(cfg, args.project)

    # Create capture config object.
    capture_cfg = CaptureConfig(capture_mode = mode,
                                batch_mode = scope.scope_cfg.batch_mode,
                                num_traces = cfg["capture"]["num_traces"],
                                num_segments = scope.scope_cfg.num_segments,
                                output_len = cfg["target"]["output_len_bytes"],
                                ptx_static = cfg["test"]["ptx_static"],
                                aad_static = cfg["test"]["aad_static"],
                                key_fixed = cfg["test"]["key_fixed"],
                                iv_fixed = cfg["test"]["iv_fixed"],
                                key_len_bytes = cfg["test"]["key_len_bytes"],
                                text_len_bytes = cfg["test"]["text_len_bytes"],
                                iv_len_bytes = 12, # Hardcoded
                                ptx_blocks = cfg["test"]["ptx_blocks"],
                                aad_blocks = cfg["test"]["aad_blocks"],
                                aad_last_block_len_bytes = cfg["test"]["aad_last_block_len_bytes"],
                                ptx_last_block_len_bytes = cfg["test"]["ptx_last_block_len_bytes"],
                                triggers = cfg["test"]["triggers"],
                                trigger_block = cfg["test"]["trigger_block"],
                                protocol = cfg["target"]["protocol"],
                                port = cfg["target"].get("port"))
    logger.info(f"Setting up capture {capture_cfg.capture_mode} batch={capture_cfg.batch_mode}...")

    # Open communication with target.
    ot_aes_gcm, ot_prng, ot_trig = establish_communication(target, capture_cfg)

    # Configure cipher.
    device_id = configure_cipher(cfg, capture_cfg, ot_aes_gcm, ot_prng)

    # Configure trigger source.
    # 0 for HW, 1 for SW.
    trigger_source = 1
    if "hw" in cfg["target"].get("trigger"):
        trigger_source = 0
    ot_trig.select_trigger(trigger_source)

    # Capture traces.
    capture(scope, ot_aes_gcm, capture_cfg, project, target)

    # Print plot.
    print_plot(project, cfg, args.project)

    # Save metadata.
    metadata = {}
    metadata["device_id"] = device_id
    metadata["datetime"] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    metadata["cfg"] = cfg
    metadata["num_samples"] = scope.scope_cfg.num_samples
    metadata["offset_samples"] = scope.scope_cfg.offset_samples
    metadata["sampling_rate"] = scope.scope_cfg.sampling_rate
    metadata["num_traces"] = capture_cfg.num_traces
    metadata["scope_gain"] = scope.scope_cfg.scope_gain
    metadata["cfg_file"] = str(args.cfg)
    # Store bitstream information.
    metadata["fpga_bitstream_path"] = cfg["target"].get("fpga_bitstream")
    if cfg["target"].get("fpga_bitstream") is not None:
        metadata["fpga_bitstream_crc"] = helpers.file_crc(cfg["target"]["fpga_bitstream"])
    if args.save_bitstream:
        metadata["fpga_bitstream"] = helpers.get_binary_blob(cfg["target"]["fpga_bitstream"])
    # Store binary information.
    metadata["fw_bin_path"] = cfg["target"]["fw_bin"]
    metadata["fw_bin_crc"] = helpers.file_crc(cfg["target"]["fw_bin"])
    if args.save_binary:
        metadata["fw_bin"] = helpers.get_binary_blob(cfg["target"]["fw_bin"])
    # Store user provided notes.
    metadata["notes"] = args.notes
    # Store the Git hash.
    metadata["git_hash"] = helpers.get_git_hash()
    # Write metadata into project database.
    project.write_metadata(metadata)

    # Finale the capture.
    project.finalize_capture(capture_cfg.num_traces)
    # Save and close project.
    project.save()


if __name__ == "__main__":
    main()
