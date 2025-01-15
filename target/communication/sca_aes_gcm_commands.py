# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for the AES-GCM SCA application on OpenTitan.

Communication with OpenTitan happens over the uJSON command interface.
"""
import json
import time
from typing import Optional


class OTAESGCM:
    def __init__(self, target, protocol: str) -> None:
        self.target = target
        self.simple_serial = True
        if protocol == "ujson":
            self.simple_serial = False

    def _ujson_aes_sca_cmd(self):
        # TODO: without the delay, the device uJSON command handler program
        # does not recognize the commands. Tracked in issue #256.
        time.sleep(0.01)
        self.target.write(json.dumps("AesSca").encode("ascii"))

    def init(self, fpga_mode_bit: int):
        """ Initializes AES on the target.
        Args:
            fpga_mode_bit: Indicates whether FPGA specific AES test is started.
        Returns:
            The device ID of the device.
        """
        if not self.simple_serial:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # Init the AES core.
            self.target.write(json.dumps("Init").encode("ascii"))
            # FPGA mode.
            time.sleep(0.01)
            fpga_mode = {"fpga_mode": fpga_mode_bit}
            self.target.write(json.dumps(fpga_mode).encode("ascii"))
            # Read back device ID from device.
            return self.read_response(max_tries=30)

    def fvsr_batch(self, num_batches: int, triggers: list[bool], trigger_block: int,
                  iv_len_bytes: int, iv: list[int], key_len_bytes: int, key: list[int],
                  num_aad_blocks: int, num_ptx_blocks: int, aad, last_aad_len,
                  ptx, last_ptx_len):
        """ Configure FvsR batch and start the operations.
        Args:
            num_batches: The number of batches we are starting.
            triggers: Boolean array. If set, trigger.
            trigger_block: Trigger at which block?
            iv_len_bytes: Number of bytes for the IV.
            iv: The current IV.
            key_iv_len_bytes: Number of bytes for the key.
            key: The current key.
            num_aad_blocks: Number of AAD blocks we are sending.
            num_ptx_blocks: Number of PTX blocks we are sending.
            aad: Array of AAD blocks.
            last_aad_len: Length of last AAD block.
            ptx: Array of PTX blocks.
            last_ptx_len: Length of last PTX block.
        """
        if self.simple_serial:
            self.target.write(cmd="f", data=bytearray(key))
        else:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # GcmFvsrBatch command.
            self.target.write(json.dumps("GcmFvsrBatch").encode("ascii"))
            time.sleep(0.01)
            # Number of batch operations we are starting.
            data = {"num_batch_ops": num_batches}
            self.target.write(json.dumps(data).encode("ascii"))
            time.sleep(0.01)
            # Trigger configuration.
            data = {"triggers": triggers, "block": trigger_block}
            self.target.write(json.dumps(data).encode("ascii"))
            time.sleep(0.01)
            # Transmit IV.
            data = {"block": iv, "num_valid_bytes": 12}
            self.target.write(json.dumps(data).encode("ascii"))
            time.sleep(0.01)
            # Transmit key.
            data = {"key": key, "key_length": 16}
            self.target.write(json.dumps(data).encode("ascii"))
            time.sleep(0.01)
            # Number of AAD blocks.
            data = {"num_blocks": num_aad_blocks}
            self.target.write(json.dumps(data).encode("ascii"))
            time.sleep(0.01)
            # Number of PTX blocks.
            data = {"num_blocks": num_ptx_blocks}
            self.target.write(json.dumps(data).encode("ascii"))
            time.sleep(0.01)
            # Transmit num_aad_blocks AADs.
            for i in range(0, num_aad_blocks):
                num_valid_bytes = 16
                if i == num_aad_blocks - 1:
                    num_valid_bytes = last_aad_len
                data = {"block": aad[i], "num_valid_bytes": num_valid_bytes}
                self.target.write(json.dumps(data).encode("ascii"))
                time.sleep(0.01)
            # Transmit num_ptx_blocks PTXs.
            for i in range(0, num_ptx_blocks):
                num_valid_bytes = 16
                if i == num_ptx_blocks - 1:
                    num_valid_bytes = last_ptx_len
                data = {"block": ptx[i], "num_valid_bytes": num_valid_bytes}
                self.target.write(json.dumps(data).encode("ascii"))
                time.sleep(0.01)

    def seed_lfsr(self, seed):
        """ Seed the LFSR.
        Args:
            seed: The 4-byte seed.
        """
        if self.simple_serial:
            self.target.write(cmd="l", data=seed)
        else:
            # AesSca command.
            self._ujson_aes_sca_cmd()
            # SeedLfsr command.
            self.target.write(json.dumps("SeedLfsr").encode("ascii"))
            # Seed payload.
            time.sleep(0.01)
            seed_data = {"seed": [x for x in seed]}
            self.target.write(json.dumps(seed_data).encode("ascii"))

    def read_tag(self, len_bytes: Optional[int] = 16):
        """ Read tag from OpenTitan AES.
        Args:
            len_bytes: Number of bytes to read.

        Returns:
            The received ciphertext.
        """
        if self.simple_serial:
            response_byte = self.target.read("r", len_bytes, ack=False)
            # Convert response into int array.
            return [x for x in response_byte]
        else:
            while True:
                read_line = str(self.target.readline())
                if "RESP_OK" in read_line:
                    json_string = read_line.split("RESP_OK:")[1].split(" CRC:")[0]
                    try:
                        ciphertext = json.loads(json_string)["block"]
                        return ciphertext[0:len_bytes]
                    except Exception:
                        pass  # noqa: E302

    def read_response(self, max_tries: Optional[int] = 1) -> str:
        """ Read response from AES SCA framework.
        Args:
            max_tries: Maximum number of attempts to read from UART.

        Returns:
            The JSON response of OpenTitan.
        """
        it = 0
        while it != max_tries:
            read_line = str(self.target.readline())
            if "RESP_OK" in read_line:
                return read_line.split("RESP_OK:")[1].split(" CRC:")[0]
            it += 1
        return ""
