# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Communication interface for the HMAC SCA application on OpenTitan.

Communication with OpenTitan happens over the uJSON command interface.
"""
import json
import time
from typing import Optional


class OTHMAC:
    def __init__(self, target, protocol: str) -> None:
        self.target = target
        if protocol == "simpleserial":
            raise Exception("Only uJSON protocol is supported for this test.")

    def _ujson_hmac_sca_cmd(self):
        # TODO: without the delay, the device uJSON command handler program
        # does not recognize the commands. Tracked in issue #256.
        time.sleep(0.01)
        self.target.write(json.dumps("HmacSca").encode("ascii"))
        time.sleep(0.01)

    def init(self, icache_disable: bool, dummy_instr_disable: bool):
        """ Initializes HMAC on the target.
        Args:
            icache_disable: If true, disable the iCache. If false, use default config
                            set in ROM.
            dummy_instr_disable: If true, disable the dummy instructions. If false,
                                 use default config set in ROM.
        Returns:
            The device ID of the device.
        """
        # HmacSca command.
        self._ujson_hmac_sca_cmd()
        # Init command.
        self.target.write(json.dumps("Init").encode("ascii"))
        # Disable iCache / dummy instructions.
        time.sleep(0.01)
        data = {"icache_disable": icache_disable, "dummy_instr_disable": dummy_instr_disable}
        self.target.write(json.dumps(data).encode("ascii"))
        # Read back device ID from device.
        return self.read_response(max_tries=30)

    def single(self, msg: list[int], key: list[int], mask: list[int]):
        """ Start a single HMAC operation using the given message and key.
        Args:
            msg: The list containing the message.
            key: The key containing the message.
            mask: The mask used for blinding the key.
        """
        # HmacSca command.
        self._ujson_hmac_sca_cmd()
        # Single command.
        self.target.write(json.dumps("Single").encode("ascii"))
        # Key and mask payload.
        time.sleep(0.01)
        key_data = {"key": key, "mask": mask}
        self.target.write(json.dumps(key_data).encode("ascii"))
        # Message payload.
        time.sleep(0.01)
        msg_data = {"message": msg}
        self.target.write(json.dumps(msg_data).encode("ascii"))

    def fvsr_batch(self, key: list[int], mask: list[int], num_segments: int):
        """ Start num_segments HMAC operation in FvsR batch mode.
        Args:
            key: The key containing the message.
            mask: The mask used for blinding the key.
            num_segments: The number of iterations.
        """
        # HmacSca command.
        self._ujson_hmac_sca_cmd()
        # BatchFvsr command.
        self.target.write(json.dumps("BatchFvsr").encode("ascii"))
        # Key and mask payload.
        time.sleep(0.01)
        key_data = {"key": key, "mask": mask}
        self.target.write(json.dumps(key_data).encode("ascii"))
        # Number of iterations payload.
        time.sleep(0.05)
        num_it_data = {"num_enc": num_segments}
        self.target.write(json.dumps(num_it_data).encode("ascii"))

    def random_batch(self, num_segments: int):
        """ Start num_segments HMAC operations in random batch mode.
        Args:
            num_segments: The number of iterations.
        """
        # HmacSca command.
        self._ujson_hmac_sca_cmd()
        # BatchRandom command.
        self.target.write(json.dumps("BatchRandom").encode("ascii"))
        # Number of iterations payload.
        time.sleep(0.01)
        num_it_data = {"num_enc": num_segments}
        self.target.write(json.dumps(num_it_data).encode("ascii"))

    def read_tag(self):
        """ Read tag from OpenTitan HMAC.

        Returns:
            The received tag.
        """
        while True:
            read_line = str(self.target.readline())
            if "RESP_OK" in read_line:
                json_string = read_line.split("RESP_OK:")[1].split(" CRC:")[0]
                try:
                    tag = json.loads(json_string)["tag"]
                    return tag
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
