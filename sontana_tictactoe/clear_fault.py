# import kortex_api.autogen as k_api
# # Create API objects
# transport = k_api.TransportClientTcp()
# router = k_api.RouterClient(transport)
# transport.connect("robot_ip", k_api.DEFAULT_PORT)

# # Create session
# session_info = k_api.Session.CreateSessionInfo()
# session_info.username = "admin"
# session_info.password = "admin"
# session_info.session_inactivity_timeout = 60000
# session_manager = k_api.SessionManager(router)
# session_manager.CreateSession(session_info)

# # Get safety information
# device_config = k_api.DeviceConfig(router)
# safety_handle = k_api.Common.SafetyHandle()
# safety_info = device_config.GetSafetyInformation(safety_handle)

# # Check fault flags
# fault_flags = safety_info.fault_flags

# # Clear specific safety status
# device_config.ClearSafetyStatus(safety_handle)

# # Clear all safety statuses
# device_config.ClearAllSafetyStatus()

#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2018 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import os
from collections.abc import MutableMapping
from collections.abc import MutableSequence
import collections
collections.MutableMapping = collections.abc.MutableMapping
collections.MutableSequence = collections.abc.MutableSequence
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient


def main():
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()
    
    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        # Create required services
        #device_manager = DeviceManagerClient(router)
        device_config = DeviceConfigClient(router)

        # Example core
        #example_routed_device_config(device_manager, device_config)
        device_config.ClearAllSafetyStatus()

if __name__ == "__main__":
    main()
    