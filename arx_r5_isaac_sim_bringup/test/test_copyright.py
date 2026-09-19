# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the ROS 2 copyright check."""

from pathlib import Path

from ament_copyright.main import main

import pytest


@pytest.mark.copyright
@pytest.mark.linter
def test_copyright():
    """Check source copyright markers."""
    package_root = Path(__file__).resolve().parents[1]
    assert main(argv=[str(package_root)]) == 0
