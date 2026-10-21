# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""The ARX door environment shared by AutoData generation and policy evaluation.

``door`` builds it through Arena, ``assets`` packages a scene into its asset bundle,
``mdp``/``recorders``/``acceptance`` are its observations, recording and strict
physical acceptance, ``physics``/``compat`` pin PhysX settings and Lab API versions.
"""
