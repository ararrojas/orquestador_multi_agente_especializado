"""Límites globales (condiciones de parada contra el "supervisor infinito")."""

import os

from dotenv import load_dotenv

load_dotenv()

MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))                 # decisiones totales del supervisor
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))         # rondas de refinamiento tras validar
MAX_CALLS_PER_AGENT = int(os.getenv("MAX_CALLS_PER_AGENT", "3"))
