import os
from seacharts.enc import ENC

config_path = os.path.join('src', 'corridor_opt', 'config', 'trondelag_1.yaml')
enc = ENC(config_path)
enc.display.start()
enc.display.show()