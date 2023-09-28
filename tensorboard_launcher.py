from libraries import *
import utilities as ut

import win32api

SPACEBAR = 0x20


#CKP_DIR = "./Data/Models/lightning_logs/"
CKP_DIR = "./Data/docker_output/lightning_logs/"


tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', CKP_DIR, '--port', '8088'])
url = tb.launch()


while True:

    if (win32api.GetAsyncKeyState(SPACEBAR)&0x8001 > 0):
            print("stop!")
            break