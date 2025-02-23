# Import thrading for non-blocking user inputs
import threading

# General get character class for wasd movements
class _getch_wasd(threading.Thread):
    def __init__(self):
        super().__init__(name='droneSimKeyboard')
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()
        self.droneSimMostRecentKey = None
        self.daemon = True
        self.hasNewPress = False
        self.killed = False

    # Used by the thread.start() 
    def run(self):
        while True:
            keyPress = self.impl()
            if keyPress == 'q':
                self.doKill()
                print('[ INFO ] [_getch_wasd] stopped.')
                break
            elif keyPress in ['w', 'a', 's', 'd']:
                self.droneSimMostRecentKey = keyPress
                self.hasNewPress = True
    
    def dummy(self):
        return 'q'
    
    # Called by modified controller to get latest key, avoids runaway inputs
    def getKey(self):
        if self.hasNewPress:
            out = self.droneSimMostRecentKey
            self.hasNewPress = False
        else:
            out = None
        return out

    # Terminate run() implicitly 
    def doKill(self):
        # print('_Getch killed.')
        self.impl = self.dummy
        self.hasNewPress = False
        self.droneSimMostRecentKey = None
        self.killed = True


# Get character class of Unix systems
class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

# Get character class of Windows
class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch().decode('utf8')


