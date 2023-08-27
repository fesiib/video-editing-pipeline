import re

class Timecode:
    def __init__(self, timecode):
        self.timecode = timecode
        self.split()

    def split(self):
        parts = self.timecode.split(':')
        print(parts)
        hours, minutes, seconds = 0, 0, 0

        if len(parts) == 3:
            hours, minutes, seconds = float(parts[0]), float(parts[1]), float(parts[2])
        elif len(parts) == 2:
            minutes, seconds = float(parts[0]), float(parts[1])
        elif len(parts) == 1:
            seconds = float(parts[0])
        self.time = {
            "hours": hours,
            "minutes": minutes,
            "seconds": seconds
        }
    '''
    Method to approximate the frame id of the provided timecode
    '''
    def approximate_frame(self, fps):
        return self.total_seconds() // fps

    def __sub__(self, other):
        return

    def total_seconds(self): return self.time["hours"] * 3600 + self.time["minutes"] * 60 + self.time["seconds"]

    def __lt__(self, other): return self.total_seconds() < other.total_seconds()
    
    def __gt__(self, other): return self.total_seconds() > other.total_seconds()
    
    def __eq__(self, other): return self.total_seconds() == other.total_seconds()



        
