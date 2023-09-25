import re

class Timecode:
    def __init__(self, timecode):
        self.timecode = timecode
        self.split()

    def get_parts(self):
        if isinstance(self.timecode, str):
            parts = self.timecode.split(':')
            while len(parts) < 3:
                parts = ['0'] + parts
            return parts
        elif isinstance(self.timecode, list):
            parts = self.timecode
            while len(parts) < 3:
                parts = [0] + parts
            return parts
        elif isinstance(self.timecode, float):
            return [self.timecode//3600, (self.timecode%3600)//60, self.timecode%60]
        elif isinstance(self.timecode, int):
            return [self.timecode//3600, (self.timecode%3600)//60, self.timecode%60]
        elif isinstance(self.timecode, Timecode):
            return [self.time["hours"], self.time["minutes"], self.time["seconds"]]
        else:
            return [0, 0, 0]

    def split(self):
        #print("time -> ", self.timecode)
        parts = self.get_parts() 
        #parts = self.timecode.split(':')
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
    
    @classmethod
    def convert_sec_to_timecode(cls, seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(secs))
    
    def convert_timecode_to_sec(self):
        return self.time["hours"] * 3600 + self.time["minutes"] * 60 + self.time["seconds"] 

    def __str__(self):
        return "{:02d}:{:02d}:{:02d}".format(int(self.time["hours"]), int(self.time["minutes"]), int(self.time["seconds"]))
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

    def is_timecode(timecode):
        return (re.match(r'^\d{2}:\d{2}:\d{2}$', timecode) is not None
            or re.match(r'^\d{1}:\d{2}:\d{2}$', timecode) is not None 
            or re.match(r'^\d{2}:\d{2}$', timecode) is not None 
            or re.match(r'^\d{1}:\d{2}$', timecode) is not None 
            or re.match(r'^\d{2}$', timecode) is not None
            or re.match(r'^\d{1}$', timecode) is not None
        )


        
