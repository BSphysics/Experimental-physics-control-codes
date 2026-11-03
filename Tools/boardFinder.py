import serial
import serial.tools.list_ports

def get_serial_connection(baud=115200, timeout=1):
    """
    Find and return a serial connection to an Arduino or clone.
    Searches all /dev/ttyUSB* and /dev/ttyACM* devices.
    """

#     # Known Arduino VID:PID pairs and common USB-serial bridges
#     known_ids = {
#         (0x2341, 0x0043),  # Arduino Uno
#         (0x2341, 0x0010),  # Arduino Mega 2560
#         (0x2A03, 0x0043),  # Arduino Uno (older vendor ID)
#         (0x1A86, 0x7523),  # CH340/CH341 (common clones)
#         (0x10C4, 0xEA60),  # CP210x USB-Serial
#         (0x1A2E, 0x7523),  # CH340 variant (your board)
#     }
    
    known_ids = {
        (0x2341, 0x0043): "Arduino Uno",
        (0x2341, 0x0010): "Arduino Mega 2560",
        (0x2A03, 0x0043): "Arduino Uno (older vendor ID)",
        (0x1A86, 0x7523): "Sparkfun Redboard",
        (0x10C4, 0xEA60): "CP210x USB-Serial",
        (0x1A2E, 0x7523): "CH340 variant",  
    }

    ports = serial.tools.list_ports.comports()
    if not ports:
        raise IOError("No serial devices found!")

    for port in ports:
        if port.vid is not None and port.pid is not None:
            if (port.vid, port.pid) in known_ids:
                #print(port.pid)
                print(f"✅ Using {port.device} ({known_ids[(port.vid, port.pid)]})")
                return serial.Serial(port.device, baudrate=baud, timeout=timeout)
                

        # Fallback: description match
        if ("Arduino" in port.description) or ("USB Serial" in port.description) or ("ttyACM" in port.device):
            return serial.Serial(port.device, baudrate=baud, timeout=timeout)

    raise IOError("No Arduino-compatible device found!")
