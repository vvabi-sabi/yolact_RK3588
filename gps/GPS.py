import time
import serial
# Import mavutil
from pymavlink import mavutil

from config import RK3588_CFG
from log import DefaultLogger

# Create the tg_bot's logger
logger = DefaultLogger("gps")


class FakeGPS:
    def __init__(
            self,
            port: str,
            lat: float = RK3588_CFG["fake_gps"]["debug_lat"],
            lon: float = RK3588_CFG["fake_gps"]["debug_lon"],
            alt: float = RK3588_CFG["fake_gps"]["debug_alt"]
    ):
        self._lat = lat
        self._lon = lon
        self._alt = alt
        self._port = port
        try:
            logger.info("FakeGPS Connecting...")
            # Create the connection
            self._master = mavutil.mavlink_connection(self._port)
            # Wait a heartbeat before sending commands
            self._master.wait_heartbeat()
            logger.info("FakeGPS Connection successful")
        except serial.SerialTimeoutException:
            logger.error(f"FakeGPS Connection failed: Timeout")
            raise SystemExit
        except serial.SerialException:
            logger.error(f"FakeGPS Connection failed: Could not find {port}")
            raise SystemExit
        except Exception as e:
            logger.error(f"FakeGPS Connection failed: {e}")
            raise SystemExit

    def setGPS(self, lat: float, lon: float, alt: float):
        self._lat = lat
        self._lon = lon
        self._alt = alt

    def getGPS(self):
        return self._lat, self._lon, self._alt

    def writeGPS(self):
        '''return gps_week and gps_week_ms for current time'''
        now = time.time()
        leapseconds = 18
        SEC_PER_WEEK = 7 * 86400
        epoch = 86400 * (10 * 365 + (1980 - 1969) / 4 + 1 + 6 - 2) - leapseconds
        epoch_seconds = int(now - epoch)
        week = int(epoch_seconds) // SEC_PER_WEEK
        t_ms = int(now * 1000) % 1000
        week_ms = (epoch_seconds % SEC_PER_WEEK) * 1000 + ((t_ms // 200) * 200)
        time_us = int(now * 1.0e6)
        self._master.mav.gps_input_send(
            time_us,  # Timestamp (micros since boot or Unix epoch)
            0,  # ID of the GPS for multiple GPS inputs
            # Flags indicating which fields to ignore (see GPS_INPUT_IGNORE_FLAGS enum).
            # All other fields must be provided.
            0,
            week_ms,  # GPS time (milliseconds from start of GPS week)
            week,  # GPS week number
            3,  # 0-1: no fix, 2: 2D fix, 3: 3D fix. 4: 3D with DGPS. 5: 3D with RTK
            int(self._lat * 1.0e7),  # Latitude (WGS84), in degrees * 1E7
            int(self._lon * 1.0e7),  # Longitude (WGS84), in degrees * 1E7
            self._alt,  # Altitude (AMSL, not WGS84), in m (positive for up)
            1,  # GPS HDOP horizontal dilution of position in m
            1,  # GPS VDOP vertical dilution of position in m
            0,  # GPS velocity in m/s in NORTH direction in earth-fixed NED frame
            0,  # GPS velocity in m/s in EAST direction in earth-fixed NED frame
            0,  # GPS velocity in m/s in DOWN direction in earth-fixed NED frame
            0.4,  # GPS speed accuracy in m/s
            1.0,  # GPS horizontal accuracy in m
            1.0,  # GPS vertical accuracy in m
            7  # Number of satellites visible.
        )

    def get_compass_data(self):
        gyro = [0] * 3

        # Getting gyro data
        msg = self._master.recv_match(type='ATTITUDE', blocking=True)
        gyro[0] = msg.roll
        gyro[1] = msg.pitch
        gyro[2] = msg.yaw

        # Getting compass and speed data
        msg = self._master.recv_match(type='VFR_HUD', blocking=True)
        heading = msg.heading
        groundspeed = msg.groundspeed

        return gyro, heading, groundspeed


class GPSMod:
    def __init__(
            self,
            port: str,
            baudrate: int = RK3588_CFG["fake_gps"]["baudrate"],
            lat: float = RK3588_CFG["fake_gps"]["debug_lat"],
            lon: float = RK3588_CFG["fake_gps"]["debug_lon"],
            alt: float = RK3588_CFG["fake_gps"]["debug_alt"]
    ):
        self._port = port
        self._baudrate = baudrate
        self._lat = lat
        self._lon = lon
        self._alt = alt
        self._correct = False
        try:
            logger.info("GPSmod connecting...")
            self._ser = serial.Serial(self._port, self._baudrate, timeout=None)
            logger.info("GPSmod Connection successful")
        except serial.SerialException:
            logger.error(f"GPSmod Connection failed: Could not find {port}")
            raise SystemExit
        except Exception as e:
            logger.error(f"GPSmod Connection failed: {e}")
            raise SystemExit

    def getGPS(self):
        line = self._ser.readline()
        if line.startswith(b'$GPGLL'): # если строка начинается с $GPGGA
            data = line.split(b',') # разбиваем строку на массив данных
            # Если корректный
            if data[6] == b'A':
                self._correct = True
                self._lat = float(data[1]) / 100  # широта
                self._lon = float(data[3]) / 100  # долгота
                # self._alt = float(data[5])  # высота над уровнем моря
            else:
                self._correct = False
        return self._correct, self._lat, self._lon, self._alt