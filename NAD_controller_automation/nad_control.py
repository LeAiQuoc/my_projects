import broadlink
import struct

# ─── CONFIG ───────────────────────────────────────────────
BROADLINK_IP  = "192.168.3.12"   # <-- replace with your RM4 Mini IP
BROADLINK_MAC = "34:8e:89:2d:c1:93"  # <-- replace with your RM4 Mini MAC
# ──────────────────────────────────────────────────────────

def pronto_to_broadlink(pronto: list[int]) -> bytes:
    """Convert Pronto hex code to Broadlink IR format."""
    if pronto[0] != 0:
        raise ValueError("Only learned Pronto codes (type 0x0000) supported")

    frequency = pronto[1]
    hz = 1000000 / (frequency * 0.241246)
    period = 1000000 / hz  # microseconds per cycle

    seq1_len = pronto[2]
    seq2_len = pronto[3]
    data = pronto[4:]

    pulses = []
    for i in range((seq1_len + seq2_len) * 2):
        pulse_cycles = data[i]
        pulse_us = int(round(pulse_cycles * period))
        pulses.append(pulse_us)

    # Convert to Broadlink format (units of 32.84 us)
    broadlink_pulses = [int(round(p / 32.84)) for p in pulses]

    # Build packet
    packet = bytearray()
    packet.append(0x26)  # IR format
    packet.append(0x00)  # repeat count
    packet += struct.pack('<H', len(broadlink_pulses))

    for p in broadlink_pulses:
        if p > 255:
            packet.append(0x00)
            packet += struct.pack('>H', p)
        else:
            packet.append(p)

    # Pad to multiple of 16
    while len(packet) % 16 != 0:
        packet.append(0x00)

    return bytes(packet)


# NAD T747 Pronto codes
VIDEO2_PRONTO = [
    0x0000,0x006d,0x0022,0x0002,0x0156,0x00ab,0x0015,0x0040,0x0015,0x0040,
    0x0015,0x0040,0x0015,0x0015,0x0015,0x0015,0x0015,0x0015,0x0015,0x0015,
    0x0015,0x0040,0x0015,0x0015,0x0015,0x0015,0x0015,0x0040,0x0015,0x0040,
    0x0015,0x0040,0x0015,0x0040,0x0015,0x0040,0x0015,0x0015,0x0015,0x0015,
    0x0015,0x0015,0x0015,0x0015,0x0015,0x0015,0x0015,0x0015,0x0015,0x0015,
    0x0015,0x0040,0x0015,0x0040,0x0015,0x0040,0x0015,0x0040,0x0015,0x0040,
    0x0015,0x0040,0x0015,0x0040,0x0015,0x0040,0x0015,0x0015,0x0015,0x0015,
    0x0015,0x05d8,0x0156,0x0055,0x0015,0x0e48
]

VIDEO3_PRONTO = [
    0x0000,0x006d,0x0022,0x0002,0x0156,0x00ab,0x0015,0x0040,0x0015,0x0040,
    0x0015,0x0040,0x0015,0x0015,0x0015,0x0015,0x0015,0x0015,0x0015,0x0015,
    0x0015,0x0040,0x0015,0x0015,0x0015,0x0015,0x0015,0x0040,0x0015,0x0040,
    0x0015,0x0040,0x0015,0x0040,0x0015,0x0040,0x0015,0x0015,0x0015,0x0040,
    0x0015,0x0015,0x0015,0x0015,0x0015,0x0015,0x0015,0x0015,0x0015,0x0015,
    0x0015,0x0040,0x0015,0x0040,0x0015,0x0015,0x0015,0x0040,0x0015,0x0040,
    0x0015,0x0040,0x0015,0x0040,0x0015,0x0040,0x0015,0x0015,0x0015,0x0015,
    0x0015,0x05d8,0x0156,0x0055,0x0015,0x0e48
]

def get_device():
    device = broadlink.rm4pro(
        host=(BROADLINK_IP, 80),
        mac=bytes.fromhex(BROADLINK_MAC.replace(':', '')),
        devtype=0x6026
    )
    device.auth()
    return device

def send_video2():
    print("Switching NAD to Video 2 (PC)")
    device = get_device()
    device.send_data(pronto_to_broadlink(VIDEO2_PRONTO))

def send_video3():
    print("Switching NAD to Video 3 (TV)")
    device = get_device()
    device.send_data(pronto_to_broadlink(VIDEO3_PRONTO))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python nad_control.py [video2|video3]")
        sys.exit(1)
    
    cmd = sys.argv[1].lower()
    if cmd == "video2":
        send_video2()
    elif cmd == "video3":
        send_video3()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)