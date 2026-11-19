"""주변 무선 신호 기록기 (4중 검사의 4번째 겹, 블루투스와 와이파이 감시의 시제품).

일정 간격으로 와이파이(접속 지점)와 블루투스 저전력(BLE) 광고를 받아 CSV로 남긴다.
받기만 하고 보내지 않는다(전파 차단이나 방해 없음).
기기 주소와 이름은 원문을 저장하지 않고 소금값을 섞은 해시(8자리)로만 남긴다.

실행: .venv/Scripts/python tools/rf_scan.py --label baseline --minutes 2 --every 15
출력: data/rf/scan_<시각>_<label>.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import ctypes
import ctypes.wintypes as wt
import datetime as dt
import hashlib
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "rf"
OUT.mkdir(parents=True, exist_ok=True)
SALT_FILE = OUT / ".salt"
if not SALT_FILE.exists():
    SALT_FILE.write_text(secrets.token_hex(16), encoding="utf-8")
SALT = SALT_FILE.read_text(encoding="utf-8").strip()


def h(x: str) -> str:
    return hashlib.sha256((SALT + x.lower()).encode()).hexdigest()[:8] if x else ""


# ---------------------------------------------------------------- 와이파이
class GUID(ctypes.Structure):
    _fields_ = [("Data1", wt.DWORD), ("Data2", wt.WORD), ("Data3", wt.WORD), ("Data4", ctypes.c_ubyte * 8)]


class WLAN_INTERFACE_INFO(ctypes.Structure):
    _fields_ = [("InterfaceGuid", GUID), ("strInterfaceDescription", ctypes.c_wchar * 256), ("isState", wt.DWORD)]


class WLAN_INTERFACE_INFO_LIST(ctypes.Structure):
    _fields_ = [("dwNumberOfItems", wt.DWORD), ("dwIndex", wt.DWORD), ("InterfaceInfo", WLAN_INTERFACE_INFO * 1)]


def wifi_force_scan() -> None:
    """Windows가 쌓아 둔 목록 대신 새로 훑게 한다."""
    try:
        api = ctypes.windll.wlanapi
        hnd, ver = wt.HANDLE(), wt.DWORD()
        if api.WlanOpenHandle(2, None, ctypes.byref(ver), ctypes.byref(hnd)) != 0:
            return
        plist = ctypes.POINTER(WLAN_INTERFACE_INFO_LIST)()
        if api.WlanEnumInterfaces(hnd, None, ctypes.byref(plist)) == 0:
            n = plist.contents.dwNumberOfItems
            arr = ctypes.cast(ctypes.addressof(plist.contents.InterfaceInfo),
                              ctypes.POINTER(WLAN_INTERFACE_INFO * n)).contents
            for info in arr:
                api.WlanScan(hnd, ctypes.byref(info.InterfaceGuid), None, None, None)
            api.WlanFreeMemory(plist)
        api.WlanCloseHandle(hnd, None)
    except Exception as e:  # 와이파이 장치가 없을 때
        print("와이파이 재검색 실패:", e)


def wifi_read() -> list[dict]:
    out = subprocess.run(["netsh", "wlan", "show", "networks", "mode=bssid"], capture_output=True,
                         text=True, encoding="utf-8", errors="ignore").stdout
    rows, ssid, auth = [], "", ""
    cur: dict | None = None
    for line in out.splitlines():
        s = line.strip()
        m = re.match(r"^SSID \d+ : ?(.*)$", s)
        if m:
            ssid, auth = m.group(1), ""
            continue
        if s.startswith("Authentication") or s.startswith("인증"):
            auth = s.split(":", 1)[1].strip()
            continue
        m = re.match(r"^BSSID \d+\s*: (\S+)", s)
        if m:
            cur = {"kind": "wifi", "dev": h(m.group(1)), "ssid_hash": h(ssid), "ssid_hidden": int(ssid == ""),
                   "auth": auth}
            rows.append(cur)
            continue
        if cur is None or ":" not in s:
            continue
        k, v = [x.strip() for x in s.split(":", 1)]
        if k in ("Signal", "신호"):
            pct = int(re.sub(r"\D", "", v) or 0)
            cur["signal_pct"], cur["rssi_dbm"] = pct, round(pct / 2 - 100, 1)  # Windows 신호%를 dBm으로 어림
        elif k in ("Radio type", "무선 송수신 장치 유형"):
            cur["radio"] = v
        elif k in ("Band", "밴드"):
            cur["band"] = v
        elif k in ("Channel", "채널"):
            cur["channel"] = v
        elif k in ("Connected Stations", "연결되는 스테이션"):
            cur["stations"] = v
        elif k in ("Channel Utilization", "채널 사용률"):
            cur["chan_util"] = v
    return rows


# ---------------------------------------------------------------- 블루투스 저전력
async def ble_read(seconds: float) -> list[dict]:
    from bleak import BleakScanner
    seen: dict[str, dict] = {}

    def cb(device, adv):
        rec = seen.setdefault(device.address, {"kind": "ble", "dev": h(device.address), "rssi_list": []})
        rec["rssi_list"].append(adv.rssi)
        rec["mfr"] = ",".join(f"0x{k:04X}" for k in sorted(adv.manufacturer_data)) or rec.get("mfr", "")
        rec["n_uuids"] = len(adv.service_uuids or [])
        rec["has_name"] = int(bool(adv.local_name or device.name))
        rec["name_hash"] = h(adv.local_name or device.name or "")
        rec["tx_power"] = adv.tx_power if adv.tx_power is not None else ""

    async with BleakScanner(detection_callback=cb):
        await asyncio.sleep(seconds)
    rows = []
    for r in seen.values():
        lst = r.pop("rssi_list")
        r["rssi_dbm"] = max(lst)
        r["n_adv"] = len(lst)
        rows.append(r)
    return rows


FIELDS = ["t", "round", "label", "kind", "dev", "rssi_dbm", "signal_pct", "ssid_hash", "ssid_hidden", "auth",
          "radio", "band", "channel", "stations", "chan_util", "mfr", "n_uuids", "has_name", "name_hash", "tx_power", "n_adv"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--minutes", type=float, default=2)
    ap.add_argument("--every", type=float, default=15, help="몇 초마다 한 번 훑을지")
    a = ap.parse_args()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT / f"scan_{stamp}_{a.label}.csv"
    end = time.time() + a.minutes * 60
    rnd = 0
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        while time.time() < end:
            rnd += 1
            t0 = time.time()
            wifi_force_scan()
            ble = asyncio.run(ble_read(min(8.0, a.every - 2)))
            wifi = wifi_read()
            now = dt.datetime.now().isoformat(timespec="seconds")
            for r in wifi + ble:
                w.writerow({"t": now, "round": rnd, "label": a.label, **r})
            f.flush()
            print(f"{now} {rnd}회: 와이파이 {len(wifi)}개, 블루투스 {len(ble)}개")
            time.sleep(max(0.0, a.every - (time.time() - t0)))
    print("저장:", path)


if __name__ == "__main__":
    main()
